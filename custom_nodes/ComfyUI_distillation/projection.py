"""
Cross-Architecture Projection Module for ComfyUI Distillation

This module provides SVD-based projection utilities for cross-architecture distillation:
- Dimension projection between different architectures
- Student subspace alignment
- Teacher-to-student basis transformation
- LoRA extraction from core deltas
"""

from typing import Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .svd_algorithms import GPUAcceleratedSVD, SVDConfig, compute_adaptive_rank
from .fp8_utils import is_float8_dtype, resolve_dequant_dtype


@dataclass
class ProjectionConfig:
    """Configuration for cross-architecture projection."""
    projection_rank: int = 256
    projection_min_rank: int = 8
    projection_adaptive_rank: bool = True
    projection_energy_threshold: float = 0.99
    projection_randomized_iter: int = 2
    projection_randomized_oversamples: int = 8

    # Subspace delta controls
    subspace_delta: str = "off"  # off | auto | diag | full
    subspace_offdiag_threshold: float = 0.55
    subspace_denom: str = "core"  # core | full
    subspace_norm_match: bool = True

    # Adaptive rank controls
    use_adaptive_rank: bool = True
    energy_threshold: float = 0.95
    min_rank: int = 8
    max_rank: int = 256


def _resize_rows_linear(mat: torch.Tensor, target_rows: int) -> torch.Tensor:
    """Resize a 2D matrix along rows using 1D linear interpolation."""
    if mat.dim() != 2:
        return mat
    if mat.shape[0] == target_rows:
        return mat
    if target_rows <= 0:
        return mat[:0]

    # Treat columns as channels, interpolate over rows
    x = mat.transpose(0, 1).unsqueeze(0)  # (1, cols, rows)
    y = F.interpolate(x, size=target_rows, mode="linear", align_corners=False)
    return y.squeeze(0).transpose(0, 1).contiguous()


def _resize_cols_linear(mat: torch.Tensor, target_cols: int) -> torch.Tensor:
    """Resize a 2D matrix along cols using 1D linear interpolation."""
    if mat.dim() != 2:
        return mat
    if mat.shape[1] == target_cols:
        return mat
    if target_cols <= 0:
        return mat[:, :0]

    # Treat rows as channels, interpolate over cols
    x = mat.unsqueeze(0)  # (1, rows, cols)
    y = F.interpolate(x, size=target_cols, mode="linear", align_corners=False)
    return y.squeeze(0).contiguous()


def _safe_output_dtype(
    src_dtype: torch.dtype,
    device: torch.device,
    ref_dtype: Optional[torch.dtype] = None
) -> torch.dtype:
    """Choose safe output dtype, avoiding float8."""
    if not is_float8_dtype(src_dtype):
        return src_dtype
    if ref_dtype is not None and not is_float8_dtype(ref_dtype) and "float" in str(ref_dtype):
        return ref_dtype
    return resolve_dequant_dtype("auto", device)


def svd_projection(
    src: torch.Tensor,
    target_shape: Tuple[int, ...],
    svd_engine: GPUAcceleratedSVD,
    config: ProjectionConfig,
    device: torch.device,
    ref: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Project a tensor to target_shape using SVD-based methods.

    Student-subspace alignment (v3.7.6 fix):
      The legacy (pool/crop/interpolate) projection loses too much information.
      If `ref` (student tensor) is provided and both `src/ref` are 2D matrices, we:
        1) Compute a truncated SVD of teacher: Wt ≈ Ut diag(St) Vt^T
        2) Compute a truncated SVD of student: Ws ≈ Us diag(Ss) Vs^T
        3) Build teacher low-rank factors Bt, At (PiSSA-style: sqrt split)
        4) Resize factors to the student's row/col sizes
        5) Project the resized teacher into the student's singular subspaces

    Args:
        src: Source tensor to project
        target_shape: Target shape tuple
        svd_engine: SVD computation engine
        config: Projection configuration
        device: Target device
        ref: Optional reference (student) tensor for subspace alignment

    Returns:
        Projected tensor with target_shape
    """
    if tuple(src.shape) == tuple(target_shape):
        return src

    # Determine safe output dtype
    ref_dtype = ref.dtype if ref is not None and isinstance(ref, torch.Tensor) else None
    out_dtype = _safe_output_dtype(src.dtype, device, ref_dtype)

    # 1D vectors: linear interpolation
    if src.dim() == 1 and len(target_shape) == 1:
        out_len = int(target_shape[0])
        x = torch.nan_to_num(src.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if x.numel() == out_len:
            return x.to(dtype=out_dtype, device=src.device)
        x3 = x.view(1, 1, -1)
        y3 = F.interpolate(x3, size=out_len, mode="linear", align_corners=False)
        return y3.view(out_len).to(dtype=out_dtype, device=src.device)

    # 2D matrices: student-subspace alignment (preferred)
    if src.dim() == 2 and len(target_shape) == 2:
        tm, tn = int(target_shape[0]), int(target_shape[1])
        m, n = int(src.shape[0]), int(src.shape[1])

        k_cap = min(m, n, tm, tn, config.projection_rank)
        if k_cap <= 0:
            return torch.zeros((tm, tn), device=src.device, dtype=out_dtype)

        # Fast fallback: if no ref matrix is provided
        if ref is None or not isinstance(ref, torch.Tensor) or ref.dim() != 2:
            src_f = torch.nan_to_num(src.to(device).float(), nan=0.0, posinf=0.0, neginf=0.0)
            try:
                Ut, St, Vht = svd_engine.randomized_svd(
                    src_f, k_cap,
                    n_oversamples=config.projection_randomized_oversamples,
                    n_iter=config.projection_randomized_iter,
                )

                if config.projection_adaptive_rank:
                    k_t = compute_adaptive_rank(
                        St,
                        energy_threshold=config.projection_energy_threshold,
                        min_rank=config.projection_min_rank,
                        max_rank=k_cap,
                    )
                    k = max(1, min(k_t, k_cap))
                    Ut, St, Vht = Ut[:, :k], St[:k], Vht[:k, :]
                else:
                    k = k_cap

                sqrtS = torch.sqrt(torch.clamp(St, min=0.0)).to(dtype=Ut.dtype)
                Bt = Ut * sqrtS.unsqueeze(0)
                At = sqrtS.unsqueeze(1) * Vht

                Bt_rs = _resize_rows_linear(Bt, tm)
                At_rs = _resize_cols_linear(At, tn)

                out = Bt_rs @ At_rs
                return out.to(dtype=out_dtype, device=src.device)
            except Exception:
                # Last resort: bilinear interpolation
                dense = torch.nan_to_num(src.to(device).float(), nan=0.0, posinf=0.0, neginf=0.0)
                out = F.interpolate(
                    dense.unsqueeze(0).unsqueeze(0),
                    size=(tm, tn),
                    mode="bilinear",
                    align_corners=False
                )
                return out.squeeze(0).squeeze(0).to(dtype=out_dtype, device=src.device)

        # Ensure ref has the expected target shape
        ref = ref.to(device)
        if tuple(ref.shape) != (tm, tn):
            return svd_projection(src, (tm, tn), svd_engine, config, device, ref=None)

        # Compute truncated SVDs (teacher + student)
        src_f = torch.nan_to_num(src.to(device).float(), nan=0.0, posinf=0.0, neginf=0.0)
        ref_f = torch.nan_to_num(ref.to(device).float(), nan=0.0, posinf=0.0, neginf=0.0)

        try:
            Ut, St, Vht = svd_engine.randomized_svd(
                src_f, k_cap,
                n_oversamples=config.projection_randomized_oversamples,
                n_iter=config.projection_randomized_iter,
            )
            Us, Ss, Vhs = svd_engine.randomized_svd(
                ref_f, k_cap,
                n_oversamples=config.projection_randomized_oversamples,
                n_iter=config.projection_randomized_iter,
            )

            if config.projection_adaptive_rank:
                k_t = compute_adaptive_rank(
                    St,
                    energy_threshold=config.projection_energy_threshold,
                    min_rank=config.projection_min_rank,
                    max_rank=k_cap,
                )
                k_s = compute_adaptive_rank(
                    Ss,
                    energy_threshold=config.projection_energy_threshold,
                    min_rank=config.projection_min_rank,
                    max_rank=k_cap,
                )
                k = max(1, min(k_t, k_s, k_cap))
            else:
                k = k_cap

            Ut, St, Vht = Ut[:, :k], St[:k], Vht[:k, :]
            Us, Ss, Vhs = Us[:, :k], Ss[:k], Vhs[:k, :]

            # Teacher low-rank factors (PiSSA sqrt split)
            sqrtSt = torch.sqrt(torch.clamp(St, min=0.0)).to(dtype=Ut.dtype)
            Bt = Ut * sqrtSt.unsqueeze(0)
            At = sqrtSt.unsqueeze(1) * Vht

            # Resize teacher factors to student dims
            Bt_rs = _resize_rows_linear(Bt, tm)
            At_rs = _resize_cols_linear(At, tn)

            # Student right singular vectors
            Vs = Vhs.transpose(0, 1).contiguous()

            # core = Us^T (Bt_rs @ At_rs) Vs
            left = Us.transpose(0, 1) @ Bt_rs
            right = At_rs @ Vs
            core = left @ right

            out = Us @ core @ Vs.transpose(0, 1)
            return out.to(dtype=out_dtype, device=src.device)

        except Exception:
            dense = torch.nan_to_num(src.to(device).float(), nan=0.0, posinf=0.0, neginf=0.0)
            out = F.interpolate(
                dense.unsqueeze(0).unsqueeze(0),
                size=(tm, tn),
                mode="bilinear",
                align_corners=False
            )
            return out.squeeze(0).squeeze(0).to(dtype=out_dtype, device=src.device)

    # Generic fallback: flatten + interpolate + reshape
    tgt_numel = 1
    for d in target_shape:
        tgt_numel *= d

    x = torch.nan_to_num(src.float().view(1, 1, -1), nan=0.0, posinf=0.0, neginf=0.0)
    y = F.interpolate(x, size=tgt_numel, mode="linear", align_corners=False)
    out = y.view(*target_shape)
    return out.to(dtype=out_dtype, device=src.device)


def compute_student_svd_basis(
    student_mat: torch.Tensor,
    svd_engine: GPUAcceleratedSVD,
    config: ProjectionConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the student's truncated SVD basis (U, S, V) used as the common subspace.

    Args:
        student_mat: Student weight matrix
        svd_engine: SVD computation engine
        config: Projection configuration
        device: Target device

    Returns:
        Tuple of (U, S, V) where V = Vh^T
    """
    if student_mat.dim() != 2:
        raise ValueError("student_mat must be 2D")

    tm, tn = int(student_mat.shape[0]), int(student_mat.shape[1])
    k_cap = min(config.projection_rank, tm, tn)
    if k_cap <= 0:
        raise ValueError("projection_rank too small for matrix")

    ref_f = torch.nan_to_num(student_mat.to(device).float(), nan=0.0, posinf=0.0, neginf=0.0)
    U, S, Vh = svd_engine.randomized_svd(
        ref_f, k_cap,
        n_oversamples=config.projection_randomized_oversamples,
        n_iter=config.projection_randomized_iter,
    )

    k = k_cap
    if config.projection_adaptive_rank:
        k = compute_adaptive_rank(
            S,
            energy_threshold=config.projection_energy_threshold,
            min_rank=config.projection_min_rank,
            max_rank=k_cap,
        )
    k = max(1, min(k, int(S.numel()), k_cap))

    U = U[:, :k].contiguous()
    S = S[:k].contiguous()
    Vh = Vh[:k, :].contiguous()
    V = Vh.transpose(0, 1).contiguous()

    return U, S, V


def teacher_core_in_student_basis(
    teacher_mat: torch.Tensor,
    U_s: torch.Tensor,
    V_s: torch.Tensor,
    svd_engine: GPUAcceleratedSVD,
    config: ProjectionConfig,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute teacher's representation in the student's SVD basis.

    Computes: core_t = U_s^T * Wt_resized * V_s

    Args:
        teacher_mat: Teacher weight matrix
        U_s: Student left singular vectors
        V_s: Student right singular vectors (transposed Vh)
        svd_engine: SVD computation engine
        config: Projection configuration
        device: Target device

    Returns:
        Core representation (k, k) in float32
    """
    if teacher_mat.dim() != 2:
        raise ValueError("teacher_mat must be 2D")
    if U_s.dim() != 2 or V_s.dim() != 2:
        raise ValueError("U_s and V_s must be 2D")

    tm, tn = int(U_s.shape[0]), int(V_s.shape[0])
    k_s = int(U_s.shape[1])

    src_f = torch.nan_to_num(teacher_mat.to(device).float(), nan=0.0, posinf=0.0, neginf=0.0)
    r_cap = min(k_s, int(src_f.shape[0]), int(src_f.shape[1]))
    if r_cap <= 0:
        return torch.zeros((k_s, k_s), device=device, dtype=torch.float32)

    U_t, S_t, Vh_t = svd_engine.randomized_svd(
        src_f, r_cap,
        n_oversamples=config.projection_randomized_oversamples,
        n_iter=config.projection_randomized_iter,
    )

    r = r_cap
    if config.projection_adaptive_rank:
        r = compute_adaptive_rank(
            S_t,
            energy_threshold=config.projection_energy_threshold,
            min_rank=config.projection_min_rank,
            max_rank=r_cap,
        )
    r = max(1, min(r, int(S_t.numel()), r_cap))

    U_t = U_t[:, :r]
    S_t = S_t[:r]
    Vh_t = Vh_t[:r, :]

    sqrt_S = torch.sqrt(torch.clamp(S_t.float(), min=1e-10)).to(dtype=U_t.dtype)
    B_t = (U_t * sqrt_S.unsqueeze(0)).contiguous()
    A_t = (sqrt_S.unsqueeze(1) * Vh_t).contiguous()

    B_rs = _resize_rows_linear(B_t, tm)
    A_rs = _resize_cols_linear(A_t, tn)

    # core = U_s^T (B_rs A_rs) V_s = (U_s^T B_rs) (A_rs V_s)
    left = U_s.transpose(0, 1) @ B_rs
    right = A_rs @ V_s
    core = left @ right

    return core.to(dtype=torch.float32)


def extract_lora_from_core_delta(
    delta_core: torch.Tensor,
    U_s: torch.Tensor,
    V_s: torch.Tensor,
    rank: int,
    config: ProjectionConfig,
    out_dtype: torch.dtype,
    svd_engine: GPUAcceleratedSVD,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    """
    Extract LoRA matrices that respect student subspace.

    Projects delta into student subspace and extracts LoRA factors:
        dc = U_s^T * delta_core * V_s   (r x r)
        SVD(dc) -> Uc, S, Vc
        lora_B = U_s * (Uc * sqrt(S))
        lora_A = (sqrt(S) * Vc^T) * V_s^T

    Args:
        delta_core: Core delta in student basis
        U_s: Student left singular vectors
        V_s: Student right singular vectors
        rank: Target LoRA rank
        config: Projection configuration
        out_dtype: Output dtype
        svd_engine: SVD computation engine

    Returns:
        Tuple of (lora_A, lora_B, actual_rank)
    """
    try:
        U32 = U_s.to(dtype=torch.float32)
        V32 = V_s.to(dtype=torch.float32)
        D32 = delta_core.to(dtype=torch.float32)
        dc = U32.T @ D32 @ V32
    except Exception as e:
        print(f"    Core delta projection failed: {e}")
        return None, None, 0

    rcap = min(rank, dc.shape[0], dc.shape[1])
    if rcap <= 0:
        return None, None, 0

    Uc = S = Vh = None

    # Use SVD engine
    try:
        Uc, S, Vh = svd_engine.svd(dc, rcap)
    except Exception as e:
        print(f"    Core delta SVD engine failed: {e}")
        Uc = S = Vh = None

    # Fallback: CPU float64 full SVD
    if Uc is None or S is None or Vh is None:
        try:
            Uc_full, S_full, Vh_full = torch.linalg.svd(
                dc.detach().cpu().double(), full_matrices=False
            )
            Uc = Uc_full.to(device=dc.device, dtype=torch.float32)
            S = S_full.to(device=dc.device, dtype=torch.float32)
            Vh = Vh_full.to(device=dc.device, dtype=torch.float32)
        except Exception as e:
            print(f"    Core delta CPU SVD fallback failed: {e}")
            return None, None, 0

    S = torch.clamp(S, min=0.0)

    # Determine actual rank based on energy threshold
    try:
        if config.use_adaptive_rank and config.energy_threshold < 1.0:
            S2 = S * S
            total_energy = S2.sum()
            if total_energy > 0:
                cumulative_energy = torch.cumsum(S2, dim=0) / total_energy
                actual_rank = int(torch.searchsorted(cumulative_energy, config.energy_threshold).item()) + 1
            else:
                actual_rank = 1
            actual_rank = max(1, min(actual_rank, rcap))
        else:
            actual_rank = rcap
    except Exception as e:
        print(f"    Core delta rank selection failed: {e}")
        actual_rank = rcap

    Uc = Uc[:, :actual_rank]
    S = S[:actual_rank]
    Vh = Vh[:actual_rank, :]

    # Construct LoRA factors in original space
    try:
        sqrt_S = torch.sqrt(torch.clamp(S, min=0.0)).unsqueeze(0)
        lora_B = (U_s @ (Uc * sqrt_S)).to(dtype=out_dtype)
        lora_A = ((sqrt_S.T * Vh) @ V_s.T).to(dtype=out_dtype)
        return lora_A, lora_B, actual_rank
    except Exception as e:
        print(f"    Core delta LoRA reconstruction failed: {e}")
        return None, None, 0


def extract_lora_pissa(
    delta: torch.Tensor,
    rank: int,
    svd_engine: GPUAcceleratedSVD,
    config: ProjectionConfig,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    """
    Extract LoRA matrices using PiSSA (sqrt-split SVD factors).

    SVD improvements:
      1) Adaptive-rank selection measured against true Frobenius energy
      2) Rank grows geometrically until energy target is met

    Args:
        delta: Delta weight matrix
        rank: Initial target rank
        svd_engine: SVD computation engine
        config: Projection configuration

    Returns:
        Tuple of (lora_A, lora_B, actual_rank)
    """
    if delta.dim() != 2:
        return None, None, 0

    out_features, in_features = delta.shape
    min_dim = min(out_features, in_features)

    # Sanitize numerical pathologies
    delta = torch.nan_to_num(delta, nan=0.0, posinf=1e6, neginf=-1e6)

    delta_norm = torch.norm(delta.float())
    if delta_norm < 1e-10:
        return None, None, 0
    total_energy = delta_norm.float() ** 2

    max_rank = min(config.max_rank, min_dim)
    min_rank = max(1, min(config.min_rank, max_rank))

    # If adaptive rank is disabled, do single SVD
    if not config.use_adaptive_rank:
        actual_rank = max(min_rank, min(rank, max_rank))
        try:
            U, S, Vh = svd_engine.svd(delta, actual_rank)
        except Exception as e:
            print(f"  SVD failed: {type(e).__name__}")
            return None, None, 0

        if S.numel() == 0 or torch.all(S < 1e-10):
            return None, None, 0

        S = torch.clamp(S, min=0.0)
        sqrtS = torch.sqrt(S).to(dtype=U.dtype)
        lora_B = U[:, :actual_rank] * sqrtS.unsqueeze(0)
        lora_A = sqrtS.unsqueeze(1) * Vh[:actual_rank, :]
        return lora_A.float(), lora_B.float(), actual_rank

    # Adaptive-rank path
    energy_threshold = float(max(0.0, min(1.0, config.energy_threshold)))
    target_energy = total_energy * energy_threshold

    r_try = max(min_rank, min(rank, max_rank))
    U = S = Vh = None

    while True:
        try:
            U, S, Vh = svd_engine.svd(delta, r_try)
        except Exception as e:
            print(f"  SVD failed at rank {r_try}: {type(e).__name__}")
            return None, None, 0

        if S.numel() == 0 or torch.all(S < 1e-10):
            return None, None, 0

        s = torch.clamp(S.detach().float(), min=0.0)
        cum_energy = torch.cumsum(s ** 2, dim=0)

        hit = (cum_energy >= target_energy)
        if bool(hit.any()):
            r_needed = int(hit.nonzero(as_tuple=True)[0][0].item() + 1)
            actual_rank = max(min_rank, min(r_needed, max_rank))
            break

        if r_try >= max_rank:
            actual_rank = max_rank
            break

        # Grow rank geometrically
        import math
        r_next = max(r_try + 1, int(math.ceil(r_try * 2.0)))
        r_try = min(max_rank, r_next)

    U = U[:, :actual_rank]
    S = S[:actual_rank]
    Vh = Vh[:actual_rank, :]

    S = torch.clamp(S, min=0.0)
    sqrtS = torch.sqrt(S).to(dtype=U.dtype)
    lora_B = U * sqrtS.unsqueeze(0)
    lora_A = sqrtS.unsqueeze(1) * Vh

    return lora_A.float(), lora_B.float(), actual_rank


class CrossArchProjector:
    """
    High-level interface for cross-architecture projection operations.

    Provides unified methods for projecting weights between different architectures.
    """

    def __init__(
        self,
        device: torch.device,
        config: Optional[ProjectionConfig] = None,
        svd_config: Optional[SVDConfig] = None,
    ):
        self.device = device
        self.config = config or ProjectionConfig()

        from .svd_algorithms import GPUMemoryManager
        memory_manager = GPUMemoryManager(device)
        self.svd_engine = GPUAcceleratedSVD(
            device=device,
            memory_manager=memory_manager,
            config=svd_config or SVDConfig(),
        )

    def project_weight(
        self,
        src: torch.Tensor,
        target_shape: Tuple[int, ...],
        ref: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Project a weight tensor to target shape.

        Args:
            src: Source weight tensor
            target_shape: Target shape
            ref: Optional reference tensor for subspace alignment

        Returns:
            Projected tensor
        """
        return svd_projection(
            src, target_shape, self.svd_engine,
            self.config, self.device, ref
        )

    def extract_lora(
        self,
        delta: torch.Tensor,
        rank: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
        """
        Extract LoRA factors from a delta using PiSSA.

        Args:
            delta: Weight delta tensor
            rank: Target LoRA rank

        Returns:
            Tuple of (lora_A, lora_B, actual_rank)
        """
        return extract_lora_pissa(delta, rank, self.svd_engine, self.config)

    def compute_subspace_delta(
        self,
        teacher_mat: torch.Tensor,
        student_mat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute delta in student's subspace.

        Args:
            teacher_mat: Teacher weight matrix
            student_mat: Student weight matrix

        Returns:
            Tuple of (delta_core, U_s, S_s, V_s)
        """
        U_s, S_s, V_s = compute_student_svd_basis(
            student_mat, self.svd_engine, self.config, self.device
        )
        core_t = teacher_core_in_student_basis(
            teacher_mat, U_s, V_s, self.svd_engine, self.config, self.device
        )
        core_s = torch.diag(S_s)
        delta_core = core_t - core_s

        return delta_core, U_s, S_s, V_s
