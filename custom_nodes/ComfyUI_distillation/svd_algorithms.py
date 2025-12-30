"""
SVD Algorithms Module for ComfyUI Distillation

This module provides comprehensive SVD backends for knowledge distillation:
- AURORA-SVD: Adaptive Unrolled Residual-Order Range Augmentation SVD
- DRF-SVD: Dual-Residual Folding SVD
- Randomized SVD: Halko et al. Algorithm 4.4
- Block Krylov SVD: Krylov subspace method
- Adaptive Range Finder: Halko et al. Algorithm 4.2
"""

import math
import gc
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
from enum import Enum

import torch
import torch.nn.functional as F


class SVDMode(Enum):
    """SVD algorithm selection modes."""
    AUTO = "auto"
    FULL = "full"
    RANDOMIZED = "randomized"
    KRYLOV = "krylov"
    ADAPTIVE = "adaptive"
    DRF = "drf"
    AURORA = "aurora"


@dataclass
class SVDConfig:
    """Configuration for SVD algorithms."""
    # General settings
    svd_mode: str = "auto"
    use_mixed_precision: bool = True
    chunk_threshold_mb: int = 512
    chunk_size_mb: int = 128
    force_cpu: bool = False
    verbose: bool = False

    # Randomized SVD parameters
    randomized_iter: int = 2
    randomized_oversamples: int = 8

    # Auto mode parameters
    auto_min_dim: int = 1024
    auto_full_rank_ratio: float = 0.6
    auto_lowrank: str = "randomized"

    # Adaptive range finder parameters
    adaptive_block_size: int = 32
    adaptive_n_test: int = 8

    # DRF-SVD parameters
    drf_steps: int = 1
    drf_theta: float = 0.5
    drf_resid_eps: float = 1e-8

    # AURORA-SVD parameters
    aurora_steps: int = 1
    aurora_order: int = 2
    aurora_theta1: float = 0.5
    aurora_theta2: float = 0.25
    aurora_resid_eps: float = 1e-8
    aurora_level2_keep: float = 1.0
    aurora_level1_keep: float = 1.0
    aurora_min_steps: int = 1
    aurora_early_stop: float = 0.005
    aurora_early_stop_patience: int = 2
    aurora_early_stop_check_every: int = 1
    aurora_core_solver: str = "svd"
    aurora_svd_driver: str = "gesvdj"


class GPUMemoryManager:
    """GPU memory management utility."""

    def __init__(
        self,
        device: Union[str, torch.device],
        memory_fraction: float = 0.85,
        verbose: bool = True
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.memory_fraction = memory_fraction
        self.verbose = verbose

        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            self.total_memory = torch.cuda.get_device_properties(self.device).total_memory
            self.max_memory = int(self.total_memory * memory_fraction)
        else:
            self.total_memory = 0
            self.max_memory = 0

    def get_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics."""
        if self.device.type != "cuda":
            return {"total_gb": 0, "used_gb": 0, "free_gb": 0}

        allocated = torch.cuda.memory_allocated(self.device)
        return {
            "total_gb": self.total_memory / (1024**3),
            "used_gb": allocated / (1024**3),
            "free_gb": (self.max_memory - allocated) / (1024**3),
        }

    def can_allocate(self, size_bytes: int) -> bool:
        """Check if we can allocate the given size."""
        if self.device.type != "cuda":
            return True
        allocated = torch.cuda.memory_allocated(self.device)
        return (allocated + size_bytes) < self.max_memory

    def clear_cache(self):
        """Clear GPU cache."""
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


class CUDAStreamManager:
    """CUDA stream management for async operations."""

    def __init__(self, device: torch.device, num_streams: int = 3):
        self.device = device
        self.num_streams = num_streams
        self.streams = []
        self.current_stream_idx = 0

        if device.type == "cuda":
            for _ in range(num_streams):
                self.streams.append(torch.cuda.Stream(device=device))

    @contextmanager
    def stream_context(self, stream_idx: Optional[int] = None):
        """Context manager for using a CUDA stream."""
        if not self.streams:
            yield
            return

        idx = stream_idx if stream_idx is not None else self.current_stream_idx
        self.current_stream_idx = (self.current_stream_idx + 1) % self.num_streams

        with torch.cuda.stream(self.streams[idx]):
            yield

    def synchronize_current(self):
        """Synchronize current stream."""
        if self.streams:
            self.streams[self.current_stream_idx].synchronize()

    def synchronize_all(self):
        """Synchronize all streams."""
        for stream in self.streams:
            stream.synchronize()


class GPUAcceleratedSVD:
    """
    SVD backend with GPU acceleration and low-memory low-rank fallbacks.

    Implemented low-rank backends:
    - Randomized SVD with numerically-stable subspace iteration (Halko et al., Alg. 4.4)
    - Block Krylov subspace SVD (stores intermediate iterates, improves accuracy per iteration)
    - Adaptive range finder (Halko et al., Alg. 4.2) for energy-targeted rank selection
    - DRF-SVD (Dual-Residual Folding SVD): residual-folding refinement for improved subspace capture
    - AURORA-SVD (Adaptive Unrolled Residual-Order Range Augmentation SVD): distillation-centric
    """

    def __init__(
        self,
        device: torch.device,
        memory_manager: Optional[GPUMemoryManager] = None,
        stream_manager: Optional[CUDAStreamManager] = None,
        config: Optional[SVDConfig] = None,
    ):
        self.device = device
        self.memory_manager = memory_manager or GPUMemoryManager(device)
        self.stream_manager = stream_manager
        self.config = config or SVDConfig()

        # Cache config values for faster access
        self._update_from_config()

        # AURORA telemetry
        self._aurora_calls = 0
        self._aurora_steps_used_total = 0

    def _update_from_config(self):
        """Update internal parameters from config."""
        cfg = self.config

        self.use_mixed_precision = cfg.use_mixed_precision
        self.chunk_threshold_bytes = int(cfg.chunk_threshold_mb) * 1024 * 1024
        self.chunk_size_bytes = int(cfg.chunk_size_mb) * 1024 * 1024
        self.force_cpu = cfg.force_cpu
        self.verbose = cfg.verbose

        self.svd_mode = cfg.svd_mode.lower() if cfg.svd_mode else "auto"
        self.randomized_iter = cfg.randomized_iter
        self.randomized_oversamples = cfg.randomized_oversamples
        self.auto_min_dim = cfg.auto_min_dim
        self.auto_full_rank_ratio = cfg.auto_full_rank_ratio
        self.auto_lowrank = cfg.auto_lowrank.lower() if cfg.auto_lowrank else "randomized"

        self.adaptive_block_size = cfg.adaptive_block_size
        self.adaptive_n_test = cfg.adaptive_n_test

        self.drf_steps = cfg.drf_steps
        self.drf_theta = cfg.drf_theta
        self.drf_resid_eps = cfg.drf_resid_eps

        self.aurora_steps = cfg.aurora_steps
        self.aurora_order = cfg.aurora_order
        self.aurora_theta1 = cfg.aurora_theta1
        self.aurora_theta2 = cfg.aurora_theta2
        self.aurora_resid_eps = cfg.aurora_resid_eps
        self.aurora_level2_keep = cfg.aurora_level2_keep
        self.aurora_level1_keep = cfg.aurora_level1_keep
        self.aurora_min_steps = cfg.aurora_min_steps
        self.aurora_early_stop = cfg.aurora_early_stop
        self.aurora_early_stop_patience = cfg.aurora_early_stop_patience
        self.aurora_early_stop_check_every = cfg.aurora_early_stop_check_every
        self.aurora_core_solver = cfg.aurora_core_solver.lower() if cfg.aurora_core_solver else "svd"
        self.aurora_svd_driver = cfg.aurora_svd_driver or ""

    @contextmanager
    def _stream_context(self):
        """Context manager for CUDA stream."""
        if self.stream_manager is None:
            yield
        else:
            with self.stream_manager.stream_context():
                yield

    # =========================================================================
    # Public APIs
    # =========================================================================

    def svd(
        self,
        tensor: torch.Tensor,
        rank: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute truncated SVD (U, S, Vh) with a bounded rank.

        Args:
            tensor: 2D input tensor
            rank: Target rank for truncation

        Returns:
            Tuple of (U, S, Vh) tensors
        """
        if tensor is None:
            raise ValueError("tensor is None")
        if tensor.dim() != 2:
            raise ValueError(f"SVD expects a 2D tensor, got dim={tensor.dim()}")

        m, n = tensor.shape
        min_dim = min(m, n)
        r = int(rank)
        if r <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")
        r = min(r, min_dim)

        tensor_bytes = int(m) * int(n) * 4

        if self.force_cpu:
            return self._cpu_svd(tensor, r)

        mode = self.svd_mode

        # Auto mode heuristics
        if mode == "auto":
            if tensor_bytes > self.chunk_threshold_bytes:
                mode = self.auto_lowrank
            elif min_dim >= self.auto_min_dim and r < int(min_dim * self.auto_full_rank_ratio):
                mode = self.auto_lowrank
            else:
                mode = "full"

        if mode == "adaptive":
            mode = self.auto_lowrank

        if mode in ("randomized", "krylov", "drf", "aurora"):
            return self._lowrank_svd(
                tensor, r, algo=mode,
                n_oversamples=self.randomized_oversamples,
                n_iter=self.randomized_iter,
            )

        # Full SVD path
        try:
            return self._gpu_svd(tensor, r)
        except RuntimeError as e:
            if self.verbose:
                print(f"[SVD] GPU full SVD failed ({type(e).__name__}: {e}); falling back to low-rank SVD.")
            return self._lowrank_svd(
                tensor, r, algo=self.auto_lowrank,
                n_oversamples=self.randomized_oversamples,
                n_iter=self.randomized_iter,
            )

    def randomized_svd(
        self,
        tensor: torch.Tensor,
        rank: int,
        n_oversamples: Optional[int] = None,
        n_iter: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Explicit low-rank SVD with per-call parameters.

        Dispatch logic:
        - If svd_mode is krylov/drf/aurora, uses that algorithm
        - Otherwise uses randomized subspace iteration (Alg. 4.4)
        """
        algo = "randomized"
        if self.svd_mode == "krylov":
            algo = "krylov"
        elif self.svd_mode == "drf":
            algo = "drf"
        elif self.svd_mode == "aurora":
            algo = "aurora"
        elif self.svd_mode == "auto":
            if self.auto_lowrank == "krylov":
                algo = "krylov"
            elif self.auto_lowrank == "drf":
                algo = "drf"
            elif self.auto_lowrank == "aurora":
                algo = "aurora"

        return self._lowrank_svd(
            tensor,
            int(rank),
            algo=algo,
            n_oversamples=self.randomized_oversamples if n_oversamples is None else int(n_oversamples),
            n_iter=self.randomized_iter if n_iter is None else int(n_iter),
        )

    def svd_energy(
        self,
        tensor: torch.Tensor,
        energy_threshold: float,
        min_rank: int,
        max_rank: int,
        block_size: Optional[int] = None,
        n_test: Optional[int] = None,
        n_iter: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Energy-targeted low-rank SVD using Adaptive range finder (Halko et al., Alg. 4.2).

        Args:
            tensor: Input matrix
            energy_threshold: Target retained Frobenius energy fraction in (0,1]
            min_rank: Minimum rank bound
            max_rank: Maximum rank bound
            block_size: Number of basis vectors added per adaptive iteration
            n_test: Number of random test vectors for residual estimation
            n_iter: Optional subspace-iteration steps applied to each new block

        Returns:
            Tuple of (U, S, Vh) tensors
        """
        if tensor is None:
            raise ValueError("tensor is None")
        if tensor.dim() != 2:
            raise ValueError(f"SVD expects a 2D tensor, got dim={tensor.dim()}")

        with self._stream_context():
            x = tensor.to(self.device, non_blocking=True).float()
            m, n = x.shape
            min_dim = min(m, n)

            max_r = min(int(max_rank), min_dim)
            min_r = max(1, min(int(min_rank), max_r))

            tau = float(max(0.0, min(1.0, energy_threshold)))
            eps = math.sqrt(max(0.0, 1.0 - tau))

            bs = self.adaptive_block_size if block_size is None else int(block_size)
            nt = self.adaptive_n_test if n_test is None else int(n_test)
            q = 0 if n_iter is None else int(n_iter)

            Q = self._adaptive_range_finder(x, eps=eps, max_rank=max_r, block_size=bs, n_test=nt, n_iter=q)

            # Ensure at least min_rank basis vectors
            if Q.shape[1] < min_r:
                add = min_r - Q.shape[1]
                omega = torch.randn(n, add, device=self.device, dtype=torch.float32)
                Y = x @ omega
                if Q.shape[1] > 0:
                    Y = Y - Q @ (Q.transpose(0, 1) @ Y)
                Qi, _ = torch.linalg.qr(Y, mode="reduced")
                Q = torch.cat([Q, Qi], dim=1)

            # Cap and re-orthonormalize
            if Q.shape[1] > max_r:
                Q = Q[:, :max_r]
            Q, _ = torch.linalg.qr(Q, mode="reduced")

            B = Q.transpose(0, 1) @ x
            Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)
            U = Q @ Ub
            return U, S, Vh

    # =========================================================================
    # Full SVD backends
    # =========================================================================

    def _gpu_svd(
        self,
        x: torch.Tensor,
        r: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full SVD on GPU, then truncate to r."""
        with self._stream_context():
            x = x.to(self.device, non_blocking=True).float()
            U, S, Vh = torch.linalg.svd(x, full_matrices=False)
            return U[:, :r], S[:r], Vh[:r, :]

    def _cpu_svd(
        self,
        x: torch.Tensor,
        r: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full SVD on CPU, then truncate to r."""
        x = x.cpu().float()
        U, S, Vh = torch.linalg.svd(x, full_matrices=False)
        return U[:, :r].to(self.device), S[:r].to(self.device), Vh[:r, :].to(self.device)

    # =========================================================================
    # Low-rank SVD backends
    # =========================================================================

    def _lowrank_svd(
        self,
        x: torch.Tensor,
        r: int,
        algo: str,
        n_oversamples: int = 8,
        n_iter: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dispatch to appropriate low-rank SVD algorithm."""
        if algo == "krylov":
            return self._krylov_svd(x, r, n_oversamples=n_oversamples, n_iter=n_iter)
        if algo == "drf":
            return self._drf_svd(x, r, n_oversamples=n_oversamples, n_iter=n_iter)
        if algo == "aurora":
            return self._aurora_svd(x, r, n_oversamples=n_oversamples, n_iter=n_iter)
        return self._randomized_svd(x, r, n_oversamples=n_oversamples, n_iter=n_iter)

    def _randomized_svd(
        self,
        x: torch.Tensor,
        r: int,
        n_oversamples: int = 8,
        n_iter: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomized SVD using numerically-stable subspace iteration (Halko et al., Algorithm 4.4).

        This alternates applications of A and A^T with intermediate QR orthonormalization.
        """
        with self._stream_context():
            x = x.to(self.device, non_blocking=True).float()
            m, n = x.shape

            r = min(int(r), min(m, n))
            p = max(0, int(n_oversamples))
            k = min(min(m, n), r + p)
            if k <= 0:
                raise ValueError("randomized_svd: computed k <= 0")

            omega = torch.randn(n, k, device=self.device, dtype=torch.float32)
            Y = x @ omega
            Q, _ = torch.linalg.qr(Y, mode="reduced")

            q = max(0, int(n_iter))
            for _ in range(q):
                Y_tilde = x.transpose(0, 1) @ Q
                Q_tilde, _ = torch.linalg.qr(Y_tilde, mode="reduced")
                Y = x @ Q_tilde
                Q, _ = torch.linalg.qr(Y, mode="reduced")

            B = Q.transpose(0, 1) @ x
            Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)
            U = Q @ Ub
            return U[:, :r], S[:r], Vh[:r, :]

    def _krylov_svd(
        self,
        x: torch.Tensor,
        r: int,
        n_oversamples: int = 8,
        n_iter: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Block Krylov subspace SVD.

        Builds the Krylov subspace span{AΩ, (AA^T)AΩ, ..., (AA^T)^q AΩ} without forming AA^T,
        using alternating multiplies by A and A^T. Intermediate orthonormal blocks are stored
        and concatenated before solving the projected SVD.
        """
        with self._stream_context():
            x = x.to(self.device, non_blocking=True).float()
            m, n = x.shape

            r = min(int(r), min(m, n))
            p = max(0, int(n_oversamples))
            k = min(min(m, n), r + p)
            if k <= 0:
                raise ValueError("krylov_svd: computed k <= 0")

            omega = torch.randn(n, k, device=self.device, dtype=torch.float32)
            Y = x @ omega
            Q, _ = torch.linalg.qr(Y, mode="reduced")

            blocks = [Q]
            q = max(0, int(n_iter))

            for _ in range(q):
                Y_tilde = x.transpose(0, 1) @ Q
                Q_tilde, _ = torch.linalg.qr(Y_tilde, mode="reduced")
                Y = x @ Q_tilde
                Q, _ = torch.linalg.qr(Y, mode="reduced")
                blocks.append(Q)

            Qbar = torch.cat(blocks, dim=1)
            Qbar, _ = torch.linalg.qr(Qbar, mode="reduced")

            B = Qbar.transpose(0, 1) @ x
            Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)
            U = Qbar @ Ub
            return U[:, :r], S[:r], Vh[:r, :]

    def _drf_svd(
        self,
        x: torch.Tensor,
        r: int,
        n_oversamples: int = 8,
        n_iter: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        DRF-SVD (Dual-Residual Folding SVD) - a residual-folding refinement on top of a
        single randomized range capture.

        Motivation (distillation-centric):
          For distillation deltas, the spectrum is often moderately ill-conditioned and
          the LoRA rank budget is tight. Standard randomized SVD accuracy can be limited
          by incomplete capture of directions that are *correlated* with the current
          approximate singular space but live in its orthogonal complement. DRF-SVD
          explicitly augments both left and right subspaces with *scaled residual*
          directions from the current approximation and then performs a Rayleigh-Ritz
          compression back to the target rank.

        Algorithm sketch (one refinement step):
          1) Compute an initial k0=(r+p) low-rank SVD approximation via randomized SVD.
          2) Form residual blocks:
                R_L = A V - U diag(S)
                R_R = A^T U - V diag(S)
             then scale them by 1/(S^theta) to stabilize small/large singular components.
          3) Fold residual blocks into the subspaces:
                Q = orth([U, R_L]),  P = orth([V, R_R])
          4) Solve the projected SVD of M = Q^T A P and truncate back to k0.
          5) Repeat for `drf_steps` rounds (default 1), then return the top-r slice.
        """
        with self._stream_context():
            A = x.to(self.device, non_blocking=True).float()
            m, n = A.shape
            min_dim = min(m, n)
            r = min(int(r), min_dim)
            p = max(0, int(n_oversamples))

            k0 = min(min_dim, r + p)
            if k0 <= 0:
                raise ValueError("drf_svd: computed k0 <= 0")

            # Initial capture
            U, S, Vh = self._randomized_svd(A, k0, n_oversamples=0, n_iter=max(0, int(n_iter)))
            U = U.contiguous()
            S = S.contiguous()
            Vh = Vh.contiguous()

            steps = max(0, self.drf_steps)
            theta = float(max(0.0, min(1.0, self.drf_theta)))
            eps = float(max(1e-30, self.drf_resid_eps))

            for _ in range(steps):
                V = Vh.transpose(0, 1).contiguous()

                AV = A @ V
                ATU = A.transpose(0, 1) @ U

                US = U * S.unsqueeze(0)
                VS = V * S.unsqueeze(0)
                R_L = AV - US
                R_R = ATU - VS

                if theta > 0.0:
                    denom = torch.pow(torch.clamp(S, min=eps), theta).unsqueeze(0)
                    R_L = R_L / denom
                    R_R = R_R / denom

                Q = torch.cat([U, R_L], dim=1)
                P = torch.cat([V, R_R], dim=1)

                Q, _ = torch.linalg.qr(Q, mode="reduced")
                P, _ = torch.linalg.qr(P, mode="reduced")

                AP = A @ P
                M = Q.transpose(0, 1) @ AP

                Uc, Sc, Vhc = torch.linalg.svd(M, full_matrices=False)

                kk = min(int(Sc.numel()), int(k0))
                if kk <= 0:
                    break

                U = (Q @ Uc[:, :kk]).contiguous()
                S = Sc[:kk].contiguous()
                Vh = (Vhc[:kk, :] @ P.transpose(0, 1)).contiguous()

                # Defensive re-orthonormalization
                U, _ = torch.linalg.qr(U, mode="reduced")
                Vt = Vh.transpose(0, 1).contiguous()
                Vt, _ = torch.linalg.qr(Vt, mode="reduced")
                Vh = Vt.transpose(0, 1).contiguous()

            return U[:, :r], S[:r], Vh[:r, :]

    def _aurora_svd(
        self,
        x: torch.Tensor,
        r: int,
        n_oversamples: int = 8,
        n_iter: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        AURORA-SVD (Adaptive Unrolled Residual-Order Range Augmentation SVD).

        Distillation-centric goal:
          Improve low-rank approximation quality under a tight LoRA rank budget by
          augmenting the approximation subspace with *structured residual orders*,
          then performing a Rayleigh-Ritz compression back to the working rank.

        Core idea (per refinement round):
          - Start from a working-rank approximation A ≈ U diag(S) V^T.
          - Form first-order residual blocks:
                R_L = A V - U diag(S)
                R_R = A^T U - V diag(S)
          - Optionally form a second-order residual response (order=2):
                T_L = A (scaled R_R),   T_R = A^T (scaled R_L)
          - Build enlarged subspaces Q, P and compute the best rank-(r+p) approximation
            inside span(Q) × span(P) by SVD of the projected core Q^T A P.

        Implementation notes (v4.2.0 speed path):
          - Uses *block-anchored orthonormalization*: keep current U/V blocks intact and
            orthonormalize residual blocks against them.
          - Avoids redundant re-orthonormalization of U/V.
          - Reuses A@V when constructing A@P (since V is an explicit leading block of P).
        """
        with self._stream_context():
            A = x.to(self.device, non_blocking=True).float()
            m, n = A.shape
            min_dim = min(m, n)

            r = min(int(r), min_dim)
            p = max(0, int(n_oversamples))
            k0 = min(min_dim, r + p)
            if k0 <= 0:
                raise ValueError("aurora_svd: computed k0 <= 0")

            # Initial capture
            U, S, Vh = self._randomized_svd(A, k0, n_oversamples=0, n_iter=max(0, int(n_iter)))
            U = U.contiguous()
            S = S.contiguous()
            Vh = Vh.contiguous()

            steps = max(0, self.aurora_steps)
            order = 1 if self.aurora_order <= 1 else 2

            theta1 = float(max(0.0, min(1.0, self.aurora_theta1)))
            theta2 = float(max(0.0, min(1.0, self.aurora_theta2)))
            eps = float(max(1e-30, self.aurora_resid_eps))

            level2_keep = float(max(0.0, min(1.0, self.aurora_level2_keep)))
            level1_keep = float(max(0.0, min(1.0, self.aurora_level1_keep)))

            min_steps = min(max(0, self.aurora_min_steps), steps)
            early_stop_tol = float(max(0.0, self.aurora_early_stop))
            early_stop_patience = max(1, self.aurora_early_stop_patience)
            early_stop_check_every = max(1, self.aurora_early_stop_check_every)

            core_solver = self.aurora_core_solver
            svd_driver = self.aurora_svd_driver

            use_amp = self.use_mixed_precision and (self.device.type == "cuda")
            if use_amp:
                try:
                    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                except:
                    amp_dtype = torch.float16
            else:
                amp_dtype = None

            def _mm(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
                if use_amp and amp_dtype is not None:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        return (X @ Y).float()
                return X @ Y

            def _orth_block(block: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
                if block.numel() == 0:
                    return block
                if basis.numel() != 0 and basis.shape[1] > 0:
                    block = block - basis @ (basis.transpose(0, 1) @ block)
                if torch.linalg.norm(block, ord="fro") < 1e-20:
                    return block[:, :0]
                Qb, _ = torch.linalg.qr(block, mode="reduced")
                return Qb

            def _core_svd(M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                if core_solver == "eigh":
                    a, b = M.shape
                    if a <= b:
                        C = M @ M.transpose(0, 1)
                        evals, Uc = torch.linalg.eigh(C)
                        evals = torch.flip(evals, dims=[0])
                        Uc = torch.flip(Uc, dims=[1])
                        Sc = torch.sqrt(torch.clamp(evals, min=0.0))
                        inv = torch.where(Sc > 1e-12, 1.0 / Sc, torch.zeros_like(Sc))
                        V = (M.transpose(0, 1) @ Uc) * inv.unsqueeze(0)
                        Vhc = V.transpose(0, 1)
                        return Uc, Sc, Vhc
                    else:
                        C = M.transpose(0, 1) @ M
                        evals, V = torch.linalg.eigh(C)
                        evals = torch.flip(evals, dims=[0])
                        V = torch.flip(V, dims=[1])
                        Sc = torch.sqrt(torch.clamp(evals, min=0.0))
                        inv = torch.where(Sc > 1e-12, 1.0 / Sc, torch.zeros_like(Sc))
                        Uc = (M @ V) * inv.unsqueeze(0)
                        Vhc = V.transpose(0, 1)
                        return Uc, Sc, Vhc

                try:
                    if (self.device.type == "cuda") and svd_driver:
                        return torch.linalg.svd(M, full_matrices=False, driver=svd_driver)
                    return torch.linalg.svd(M, full_matrices=False)
                except TypeError:
                    return torch.linalg.svd(M, full_matrices=False)

            prev_res2: Optional[torch.Tensor] = None
            stall = 0
            steps_used = 0

            for it in range(steps):
                V = Vh.transpose(0, 1).contiguous()

                # First-order residuals
                AV = _mm(A, V)
                ATU = _mm(A.transpose(0, 1), U)

                US = U * S.unsqueeze(0)
                VS = V * S.unsqueeze(0)

                R_L = AV - US
                R_R = ATU - VS

                # Early-stop check
                if early_stop_tol > 0.0:
                    res2 = (R_L.float() * R_L.float()).sum() + (R_R.float() * R_R.float()).sum()
                    if prev_res2 is not None and (it >= min_steps) and ((it % early_stop_check_every) == 0):
                        rel_impr = ((prev_res2 - res2) / (prev_res2 + eps)).clamp(min=-1.0, max=1.0)
                        if float(rel_impr.item()) < early_stop_tol:
                            stall += 1
                        else:
                            stall = 0
                        if stall >= early_stop_patience:
                            break
                    prev_res2 = res2

                # Residual scaling
                if theta1 > 0.0:
                    denom1 = torch.pow(torch.clamp(S, min=eps), theta1).unsqueeze(0)
                    R_Ls = R_L / denom1
                    R_Rs = R_R / denom1
                else:
                    R_Ls = R_L
                    R_Rs = R_R

                Q_blocks: List[torch.Tensor] = [U]
                P_blocks: List[torch.Tensor] = [V]

                # Optional level-1 pruning
                R_Ls1 = R_Ls
                R_Rs1 = R_Rs
                if level1_keep <= 0.0:
                    R_Ls1 = R_Ls[:, :0]
                    R_Rs1 = R_Rs[:, :0]
                elif level1_keep < 1.0 and k0 > 1:
                    eL1 = torch.sum(R_L.float() * R_L.float(), dim=0)
                    eR1 = torch.sum(R_R.float() * R_R.float(), dim=0)
                    e1 = (eL1 + eR1).contiguous()
                    keep1 = max(1, int(round(level1_keep * k0)))
                    keep1 = min(keep1, k0)
                    _, topi1 = torch.topk(e1, k=keep1, largest=True, sorted=False)
                    R_Ls1 = R_Ls[:, topi1]
                    R_Rs1 = R_Rs[:, topi1]

                Q1 = _orth_block(R_Ls1, U)
                if Q1.numel() != 0 and Q1.shape[1] > 0:
                    Q_blocks.append(Q1)

                P1 = _orth_block(R_Rs1, V)
                if P1.numel() != 0 and P1.shape[1] > 0:
                    P_blocks.append(P1)

                if order >= 2:
                    # Level-2 component selection
                    if level2_keep <= 0.0:
                        idx = None
                    elif level2_keep >= 1.0 or k0 <= 1:
                        idx = None
                    else:
                        eL = torch.sum(R_L.float() * R_L.float(), dim=0)
                        eR = torch.sum(R_R.float() * R_R.float(), dim=0)
                        e = (eL + eR).contiguous()
                        keep = max(1, int(round(level2_keep * k0)))
                        keep = min(keep, k0)
                        _, topi = torch.topk(e, k=keep, largest=True, sorted=False)
                        idx = topi

                    if idx is None:
                        R_L2 = R_Ls
                        R_R2 = R_Rs
                        S2 = S
                    else:
                        R_L2 = R_Ls[:, idx]
                        R_R2 = R_Rs[:, idx]
                        S2 = S[idx]

                    T_L = _mm(A, R_R2)
                    T_R = _mm(A.transpose(0, 1), R_L2)

                    Q_basis = torch.cat(Q_blocks, dim=1)
                    P_basis = torch.cat(P_blocks, dim=1)
                    T_L = T_L - Q_basis @ (Q_basis.transpose(0, 1) @ T_L)
                    T_R = T_R - P_basis @ (P_basis.transpose(0, 1) @ T_R)

                    if theta2 > 0.0:
                        denom2 = torch.pow(torch.clamp(S2, min=eps), theta2).unsqueeze(0)
                        T_L = T_L / denom2
                        T_R = T_R / denom2

                    Q2 = _orth_block(T_L, Q_basis)
                    if Q2.numel() != 0 and Q2.shape[1] > 0:
                        Q_blocks.append(Q2)

                    P2 = _orth_block(T_R, P_basis)
                    if P2.numel() != 0 and P2.shape[1] > 0:
                        P_blocks.append(P2)

                Q = torch.cat(Q_blocks, dim=1)
                P = torch.cat(P_blocks, dim=1)

                # Compute A@P with reuse of A@V
                if len(P_blocks) == 1:
                    AP = AV
                else:
                    P_rest = torch.cat(P_blocks[1:], dim=1)
                    AP_rest = _mm(A, P_rest)
                    AP = torch.cat([AV, AP_rest], dim=1)

                # Project and solve small SVD
                M = Q.transpose(0, 1) @ AP
                Uc, Sc, Vhc = _core_svd(M)

                kk = min(int(Sc.numel()), int(k0))
                if kk <= 0:
                    break

                U = (Q @ Uc[:, :kk]).contiguous()
                S = Sc[:kk].contiguous()
                Vh = (Vhc[:kk, :] @ P.transpose(0, 1)).contiguous()

                steps_used += 1

            self._aurora_calls += 1
            self._aurora_steps_used_total += int(steps_used)

            return U[:, :r], S[:r], Vh[:r, :]

    def _adaptive_range_finder(
        self,
        x: torch.Tensor,
        eps: float,
        max_rank: int,
        block_size: int,
        n_test: int,
        n_iter: int = 0,
    ) -> torch.Tensor:
        """
        Adaptive randomized range finder (Halko et al., Alg. 4.2).

        Maintains an estimate of the residual via random test vectors:
          H = (I - QQ^T) A G
        and stops when ||H||_F <= eps ||A||_F or max_rank is reached.
        """
        m, n = x.shape
        max_r = min(int(max_rank), min(m, n))
        if max_r <= 0:
            return torch.zeros((m, 0), device=self.device, dtype=torch.float32)

        bs = max(1, int(block_size))
        nt = max(1, int(n_test))
        q = max(0, int(n_iter))
        eps = float(max(0.0, eps))

        normA = torch.linalg.norm(x, ord="fro")
        if normA < 1e-20:
            return torch.zeros((m, 0), device=self.device, dtype=torch.float32)

        G = torch.randn(n, nt, device=self.device, dtype=torch.float32)
        H = x @ G

        Q: Optional[torch.Tensor] = None
        k = 0

        res = torch.linalg.norm(H, ord="fro")
        tol = eps * normA

        while (res > tol) and (k < max_r):
            b = min(bs, max_r - k)

            omega = torch.randn(n, b, device=self.device, dtype=torch.float32)
            Y = x @ omega

            if q > 0:
                Qi, _ = torch.linalg.qr(Y, mode="reduced")
                for _ in range(q):
                    Z = x.transpose(0, 1) @ Qi
                    Qz, _ = torch.linalg.qr(Z, mode="reduced")
                    Y = x @ Qz
                    Qi, _ = torch.linalg.qr(Y, mode="reduced")
                Y = Qi

            if Q is not None and Q.shape[1] > 0:
                Y = Y - Q @ (Q.transpose(0, 1) @ Y)

            Qi, _ = torch.linalg.qr(Y, mode="reduced")

            # Double-orthogonalize
            if Q is not None and Q.shape[1] > 0:
                Qi = Qi - Q @ (Q.transpose(0, 1) @ Qi)
                Qi, _ = torch.linalg.qr(Qi, mode="reduced")

            if Qi.numel() == 0 or Qi.shape[1] == 0:
                break

            Q = Qi if Q is None else torch.cat([Q, Qi], dim=1)
            k = Q.shape[1]

            H = H - Qi @ (Qi.transpose(0, 1) @ H)
            res = torch.linalg.norm(H, ord="fro")

        if Q is None:
            return torch.zeros((m, 0), device=self.device, dtype=torch.float32)

        Q, _ = torch.linalg.qr(Q, mode="reduced")
        if Q.shape[1] > max_r:
            Q = Q[:, :max_r]
        return Q


def compute_adaptive_rank(
    singular_values: torch.Tensor,
    energy_threshold: float = 0.95,
    min_rank: int = 8,
    max_rank: int = 256,
    total_energy: Optional[Union[float, torch.Tensor]] = None,
) -> int:
    """
    Compute an adaptive rank based on retained Frobenius-energy.

    Args:
        singular_values: 1D tensor of (estimated) singular values
        energy_threshold: Target fraction of Frobenius energy to retain
        min_rank: Minimum rank bound
        max_rank: Maximum rank bound
        total_energy: If provided, ||A||_F^2 for the original matrix A

    Returns:
        Computed rank as integer
    """
    s = singular_values.detach().float()
    if s.numel() == 0:
        return max(0, int(min_rank))

    s = torch.clamp(s, min=0.0)

    if total_energy is None:
        total = torch.sum(s ** 2)
    else:
        if torch.is_tensor(total_energy):
            total = total_energy.to(device=s.device, dtype=s.dtype)
        else:
            total = torch.tensor(float(total_energy), device=s.device, dtype=s.dtype)

    if total < 1e-20:
        return max(1, int(min_rank))

    cumulative = torch.cumsum(s ** 2, dim=0)
    target = float(energy_threshold) * total

    hit = (cumulative >= target)
    if bool(hit.any()):
        r = int(hit.nonzero(as_tuple=True)[0][0].item() + 1)
    else:
        r = int(s.numel())

    r = max(int(min_rank), min(r, int(max_rank), int(s.numel())))
    return r
