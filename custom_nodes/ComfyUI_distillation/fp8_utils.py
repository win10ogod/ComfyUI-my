"""
FP8 and Quantization Utilities for ComfyUI Distillation

This module provides utilities for handling FP8 (float8) and quantized weights:
- FP8 dtype detection and safety checks
- Block-wise FP8 dequantization (DeepSeek/MiniMax style)
- Safe weight dtype resolution
"""

from typing import List, Optional, Set, Tuple, Union

import torch
import torch.nn.functional as F


def _collect_float8_dtypes() -> Tuple[torch.dtype, ...]:
    """Return the float8 dtypes available in the current torch build."""
    names = (
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
    )
    dtypes: List[torch.dtype] = []
    for n in names:
        if hasattr(torch, n):
            try:
                dtypes.append(getattr(torch, n))
            except Exception:
                continue
    # Deduplicate while preserving order
    seen: Set[str] = set()
    out: List[torch.dtype] = []
    for dt in dtypes:
        s = str(dt)
        if s not in seen:
            out.append(dt)
            seen.add(s)
    return tuple(out)


# Cache float8 dtypes at module load
_FLOAT8_DTYPES: Tuple[torch.dtype, ...] = _collect_float8_dtypes()


def is_float8_dtype(dtype: torch.dtype) -> bool:
    """
    Best-effort float8 dtype check (supports future torch additions).

    Args:
        dtype: PyTorch dtype to check

    Returns:
        True if dtype is a float8 variant
    """
    try:
        if dtype in _FLOAT8_DTYPES:
            return True
    except Exception:
        pass
    return "float8" in str(dtype)


def get_available_float8_dtypes() -> Tuple[torch.dtype, ...]:
    """Get all available float8 dtypes in current PyTorch build."""
    return _FLOAT8_DTYPES


def resolve_dequant_dtype(
    target_dtype: str = "auto",
    device: Optional[torch.device] = None
) -> torch.dtype:
    """
    Resolve target dtype for streaming dequantization / float8 safety casts.

    Args:
        target_dtype: Target dtype specification ('auto', 'bf16', 'fp16', 'fp32')
        device: Target device for dtype selection

    Returns:
        Resolved PyTorch dtype
    """
    mode = str(target_dtype or "auto").lower().strip()

    if mode in ("bf16", "bfloat16"):
        return torch.bfloat16
    if mode in ("fp16", "float16", "half"):
        return torch.float16
    if mode in ("fp32", "float32", "float"):
        return torch.float32

    # Auto mode
    if device is not None and device.type == "cuda":
        try:
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16

    # CPU / other: prefer bf16 for range; torch will emulate if needed
    return torch.bfloat16


def safe_weight_dtype(
    src_dtype: torch.dtype,
    target_dtype: str = "auto",
    device: Optional[torch.device] = None,
    ref_dtype: Optional[torch.dtype] = None
) -> torch.dtype:
    """
    Choose an output dtype that avoids float8 mixed arithmetic pitfalls.

    Args:
        src_dtype: Source tensor dtype
        target_dtype: Desired target dtype specification
        device: Target device
        ref_dtype: Reference dtype (e.g., from student tensor)

    Returns:
        Safe output dtype
    """
    if not is_float8_dtype(src_dtype):
        return src_dtype
    if ref_dtype is not None and (not is_float8_dtype(ref_dtype)) and ("float" in str(ref_dtype)):
        return ref_dtype
    return resolve_dequant_dtype(target_dtype, device)


def fp8_dequantize(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: int = 128
) -> torch.Tensor:
    """
    FP8 -> float32 dequantization (pure PyTorch, no Triton) with support for:

    1) DeepSeek / MiniMax per-group block-wise FP8:
       - weight shape: (M, N)
       - scale_inv shape: (ceil(M/bs), ceil(N/bs)) OR flattened
       - kernel logic: y = x * scale_inv (multiply!)

    2) TransformerEngine-style scalar / broadcastable scale_inv:
       - falls back to broadcasting multiplication.

    Args:
        weight: FP8 weight tensor (M, N)
        scale_inv: Inverse scale tensor
        block_size: Per-group block size (default: 128)

    Returns:
        float32 tensor (caller can cast to BF16/FP16 as needed)
    """
    if weight is None or scale_inv is None:
        raise ValueError("weight and scale_inv must be non-None")

    # Scalar fast-path
    try:
        if scale_inv.numel() == 1:
            return weight.float() * scale_inv.float()
    except Exception:
        pass

    # DeepSeek/MiniMax block-wise path (2D only)
    if weight.dim() == 2:
        M, N = weight.shape
        bs = int(block_size)
        if bs <= 0:
            bs = 128

        bm = (M + bs - 1) // bs
        bn = (N + bs - 1) // bs

        s = scale_inv
        # Reshape flattened scale_inv -> (bm, bn)
        if s.dim() == 1 and s.numel() == bm * bn:
            s = s.view(bm, bn)
        # Some checkpoints store 2D but not in the expected shape
        elif s.dim() == 2 and s.shape != (bm, bn) and s.numel() == bm * bn:
            s = s.reshape(bm, bn)

        if s.dim() == 2 and s.shape == (bm, bn):
            w32 = weight.float()
            s32 = s.float()

            pad_m = bm * bs - M
            pad_n = bn * bs - N

            if pad_m or pad_n:
                # pad=(left, right, top, bottom) for 2D
                w32 = F.pad(w32, (0, pad_n, 0, pad_m))

            # (bm*bs, bn*bs) -> (bm, bs, bn, bs)
            w4 = w32.view(bm, bs, bn, bs)
            # (bm, bn) -> (bm, 1, bn, 1)
            s4 = s32.view(bm, 1, bn, 1)

            y4 = w4 * s4
            y = y4.reshape(bm * bs, bn * bs)

            if pad_m or pad_n:
                y = y[:M, :N]

            return y

    # Fallback broadcast multiply (TransformerEngine / unexpected shapes)
    return weight.float() * scale_inv.float()


def quantize_to_fp8(
    tensor: torch.Tensor,
    dtype_name: str = "float8_e4m3fn",
    per_channel: bool = False,
    block_size: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to FP8 format.

    Args:
        tensor: Input tensor to quantize
        dtype_name: FP8 dtype name ('float8_e4m3fn', 'float8_e5m2', etc.)
        per_channel: Use per-channel scaling
        block_size: Block size for block-wise quantization

    Returns:
        Tuple of (quantized_tensor, scale_inv)
    """
    if not hasattr(torch, dtype_name):
        raise ValueError(f"FP8 dtype {dtype_name} not available in this PyTorch build")

    fp8_dtype = getattr(torch, dtype_name)

    # Determine scaling mode
    if block_size is not None and tensor.dim() == 2:
        # Block-wise quantization
        M, N = tensor.shape
        bs = block_size
        bm = (M + bs - 1) // bs
        bn = (N + bs - 1) // bs

        # Pad if necessary
        pad_m = bm * bs - M
        pad_n = bn * bs - N
        if pad_m or pad_n:
            tensor = F.pad(tensor.float(), (0, pad_n, 0, pad_m))

        # Reshape to blocks
        t_blocks = tensor.view(bm, bs, bn, bs)

        # Compute max abs per block
        max_abs = t_blocks.abs().amax(dim=(1, 3), keepdim=True).float()
        max_abs = torch.clamp(max_abs, min=1e-12)

        # FP8 e4m3fn max value is ~448
        fp8_max = 448.0 if "e4m3" in dtype_name else 57344.0
        scale = fp8_max / max_abs
        scale_inv = max_abs / fp8_max

        # Quantize
        scaled = t_blocks * scale
        quantized = scaled.to(fp8_dtype)

        # Reshape back
        quantized = quantized.view(bm * bs, bn * bs)
        if pad_m or pad_n:
            quantized = quantized[:M, :N]

        scale_inv = scale_inv.squeeze().view(bm, bn)
        return quantized, scale_inv

    elif per_channel and tensor.dim() == 2:
        # Per-channel (per-row) quantization
        max_abs = tensor.abs().amax(dim=1, keepdim=True).float()
        max_abs = torch.clamp(max_abs, min=1e-12)

        fp8_max = 448.0 if "e4m3" in dtype_name else 57344.0
        scale = fp8_max / max_abs
        scale_inv = max_abs / fp8_max

        scaled = tensor.float() * scale
        quantized = scaled.to(fp8_dtype)

        return quantized, scale_inv.squeeze()

    else:
        # Scalar quantization
        max_abs = tensor.abs().max().float()
        max_abs = torch.clamp(max_abs, min=1e-12)

        fp8_max = 448.0 if "e4m3" in dtype_name else 57344.0
        scale = fp8_max / max_abs
        scale_inv = max_abs / fp8_max

        scaled = tensor.float() * scale
        quantized = scaled.to(fp8_dtype)

        return quantized, scale_inv


def check_fp8_compatibility(device: Optional[torch.device] = None) -> dict:
    """
    Check FP8 compatibility for the given device.

    Args:
        device: Device to check (default: current CUDA device)

    Returns:
        Dictionary with compatibility information
    """
    result = {
        "available_dtypes": list(str(dt) for dt in _FLOAT8_DTYPES),
        "cuda_available": torch.cuda.is_available(),
        "compute_capability": None,
        "native_fp8_support": False,
        "recommended_dtype": None,
    }

    if not torch.cuda.is_available():
        return result

    try:
        if device is None:
            device = torch.device("cuda:0")

        props = torch.cuda.get_device_properties(device)
        cc = (props.major, props.minor)
        result["compute_capability"] = f"{cc[0]}.{cc[1]}"

        # FP8 is natively supported on Ada Lovelace (SM 89) and Hopper (SM 90)
        if cc[0] >= 9 or (cc[0] == 8 and cc[1] >= 9):
            result["native_fp8_support"] = True
            result["recommended_dtype"] = "float8_e4m3fn"
        elif len(_FLOAT8_DTYPES) > 0:
            result["recommended_dtype"] = str(_FLOAT8_DTYPES[0])
    except Exception as e:
        result["error"] = str(e)

    return result


def ensure_contiguous_and_safe(
    tensor: torch.Tensor,
    target_dtype: str = "auto",
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Ensure tensor is contiguous and in a safe dtype for computation.

    Args:
        tensor: Input tensor
        target_dtype: Target dtype specification
        device: Target device

    Returns:
        Contiguous tensor in safe dtype
    """
    # Handle float8 tensors
    if is_float8_dtype(tensor.dtype):
        safe_dt = resolve_dequant_dtype(target_dtype, device)
        tensor = tensor.to(safe_dt)

    # Ensure contiguous
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    return tensor


class FP8WeightHandler:
    """
    Handler for FP8 weights with associated scale tensors.

    Provides a unified interface for working with FP8-quantized model weights.
    """

    def __init__(
        self,
        block_size: int = 128,
        target_dtype: str = "auto",
        device: Optional[torch.device] = None
    ):
        self.block_size = block_size
        self.target_dtype = target_dtype
        self.device = device
        self._scale_cache: dict = {}

    def register_scale(self, weight_key: str, scale_inv: torch.Tensor):
        """Register a scale tensor for a weight key."""
        self._scale_cache[weight_key] = scale_inv

    def get_scale(self, weight_key: str) -> Optional[torch.Tensor]:
        """Get the scale tensor for a weight key."""
        return self._scale_cache.get(weight_key)

    def dequantize(
        self,
        weight: torch.Tensor,
        weight_key: Optional[str] = None,
        scale_inv: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Dequantize an FP8 weight tensor.

        Args:
            weight: FP8 weight tensor
            weight_key: Key to look up scale in cache
            scale_inv: Explicit scale tensor (overrides cache)

        Returns:
            Dequantized float32 tensor
        """
        if not is_float8_dtype(weight.dtype):
            return weight

        if scale_inv is None and weight_key is not None:
            scale_inv = self._scale_cache.get(weight_key)

        if scale_inv is None:
            # Fallback: assume no scaling needed
            safe_dt = resolve_dequant_dtype(self.target_dtype, self.device)
            return weight.to(safe_dt)

        return fp8_dequantize(weight, scale_inv, self.block_size)

    def find_scale_key(
        self,
        weight_key: str,
        available_keys: list
    ) -> Optional[str]:
        """
        Find the corresponding scale key for a weight key.

        Common patterns:
        - weight_key + "_scale_inv"
        - weight_key.replace(".weight", "_scale_inv")
        - weight_key.replace(".weight", ".scale_inv")
        """
        patterns = [
            weight_key + "_scale_inv",
            weight_key.replace(".weight", "_scale_inv"),
            weight_key.replace(".weight", ".scale_inv"),
            weight_key + ".scale_inv",
        ]

        for pattern in patterns:
            if pattern in available_keys:
                return pattern

        return None

    def clear_cache(self):
        """Clear the scale cache."""
        self._scale_cache.clear()
