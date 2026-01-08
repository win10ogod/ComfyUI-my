"""
Merge Strategies for ComfyUI Distillation

This module provides delta merging strategies for model distillation:
- DARE: Drop And REscale
- TIES: Trim, Elect Sign, and Disjoint Merge
- Least-squares weight mixing
"""

import math
from typing import List, Optional, Tuple, Union

import torch


def apply_dare(
    delta: torch.Tensor,
    drop_rate: float = 0.7,
    rescale: bool = True,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    DARE: Drop And REscale for delta parameters.

    Randomly drops elements of the delta tensor and rescales to maintain
    expected magnitude.

    Args:
        delta: Input delta tensor (weight difference)
        drop_rate: Fraction of elements to drop (0 to 1)
        rescale: Whether to rescale remaining elements
        seed: Random seed for reproducibility

    Returns:
        Sparse, rescaled delta tensor
    """
    if delta.dim() != 2 or drop_rate <= 0:
        return delta

    if seed is not None:
        gen = torch.Generator(device=delta.device)
        gen.manual_seed(seed)
        mask = torch.rand(delta.shape, device=delta.device, dtype=delta.dtype, generator=gen) > drop_rate
    else:
        mask = torch.rand_like(delta.float()) > drop_rate

    sparse_delta = delta * mask

    if rescale and drop_rate < 1.0:
        sparse_delta = sparse_delta / (1.0 - drop_rate)

    return sparse_delta


def apply_ties_single(
    delta: torch.Tensor,
    density: float = 0.3
) -> torch.Tensor:
    """
    Apply TIES-style magnitude trimming to a single delta.

    This is the "Trim" step of TIES: keep only the top-k parameters by magnitude.
    Useful for sparsification even when there's only one delta.

    Args:
        delta: Input delta tensor
        density: Fraction of parameters to keep (0 to 1)

    Returns:
        Trimmed delta tensor
    """
    if delta.dim() != 2 or density >= 1.0:
        return delta

    dtype = delta.dtype
    device = delta.device

    # Flatten and find threshold
    flat = delta.float().view(-1)
    k = max(1, int(flat.numel() * density))

    # Get threshold value (k-th largest magnitude)
    threshold = torch.topk(flat.abs(), k, largest=True).values[-1]

    # Create mask and apply
    mask = flat.abs() >= threshold
    trimmed = flat * mask.float()

    return trimmed.view(delta.shape).to(dtype)


def apply_ties(
    deltas: List[torch.Tensor],
    density: float = 0.3
) -> torch.Tensor:
    """
    TIES-Merging: Trim, Elect Sign, Disjoint Merge.

    Full TIES algorithm for merging multiple delta tensors:
    1. Trim: Keep only top-k parameters by magnitude for each delta
    2. Elect: Choose sign by majority vote
    3. Merge: Average parameters that agree with elected sign

    Args:
        deltas: List of delta tensors to merge
        density: Fraction of parameters to keep in trimming

    Returns:
        Merged delta tensor
    """
    if not deltas:
        return torch.zeros(1)

    if len(deltas) == 1:
        return deltas[0]

    device = deltas[0].device
    dtype = deltas[0].dtype
    shape = deltas[0].shape

    # Ensure all deltas have same shape
    deltas = [
        d if d.shape == shape else torch.zeros(shape, device=device, dtype=dtype)
        for d in deltas
    ]

    stacked = torch.stack([d.float() for d in deltas], dim=0)

    # Trim step: keep top-k by magnitude for each delta
    k = max(1, int(stacked[0].numel() * density))
    abs_flat = stacked.abs().view(len(deltas), -1)

    trimmed = []
    for i in range(len(deltas)):
        threshold = torch.topk(abs_flat[i], k, largest=True).values[-1]
        mask = abs_flat[i] >= threshold
        trimmed_flat = stacked[i].view(-1) * mask.float()
        trimmed.append(trimmed_flat.view(shape))

    stacked_trimmed = torch.stack(trimmed, dim=0)

    # Elect step: majority sign vote
    signs = torch.sign(stacked_trimmed)
    sign_sum = signs.sum(dim=0)
    elected_sign = torch.sign(sign_sum)
    elected_sign = torch.where(elected_sign == 0, torch.ones_like(elected_sign), elected_sign)

    # Merge step: average parameters with matching sign
    merged = torch.zeros(shape, device=device, dtype=dtype)
    counts = torch.zeros(shape, device=device, dtype=dtype)

    for t in stacked_trimmed:
        match = (torch.sign(t) == elected_sign) | (t == 0)
        merged += torch.where(match, t, torch.zeros_like(t))
        counts += match.float()

    counts = torch.clamp(counts, min=1)
    return (merged / counts).to(dtype)


def lsq_mix_weight_from_deltas(
    delta_floor: torch.Tensor,
    delta_ceil: torch.Tensor,
    fallback: float,
    eps: float = 1e-12,
) -> float:
    """
    Compute convex mixing weight w in [0,1] minimizing ||(1-w)Δ_floor + wΔ_ceil||_F.

    Weight-space least-squares mixing rule:

      argmin_w || Δ_floor + w(Δ_ceil - Δ_floor) ||_F^2

      w* = - <Δ_floor, Δ_ceil - Δ_floor> / ||Δ_ceil - Δ_floor||_F^2

    Args:
        delta_floor: First delta tensor
        delta_ceil: Second delta tensor
        fallback: Fallback value if computation fails
        eps: Numerical stability epsilon

    Returns:
        Optimal mixing weight in [0, 1]
    """
    try:
        if delta_floor is None or delta_ceil is None:
            return float(fallback)

        if delta_floor.shape != delta_ceil.shape:
            return float(fallback)

        d0 = delta_floor.float()
        d1 = delta_ceil.float()
        d = d1 - d0

        denom = float(torch.sum(d * d).item())
        if not (denom > eps) or not math.isfinite(denom):
            return float(fallback)

        numer = float((-torch.sum(d0 * d)).item())
        if not math.isfinite(numer):
            return float(fallback)

        w = numer / denom
        if not math.isfinite(w):
            return float(fallback)

        w = max(0.0, min(1.0, w))
        return float(w)
    except Exception:
        return float(fallback)


def weighted_average_deltas(
    deltas: List[torch.Tensor],
    weights: Optional[List[float]] = None
) -> torch.Tensor:
    """
    Compute weighted average of delta tensors.

    Args:
        deltas: List of delta tensors
        weights: Optional weights (default: uniform)

    Returns:
        Weighted average delta
    """
    if not deltas:
        return torch.zeros(1)

    if len(deltas) == 1:
        return deltas[0]

    if weights is None:
        weights = [1.0 / len(deltas)] * len(deltas)
    else:
        # Normalize weights
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        else:
            weights = [1.0 / len(deltas)] * len(deltas)

    device = deltas[0].device
    dtype = deltas[0].dtype
    result = torch.zeros_like(deltas[0])

    for delta, weight in zip(deltas, weights):
        result += weight * delta.to(device=device, dtype=dtype)

    return result


def magnitude_prune_delta(
    delta: torch.Tensor,
    prune_ratio: float = 0.5,
    structured: bool = False,
    dim: int = 0
) -> torch.Tensor:
    """
    Prune delta by magnitude.

    Args:
        delta: Input delta tensor
        prune_ratio: Fraction to prune (0 to 1)
        structured: Use structured (row/column) pruning
        dim: Dimension for structured pruning

    Returns:
        Pruned delta tensor
    """
    if prune_ratio <= 0:
        return delta
    if prune_ratio >= 1:
        return torch.zeros_like(delta)

    if structured and delta.dim() >= 2:
        # Compute norm along the specified dimension
        norms = torch.norm(delta.float(), dim=dim)
        k = max(1, int(norms.numel() * (1 - prune_ratio)))
        threshold = torch.topk(norms.view(-1), k, largest=True).values[-1]
        mask = norms >= threshold

        # Expand mask to match delta shape
        if dim == 0:
            mask = mask.unsqueeze(1).expand_as(delta)
        else:
            mask = mask.unsqueeze(0).expand_as(delta)

        return delta * mask.to(delta.dtype)
    else:
        # Unstructured pruning
        flat = delta.float().view(-1)
        k = max(1, int(flat.numel() * (1 - prune_ratio)))
        threshold = torch.topk(flat.abs(), k, largest=True).values[-1]
        mask = flat.abs() >= threshold
        return (flat * mask.float()).view(delta.shape).to(delta.dtype)


def slerp_delta(
    delta1: torch.Tensor,
    delta2: torch.Tensor,
    t: float = 0.5,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Spherical linear interpolation between two deltas.

    SLERP preserves magnitude better than linear interpolation.

    Args:
        delta1: First delta tensor
        delta2: Second delta tensor
        t: Interpolation parameter (0 to 1)
        eps: Numerical stability epsilon

    Returns:
        Interpolated delta
    """
    if t <= 0:
        return delta1
    if t >= 1:
        return delta2

    # Flatten for computation
    v1 = delta1.float().view(-1)
    v2 = delta2.float().view(-1)

    # Compute cosine similarity
    dot = torch.dot(v1, v2)
    norm1 = torch.norm(v1)
    norm2 = torch.norm(v2)

    if norm1 < eps or norm2 < eps:
        # Fallback to linear interpolation
        return (1 - t) * delta1 + t * delta2

    cos_angle = torch.clamp(dot / (norm1 * norm2), -1.0, 1.0)

    if cos_angle.abs() > 0.9995:
        # Nearly parallel, use linear interpolation
        return (1 - t) * delta1 + t * delta2

    angle = torch.acos(cos_angle)
    sin_angle = torch.sin(angle)

    if sin_angle.abs() < eps:
        return (1 - t) * delta1 + t * delta2

    # SLERP formula
    s1 = torch.sin((1 - t) * angle) / sin_angle
    s2 = torch.sin(t * angle) / sin_angle

    result = s1 * v1 + s2 * v2
    return result.view(delta1.shape).to(delta1.dtype)


class DeltaMerger:
    """
    Unified interface for delta merging strategies.

    Provides methods for various merging approaches with consistent API.
    """

    def __init__(
        self,
        method: str = "average",
        dare_drop_rate: float = 0.3,
        ties_density: float = 0.3,
        seed: Optional[int] = None
    ):
        """
        Initialize merger with specified method.

        Args:
            method: Merging method ('average', 'dare', 'ties', 'slerp')
            dare_drop_rate: Drop rate for DARE method
            ties_density: Density for TIES method
            seed: Random seed for reproducibility
        """
        self.method = method.lower()
        self.dare_drop_rate = dare_drop_rate
        self.ties_density = ties_density
        self.seed = seed

    def merge(
        self,
        deltas: List[torch.Tensor],
        weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        """
        Merge multiple deltas using configured method.

        Args:
            deltas: List of delta tensors
            weights: Optional weights for weighted methods

        Returns:
            Merged delta tensor
        """
        if not deltas:
            return torch.zeros(1)

        if len(deltas) == 1:
            if self.method == "dare":
                return apply_dare(deltas[0], self.dare_drop_rate, seed=self.seed)
            elif self.method == "ties":
                return apply_ties_single(deltas[0], self.ties_density)
            return deltas[0]

        if self.method == "average":
            return weighted_average_deltas(deltas, weights)
        elif self.method == "dare":
            # Apply DARE to each, then average
            processed = [apply_dare(d, self.dare_drop_rate, seed=self.seed) for d in deltas]
            return weighted_average_deltas(processed, weights)
        elif self.method == "ties":
            return apply_ties(deltas, self.ties_density)
        elif self.method == "slerp" and len(deltas) == 2:
            t = 0.5 if weights is None else weights[1] / sum(weights)
            return slerp_delta(deltas[0], deltas[1], t)
        else:
            # Fallback to average
            return weighted_average_deltas(deltas, weights)

    def process_single(self, delta: torch.Tensor) -> torch.Tensor:
        """
        Process a single delta with configured sparsification.

        Args:
            delta: Input delta tensor

        Returns:
            Processed delta
        """
        if self.method == "dare":
            return apply_dare(delta, self.dare_drop_rate, seed=self.seed)
        elif self.method == "ties":
            return apply_ties_single(delta, self.ties_density)
        return delta


def compute_delta_stats(delta: torch.Tensor) -> dict:
    """
    Compute statistics for a delta tensor.

    Args:
        delta: Input delta tensor

    Returns:
        Dictionary with statistics
    """
    flat = delta.float().view(-1)

    return {
        "shape": list(delta.shape),
        "dtype": str(delta.dtype),
        "numel": delta.numel(),
        "mean": float(flat.mean().item()),
        "std": float(flat.std().item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "abs_mean": float(flat.abs().mean().item()),
        "sparsity": float((flat == 0).float().mean().item()),
        "frobenius_norm": float(torch.norm(flat).item()),
    }
