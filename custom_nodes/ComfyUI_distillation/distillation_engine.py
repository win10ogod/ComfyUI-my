"""
Main Distillation Engine for ComfyUI

This module provides a comprehensive distillation engine that integrates:
- Multiple SVD backends (AURORA, DRF, Randomized, Krylov)
- Cross-architecture projection
- Memory-efficient layer processing
- DARE/TIES merging strategies
- LoRA extraction with PiSSA
"""

import os
import re
import json
import gc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from enum import Enum
from pathlib import Path

import torch
import torch.nn.functional as F

from .svd_algorithms import (
    GPUAcceleratedSVD, SVDConfig, GPUMemoryManager, CUDAStreamManager,
    SVDMode, compute_adaptive_rank
)
from .memory_utils import (
    LayerOffloader, OffloadStrategy, TensorPool,
    get_optimal_offload_strategy, estimate_model_memory
)
from .fp8_utils import (
    is_float8_dtype, fp8_dequantize, FP8WeightHandler,
    resolve_dequant_dtype, safe_weight_dtype
)
from .merge_strategies import (
    apply_dare, apply_ties, apply_ties_single,
    DeltaMerger, compute_delta_stats
)
from .projection import (
    ProjectionConfig, svd_projection, extract_lora_pissa,
    compute_student_svd_basis, teacher_core_in_student_basis,
    extract_lora_from_core_delta, CrossArchProjector
)


class DistillationMode(Enum):
    """Distillation operation modes."""
    LORA = "lora"
    FULL = "full"
    HYBRID = "hybrid"


class LayerType(Enum):
    """Layer type classification."""
    ATTENTION = "attention"
    MLP = "mlp"
    NORM = "norm"
    EMBED = "embed"
    LM_HEAD = "lm_head"
    OTHER = "other"


@dataclass
class DistillationConfig:
    """Comprehensive configuration for distillation."""

    # Paths
    teacher_path: str = ""
    student_path: str = ""
    output_path: str = ""

    # LoRA configuration
    rank_default: int = 64
    rank_attn: Optional[int] = None
    rank_mlp: Optional[int] = None
    lora_alpha: Optional[int] = None
    alpha_mode: str = "auto"  # auto | rank | fixed

    # Adaptive rank
    use_adaptive_rank: bool = True
    energy_threshold: float = 0.95
    min_rank: int = 8
    max_rank: int = 256

    # Delta regularization
    use_dare: bool = False
    dare_drop_rate: float = 0.3
    use_ties: bool = True
    ties_density: float = 0.3
    ties_trim_single: bool = True

    # Memory controls
    gpu_memory_fraction: float = 0.95
    use_cuda_streams: bool = True
    num_cuda_streams: int = 3
    use_mixed_precision: bool = True
    offload_strategy: str = "cpu"  # cpu | disk | none
    offload_dir: Optional[str] = None
    prefetch_layers: int = 2
    use_pinned_memory: bool = True
    max_cpu_memory_gb: float = 64.0

    # SVD configuration
    svd_mode: str = "auto"  # auto | full | randomized | krylov | drf | aurora
    svd_randomized_iter: int = 2
    svd_randomized_oversamples: int = 8
    svd_auto_lowrank: str = "randomized"

    # AURORA-SVD parameters
    aurora_steps: int = 1
    aurora_order: int = 2
    aurora_theta1: float = 0.5
    aurora_theta2: float = 0.25
    aurora_early_stop: float = 0.005

    # DRF-SVD parameters
    drf_steps: int = 1
    drf_theta: float = 0.5

    # Projection configuration
    use_svd_projection: bool = True
    projection_rank: int = 256
    projection_min_rank: int = 8
    projection_adaptive_rank: bool = True
    projection_energy_threshold: float = 0.99

    # FP8 handling
    dequantize_float8: bool = True
    dequantize_dtype: str = "auto"
    fp8_block_size: int = 128

    # Module selection
    include_pattern: str = "self_attn|mlp"
    exclude_pattern: str = ""
    include_embed_lm_head: bool = False

    # Delta stability
    max_delta_ratio: float = 0.35

    # Misc
    seed: int = 42
    verbose: bool = True

    def to_svd_config(self) -> SVDConfig:
        """Convert to SVDConfig."""
        return SVDConfig(
            svd_mode=self.svd_mode,
            use_mixed_precision=self.use_mixed_precision,
            randomized_iter=self.svd_randomized_iter,
            randomized_oversamples=self.svd_randomized_oversamples,
            auto_lowrank=self.svd_auto_lowrank,
            aurora_steps=self.aurora_steps,
            aurora_order=self.aurora_order,
            aurora_theta1=self.aurora_theta1,
            aurora_theta2=self.aurora_theta2,
            aurora_early_stop=self.aurora_early_stop,
            drf_steps=self.drf_steps,
            drf_theta=self.drf_theta,
            verbose=self.verbose,
        )

    def to_projection_config(self) -> ProjectionConfig:
        """Convert to ProjectionConfig."""
        return ProjectionConfig(
            projection_rank=self.projection_rank,
            projection_min_rank=self.projection_min_rank,
            projection_adaptive_rank=self.projection_adaptive_rank,
            projection_energy_threshold=self.projection_energy_threshold,
            use_adaptive_rank=self.use_adaptive_rank,
            energy_threshold=self.energy_threshold,
            min_rank=self.min_rank,
            max_rank=self.max_rank,
        )


@dataclass
class LoRAOutput:
    """Container for LoRA extraction results."""
    lora_A: torch.Tensor
    lora_B: torch.Tensor
    rank: int
    key: str
    alpha: int
    original_shape: Tuple[int, ...]


@dataclass
class DistillationResult:
    """Container for distillation results."""
    lora_outputs: Dict[str, LoRAOutput] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    skipped_keys: List[str] = field(default_factory=list)
    error_keys: Dict[str, str] = field(default_factory=dict)


# Layer classification patterns
NORM_TOKENS = ["layernorm", "rmsnorm", "layer_norm", "rms_norm", "ln_", "norm"]
EMBED_TOKENS = ["embed", "wte", "wpe"]
LM_HEAD_TOKENS = ["lm_head", "output.weight"]
ATTN_TOKENS = ["self_attn", "attention", "attn", "q_proj", "k_proj", "v_proj", "o_proj"]
MLP_TOKENS = ["mlp", "feed_forward", "ffn", "gate_proj", "up_proj", "down_proj"]


def classify_layer(key: str) -> LayerType:
    """Classify a layer by its key."""
    key_lower = key.lower()

    for token in NORM_TOKENS:
        if token in key_lower:
            return LayerType.NORM

    for token in LM_HEAD_TOKENS:
        if token in key_lower:
            return LayerType.LM_HEAD

    for token in EMBED_TOKENS:
        if token in key_lower:
            return LayerType.EMBED

    for token in ATTN_TOKENS:
        if token in key_lower:
            return LayerType.ATTENTION

    for token in MLP_TOKENS:
        if token in key_lower:
            return LayerType.MLP

    return LayerType.OTHER


def parse_layer_index(key: str) -> Optional[int]:
    """Extract layer index from key."""
    patterns = [
        r'layers\.(\d+)\.',
        r'\.h\.(\d+)\.',
        r'blocks\.(\d+)\.',
        r'layer\.(\d+)\.',
    ]

    for pattern in patterns:
        match = re.search(pattern, key)
        if match:
            return int(match.group(1))

    return None


def should_include_key(
    key: str,
    include_pattern: str,
    exclude_pattern: str,
    include_embed_lm_head: bool
) -> bool:
    """Determine if a key should be included for distillation."""
    layer_type = classify_layer(key)

    # Skip norms
    if layer_type == LayerType.NORM:
        return False

    # Handle embed/lm_head
    if layer_type in (LayerType.EMBED, LayerType.LM_HEAD):
        return include_embed_lm_head

    # Apply include/exclude patterns
    if include_pattern:
        if not re.search(include_pattern, key):
            return False

    if exclude_pattern:
        if re.search(exclude_pattern, key):
            return False

    return True


class DistillationEngine:
    """
    Main distillation engine for cross-architecture knowledge distillation.

    Features:
    - Multiple SVD backends (AURORA, DRF, Randomized, Krylov)
    - Memory-efficient layer-by-layer processing
    - Cross-architecture projection with student subspace alignment
    - DARE/TIES delta merging strategies
    - FP8/quantized weight handling
    - LoRA extraction with adaptive rank selection
    """

    def __init__(
        self,
        config: DistillationConfig,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.device = device or torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        # Initialize components
        self._init_memory_manager()
        self._init_svd_engine()
        self._init_offloader()
        self._init_fp8_handler()
        self._init_delta_merger()

        # Projector for cross-arch operations
        self.projector = CrossArchProjector(
            device=self.device,
            config=config.to_projection_config(),
            svd_config=config.to_svd_config(),
        )

        # Statistics tracking
        self.stats = {
            "layers_processed": 0,
            "layers_skipped": 0,
            "total_params": 0,
            "lora_params": 0,
        }

    def _init_memory_manager(self):
        """Initialize memory management."""
        self.memory_manager = GPUMemoryManager(
            self.device,
            memory_fraction=self.config.gpu_memory_fraction,
            verbose=self.config.verbose,
        )

        if self.config.use_cuda_streams and self.device.type == "cuda":
            self.stream_manager = CUDAStreamManager(
                self.device,
                num_streams=self.config.num_cuda_streams
            )
        else:
            self.stream_manager = None

    def _init_svd_engine(self):
        """Initialize SVD engine."""
        self.svd_engine = GPUAcceleratedSVD(
            device=self.device,
            memory_manager=self.memory_manager,
            stream_manager=self.stream_manager,
            config=self.config.to_svd_config(),
        )

    def _init_offloader(self):
        """Initialize layer offloader."""
        strategy = OffloadStrategy(self.config.offload_strategy.lower())

        self.offloader = LayerOffloader(
            strategy=strategy,
            device=self.device,
            offload_dir=self.config.offload_dir,
            use_pinned=self.config.use_pinned_memory,
            max_cpu_gb=self.config.max_cpu_memory_gb,
            prefetch_count=self.config.prefetch_layers,
            verbose=self.config.verbose,
        )

    def _init_fp8_handler(self):
        """Initialize FP8 handler."""
        self.fp8_handler = FP8WeightHandler(
            block_size=self.config.fp8_block_size,
            target_dtype=self.config.dequantize_dtype,
            device=self.device,
        )

    def _init_delta_merger(self):
        """Initialize delta merger."""
        if self.config.use_dare:
            method = "dare"
        elif self.config.use_ties:
            method = "ties"
        else:
            method = "average"

        self.delta_merger = DeltaMerger(
            method=method,
            dare_drop_rate=self.config.dare_drop_rate,
            ties_density=self.config.ties_density,
            seed=self.config.seed,
        )

    def _get_rank_for_key(self, key: str) -> int:
        """Get appropriate rank for a layer key."""
        layer_type = classify_layer(key)

        if layer_type == LayerType.ATTENTION and self.config.rank_attn is not None:
            return self.config.rank_attn
        elif layer_type == LayerType.MLP and self.config.rank_mlp is not None:
            return self.config.rank_mlp
        else:
            return self.config.rank_default

    def _get_alpha_for_rank(self, rank: int) -> int:
        """Get LoRA alpha for a given rank."""
        if self.config.lora_alpha is not None:
            return self.config.lora_alpha

        mode = self.config.alpha_mode.lower()
        if mode == "rank":
            return rank
        elif mode == "fixed":
            return self.config.rank_default
        else:  # auto
            return rank

    def _load_weight(
        self,
        key: str,
        weight: torch.Tensor,
        scale_key: Optional[str] = None,
        scale_inv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Load and dequantize a weight tensor."""
        if self.config.dequantize_float8 and is_float8_dtype(weight.dtype):
            if scale_inv is not None:
                weight = fp8_dequantize(weight, scale_inv, self.config.fp8_block_size)
            else:
                weight = self.fp8_handler.dequantize(weight, key)

        return weight.to(self.device).float()

    def _compute_delta(
        self,
        teacher_weight: torch.Tensor,
        student_weight: torch.Tensor,
        key: str,
    ) -> torch.Tensor:
        """Compute delta between teacher and student weights."""
        # Handle shape mismatch with projection
        if teacher_weight.shape != student_weight.shape:
            if self.config.use_svd_projection:
                teacher_weight = svd_projection(
                    teacher_weight,
                    student_weight.shape,
                    self.svd_engine,
                    self.config.to_projection_config(),
                    self.device,
                    ref=student_weight,
                )
            else:
                # Simple resize fallback
                if teacher_weight.dim() == 2:
                    teacher_weight = F.interpolate(
                        teacher_weight.unsqueeze(0).unsqueeze(0),
                        size=student_weight.shape,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0)

        delta = teacher_weight - student_weight

        # Apply delta stability guard
        if self.config.max_delta_ratio > 0:
            delta_norm = torch.norm(delta)
            student_norm = torch.norm(student_weight)
            if student_norm > 1e-10:
                ratio = delta_norm / student_norm
                if ratio > self.config.max_delta_ratio:
                    scale = self.config.max_delta_ratio / ratio
                    delta = delta * scale
                    if self.config.verbose:
                        print(f"    Scaled delta for {key}: ratio {ratio:.3f} -> {self.config.max_delta_ratio:.3f}")

        return delta

    def extract_lora_from_delta(
        self,
        delta: torch.Tensor,
        key: str,
        student_weight: Optional[torch.Tensor] = None,
    ) -> Optional[LoRAOutput]:
        """Extract LoRA factors from a delta tensor."""
        if delta.dim() != 2:
            return None

        rank = self._get_rank_for_key(key)

        # Apply DARE/TIES if configured
        if self.config.use_dare:
            delta = apply_dare(delta, self.config.dare_drop_rate)
        elif self.config.use_ties and self.config.ties_trim_single:
            delta = apply_ties_single(delta, self.config.ties_density)

        # Extract LoRA using PiSSA
        lora_A, lora_B, actual_rank = extract_lora_pissa(
            delta,
            rank,
            self.svd_engine,
            self.config.to_projection_config(),
        )

        if lora_A is None or lora_B is None:
            return None

        alpha = self._get_alpha_for_rank(actual_rank)

        return LoRAOutput(
            lora_A=lora_A,
            lora_B=lora_B,
            rank=actual_rank,
            key=key,
            alpha=alpha,
            original_shape=tuple(delta.shape),
        )

    def process_layer(
        self,
        key: str,
        teacher_weight: torch.Tensor,
        student_weight: torch.Tensor,
        teacher_scale_inv: Optional[torch.Tensor] = None,
        student_scale_inv: Optional[torch.Tensor] = None,
    ) -> Optional[LoRAOutput]:
        """Process a single layer for distillation."""
        try:
            # Load and dequantize weights
            t_weight = self._load_weight(key, teacher_weight, scale_inv=teacher_scale_inv)
            s_weight = self._load_weight(key, student_weight, scale_inv=student_scale_inv)

            # Compute delta
            delta = self._compute_delta(t_weight, s_weight, key)

            # Extract LoRA
            result = self.extract_lora_from_delta(delta, key, s_weight)

            if result is not None:
                self.stats["layers_processed"] += 1
                self.stats["total_params"] += delta.numel()
                self.stats["lora_params"] += result.lora_A.numel() + result.lora_B.numel()

            return result

        except Exception as e:
            if self.config.verbose:
                print(f"Error processing {key}: {e}")
            return None
        finally:
            # Clean up
            self.memory_manager.clear_cache()

    def distill_models(
        self,
        teacher_state_dict: Dict[str, torch.Tensor],
        student_state_dict: Dict[str, torch.Tensor],
        teacher_scales: Optional[Dict[str, torch.Tensor]] = None,
        student_scales: Optional[Dict[str, torch.Tensor]] = None,
    ) -> DistillationResult:
        """
        Distill knowledge from teacher to student model.

        Args:
            teacher_state_dict: Teacher model weights
            student_state_dict: Student model weights
            teacher_scales: Optional FP8 scale tensors for teacher
            student_scales: Optional FP8 scale tensors for student

        Returns:
            DistillationResult with LoRA outputs and statistics
        """
        result = DistillationResult()
        teacher_scales = teacher_scales or {}
        student_scales = student_scales or {}

        # Find common keys
        teacher_keys = set(teacher_state_dict.keys())
        student_keys = set(student_state_dict.keys())

        # Get processable keys
        common_weight_keys = []
        for key in teacher_keys:
            if key in student_keys:
                if should_include_key(
                    key,
                    self.config.include_pattern,
                    self.config.exclude_pattern,
                    self.config.include_embed_lm_head,
                ):
                    if ".weight" in key:  # Only process weight tensors
                        common_weight_keys.append(key)

        if self.config.verbose:
            print(f"Found {len(common_weight_keys)} layers to process")

        # Sort by layer index for better prefetching
        common_weight_keys.sort(key=lambda k: (parse_layer_index(k) or 0, k))

        # Process each layer
        for i, key in enumerate(common_weight_keys):
            if self.config.verbose and (i + 1) % 10 == 0:
                print(f"Processing layer {i + 1}/{len(common_weight_keys)}: {key}")

            # Get scale tensors if available
            t_scale = teacher_scales.get(key.replace(".weight", "_scale_inv"))
            s_scale = student_scales.get(key.replace(".weight", "_scale_inv"))

            lora_output = self.process_layer(
                key,
                teacher_state_dict[key],
                student_state_dict[key],
                teacher_scale_inv=t_scale,
                student_scale_inv=s_scale,
            )

            if lora_output is not None:
                result.lora_outputs[key] = lora_output
            else:
                result.skipped_keys.append(key)
                self.stats["layers_skipped"] += 1

        result.statistics = self.stats.copy()
        return result

    def save_lora_adapter(
        self,
        result: DistillationResult,
        output_path: str,
        base_model_name: Optional[str] = None,
    ):
        """
        Save LoRA adapter in PEFT-compatible format.

        Args:
            result: Distillation result
            output_path: Output directory path
            base_model_name: Optional base model name for config
        """
        os.makedirs(output_path, exist_ok=True)

        # Prepare state dict
        lora_state_dict = {}
        target_modules = set()

        for key, lora_output in result.lora_outputs.items():
            # Convert key to PEFT format
            base_key = key.replace(".weight", "")
            lora_state_dict[f"base_model.model.{base_key}.lora_A.weight"] = lora_output.lora_A
            lora_state_dict[f"base_model.model.{base_key}.lora_B.weight"] = lora_output.lora_B

            # Track target modules
            parts = base_key.split(".")
            if len(parts) >= 2:
                target_modules.add(parts[-1])

        # Save weights
        try:
            from safetensors.torch import save_file
            save_file(lora_state_dict, os.path.join(output_path, "adapter_model.safetensors"))
        except ImportError:
            torch.save(lora_state_dict, os.path.join(output_path, "adapter_model.bin"))

        # Create adapter config
        # Get representative rank (use most common)
        ranks = [o.rank for o in result.lora_outputs.values()]
        alphas = [o.alpha for o in result.lora_outputs.values()]
        r = max(set(ranks), key=ranks.count) if ranks else self.config.rank_default
        alpha = max(set(alphas), key=alphas.count) if alphas else r

        adapter_config = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": 0.0,
            "target_modules": list(target_modules),
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "peft_type": "LORA",
        }

        if base_model_name:
            adapter_config["base_model_name_or_path"] = base_model_name

        with open(os.path.join(output_path, "adapter_config.json"), "w") as f:
            json.dump(adapter_config, f, indent=2)

        if self.config.verbose:
            print(f"Saved LoRA adapter to {output_path}")
            print(f"  - Layers: {len(result.lora_outputs)}")
            print(f"  - Rank: {r}, Alpha: {alpha}")
            print(f"  - Target modules: {sorted(target_modules)}")

    def cleanup(self):
        """Clean up resources."""
        self.offloader.cleanup()
        self.memory_manager.clear_cache()


def create_distillation_engine(
    teacher_path: Optional[str] = None,
    student_path: Optional[str] = None,
    output_path: Optional[str] = None,
    rank: int = 64,
    svd_mode: str = "aurora",
    use_dare: bool = False,
    use_ties: bool = True,
    device: Optional[str] = None,
    **kwargs
) -> DistillationEngine:
    """
    Factory function to create a distillation engine with common settings.

    Args:
        teacher_path: Path to teacher model
        student_path: Path to student model
        output_path: Path for output
        rank: Default LoRA rank
        svd_mode: SVD algorithm mode
        use_dare: Enable DARE sparsification
        use_ties: Enable TIES trimming
        device: Target device
        **kwargs: Additional config options

    Returns:
        Configured DistillationEngine
    """
    config = DistillationConfig(
        teacher_path=teacher_path or "",
        student_path=student_path or "",
        output_path=output_path or "",
        rank_default=rank,
        svd_mode=svd_mode,
        use_dare=use_dare,
        use_ties=use_ties,
        **kwargs
    )

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    return DistillationEngine(config, torch.device(device))
