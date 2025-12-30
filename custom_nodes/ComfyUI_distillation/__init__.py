"""
ComfyUI Cross-Architecture Distillation Nodes

This package provides comprehensive support for knowledge distillation between
different diffusion model architectures supported by ComfyUI.

Supported architectures and distillation pairs:
- SDXL → SD3
- SDXL → FLUX.1-dev
- SDXL → Z-Image-Turbo
- Z-Image-Turbo → SD3
- Z-Image-Turbo → FLUX.1-dev
- FLUX.1-dev → FLUX.2-dev
- Qwen-Image → SDXL
- Qwen-Image → Z-Image-Turbo
- And all other ComfyUI supported models

Features:
- Single-file distilled checkpoint output
- Style transfer between architectures
- Knowledge distillation with SVD-based LoRA
- Cross-architecture feature mapping with learned projections
- Support for different latent formats and context dimensions
- Memory-efficient streaming distillation

SVD Backends:
- AURORA-SVD: Adaptive Unrolled Residual-Order Range Augmentation SVD
- DRF-SVD: Dual-Residual Folding SVD
- Randomized SVD: Halko et al. Algorithm 4.4
- Block Krylov SVD: Krylov subspace method
- Adaptive Range Finder: Halko et al. Algorithm 4.2

Additional Features:
- DARE: Drop And REscale for delta sparsification
- TIES: Trim, Elect Sign, and Disjoint Merge
- FP8/Quantized weight handling
- Memory-efficient layer offloading (CPU/disk)
- PiSSA-style LoRA extraction
"""

# Core algorithm modules (can be imported directly)
from .svd_algorithms import (
    GPUAcceleratedSVD,
    SVDConfig,
    SVDMode,
    GPUMemoryManager,
    CUDAStreamManager,
    compute_adaptive_rank,
)

from .memory_utils import (
    LayerOffloader,
    OffloadStrategy,
    TensorPool,
    get_optimal_offload_strategy,
    estimate_model_memory,
)

from .fp8_utils import (
    is_float8_dtype,
    fp8_dequantize,
    FP8WeightHandler,
    resolve_dequant_dtype,
    check_fp8_compatibility,
)

from .merge_strategies import (
    apply_dare,
    apply_ties,
    apply_ties_single,
    DeltaMerger,
    compute_delta_stats,
    slerp_delta,
)

from .projection import (
    ProjectionConfig,
    svd_projection,
    extract_lora_pissa,
    CrossArchProjector,
    compute_student_svd_basis,
    teacher_core_in_student_basis,
)

from .distillation_engine import (
    DistillationEngine,
    DistillationConfig,
    DistillationMode,
    DistillationResult,
    LoRAOutput,
    create_distillation_engine,
)

# Import base distillation nodes
from .cross_arch_distillation import (
    NODE_CLASS_MAPPINGS as BASE_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as BASE_NODE_DISPLAY_NAME_MAPPINGS,
)

# Import universal cross-architecture distillation nodes
from .universal_cross_arch_distillation import (
    NODE_CLASS_MAPPINGS as UNIVERSAL_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as UNIVERSAL_NODE_DISPLAY_NAME_MAPPINGS,
)

# Merge all node mappings
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(BASE_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(UNIVERSAL_NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(BASE_NODE_DISPLAY_NAME_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(UNIVERSAL_NODE_DISPLAY_NAME_MAPPINGS)

__all__ = [
    # Node mappings for ComfyUI
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    # SVD algorithms
    "GPUAcceleratedSVD",
    "SVDConfig",
    "SVDMode",
    "GPUMemoryManager",
    "CUDAStreamManager",
    "compute_adaptive_rank",
    # Memory utilities
    "LayerOffloader",
    "OffloadStrategy",
    "TensorPool",
    "get_optimal_offload_strategy",
    "estimate_model_memory",
    # FP8 utilities
    "is_float8_dtype",
    "fp8_dequantize",
    "FP8WeightHandler",
    "resolve_dequant_dtype",
    "check_fp8_compatibility",
    # Merge strategies
    "apply_dare",
    "apply_ties",
    "apply_ties_single",
    "DeltaMerger",
    "compute_delta_stats",
    "slerp_delta",
    # Projection
    "ProjectionConfig",
    "svd_projection",
    "extract_lora_pissa",
    "CrossArchProjector",
    "compute_student_svd_basis",
    "teacher_core_in_student_basis",
    # Distillation engine
    "DistillationEngine",
    "DistillationConfig",
    "DistillationMode",
    "DistillationResult",
    "LoRAOutput",
    "create_distillation_engine",
]
__version__ = "2.0.0"
