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
"""

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

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
__version__ = "1.0.0"
