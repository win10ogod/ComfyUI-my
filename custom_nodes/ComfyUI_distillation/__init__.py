"""
ComfyUI Distillation Nodes

Simple SVD-based distillation for extracting LoRA from model deltas.
Uses ComfyUI native MODEL type directly.

Nodes:
- ExtractLoRAFromModels: Extract LoRA by computing delta between two models
- SaveLoRAWeights: Save LoRA weights to file
- LoadLoRAWeights: Load LoRA weights from file
- ApplyLoRAWeightsToModel: Apply LoRA to model
- QuickDistill: All-in-one distillation node
"""

from .cross_arch_distillation import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
__version__ = "3.0.0"
