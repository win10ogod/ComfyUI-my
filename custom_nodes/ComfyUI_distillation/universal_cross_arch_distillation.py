#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal Cross-Architecture Distillation for ComfyUI

Supports distillation between any ComfyUI supported model architecture:
- SDXL ↔ SD3
- SDXL ↔ Z-Image-Turbo
- SDXL ↔ FLUX.1-dev
- Z-Image-Turbo ↔ SD3
- Z-Image-Turbo ↔ FLUX.1-dev
- FLUX.1-dev ↔ FLUX.2-dev
- Qwen-Image ↔ SDXL
- Qwen-Image ↔ Z-Image-Turbo
- And all other architecture combinations

Outputs:
- Single-file distilled checkpoint (safetensors)
- LoRA adapter for style transfer
"""

import os
import gc
import json
import math
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from safetensors.torch import save_file, load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    logging.warning("safetensors not found, will use torch.save for output")

import folder_paths
import comfy.model_management
import comfy.utils
import comfy.sd
import comfy.model_patcher
import comfy.model_base
import comfy.supported_models
import comfy.latent_formats


# =============================================================================
#                    ARCHITECTURE SPECIFICATIONS
# =============================================================================

class ModelArchType(Enum):
    """Supported model architecture types."""
    SDXL = "sdxl"
    SD3 = "sd3"
    FLUX_DEV = "flux_dev"
    FLUX2_DEV = "flux2_dev"
    Z_IMAGE_TURBO = "z_image_turbo"
    QWEN_IMAGE = "qwen_image"
    HUNYUAN_IMAGE = "hunyuan_image"
    LUMINA2 = "lumina2"
    PIXART_SIGMA = "pixart_sigma"
    SD15 = "sd15"
    SD21 = "sd21"
    UNKNOWN = "unknown"


@dataclass
class ArchSpec:
    """Specification of a model architecture for distillation."""
    arch_type: ModelArchType
    context_dim: int
    model_channels: int
    latent_channels: int = 4
    latent_format: str = "SD15"
    is_dit: bool = False
    is_flow: bool = False
    use_guidance_embed: bool = False
    hidden_size: int = 0
    num_layers: int = 0
    patch_size: Tuple[int, ...] = (2, 2)
    supports_cfg: bool = True


# Architecture specifications database
ARCH_SPECS = {
    ModelArchType.SDXL: ArchSpec(
        arch_type=ModelArchType.SDXL,
        context_dim=2048,
        model_channels=320,
        latent_channels=4,
        latent_format="SDXL",
        is_dit=False,
        is_flow=False,
        hidden_size=2816,
    ),
    ModelArchType.SD3: ArchSpec(
        arch_type=ModelArchType.SD3,
        context_dim=4096,
        model_channels=1536,
        latent_channels=16,
        latent_format="SD3",
        is_dit=True,
        is_flow=True,
        hidden_size=1536,
        num_layers=24,
    ),
    ModelArchType.FLUX_DEV: ArchSpec(
        arch_type=ModelArchType.FLUX_DEV,
        context_dim=4096,
        model_channels=3072,
        latent_channels=16,
        latent_format="Flux",
        is_dit=True,
        is_flow=True,
        use_guidance_embed=True,
        hidden_size=3072,
        num_layers=19,
    ),
    ModelArchType.FLUX2_DEV: ArchSpec(
        arch_type=ModelArchType.FLUX2_DEV,
        context_dim=4096,
        model_channels=3072,
        latent_channels=16,
        latent_format="Flux2",
        is_dit=True,
        is_flow=True,
        use_guidance_embed=False,
        hidden_size=3072,
        num_layers=19,
    ),
    ModelArchType.Z_IMAGE_TURBO: ArchSpec(
        arch_type=ModelArchType.Z_IMAGE_TURBO,
        context_dim=4096,
        model_channels=3840,
        latent_channels=16,
        latent_format="Flux",
        is_dit=True,
        is_flow=True,
        hidden_size=3840,
        num_layers=28,
    ),
    ModelArchType.QWEN_IMAGE: ArchSpec(
        arch_type=ModelArchType.QWEN_IMAGE,
        context_dim=4096,
        model_channels=2560,
        latent_channels=16,
        latent_format="Wan21",
        is_dit=True,
        is_flow=True,
        hidden_size=2560,
    ),
    ModelArchType.HUNYUAN_IMAGE: ArchSpec(
        arch_type=ModelArchType.HUNYUAN_IMAGE,
        context_dim=4096,
        model_channels=3072,
        latent_channels=16,
        latent_format="HunyuanImage21",
        is_dit=True,
        is_flow=True,
        hidden_size=3072,
    ),
    ModelArchType.LUMINA2: ArchSpec(
        arch_type=ModelArchType.LUMINA2,
        context_dim=4096,
        model_channels=2048,
        latent_channels=16,
        latent_format="Flux",
        is_dit=True,
        is_flow=True,
        hidden_size=2048,
    ),
    ModelArchType.PIXART_SIGMA: ArchSpec(
        arch_type=ModelArchType.PIXART_SIGMA,
        context_dim=4096,
        model_channels=1152,
        latent_channels=4,
        latent_format="SDXL",
        is_dit=True,
        is_flow=False,
        hidden_size=1152,
    ),
    ModelArchType.SD15: ArchSpec(
        arch_type=ModelArchType.SD15,
        context_dim=768,
        model_channels=320,
        latent_channels=4,
        latent_format="SD15",
        is_dit=False,
        is_flow=False,
    ),
    ModelArchType.SD21: ArchSpec(
        arch_type=ModelArchType.SD21,
        context_dim=1024,
        model_channels=320,
        latent_channels=4,
        latent_format="SD15",
        is_dit=False,
        is_flow=False,
    ),
}


def detect_model_arch_type(model_patcher: comfy.model_patcher.ModelPatcher) -> ModelArchType:
    """Detect the architecture type of a loaded model."""
    model = model_patcher.model
    model_config = model.model_config

    # Get config info
    unet_config = model_config.unet_config.copy() if hasattr(model_config, 'unet_config') else {}
    class_name = model_config.__class__.__name__ if hasattr(model_config, '__class__') else ""
    image_model = unet_config.get("image_model", "")

    # Check specific architectures
    if "flux2" in image_model.lower() or "Flux2" in class_name:
        return ModelArchType.FLUX2_DEV
    if "flux" in image_model.lower() or "Flux" in class_name:
        if unet_config.get("guidance_embed", False):
            return ModelArchType.FLUX_DEV
        return ModelArchType.FLUX_DEV
    if "lumina2" in image_model.lower() or "ZImage" in class_name or "Lumina2" in class_name:
        dim = unet_config.get("dim", 0)
        if dim >= 3500:
            return ModelArchType.Z_IMAGE_TURBO
        return ModelArchType.LUMINA2
    if "qwen_image" in image_model.lower() or "QwenImage" in class_name:
        return ModelArchType.QWEN_IMAGE
    if "hunyuan" in image_model.lower() and "image" in image_model.lower():
        return ModelArchType.HUNYUAN_IMAGE
    if "SD3" in class_name or (unet_config.get("in_channels") == 16 and "pos_embed" in str(unet_config)):
        return ModelArchType.SD3
    if "SDXL" in class_name:
        return ModelArchType.SDXL
    if "pixart" in image_model.lower() or "PixArt" in class_name:
        return ModelArchType.PIXART_SIGMA
    if "SD21" in class_name or "SD20" in class_name:
        return ModelArchType.SD21

    # Default to SD15 for unknown
    return ModelArchType.SD15


def get_arch_spec(model_patcher: comfy.model_patcher.ModelPatcher) -> ArchSpec:
    """Get the architecture specification for a model."""
    arch_type = detect_model_arch_type(model_patcher)
    return ARCH_SPECS.get(arch_type, ARCH_SPECS[ModelArchType.SD15])


# =============================================================================
#                    CROSS-ARCHITECTURE CONVERSION
# =============================================================================

class CrossArchConverter(nn.Module):
    """Converts weights between different architectures."""

    def __init__(
        self,
        source_spec: ArchSpec,
        target_spec: ArchSpec,
        rank: int = 64,
        use_svd: bool = True,
    ):
        super().__init__()
        self.source_spec = source_spec
        self.target_spec = target_spec
        self.rank = rank
        self.use_svd = use_svd

        # Context dimension projector
        if source_spec.context_dim != target_spec.context_dim:
            self.context_proj = nn.Linear(
                source_spec.context_dim,
                target_spec.context_dim,
                bias=False
            )
        else:
            self.context_proj = None

        # Hidden dimension projector
        if source_spec.model_channels != target_spec.model_channels:
            self.hidden_proj = nn.Linear(
                source_spec.model_channels,
                target_spec.model_channels,
                bias=False
            )
        else:
            self.hidden_proj = None

    def project_context(self, x: torch.Tensor) -> torch.Tensor:
        """Project context embeddings."""
        if self.context_proj is not None:
            return self.context_proj(x)
        return x

    def project_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Project hidden states."""
        if self.hidden_proj is not None:
            return self.hidden_proj(x)
        return x


# =============================================================================
#                    SVD-BASED WEIGHT TRANSFER
# =============================================================================

class WeightTransfer:
    """Transfer weights between architectures using SVD decomposition."""

    def __init__(
        self,
        rank: int = 64,
        alpha: Optional[float] = None,
        energy_threshold: float = 0.95,
    ):
        self.rank = rank
        self.alpha = alpha if alpha is not None else float(rank)
        self.energy_threshold = energy_threshold

    def compute_transfer_weights(
        self,
        source_weight: torch.Tensor,
        target_shape: Tuple[int, ...],
        mode: str = "svd",  # svd, interpolate, project
    ) -> torch.Tensor:
        """Compute transferred weights for target shape."""
        device = source_weight.device
        dtype = source_weight.dtype

        if source_weight.shape == target_shape:
            return source_weight.clone()

        source_2d = source_weight.reshape(source_weight.shape[0], -1)
        target_2d_shape = (target_shape[0], np.prod(target_shape[1:]) if len(target_shape) > 1 else 1)

        if mode == "svd":
            return self._svd_transfer(source_2d, target_2d_shape, target_shape, device, dtype)
        elif mode == "interpolate":
            return self._interpolate_transfer(source_weight, target_shape, device, dtype)
        else:  # project
            return self._project_transfer(source_2d, target_2d_shape, target_shape, device, dtype)

    def _svd_transfer(
        self,
        source_2d: torch.Tensor,
        target_2d_shape: Tuple[int, int],
        target_shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Transfer using SVD decomposition and recomposition."""
        try:
            U, S, Vh = torch.linalg.svd(source_2d.float(), full_matrices=False)
        except Exception:
            # Fallback to randomized SVD for large matrices
            U, S, Vh = self._randomized_svd(source_2d.float(), min(self.rank, min(source_2d.shape)))

        # Select effective rank
        effective_rank = min(self.rank, len(S))

        # Energy-based rank selection
        total_energy = (S ** 2).sum()
        cumsum = torch.cumsum(S ** 2, dim=0)
        energy_ratio = cumsum / total_energy
        mask = energy_ratio >= self.energy_threshold
        if mask.any():
            adaptive_rank = min(mask.nonzero()[0].item() + 1, effective_rank)
        else:
            adaptive_rank = effective_rank

        # Create scaled factors
        U_r = U[:, :adaptive_rank]
        S_r = S[:adaptive_rank]
        Vh_r = Vh[:adaptive_rank, :]

        # Resize to target dimensions
        target_out, target_in = target_2d_shape
        source_out, source_in = source_2d.shape

        # Scale factors for dimension change
        out_scale = math.sqrt(target_out / source_out)
        in_scale = math.sqrt(target_in / source_in)

        # Create target-sized U and Vh
        if target_out != source_out:
            U_target = F.interpolate(
                U_r.unsqueeze(0).unsqueeze(0),
                size=(target_out, adaptive_rank),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0) * out_scale
        else:
            U_target = U_r

        if target_in != source_in:
            Vh_target = F.interpolate(
                Vh_r.unsqueeze(0).unsqueeze(0),
                size=(adaptive_rank, target_in),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0) * in_scale
        else:
            Vh_target = Vh_r

        # Reconstruct
        result_2d = U_target @ torch.diag(S_r) @ Vh_target
        return result_2d.reshape(target_shape).to(dtype)

    def _interpolate_transfer(
        self,
        source_weight: torch.Tensor,
        target_shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Transfer using interpolation."""
        if len(source_weight.shape) == 2:
            # Linear layer weight
            result = F.interpolate(
                source_weight.float().unsqueeze(0).unsqueeze(0),
                size=target_shape,
                mode='bilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
        elif len(source_weight.shape) == 4:
            # Conv layer weight
            result = F.interpolate(
                source_weight.float(),
                size=target_shape[2:],
                mode='bilinear',
                align_corners=False
            )
            # Handle channel dimension changes
            if source_weight.shape[0] != target_shape[0] or source_weight.shape[1] != target_shape[1]:
                result = F.interpolate(
                    result.permute(2, 3, 0, 1),
                    size=(target_shape[0], target_shape[1]),
                    mode='bilinear',
                    align_corners=False
                ).permute(2, 3, 0, 1)
        else:
            result = source_weight.float()

        return result.to(dtype)

    def _project_transfer(
        self,
        source_2d: torch.Tensor,
        target_2d_shape: Tuple[int, int],
        target_shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Transfer using random projection."""
        target_out, target_in = target_2d_shape
        source_out, source_in = source_2d.shape

        # Create projection matrices
        if source_out != target_out:
            proj_out = torch.randn(target_out, source_out, device=device) / math.sqrt(source_out)
        else:
            proj_out = torch.eye(source_out, device=device)

        if source_in != target_in:
            proj_in = torch.randn(source_in, target_in, device=device) / math.sqrt(source_in)
        else:
            proj_in = torch.eye(source_in, device=device)

        result_2d = proj_out @ source_2d.float() @ proj_in
        return result_2d.reshape(target_shape).to(dtype)

    def _randomized_svd(
        self,
        A: torch.Tensor,
        k: int,
        n_iter: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomized SVD for large matrices."""
        m, n = A.shape
        k = min(k, m, n)

        # Random projection
        P = torch.randn(n, k + 10, device=A.device, dtype=A.dtype)
        Q = A @ P

        for _ in range(n_iter):
            Q, _ = torch.linalg.qr(Q)
            Q = A @ (A.T @ Q)
            Q, _ = torch.linalg.qr(Q)

        # Final QR
        Q, _ = torch.linalg.qr(A @ P)

        # Project and compute SVD
        B = Q.T @ A
        U_small, S, Vh = torch.linalg.svd(B, full_matrices=False)
        U = Q @ U_small

        return U[:, :k], S[:k], Vh[:k, :]


# =============================================================================
#                    DISTILLATION TRAINER
# =============================================================================

class UniversalDistillationTrainer:
    """Universal distillation trainer supporting all architecture pairs."""

    def __init__(
        self,
        teacher_patcher: comfy.model_patcher.ModelPatcher,
        student_patcher: comfy.model_patcher.ModelPatcher,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        rank: int = 64,
        alpha: Optional[float] = None,
        distill_mode: str = "style",  # style, knowledge, full
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.rank = rank
        self.alpha = alpha if alpha is not None else float(rank)
        self.distill_mode = distill_mode

        self.teacher_patcher = teacher_patcher
        self.student_patcher = student_patcher

        # Get architecture specs
        self.teacher_spec = get_arch_spec(teacher_patcher)
        self.student_spec = get_arch_spec(student_patcher)

        logging.info(f"Teacher: {self.teacher_spec.arch_type.value}")
        logging.info(f"Student: {self.student_spec.arch_type.value}")

        # Create converter
        self.converter = CrossArchConverter(
            self.teacher_spec,
            self.student_spec,
            rank=rank,
        ).to(self.device)

        # Weight transfer utility
        self.weight_transfer = WeightTransfer(
            rank=rank,
            alpha=alpha,
        )

    def extract_distilled_weights(
        self,
        output_format: str = "full",  # full, lora, delta
        progress_callback=None,
    ) -> Dict[str, torch.Tensor]:
        """Extract distilled weights from teacher to student."""
        teacher_sd = self.teacher_patcher.model.diffusion_model.state_dict()
        student_sd = self.student_patcher.model.diffusion_model.state_dict()

        result = {}

        # Determine which layers to process
        if self.distill_mode == "style":
            # Focus on attention and normalization for style transfer
            include_patterns = ["attn", "attention", "norm", "ln"]
        elif self.distill_mode == "knowledge":
            # Focus on MLP/FFN for knowledge transfer
            include_patterns = ["mlp", "ff", "proj", "linear"]
        else:  # full
            include_patterns = ["attn", "mlp", "ff", "proj", "linear", "conv"]

        # Find matching layers
        layer_pairs = self._find_layer_pairs(teacher_sd, student_sd, include_patterns)

        total = len(layer_pairs)
        for i, (t_key, s_key) in enumerate(layer_pairs):
            try:
                t_weight = teacher_sd[t_key].to(self.device)
                s_weight = student_sd[s_key].to(self.device)

                if output_format == "full":
                    # Full weight transfer
                    if t_weight.shape == s_weight.shape:
                        result[s_key] = t_weight.cpu()
                    else:
                        transferred = self.weight_transfer.compute_transfer_weights(
                            t_weight, s_weight.shape, mode="svd"
                        )
                        result[s_key] = transferred.cpu()

                elif output_format == "lora":
                    # LoRA extraction
                    if t_weight.shape == s_weight.shape:
                        delta = t_weight - s_weight
                        lora_A, lora_B = self._extract_lora(delta)
                        base_key = s_key.replace(".weight", "")
                        result[f"{base_key}.lora_A.weight"] = lora_A.cpu()
                        result[f"{base_key}.lora_B.weight"] = lora_B.cpu()

                elif output_format == "delta":
                    # Delta weights
                    if t_weight.shape == s_weight.shape:
                        result[s_key] = (t_weight - s_weight).cpu()

                if progress_callback:
                    progress_callback(i + 1, total, s_key)

            except Exception as e:
                logging.warning(f"Failed to process {s_key}: {e}")
                continue

            # Clear cache periodically
            if i % 20 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return result

    def _find_layer_pairs(
        self,
        teacher_sd: Dict[str, torch.Tensor],
        student_sd: Dict[str, torch.Tensor],
        include_patterns: List[str],
    ) -> List[Tuple[str, str]]:
        """Find matching layer pairs between teacher and student."""
        pairs = []

        # Group teacher layers by semantic type
        teacher_groups = self._group_layers_by_type(teacher_sd, include_patterns)
        student_groups = self._group_layers_by_type(student_sd, include_patterns)

        # Match groups
        for group_type in teacher_groups:
            if group_type not in student_groups:
                continue

            t_layers = sorted(teacher_groups[group_type])
            s_layers = sorted(student_groups[group_type])

            # Proportional matching
            for i, t_layer in enumerate(t_layers):
                if s_layers:
                    s_idx = int(i * len(s_layers) / len(t_layers))
                    s_idx = min(s_idx, len(s_layers) - 1)
                    pairs.append((t_layer, s_layers[s_idx]))

        return pairs

    def _group_layers_by_type(
        self,
        state_dict: Dict[str, torch.Tensor],
        include_patterns: List[str],
    ) -> Dict[str, List[str]]:
        """Group layers by their semantic type."""
        groups = {}

        for key in state_dict:
            if not any(p in key.lower() for p in include_patterns):
                continue
            if "norm" in key.lower():
                continue  # Skip norms for now

            # Determine group
            if "attn" in key.lower() or "attention" in key.lower():
                if "q" in key.lower().split(".")[-1]:
                    group = "attn_q"
                elif "k" in key.lower().split(".")[-1]:
                    group = "attn_k"
                elif "v" in key.lower().split(".")[-1]:
                    group = "attn_v"
                elif "out" in key.lower() or "proj" in key.lower():
                    group = "attn_out"
                else:
                    group = "attn_other"
            elif "mlp" in key.lower() or "ff" in key.lower():
                if "up" in key.lower() or "fc1" in key.lower() or "gate" in key.lower():
                    group = "mlp_up"
                elif "down" in key.lower() or "fc2" in key.lower():
                    group = "mlp_down"
                else:
                    group = "mlp_other"
            else:
                group = "other"

            if group not in groups:
                groups[group] = []
            groups[group].append(key)

        return groups

    def _extract_lora(
        self,
        delta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract LoRA matrices from weight delta."""
        delta_2d = delta.reshape(delta.shape[0], -1).float()

        try:
            U, S, Vh = torch.linalg.svd(delta_2d, full_matrices=False)
        except Exception:
            U, S, Vh = self.weight_transfer._randomized_svd(delta_2d, self.rank)

        r = min(self.rank, len(S))

        sqrt_S = torch.sqrt(S[:r])
        lora_A = (Vh[:r, :].T * sqrt_S).T  # (r, in_features)
        lora_B = U[:, :r] * sqrt_S  # (out_features, r)

        return lora_A.to(delta.dtype), lora_B.to(delta.dtype)

    def create_distilled_checkpoint(
        self,
        output_path: str,
        include_clip: bool = True,
        include_vae: bool = True,
        progress_callback=None,
    ) -> str:
        """Create a complete distilled checkpoint file."""
        # Get distilled weights
        distilled_weights = self.extract_distilled_weights(
            output_format="full",
            progress_callback=progress_callback,
        )

        # Get student base weights
        student_full_sd = self.student_patcher.model.diffusion_model.state_dict()

        # Merge distilled weights into student
        for key, weight in distilled_weights.items():
            if key in student_full_sd:
                student_full_sd[key] = weight

        # Add CLIP and VAE if requested
        checkpoint = {"model": student_full_sd}

        # Save checkpoint
        if HAS_SAFETENSORS:
            # Flatten for safetensors
            flat_sd = {}
            for key, value in student_full_sd.items():
                flat_sd[f"model.diffusion_model.{key}"] = value.contiguous()

            save_file(flat_sd, output_path)
        else:
            torch.save(checkpoint, output_path)

        return output_path


# =============================================================================
#                    COMFYUI NODES
# =============================================================================

class UniversalCrossArchDistill:
    """Universal cross-architecture distillation node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "teacher_model": ("MODEL",),
                "student_model": ("MODEL",),
                "output_name": ("STRING", {"default": "distilled_model"}),
                "distill_mode": (["style", "knowledge", "full"], {"default": "style"}),
                "rank": ("INT", {"default": 64, "min": 1, "max": 512}),
                "output_format": (["full_checkpoint", "lora", "delta"], {"default": "full_checkpoint"}),
            },
            "optional": {
                "alpha": ("FLOAT", {"default": 64.0, "min": 1.0, "max": 512.0}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_path", "report")
    FUNCTION = "distill"
    CATEGORY = "distillation/universal"
    OUTPUT_NODE = True

    def distill(
        self,
        teacher_model,
        student_model,
        output_name,
        distill_mode,
        rank,
        output_format,
        alpha=None,
    ):
        device = comfy.model_management.get_torch_device()

        # Get architecture info
        teacher_spec = get_arch_spec(teacher_model)
        student_spec = get_arch_spec(student_model)

        # Create trainer
        trainer = UniversalDistillationTrainer(
            teacher_patcher=teacher_model,
            student_patcher=student_model,
            device=str(device),
            rank=rank,
            alpha=alpha,
            distill_mode=distill_mode,
        )

        # Determine output path
        output_dir = folder_paths.get_output_directory()
        if output_format == "full_checkpoint":
            output_path = os.path.join(output_dir, f"{output_name}.safetensors")
            output_path = trainer.create_distilled_checkpoint(output_path)
        elif output_format == "lora":
            output_path = os.path.join(output_dir, f"{output_name}_lora.safetensors")
            lora_weights = trainer.extract_distilled_weights(output_format="lora")
            if HAS_SAFETENSORS:
                save_file(lora_weights, output_path)
            else:
                output_path = output_path.replace(".safetensors", ".pt")
                torch.save(lora_weights, output_path)
        else:  # delta
            output_path = os.path.join(output_dir, f"{output_name}_delta.safetensors")
            delta_weights = trainer.extract_distilled_weights(output_format="delta")
            if HAS_SAFETENSORS:
                save_file(delta_weights, output_path)
            else:
                output_path = output_path.replace(".safetensors", ".pt")
                torch.save(delta_weights, output_path)

        # Generate report
        report = f"""=== Universal Cross-Architecture Distillation Report ===

Teacher Model:
  Architecture: {teacher_spec.arch_type.value}
  Context Dim: {teacher_spec.context_dim}
  Model Channels: {teacher_spec.model_channels}
  Is DiT: {teacher_spec.is_dit}
  Is Flow: {teacher_spec.is_flow}

Student Model:
  Architecture: {student_spec.arch_type.value}
  Context Dim: {student_spec.context_dim}
  Model Channels: {student_spec.model_channels}
  Is DiT: {student_spec.is_dit}
  Is Flow: {student_spec.is_flow}

Distillation Settings:
  Mode: {distill_mode}
  Rank: {rank}
  Alpha: {alpha if alpha else rank}
  Output Format: {output_format}

Output: {output_path}
"""

        return (output_path, report)


class StyleDistillation:
    """Distill style/aesthetics from one model to another."""

    ARCH_PAIRS = [
        "SDXL → SD3",
        "SDXL → FLUX.1-dev",
        "SDXL → Z-Image-Turbo",
        "Z-Image-Turbo → SD3",
        "Z-Image-Turbo → FLUX.1-dev",
        "FLUX.1-dev → FLUX.2-dev",
        "Qwen-Image → SDXL",
        "Qwen-Image → Z-Image-Turbo",
        "Custom (use models below)",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "teacher_model": ("MODEL",),
                "student_model": ("MODEL",),
                "preset": (cls.ARCH_PAIRS, {"default": "Custom (use models below)"}),
                "output_name": ("STRING", {"default": "style_distilled"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "rank": ("INT", {"default": 64, "min": 1, "max": 512}),
                "focus": (["attention", "all"], {"default": "attention"}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("distilled_model", "report")
    FUNCTION = "distill_style"
    CATEGORY = "distillation/style"

    def distill_style(
        self,
        teacher_model,
        student_model,
        preset,
        output_name,
        strength,
        rank=64,
        focus="attention",
    ):
        device = comfy.model_management.get_torch_device()

        # Get specs
        teacher_spec = get_arch_spec(teacher_model)
        student_spec = get_arch_spec(student_model)

        # Clone student for modification
        student_clone = student_model.clone()

        # Create trainer with style mode
        trainer = UniversalDistillationTrainer(
            teacher_patcher=teacher_model,
            student_patcher=student_clone,
            device=str(device),
            rank=rank,
            distill_mode="style",
        )

        # Extract style weights
        style_weights = trainer.extract_distilled_weights(output_format="lora")

        # Apply style weights to student with strength
        if style_weights:
            # Apply LoRA patches
            for key in list(style_weights.keys()):
                if "lora_A" in key:
                    base_key = key.replace(".lora_A.weight", "")
                    lora_a_key = key
                    lora_b_key = key.replace("lora_A", "lora_B")

                    if lora_b_key in style_weights:
                        student_clone.add_patches({
                            f"{base_key}.weight": (
                                style_weights[lora_a_key].to(device),
                                style_weights[lora_b_key].to(device),
                                strength,
                            )
                        })

        report = f"""=== Style Distillation Report ===

Teacher: {teacher_spec.arch_type.value}
Student: {student_spec.arch_type.value}
Preset: {preset}
Strength: {strength}
Rank: {rank}
Focus: {focus}

Style weights extracted: {len(style_weights) // 2} layer pairs
"""

        return (student_clone, report)


class ArchitectureInfo:
    """Display architecture information for models."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "get_info"
    CATEGORY = "distillation/utils"

    def get_info(self, model):
        spec = get_arch_spec(model)

        info = f"""=== Model Architecture Info ===

Type: {spec.arch_type.value}
Context Dimension: {spec.context_dim}
Model Channels: {spec.model_channels}
Latent Channels: {spec.latent_channels}
Latent Format: {spec.latent_format}
Is DiT: {spec.is_dit}
Is Flow: {spec.is_flow}
Uses Guidance Embed: {spec.use_guidance_embed}
Hidden Size: {spec.hidden_size}
Num Layers: {spec.num_layers}
"""

        return (info,)


# =============================================================================
#                    NODE REGISTRATION
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "UniversalCrossArchDistill": UniversalCrossArchDistill,
    "StyleDistillation": StyleDistillation,
    "ArchitectureInfo": ArchitectureInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalCrossArchDistill": "Universal Cross-Arch Distillation",
    "StyleDistillation": "Style Distillation",
    "ArchitectureInfo": "Architecture Info",
}
