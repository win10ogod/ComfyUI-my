#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Architecture Distillation Nodes for ComfyUI

Enables knowledge distillation between any ComfyUI supported model architectures.
Supports SD1.5, SD2.x, SDXL, Flux, SD3, video models, and more.
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from safetensors.torch import save_file, load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    logging.warning("safetensors not found, will use torch.save for LoRA output")

import folder_paths
import comfy.model_management
import comfy.utils
import comfy.sd
import comfy.model_patcher
import comfy.model_base
import comfy.supported_models
import comfy.latent_formats


# =============================================================================
#                        MODEL ARCHITECTURE INFO
# =============================================================================

class ModelFamily(Enum):
    """Model architecture families."""
    SD15 = "sd15"
    SD2X = "sd2x"
    SDXL = "sdxl"
    FLUX = "flux"
    SD3 = "sd3"
    AURA = "auraflow"
    PIXART = "pixart"
    HUNYUAN_DIT = "hunyuan_dit"
    VIDEO_SVD = "svd"
    VIDEO_HUNYUAN = "hunyuan_video"
    VIDEO_WAN = "wan"
    VIDEO_MOCHI = "mochi"
    VIDEO_LTXV = "ltxv"
    VIDEO_COSMOS = "cosmos"
    CASCADE = "cascade"
    LUMINA = "lumina"
    OTHER = "other"


@dataclass
class ArchitectureSpec:
    """Specification of a model architecture."""
    family: ModelFamily
    name: str
    context_dim: int
    model_channels: int
    num_heads: int = -1
    num_head_channels: int = -1
    transformer_depth: List[int] = field(default_factory=list)
    in_channels: int = 4
    out_channels: int = 4
    use_linear_in_transformer: bool = False
    adm_in_channels: Optional[int] = None
    use_temporal_attention: bool = False
    image_model: Optional[str] = None
    latent_format: str = "SD15"
    is_video: bool = False
    is_dit: bool = False


def detect_architecture(model_patcher: comfy.model_patcher.ModelPatcher) -> ArchitectureSpec:
    """Detect the architecture of a loaded model."""
    model = model_patcher.model
    model_config = model.model_config

    # Get unet_config
    unet_config = model_config.unet_config.copy() if hasattr(model_config, 'unet_config') else {}

    # Detect family based on model class and config
    family = ModelFamily.OTHER
    name = model_config.__class__.__name__ if hasattr(model_config, '__class__') else "Unknown"

    # Check for image_model identifier (newer architectures)
    image_model = unet_config.get("image_model", None)

    if image_model:
        if "flux" in image_model:
            family = ModelFamily.FLUX
        elif "mochi" in image_model:
            family = ModelFamily.VIDEO_MOCHI
        elif "hunyuan_video" in image_model:
            family = ModelFamily.VIDEO_HUNYUAN
        elif "wan" in image_model:
            family = ModelFamily.VIDEO_WAN
        elif "cosmos" in image_model:
            family = ModelFamily.VIDEO_COSMOS
        elif "ltxv" in image_model:
            family = ModelFamily.VIDEO_LTXV
        elif "lumina" in image_model:
            family = ModelFamily.LUMINA
        elif "hydit" in image_model:
            family = ModelFamily.HUNYUAN_DIT
        elif "pixart" in image_model:
            family = ModelFamily.PIXART
    elif unet_config.get("stable_cascade_stage"):
        family = ModelFamily.CASCADE
    elif hasattr(model_config, 'latent_format'):
        lf = model_config.latent_format
        lf_name = lf.__class__.__name__ if lf else ""
        if "SDXL" in lf_name:
            family = ModelFamily.SDXL
        elif "SD3" in lf_name:
            family = ModelFamily.SD3
        elif "Flux" in lf_name:
            family = ModelFamily.FLUX

    # If still unknown, try to infer from context_dim
    context_dim = unet_config.get("context_dim", 768)
    if family == ModelFamily.OTHER:
        if context_dim == 768:
            family = ModelFamily.SD15
        elif context_dim == 1024:
            family = ModelFamily.SD2X
        elif context_dim == 2048:
            family = ModelFamily.SDXL

    # Determine if it's a DiT model
    is_dit = image_model is not None or unet_config.get("stable_cascade_stage") is not None

    # Determine if it's a video model
    is_video = unet_config.get("use_temporal_attention", False) or family in [
        ModelFamily.VIDEO_SVD, ModelFamily.VIDEO_HUNYUAN, ModelFamily.VIDEO_WAN,
        ModelFamily.VIDEO_MOCHI, ModelFamily.VIDEO_LTXV, ModelFamily.VIDEO_COSMOS
    ]

    # Get latent format name
    latent_format = "SD15"
    if hasattr(model_config, 'latent_format') and model_config.latent_format:
        latent_format = model_config.latent_format.__class__.__name__

    return ArchitectureSpec(
        family=family,
        name=name,
        context_dim=context_dim,
        model_channels=unet_config.get("model_channels", 320),
        num_heads=unet_config.get("num_heads", -1),
        num_head_channels=unet_config.get("num_head_channels", -1),
        transformer_depth=unet_config.get("transformer_depth", []),
        in_channels=unet_config.get("in_channels", 4),
        out_channels=unet_config.get("out_channels", 4),
        use_linear_in_transformer=unet_config.get("use_linear_in_transformer", False),
        adm_in_channels=unet_config.get("adm_in_channels"),
        use_temporal_attention=unet_config.get("use_temporal_attention", False),
        image_model=image_model,
        latent_format=latent_format,
        is_video=is_video,
        is_dit=is_dit,
    )


# =============================================================================
#                      FEATURE EXTRACTOR HOOKS
# =============================================================================

class FeatureExtractor:
    """Extracts intermediate features from diffusion models."""

    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = layer_names
        self.features: Dict[str, torch.Tensor] = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on specified layers."""
        for name, module in self.model.named_modules():
            if any(ln in name for ln in self.layer_names):
                hook = module.register_forward_hook(self._create_hook(name))
                self.hooks.append(hook)

    def _create_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.features[name] = output.detach()
        return hook

    def clear(self):
        """Clear stored features."""
        self.features.clear()

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __del__(self):
        self.remove_hooks()


# =============================================================================
#                    CROSS-ARCHITECTURE PROJECTIONS
# =============================================================================

class FeatureProjector(nn.Module):
    """Projects features from one architecture dimension to another."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: Optional[int] = None,
        use_bias: bool = True,
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if hidden_dim is None:
            hidden_dim = min(in_features, out_features) * 2

        # Simple linear projection with optional hidden layer
        if in_features == out_features:
            self.proj = nn.Identity()
        elif hidden_dim > 0 and abs(in_features - out_features) > 256:
            self.proj = nn.Sequential(
                nn.Linear(in_features, hidden_dim, bias=use_bias),
                nn.GELU(),
                nn.Linear(hidden_dim, out_features, bias=use_bias),
            )
            # Initialize with small weights
            for m in self.proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=init_scale)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        else:
            self.proj = nn.Linear(in_features, out_features, bias=use_bias)
            nn.init.normal_(self.proj.weight, std=init_scale)
            if use_bias:
                nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle different tensor shapes
        original_shape = x.shape

        if len(original_shape) > 2:
            # Reshape to (batch, -1, features)
            x_flat = x.reshape(-1, self.in_features)
            out = self.proj(x_flat)
            # Reshape back
            new_shape = original_shape[:-1] + (self.out_features,)
            return out.reshape(new_shape)
        else:
            return self.proj(x)


class CrossArchitectureMapper(nn.Module):
    """Maps features between different model architectures."""

    def __init__(
        self,
        teacher_spec: ArchitectureSpec,
        student_spec: ArchitectureSpec,
        num_matching_layers: int = 8,
        projection_dim: int = 512,
    ):
        super().__init__()
        self.teacher_spec = teacher_spec
        self.student_spec = student_spec

        # Context dimension projector (for cross-attention features)
        if teacher_spec.context_dim != student_spec.context_dim:
            self.context_proj = FeatureProjector(
                teacher_spec.context_dim,
                student_spec.context_dim,
            )
        else:
            self.context_proj = nn.Identity()

        # Model channel projector (for hidden states)
        if teacher_spec.model_channels != student_spec.model_channels:
            self.channel_proj = FeatureProjector(
                teacher_spec.model_channels,
                student_spec.model_channels,
            )
        else:
            self.channel_proj = nn.Identity()

        # Per-layer feature alignment projectors
        self.layer_projectors = nn.ModuleDict()

        # Temporal projection for video models
        self.has_temporal_proj = (
            teacher_spec.is_video != student_spec.is_video or
            teacher_spec.use_temporal_attention != student_spec.use_temporal_attention
        )

    def add_layer_projector(self, name: str, in_dim: int, out_dim: int):
        """Add a projector for a specific layer."""
        if in_dim != out_dim:
            self.layer_projectors[name.replace(".", "_")] = FeatureProjector(in_dim, out_dim)

    def forward(
        self,
        teacher_features: Dict[str, torch.Tensor],
        feature_type: str = "hidden"
    ) -> Dict[str, torch.Tensor]:
        """Project teacher features to match student dimensions."""
        projected = {}

        for name, feat in teacher_features.items():
            safe_name = name.replace(".", "_")

            if safe_name in self.layer_projectors:
                projected[name] = self.layer_projectors[safe_name](feat)
            elif feature_type == "context":
                projected[name] = self.context_proj(feat)
            elif feature_type == "hidden":
                projected[name] = self.channel_proj(feat)
            else:
                projected[name] = feat

        return projected


# =============================================================================
#                        DISTILLATION LOSSES
# =============================================================================

class DistillationLoss(nn.Module):
    """Combined distillation loss functions."""

    def __init__(
        self,
        alpha_output: float = 1.0,
        alpha_feature: float = 0.5,
        alpha_attention: float = 0.3,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.alpha_output = alpha_output
        self.alpha_feature = alpha_feature
        self.alpha_attention = alpha_attention
        self.temperature = temperature

    def output_loss(
        self,
        teacher_output: torch.Tensor,
        student_output: torch.Tensor,
    ) -> torch.Tensor:
        """MSE loss on model outputs."""
        return F.mse_loss(student_output, teacher_output)

    def feature_loss(
        self,
        teacher_features: Dict[str, torch.Tensor],
        student_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Feature matching loss across layers."""
        total_loss = 0.0
        count = 0

        for name in teacher_features:
            if name in student_features:
                t_feat = teacher_features[name]
                s_feat = student_features[name]

                # Handle dimension mismatch by interpolation
                if t_feat.shape != s_feat.shape:
                    # Simple linear interpolation for mismatched shapes
                    if len(t_feat.shape) == len(s_feat.shape):
                        t_feat = F.interpolate(
                            t_feat.float().unsqueeze(0) if len(t_feat.shape) == 3 else t_feat.float(),
                            size=s_feat.shape[-1],
                            mode='linear' if len(t_feat.shape) <= 3 else 'nearest'
                        )
                        if len(t_feat.shape) == 4:
                            t_feat = t_feat.squeeze(0)

                if t_feat.shape == s_feat.shape:
                    total_loss = total_loss + F.mse_loss(s_feat, t_feat)
                    count += 1

        return total_loss / max(count, 1)

    def forward(
        self,
        teacher_output: torch.Tensor,
        student_output: torch.Tensor,
        teacher_features: Optional[Dict[str, torch.Tensor]] = None,
        student_features: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined distillation loss."""
        losses = {}

        # Output matching loss
        out_loss = self.output_loss(teacher_output, student_output)
        losses["output"] = out_loss.item()
        total_loss = self.alpha_output * out_loss

        # Feature matching loss
        if teacher_features and student_features:
            feat_loss = self.feature_loss(teacher_features, student_features)
            losses["feature"] = feat_loss.item()
            total_loss = total_loss + self.alpha_feature * feat_loss

        losses["total"] = total_loss.item()
        return total_loss, losses


# =============================================================================
#                        SVD-LORA EXTRACTION
# =============================================================================

class SVDLoRAExtractor:
    """Extract LoRA weights using SVD decomposition."""

    def __init__(
        self,
        rank: int = 64,
        alpha: Optional[int] = None,
        min_rank: int = 4,
        max_rank: int = 512,
        use_adaptive_rank: bool = True,
        energy_threshold: float = 0.95,
    ):
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank
        self.min_rank = min_rank
        self.max_rank = max_rank
        self.use_adaptive_rank = use_adaptive_rank
        self.energy_threshold = energy_threshold

    def compute_delta(
        self,
        original_weight: torch.Tensor,
        distilled_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weight delta."""
        return distilled_weight - original_weight

    def select_rank(self, S: torch.Tensor, target_rank: int) -> int:
        """Adaptively select rank based on energy threshold."""
        if not self.use_adaptive_rank:
            return min(target_rank, len(S))

        # Compute cumulative energy
        total_energy = (S ** 2).sum()
        cumsum = torch.cumsum(S ** 2, dim=0)
        energy_ratio = cumsum / total_energy

        # Find rank that captures enough energy
        mask = energy_ratio >= self.energy_threshold
        if mask.any():
            adaptive_rank = mask.nonzero()[0].item() + 1
        else:
            adaptive_rank = len(S)

        # Clamp to bounds
        final_rank = max(self.min_rank, min(adaptive_rank, target_rank, self.max_rank))
        return int(final_rank)

    def extract_lora(
        self,
        delta: torch.Tensor,
        target_rank: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, float]:
        """Extract LoRA matrices from weight delta using SVD."""
        if target_rank is None:
            target_rank = self.rank

        original_shape = delta.shape
        dtype = delta.dtype
        device = delta.device

        # Reshape to 2D for SVD
        if len(original_shape) > 2:
            delta_2d = delta.reshape(original_shape[0], -1)
        else:
            delta_2d = delta

        # Compute SVD
        delta_float = delta_2d.float()
        try:
            U, S, Vh = torch.linalg.svd(delta_float, full_matrices=False)
        except Exception as e:
            logging.warning(f"SVD failed, using randomized SVD: {e}")
            U, S, Vh = self._randomized_svd(delta_float, target_rank)

        # Select rank
        r = self.select_rank(S, target_rank)

        # Compute reconstruction error
        recon = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]
        error = torch.norm(delta_float - recon) / torch.norm(delta_float)

        # Extract LoRA matrices
        # lora_A: down projection (input -> rank)
        # lora_B: up projection (rank -> output)
        # W_delta ≈ lora_B @ lora_A
        sqrt_S = torch.sqrt(S[:r])
        lora_A = (Vh[:r, :].T * sqrt_S).T  # (r, in_features)
        lora_B = U[:, :r] * sqrt_S  # (out_features, r)

        return lora_A.to(dtype), lora_B.to(dtype), r, error.item()

    def _randomized_svd(
        self,
        A: torch.Tensor,
        k: int,
        n_iter: int = 2,
        n_oversamples: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomized SVD for large matrices (Halko et al.)"""
        m, n = A.shape
        k = min(k, m, n)

        # Random projection
        P = torch.randn(n, k + n_oversamples, device=A.device, dtype=A.dtype)

        # Power iteration
        Q = A @ P
        for _ in range(n_iter):
            Q, _ = torch.linalg.qr(Q)
            Q = A @ (A.T @ Q)
            Q, _ = torch.linalg.qr(Q)

        Q, _ = torch.linalg.qr(A @ P)

        # Project to low-rank space
        B = Q.T @ A

        # SVD on smaller matrix
        U_small, S, Vh = torch.linalg.svd(B, full_matrices=False)
        U = Q @ U_small

        return U[:, :k], S[:k], Vh[:k, :]


def extract_lora_from_models(
    original_state_dict: Dict[str, torch.Tensor],
    distilled_state_dict: Dict[str, torch.Tensor],
    rank: int = 64,
    alpha: Optional[int] = None,
    include_patterns: List[str] = None,
    exclude_patterns: List[str] = None,
    device: str = "cuda",
    progress_callback=None,
) -> Dict[str, torch.Tensor]:
    """Extract LoRA weights from distilled model."""

    extractor = SVDLoRAExtractor(rank=rank, alpha=alpha)
    lora_state_dict = {}

    # Default patterns
    if include_patterns is None:
        include_patterns = ["attn", "mlp", "ff", "proj", "linear", "conv"]
    if exclude_patterns is None:
        exclude_patterns = ["norm", "embed", "time"]

    # Find matching keys
    matching_keys = []
    for key in original_state_dict:
        if key in distilled_state_dict:
            # Check include patterns
            if any(p in key.lower() for p in include_patterns):
                # Check exclude patterns
                if not any(p in key.lower() for p in exclude_patterns):
                    matching_keys.append(key)

    total_keys = len(matching_keys)
    for i, key in enumerate(matching_keys):
        try:
            orig_w = original_state_dict[key].to(device).float()
            dist_w = distilled_state_dict[key].to(device).float()

            if orig_w.shape != dist_w.shape:
                logging.warning(f"Shape mismatch for {key}: {orig_w.shape} vs {dist_w.shape}")
                continue

            # Skip small layers
            if orig_w.numel() < 1000:
                continue

            delta = extractor.compute_delta(orig_w, dist_w)
            lora_A, lora_B, actual_rank, error = extractor.extract_lora(delta)

            # Store LoRA weights
            base_key = key.replace(".weight", "")
            lora_state_dict[f"{base_key}.lora_A.weight"] = lora_A.cpu()
            lora_state_dict[f"{base_key}.lora_B.weight"] = lora_B.cpu()

            if progress_callback:
                progress_callback(i + 1, total_keys, key, actual_rank, error)

        except Exception as e:
            logging.warning(f"Failed to extract LoRA for {key}: {e}")
            continue

        # Clear cache periodically
        if i % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return lora_state_dict


# =============================================================================
#                        DISTILLATION TRAINER
# =============================================================================

class DistillationTrainer:
    """Trains student model using knowledge from teacher."""

    def __init__(
        self,
        teacher_patcher: comfy.model_patcher.ModelPatcher,
        student_patcher: comfy.model_patcher.ModelPatcher,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        learning_rate: float = 1e-4,
        num_steps: int = 1000,
        batch_size: int = 1,
        feature_matching_layers: List[str] = None,
    ):
        self.device = torch.device(device)
        self.dtype = dtype

        self.teacher_patcher = teacher_patcher
        self.student_patcher = student_patcher
        self.teacher_model = teacher_patcher.model
        self.student_model = student_patcher.model

        # Detect architectures
        self.teacher_spec = detect_architecture(teacher_patcher)
        self.student_spec = detect_architecture(student_patcher)

        logging.info(f"Teacher: {self.teacher_spec.name} ({self.teacher_spec.family.value})")
        logging.info(f"Student: {self.student_spec.name} ({self.student_spec.family.value})")

        # Create cross-architecture mapper
        self.mapper = CrossArchitectureMapper(
            self.teacher_spec,
            self.student_spec,
        ).to(self.device)

        # Setup loss
        self.loss_fn = DistillationLoss()

        # Default feature matching layers
        if feature_matching_layers is None:
            feature_matching_layers = ["transformer_blocks", "joint_blocks", "input_blocks", "output_blocks"]

        self.feature_layers = feature_matching_layers

        # Training params
        self.lr = learning_rate
        self.num_steps = num_steps
        self.batch_size = batch_size

        # Feature extractors
        self.teacher_extractor = None
        self.student_extractor = None

    def setup_feature_extractors(self):
        """Setup feature extraction hooks."""
        if hasattr(self.teacher_model, 'diffusion_model'):
            self.teacher_extractor = FeatureExtractor(
                self.teacher_model.diffusion_model,
                self.feature_layers
            )
            self.student_extractor = FeatureExtractor(
                self.student_model.diffusion_model,
                self.feature_layers
            )

    def cleanup_feature_extractors(self):
        """Remove feature extraction hooks."""
        if self.teacher_extractor:
            self.teacher_extractor.remove_hooks()
        if self.student_extractor:
            self.student_extractor.remove_hooks()

    def generate_training_data(
        self,
        num_samples: int,
        latent_shape: Tuple[int, ...] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Generate synthetic training data."""
        if latent_shape is None:
            # Default SD latent shape
            latent_shape = (self.batch_size, 4, 64, 64)

        data = []
        for _ in range(num_samples):
            # Random latent
            latent = torch.randn(latent_shape, device=self.device, dtype=self.dtype)
            # Random timestep
            timestep = torch.randint(1, 1000, (self.batch_size,), device=self.device)
            # Random context (text embedding placeholder)
            context_dim = self.teacher_spec.context_dim
            context = torch.randn(self.batch_size, 77, context_dim, device=self.device, dtype=self.dtype)

            data.append((latent, timestep, context))

        return data

    def train_step(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
    ) -> Dict[str, float]:
        """Single training step."""
        # Clear feature caches
        if self.teacher_extractor:
            self.teacher_extractor.clear()
        if self.student_extractor:
            self.student_extractor.clear()

        with torch.no_grad():
            # Get teacher output
            teacher_out = self.teacher_model.apply_model(
                latent, timestep, c_crossattn=context
            )

        # Project teacher features if needed
        teacher_features = {}
        if self.teacher_extractor:
            teacher_features = self.mapper(
                self.teacher_extractor.features,
                feature_type="hidden"
            )

        # Get student output
        student_out = self.student_model.apply_model(
            latent, timestep, c_crossattn=context
        )

        student_features = {}
        if self.student_extractor:
            student_features = self.student_extractor.features

        # Compute loss
        loss, loss_dict = self.loss_fn(
            teacher_out, student_out,
            teacher_features, student_features
        )

        return loss_dict

    def train(
        self,
        training_data: List[Tuple] = None,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """Run distillation training."""
        self.setup_feature_extractors()

        try:
            if training_data is None:
                training_data = self.generate_training_data(self.num_steps)

            losses_history = []

            for step, (latent, timestep, context) in enumerate(training_data):
                loss_dict = self.train_step(latent, timestep, context)
                losses_history.append(loss_dict)

                if progress_callback:
                    progress_callback(step + 1, len(training_data), loss_dict)

                if step % 100 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            return {
                "losses": losses_history,
                "final_loss": losses_history[-1] if losses_history else {},
            }

        finally:
            self.cleanup_feature_extractors()


# =============================================================================
#                        COMFYUI NODES
# =============================================================================

class LoadModelForDistillation:
    """Load a model for use in distillation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"tooltip": "Checkpoint to load"}),
                "role": (["teacher", "student"], {"default": "teacher"}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "ARCH_INFO")
    RETURN_NAMES = ("model", "clip", "vae", "arch_info")
    FUNCTION = "load"
    CATEGORY = "distillation"

    def load(self, ckpt_name, role):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
        )

        model_patcher = out[0]
        arch_spec = detect_architecture(model_patcher)

        arch_info = {
            "role": role,
            "name": arch_spec.name,
            "family": arch_spec.family.value,
            "context_dim": arch_spec.context_dim,
            "model_channels": arch_spec.model_channels,
            "is_dit": arch_spec.is_dit,
            "is_video": arch_spec.is_video,
            "latent_format": arch_spec.latent_format,
        }

        return (out[0], out[1], out[2], arch_info)


class ArchitectureAnalyzer:
    """Analyze and compare model architectures."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "teacher_model": ("MODEL",),
                "student_model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("ARCH_COMPARISON", "STRING")
    RETURN_NAMES = ("comparison", "report")
    FUNCTION = "analyze"
    CATEGORY = "distillation"

    def analyze(self, teacher_model, student_model):
        teacher_spec = detect_architecture(teacher_model)
        student_spec = detect_architecture(student_model)

        comparison = {
            "teacher": {
                "name": teacher_spec.name,
                "family": teacher_spec.family.value,
                "context_dim": teacher_spec.context_dim,
                "model_channels": teacher_spec.model_channels,
                "is_dit": teacher_spec.is_dit,
                "is_video": teacher_spec.is_video,
            },
            "student": {
                "name": student_spec.name,
                "family": student_spec.family.value,
                "context_dim": student_spec.context_dim,
                "model_channels": student_spec.model_channels,
                "is_dit": student_spec.is_dit,
                "is_video": student_spec.is_video,
            },
            "compatibility": {
                "same_family": teacher_spec.family == student_spec.family,
                "same_context_dim": teacher_spec.context_dim == student_spec.context_dim,
                "same_channels": teacher_spec.model_channels == student_spec.model_channels,
                "same_latent_format": teacher_spec.latent_format == student_spec.latent_format,
                "requires_projection": (
                    teacher_spec.context_dim != student_spec.context_dim or
                    teacher_spec.model_channels != student_spec.model_channels
                ),
            },
        }

        # Generate report
        report = f"""=== Architecture Comparison Report ===

Teacher Model:
  Name: {teacher_spec.name}
  Family: {teacher_spec.family.value}
  Context Dim: {teacher_spec.context_dim}
  Model Channels: {teacher_spec.model_channels}
  Is DiT: {teacher_spec.is_dit}
  Is Video: {teacher_spec.is_video}
  Latent Format: {teacher_spec.latent_format}

Student Model:
  Name: {student_spec.name}
  Family: {student_spec.family.value}
  Context Dim: {student_spec.context_dim}
  Model Channels: {student_spec.model_channels}
  Is DiT: {student_spec.is_dit}
  Is Video: {student_spec.is_video}
  Latent Format: {student_spec.latent_format}

Compatibility:
  Same Family: {comparison['compatibility']['same_family']}
  Same Context Dim: {comparison['compatibility']['same_context_dim']}
  Same Channels: {comparison['compatibility']['same_channels']}
  Same Latent Format: {comparison['compatibility']['same_latent_format']}
  Requires Projection: {comparison['compatibility']['requires_projection']}
"""

        return (comparison, report)


class CrossArchDistillation:
    """Perform cross-architecture distillation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "teacher_model": ("MODEL",),
                "student_model": ("MODEL",),
                "num_steps": ("INT", {"default": 100, "min": 1, "max": 10000}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
                "learning_rate": ("FLOAT", {"default": 1e-4, "min": 1e-8, "max": 1.0, "step": 1e-5}),
                "feature_matching": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "latent_shape_h": ("INT", {"default": 64, "min": 8, "max": 256}),
                "latent_shape_w": ("INT", {"default": 64, "min": 8, "max": 256}),
                "seed": ("INT", {"default": 42}),
            },
        }

    RETURN_TYPES = ("MODEL", "DISTILL_RESULT")
    RETURN_NAMES = ("distilled_model", "result")
    FUNCTION = "distill"
    CATEGORY = "distillation"

    def distill(
        self,
        teacher_model,
        student_model,
        num_steps,
        batch_size,
        learning_rate,
        feature_matching,
        latent_shape_h=64,
        latent_shape_w=64,
        seed=42,
    ):
        torch.manual_seed(seed)

        device = comfy.model_management.get_torch_device()

        # Clone student for training
        student_clone = student_model.clone()

        # Create trainer
        feature_layers = ["transformer", "joint", "input", "output"] if feature_matching else []

        trainer = DistillationTrainer(
            teacher_patcher=teacher_model,
            student_patcher=student_clone,
            device=str(device),
            learning_rate=learning_rate,
            num_steps=num_steps,
            batch_size=batch_size,
            feature_matching_layers=feature_layers,
        )

        # Generate training data
        latent_shape = (batch_size, 4, latent_shape_h, latent_shape_w)
        training_data = trainer.generate_training_data(num_steps, latent_shape)

        # Run training
        result = trainer.train(training_data)

        return (student_clone, result)


class ExtractLoRAFromDistillation:
    """Extract LoRA weights from distilled model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_model": ("MODEL",),
                "distilled_model": ("MODEL",),
                "rank": ("INT", {"default": 64, "min": 1, "max": 512}),
                "alpha": ("INT", {"default": 64, "min": 1, "max": 512}),
                "output_name": ("STRING", {"default": "distilled_lora"}),
            },
            "optional": {
                "include_attn": ("BOOLEAN", {"default": True}),
                "include_mlp": ("BOOLEAN", {"default": True}),
                "include_proj": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_path",)
    FUNCTION = "extract"
    CATEGORY = "distillation"
    OUTPUT_NODE = True

    def extract(
        self,
        original_model,
        distilled_model,
        rank,
        alpha,
        output_name,
        include_attn=True,
        include_mlp=True,
        include_proj=True,
    ):
        device = comfy.model_management.get_torch_device()

        # Get state dicts
        original_sd = original_model.model.diffusion_model.state_dict()
        distilled_sd = distilled_model.model.diffusion_model.state_dict()

        # Build include patterns
        include_patterns = []
        if include_attn:
            include_patterns.extend(["attn", "attention", "self_attn"])
        if include_mlp:
            include_patterns.extend(["mlp", "ff", "feedforward"])
        if include_proj:
            include_patterns.extend(["proj", "linear"])

        # Extract LoRA
        lora_sd = extract_lora_from_models(
            original_sd,
            distilled_sd,
            rank=rank,
            alpha=alpha,
            include_patterns=include_patterns,
            device=str(device),
        )

        if not lora_sd:
            raise RuntimeError("No LoRA weights extracted. Check if models have compatible weights.")

        # Add metadata
        lora_sd["lora_config"] = torch.tensor([rank, alpha], dtype=torch.float32)

        # Save LoRA
        output_dir = folder_paths.get_output_directory()
        lora_path = os.path.join(output_dir, f"{output_name}.safetensors")

        if HAS_SAFETENSORS:
            save_file(lora_sd, lora_path)
        else:
            torch.save(lora_sd, lora_path.replace(".safetensors", ".pt"))
            lora_path = lora_path.replace(".safetensors", ".pt")

        logging.info(f"Saved LoRA to {lora_path}")

        return (lora_path,)


class QuickDistillation:
    """Quick distillation using weight delta extraction (no training)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "teacher_model": ("MODEL",),
                "student_model": ("MODEL",),
                "rank": ("INT", {"default": 64, "min": 1, "max": 512}),
                "output_name": ("STRING", {"default": "quick_distill_lora"}),
            },
            "optional": {
                "use_adaptive_rank": ("BOOLEAN", {"default": True}),
                "energy_threshold": ("FLOAT", {"default": 0.95, "min": 0.5, "max": 0.999}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("lora_path", "report")
    FUNCTION = "distill"
    CATEGORY = "distillation"
    OUTPUT_NODE = True

    def distill(
        self,
        teacher_model,
        student_model,
        rank,
        output_name,
        use_adaptive_rank=True,
        energy_threshold=0.95,
    ):
        device = comfy.model_management.get_torch_device()

        # Analyze architectures
        teacher_spec = detect_architecture(teacher_model)
        student_spec = detect_architecture(student_model)

        # Get state dicts
        teacher_sd = teacher_model.model.diffusion_model.state_dict()
        student_sd = student_model.model.diffusion_model.state_dict()

        # Create extractor
        extractor = SVDLoRAExtractor(
            rank=rank,
            use_adaptive_rank=use_adaptive_rank,
            energy_threshold=energy_threshold,
        )

        lora_sd = {}
        stats = []

        # Find common keys
        common_keys = set(teacher_sd.keys()) & set(student_sd.keys())
        weight_keys = [k for k in common_keys if ".weight" in k and "norm" not in k.lower()]

        for key in weight_keys:
            try:
                t_w = teacher_sd[key].to(device).float()
                s_w = student_sd[key].to(device).float()

                if t_w.shape != s_w.shape:
                    continue

                if t_w.numel() < 1000:
                    continue

                delta = extractor.compute_delta(s_w, t_w)  # student toward teacher
                lora_A, lora_B, actual_rank, error = extractor.extract_lora(delta)

                base_key = key.replace(".weight", "")
                lora_sd[f"{base_key}.lora_A.weight"] = lora_A.cpu().half()
                lora_sd[f"{base_key}.lora_B.weight"] = lora_B.cpu().half()

                stats.append({
                    "key": key,
                    "rank": actual_rank,
                    "error": error,
                    "shape": list(t_w.shape),
                })

            except Exception as e:
                logging.warning(f"Failed for {key}: {e}")
                continue

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not lora_sd:
            raise RuntimeError("No LoRA weights extracted")

        # Save
        output_dir = folder_paths.get_output_directory()
        lora_path = os.path.join(output_dir, f"{output_name}.safetensors")

        if HAS_SAFETENSORS:
            save_file(lora_sd, lora_path)
        else:
            lora_path = lora_path.replace(".safetensors", ".pt")
            torch.save(lora_sd, lora_path)

        # Generate report
        avg_rank = sum(s["rank"] for s in stats) / len(stats) if stats else 0
        avg_error = sum(s["error"] for s in stats) / len(stats) if stats else 0

        report = f"""=== Quick Distillation Report ===
Teacher: {teacher_spec.name} ({teacher_spec.family.value})
Student: {student_spec.name} ({student_spec.family.value})

Extracted Weights: {len(stats)}
Average Rank: {avg_rank:.1f}
Average Error: {avg_error:.4f}

Output: {lora_path}
"""

        return (lora_path, report)


class LatentSpaceAdapter:
    """Adapts latents between different model architectures."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "source_format": (["SD15", "SDXL", "SD3", "Flux", "Flux2", "Mochi", "HunyuanVideo", "WAN21", "auto"], {"default": "auto"}),
                "target_format": (["SD15", "SDXL", "SD3", "Flux", "Flux2", "Mochi", "HunyuanVideo", "WAN21"],),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "adapt"
    CATEGORY = "distillation/utils"

    def adapt(self, latent, source_format, target_format):
        samples = latent["samples"]

        # Get format classes
        format_map = {
            "SD15": comfy.latent_formats.SD15,
            "SDXL": comfy.latent_formats.SDXL,
            "SD3": comfy.latent_formats.SD3,
            "Flux": comfy.latent_formats.Flux,
            "Flux2": getattr(comfy.latent_formats, "Flux2", comfy.latent_formats.Flux),
            "Mochi": getattr(comfy.latent_formats, "Mochi", comfy.latent_formats.SD15),
            "HunyuanVideo": getattr(comfy.latent_formats, "HunyuanVideo", comfy.latent_formats.SD15),
            "WAN21": getattr(comfy.latent_formats, "Wan21", comfy.latent_formats.SD15),
        }

        # Auto-detect source format based on channels
        if source_format == "auto":
            c = samples.shape[1]
            if c == 4:
                source_format = "SD15"
            elif c == 16:
                source_format = "SD3"
            else:
                source_format = "SD15"

        src_fmt = format_map.get(source_format, comfy.latent_formats.SD15)()
        tgt_fmt = format_map.get(target_format, comfy.latent_formats.SD15)()

        # Process out from source, then in to target
        if hasattr(src_fmt, "process_out") and hasattr(tgt_fmt, "process_in"):
            # Decode from source format
            decoded = src_fmt.process_out(samples)
            # Encode to target format
            encoded = tgt_fmt.process_in(decoded)
            samples = encoded

        return ({"samples": samples},)


class DistillationPreview:
    """Preview distillation by comparing model outputs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "teacher_model": ("MODEL",),
                "student_model": ("MODEL",),
                "latent": ("LATENT",),
                "seed": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "FLOAT")
    RETURN_NAMES = ("teacher_output", "student_output", "difference")
    FUNCTION = "preview"
    CATEGORY = "distillation"

    def preview(self, teacher_model, student_model, latent, seed):
        device = comfy.model_management.get_torch_device()
        samples = latent["samples"].to(device)

        # Random timestep
        torch.manual_seed(seed)
        timestep = torch.randint(1, 1000, (samples.shape[0],), device=device)

        # Get context dimension from teacher
        teacher_spec = detect_architecture(teacher_model)
        context = torch.randn(samples.shape[0], 77, teacher_spec.context_dim, device=device)

        with torch.no_grad():
            teacher_out = teacher_model.model.apply_model(samples, timestep, c_crossattn=context)
            student_out = student_model.model.apply_model(samples, timestep, c_crossattn=context)

        # Compute difference
        diff = F.mse_loss(student_out, teacher_out).item()

        return (
            {"samples": teacher_out.cpu()},
            {"samples": student_out.cpu()},
            diff,
        )


# =============================================================================
#                        NODE REGISTRATION
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "LoadModelForDistillation": LoadModelForDistillation,
    "ArchitectureAnalyzer": ArchitectureAnalyzer,
    "CrossArchDistillation": CrossArchDistillation,
    "ExtractLoRAFromDistillation": ExtractLoRAFromDistillation,
    "QuickDistillation": QuickDistillation,
    "LatentSpaceAdapter": LatentSpaceAdapter,
    "DistillationPreview": DistillationPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadModelForDistillation": "Load Model (Distillation)",
    "ArchitectureAnalyzer": "Analyze Architectures",
    "CrossArchDistillation": "Cross-Architecture Distillation",
    "ExtractLoRAFromDistillation": "Extract LoRA from Distillation",
    "QuickDistillation": "Quick Distillation (SVD)",
    "LatentSpaceAdapter": "Latent Space Adapter",
    "DistillationPreview": "Distillation Preview",
}
