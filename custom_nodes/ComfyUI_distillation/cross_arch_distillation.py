#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple SVD Distillation Nodes for ComfyUI

Simple nodes for distilling knowledge from teacher model to LoRA.
Uses ComfyUI native MODEL type directly.
"""

import os
import gc
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path

import torch

try:
    from safetensors.torch import save_file, load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    logging.warning("safetensors not found, will use torch.save for LoRA output")

import folder_paths
import comfy.model_management
import comfy.utils
import comfy.lora


# =============================================================================
#                           SIMPLE SVD DISTILLATION
# =============================================================================

def compute_delta_weight(teacher_weight: torch.Tensor, student_weight: torch.Tensor) -> torch.Tensor:
    """Compute weight delta between teacher and student."""
    # Handle shape mismatch
    if teacher_weight.shape != student_weight.shape:
        # Try to align shapes through interpolation for conv layers
        if len(teacher_weight.shape) == 4:  # Conv2d
            student_weight = torch.nn.functional.interpolate(
                student_weight.unsqueeze(0),
                size=teacher_weight.shape[2:],
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            if student_weight.shape[:2] != teacher_weight.shape[:2]:
                # Channel mismatch - pad or truncate
                if student_weight.shape[0] < teacher_weight.shape[0]:
                    pad = torch.zeros(teacher_weight.shape[0] - student_weight.shape[0], *student_weight.shape[1:], device=student_weight.device, dtype=student_weight.dtype)
                    student_weight = torch.cat([student_weight, pad], dim=0)
                else:
                    student_weight = student_weight[:teacher_weight.shape[0]]
                if student_weight.shape[1] < teacher_weight.shape[1]:
                    pad = torch.zeros(student_weight.shape[0], teacher_weight.shape[1] - student_weight.shape[1], *student_weight.shape[2:], device=student_weight.device, dtype=student_weight.dtype)
                    student_weight = torch.cat([student_weight, pad], dim=1)
                else:
                    student_weight = student_weight[:, :teacher_weight.shape[1]]
        elif len(teacher_weight.shape) == 2:  # Linear
            # Pad or truncate for linear layers
            if student_weight.shape[0] != teacher_weight.shape[0]:
                if student_weight.shape[0] < teacher_weight.shape[0]:
                    pad = torch.zeros(teacher_weight.shape[0] - student_weight.shape[0], student_weight.shape[1], device=student_weight.device, dtype=student_weight.dtype)
                    student_weight = torch.cat([student_weight, pad], dim=0)
                else:
                    student_weight = student_weight[:teacher_weight.shape[0]]
            if student_weight.shape[1] != teacher_weight.shape[1]:
                if student_weight.shape[1] < teacher_weight.shape[1]:
                    pad = torch.zeros(student_weight.shape[0], teacher_weight.shape[1] - student_weight.shape[1], device=student_weight.device, dtype=student_weight.dtype)
                    student_weight = torch.cat([student_weight, pad], dim=1)
                else:
                    student_weight = student_weight[:, :teacher_weight.shape[1]]
        else:
            return None  # Skip incompatible layers

    return teacher_weight - student_weight


def svd_decompose(delta: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decompose delta weight into LoRA A and B matrices using SVD."""
    original_shape = delta.shape
    dtype = delta.dtype
    device = delta.device

    # Reshape to 2D for SVD
    if len(original_shape) == 4:  # Conv2d: (out, in, h, w) -> (out, in*h*w)
        delta_2d = delta.reshape(original_shape[0], -1)
    elif len(original_shape) == 2:  # Linear: already 2D
        delta_2d = delta
    else:
        return None, None

    # SVD decomposition
    try:
        delta_2d = delta_2d.float()  # SVD needs float32
        U, S, Vh = torch.linalg.svd(delta_2d, full_matrices=False)

        # Truncate to rank
        rank = min(rank, S.shape[0], U.shape[1], Vh.shape[0])
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]

        # Split sqrt(S) between A and B (PiSSA style)
        sqrt_S = torch.sqrt(S)
        lora_B = (U * sqrt_S.unsqueeze(0)).to(dtype)  # (out, rank)
        lora_A = (sqrt_S.unsqueeze(1) * Vh).to(dtype)  # (rank, in*h*w)

        # Reshape A back for conv
        if len(original_shape) == 4:
            lora_A = lora_A.reshape(rank, original_shape[1], original_shape[2], original_shape[3])

        return lora_A, lora_B

    except Exception as e:
        logging.warning(f"SVD failed: {e}")
        return None, None


def get_matchable_keys(teacher_sd: Dict, student_sd: Dict) -> Dict[str, str]:
    """Find matching keys between teacher and student state dicts."""
    matches = {}

    # Direct matching
    for key in teacher_sd.keys():
        if key in student_sd:
            matches[key] = key

    return matches


class ExtractLoRAFromModels:
    """Extract LoRA weights by computing delta between two models and SVD decomposition."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "teacher_model": ("MODEL", {"tooltip": "Teacher model (the one you want to learn from)"}),
                "student_model": ("MODEL", {"tooltip": "Student/base model (the one to be enhanced)"}),
                "rank": ("INT", {"default": 64, "min": 1, "max": 512, "step": 1, "tooltip": "LoRA rank (higher = more capacity, more VRAM)"}),
            },
            "optional": {
                "filter_keys": ("STRING", {"default": "", "multiline": False, "tooltip": "Filter keys containing this string (empty = all)"}),
                "skip_keys": ("STRING", {"default": "time_embed,label_emb", "multiline": False, "tooltip": "Skip keys containing these (comma-separated)"}),
            },
        }

    RETURN_TYPES = ("LORA_WEIGHTS",)
    RETURN_NAMES = ("lora_weights",)
    FUNCTION = "extract"
    CATEGORY = "distillation"

    def extract(self, teacher_model, student_model, rank, filter_keys="", skip_keys="time_embed,label_emb"):
        device = comfy.model_management.get_torch_device()

        # Get state dicts
        teacher_sd = teacher_model.model.state_dict()
        student_sd = student_model.model.state_dict()

        # Parse skip keys
        skip_list = [s.strip() for s in skip_keys.split(",") if s.strip()]

        # Find matching keys
        matches = get_matchable_keys(teacher_sd, student_sd)

        lora_weights = {}
        processed = 0
        skipped = 0

        for t_key, s_key in matches.items():
            # Apply filter
            if filter_keys and filter_keys not in t_key:
                continue

            # Apply skip
            if any(skip in t_key for skip in skip_list):
                skipped += 1
                continue

            # Only process weight layers
            if not t_key.endswith('.weight'):
                continue

            teacher_w = teacher_sd[t_key].to(device)
            student_w = student_sd[s_key].to(device)

            # Skip 1D tensors (biases, norms)
            if len(teacher_w.shape) < 2:
                continue

            # Compute delta
            delta = compute_delta_weight(teacher_w, student_w)
            if delta is None:
                skipped += 1
                continue

            # SVD decompose
            lora_A, lora_B = svd_decompose(delta, rank)
            if lora_A is None:
                skipped += 1
                continue

            # Store in LoRA format
            base_key = t_key.replace('.weight', '')
            lora_weights[f"{base_key}.lora_down.weight"] = lora_A.cpu()
            lora_weights[f"{base_key}.lora_up.weight"] = lora_B.cpu()
            lora_weights[f"{base_key}.alpha"] = torch.tensor(rank, dtype=torch.float32)

            processed += 1

            # Free memory
            del teacher_w, student_w, delta, lora_A, lora_B
            if processed % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        logging.info(f"Extracted LoRA: {processed} layers processed, {skipped} skipped")

        return (lora_weights,)


class SaveLoRAWeights:
    """Save LoRA weights to file."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_weights": ("LORA_WEIGHTS",),
                "filename": ("STRING", {"default": "distilled_lora"}),
            },
            "optional": {
                "save_format": (["safetensors", "pytorch"],),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save"
    CATEGORY = "distillation"
    OUTPUT_NODE = True

    def save(self, lora_weights, filename, save_format="safetensors"):
        # Get output directory
        output_dir = folder_paths.get_output_directory()
        lora_dir = os.path.join(output_dir, "loras")
        os.makedirs(lora_dir, exist_ok=True)

        # Determine extension and path
        if save_format == "safetensors" and HAS_SAFETENSORS:
            ext = ".safetensors"
            filepath = os.path.join(lora_dir, f"{filename}{ext}")
            save_file(lora_weights, filepath)
        else:
            ext = ".pt"
            filepath = os.path.join(lora_dir, f"{filename}{ext}")
            torch.save(lora_weights, filepath)

        logging.info(f"LoRA saved to: {filepath}")
        return (filepath,)


class LoadLoRAWeights:
    """Load LoRA weights from file."""

    @classmethod
    def INPUT_TYPES(cls):
        lora_files = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "lora_name": (lora_files,),
            },
        }

    RETURN_TYPES = ("LORA_WEIGHTS",)
    RETURN_NAMES = ("lora_weights",)
    FUNCTION = "load"
    CATEGORY = "distillation"

    def load(self, lora_name):
        lora_path = folder_paths.get_full_path("loras", lora_name)

        if lora_path.endswith('.safetensors') and HAS_SAFETENSORS:
            lora_weights = load_file(lora_path)
        else:
            lora_weights = torch.load(lora_path, map_location='cpu')

        return (lora_weights,)


class ApplyLoRAWeightsToModel:
    """Apply LoRA weights to a model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_weights": ("LORA_WEIGHTS",),
                "strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "distillation"

    def apply(self, model, lora_weights, strength):
        if strength == 0:
            return (model,)

        # Clone model
        model_lora = model.clone()

        # Build key map
        key_map = comfy.lora.model_lora_keys_unet(model_lora.model, {})

        # Load and apply
        loaded = comfy.lora.load_lora(lora_weights, key_map)
        model_lora.add_patches(loaded, strength)

        return (model_lora,)


class QuickDistill:
    """One-node solution: Load teacher, distill to LoRA, save - all in one."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model": ("MODEL", {"tooltip": "Base/student model"}),
                "teacher_model": ("MODEL", {"tooltip": "Teacher model to distill from"}),
                "output_name": ("STRING", {"default": "quick_distill_lora"}),
                "rank": ("INT", {"default": 64, "min": 1, "max": 256, "step": 1}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING",)
    RETURN_NAMES = ("model_with_lora", "lora_path",)
    FUNCTION = "distill"
    CATEGORY = "distillation"
    OUTPUT_NODE = True

    def distill(self, base_model, teacher_model, output_name, rank):
        # Step 1: Extract LoRA
        extractor = ExtractLoRAFromModels()
        lora_weights, = extractor.extract(teacher_model, base_model, rank)

        # Step 2: Save LoRA
        saver = SaveLoRAWeights()
        filepath, = saver.save(lora_weights, output_name)

        # Step 3: Apply LoRA to base model
        applier = ApplyLoRAWeightsToModel()
        model_with_lora, = applier.apply(base_model, lora_weights, 1.0)

        return (model_with_lora, filepath,)


# =============================================================================
#                              NODE MAPPINGS
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "ExtractLoRAFromModels": ExtractLoRAFromModels,
    "SaveLoRAWeights": SaveLoRAWeights,
    "LoadLoRAWeights": LoadLoRAWeights,
    "ApplyLoRAWeightsToModel": ApplyLoRAWeightsToModel,
    "QuickDistill": QuickDistill,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExtractLoRAFromModels": "Extract LoRA (SVD Distill)",
    "SaveLoRAWeights": "Save LoRA Weights",
    "LoadLoRAWeights": "Load LoRA Weights",
    "ApplyLoRAWeightsToModel": "Apply LoRA to Model",
    "QuickDistill": "Quick Distill (All-in-One)",
}
