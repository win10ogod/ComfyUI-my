#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Layer Mapping for Cross-Architecture Distillation

Provides intelligent mapping between different model architectures:
- UNet ↔ DiT mappings
- Attention layer alignment
- FFN/MLP mapping
- Temporal attention handling for video models
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn


class LayerType(Enum):
    """Types of layers in diffusion models."""
    ATTENTION_SELF = "self_attention"
    ATTENTION_CROSS = "cross_attention"
    ATTENTION_TEMPORAL = "temporal_attention"
    ATTENTION_JOINT = "joint_attention"
    FFN = "feedforward"
    MLP = "mlp"
    CONV = "convolution"
    NORM = "normalization"
    EMBED = "embedding"
    PROJ_IN = "projection_in"
    PROJ_OUT = "projection_out"
    TIME_EMBED = "time_embedding"
    TEXT_PROJ = "text_projection"
    UNKNOWN = "unknown"


@dataclass
class LayerInfo:
    """Information about a layer."""
    name: str
    layer_type: LayerType
    block_idx: int = -1
    transformer_idx: int = -1
    in_features: int = 0
    out_features: int = 0
    is_input_block: bool = False
    is_output_block: bool = False
    is_middle_block: bool = False
    depth_level: float = 0.0  # Normalized depth (0=input, 1=output)


@dataclass
class LayerMapping:
    """Mapping between teacher and student layers."""
    teacher_layer: str
    student_layer: str
    teacher_info: LayerInfo
    student_info: LayerInfo
    projection_needed: bool = False
    weight_scale: float = 1.0


class LayerParser:
    """Parse layer names to extract structure information."""

    # Common patterns for different architectures
    UNET_PATTERNS = {
        "input": r"input_blocks\.(\d+)",
        "output": r"output_blocks\.(\d+)",
        "middle": r"middle_block",
        "transformer": r"transformer_blocks\.(\d+)",
        "attn1": r"attn1",
        "attn2": r"attn2",
        "ff": r"ff\.",
        "mlp": r"mlp\.",
    }

    DIT_PATTERNS = {
        "joint": r"joint_blocks\.(\d+)",
        "single": r"single_blocks\.(\d+)",
        "double": r"double_blocks\.(\d+)",
        "transformer": r"transformer_blocks\.(\d+)",
        "attn": r"attn\.",
        "mlp": r"mlp\.",
        "ff": r"ff\.",
    }

    FLUX_PATTERNS = {
        "double": r"double_blocks\.(\d+)",
        "single": r"single_blocks\.(\d+)",
        "img_attn": r"img_attn",
        "txt_attn": r"txt_attn",
        "img_mlp": r"img_mlp",
        "txt_mlp": r"txt_mlp",
    }

    @classmethod
    def detect_layer_type(cls, name: str) -> LayerType:
        """Detect the type of layer from its name."""
        name_lower = name.lower()

        # Attention layers
        if "temporal" in name_lower and "attn" in name_lower:
            return LayerType.ATTENTION_TEMPORAL
        if "joint" in name_lower and "attn" in name_lower:
            return LayerType.ATTENTION_JOINT
        if "attn2" in name_lower or "cross_attn" in name_lower:
            return LayerType.ATTENTION_CROSS
        if "attn1" in name_lower or "self_attn" in name_lower or "attn" in name_lower:
            return LayerType.ATTENTION_SELF

        # FFN/MLP
        if any(x in name_lower for x in ["ff.", "feedforward", "ff_"]):
            return LayerType.FFN
        if "mlp" in name_lower:
            return LayerType.MLP

        # Projections
        if "proj_in" in name_lower or "input_proj" in name_lower:
            return LayerType.PROJ_IN
        if "proj_out" in name_lower or "output_proj" in name_lower:
            return LayerType.PROJ_OUT

        # Embeddings
        if "time_embed" in name_lower or "timestep" in name_lower:
            return LayerType.TIME_EMBED
        if "text" in name_lower and "proj" in name_lower:
            return LayerType.TEXT_PROJ
        if "embed" in name_lower:
            return LayerType.EMBED

        # Normalization
        if any(x in name_lower for x in ["norm", "ln", "layernorm", "groupnorm"]):
            return LayerType.NORM

        # Convolution
        if "conv" in name_lower:
            return LayerType.CONV

        return LayerType.UNKNOWN

    @classmethod
    def parse_unet_layer(cls, name: str) -> LayerInfo:
        """Parse a UNet-style layer name."""
        layer_type = cls.detect_layer_type(name)
        info = LayerInfo(name=name, layer_type=layer_type)

        # Check block type
        if re.search(cls.UNET_PATTERNS["input"], name):
            info.is_input_block = True
            match = re.search(cls.UNET_PATTERNS["input"], name)
            if match:
                info.block_idx = int(match.group(1))
        elif re.search(cls.UNET_PATTERNS["output"], name):
            info.is_output_block = True
            match = re.search(cls.UNET_PATTERNS["output"], name)
            if match:
                info.block_idx = int(match.group(1))
        elif re.search(cls.UNET_PATTERNS["middle"], name):
            info.is_middle_block = True
            info.block_idx = 0

        # Get transformer index
        trans_match = re.search(cls.UNET_PATTERNS["transformer"], name)
        if trans_match:
            info.transformer_idx = int(trans_match.group(1))

        return info

    @classmethod
    def parse_dit_layer(cls, name: str) -> LayerInfo:
        """Parse a DiT-style layer name."""
        layer_type = cls.detect_layer_type(name)
        info = LayerInfo(name=name, layer_type=layer_type)

        # Check for joint/single/double blocks
        for pattern_name, pattern in cls.DIT_PATTERNS.items():
            match = re.search(pattern, name)
            if match and match.groups():
                info.block_idx = int(match.group(1))
                break

        return info

    @classmethod
    def parse_flux_layer(cls, name: str) -> LayerInfo:
        """Parse a Flux-style layer name."""
        layer_type = cls.detect_layer_type(name)
        info = LayerInfo(name=name, layer_type=layer_type)

        # Check for double/single blocks
        for pattern_name in ["double", "single"]:
            match = re.search(cls.FLUX_PATTERNS[pattern_name], name)
            if match:
                info.block_idx = int(match.group(1))
                break

        return info

    @classmethod
    def parse_layer(cls, name: str, arch_type: str = "auto") -> LayerInfo:
        """Parse a layer name based on architecture type."""
        if arch_type == "auto":
            # Auto-detect based on patterns
            if re.search(r"(input_blocks|output_blocks|middle_block)", name):
                return cls.parse_unet_layer(name)
            elif re.search(r"(double_blocks|single_blocks)", name):
                return cls.parse_flux_layer(name)
            elif re.search(r"(joint_blocks|transformer_blocks)", name):
                return cls.parse_dit_layer(name)
            else:
                return cls.parse_unet_layer(name)  # Default to UNet parsing
        elif arch_type == "unet":
            return cls.parse_unet_layer(name)
        elif arch_type == "dit":
            return cls.parse_dit_layer(name)
        elif arch_type == "flux":
            return cls.parse_flux_layer(name)
        else:
            return cls.parse_unet_layer(name)


class LayerMapper:
    """Maps layers between different architectures."""

    def __init__(
        self,
        teacher_layers: Dict[str, torch.Tensor],
        student_layers: Dict[str, torch.Tensor],
        teacher_arch: str = "auto",
        student_arch: str = "auto",
    ):
        self.teacher_layers = teacher_layers
        self.student_layers = student_layers
        self.teacher_arch = teacher_arch
        self.student_arch = student_arch

        # Parse all layers
        self.teacher_info = self._parse_all_layers(teacher_layers, teacher_arch)
        self.student_info = self._parse_all_layers(student_layers, student_arch)

        # Compute depth levels
        self._compute_depth_levels(self.teacher_info)
        self._compute_depth_levels(self.student_info)

    def _parse_all_layers(
        self,
        layers: Dict[str, torch.Tensor],
        arch_type: str
    ) -> Dict[str, LayerInfo]:
        """Parse all layer names."""
        info_dict = {}
        for name, tensor in layers.items():
            info = LayerParser.parse_layer(name, arch_type)

            # Add dimension info
            if tensor.dim() >= 2:
                info.out_features = tensor.shape[0]
                info.in_features = tensor.shape[1] if tensor.dim() > 1 else tensor.shape[0]

            info_dict[name] = info
        return info_dict

    def _compute_depth_levels(self, info_dict: Dict[str, LayerInfo]):
        """Compute normalized depth levels for layers."""
        # Find max block indices
        max_input_idx = 0
        max_output_idx = 0
        max_block_idx = 0

        for info in info_dict.values():
            if info.is_input_block:
                max_input_idx = max(max_input_idx, info.block_idx)
            elif info.is_output_block:
                max_output_idx = max(max_output_idx, info.block_idx)
            max_block_idx = max(max_block_idx, info.block_idx)

        total_blocks = max_input_idx + max_output_idx + 1 if max_input_idx > 0 else max_block_idx

        for info in info_dict.values():
            if total_blocks > 0:
                if info.is_input_block:
                    info.depth_level = info.block_idx / (2 * total_blocks)
                elif info.is_middle_block:
                    info.depth_level = 0.5
                elif info.is_output_block:
                    info.depth_level = 0.5 + (info.block_idx + 1) / (2 * total_blocks)
                else:
                    info.depth_level = info.block_idx / max(max_block_idx, 1)

    def find_layer_mappings(
        self,
        match_by_type: bool = True,
        match_by_depth: bool = True,
        depth_tolerance: float = 0.15,
    ) -> List[LayerMapping]:
        """Find mappings between teacher and student layers."""
        mappings = []
        used_student_layers: Set[str] = set()

        # Group layers by type
        teacher_by_type: Dict[LayerType, List[str]] = {}
        student_by_type: Dict[LayerType, List[str]] = {}

        for name, info in self.teacher_info.items():
            if info.layer_type not in teacher_by_type:
                teacher_by_type[info.layer_type] = []
            teacher_by_type[info.layer_type].append(name)

        for name, info in self.student_info.items():
            if info.layer_type not in student_by_type:
                student_by_type[info.layer_type] = []
            student_by_type[info.layer_type].append(name)

        # Match layers of the same type
        for layer_type, teacher_names in teacher_by_type.items():
            if layer_type not in student_by_type:
                continue

            student_names = student_by_type[layer_type]

            # Sort by depth level
            teacher_sorted = sorted(
                teacher_names,
                key=lambda x: self.teacher_info[x].depth_level
            )
            student_sorted = sorted(
                student_names,
                key=lambda x: self.student_info[x].depth_level
            )

            # Map by relative position
            if len(teacher_sorted) > 0 and len(student_sorted) > 0:
                for i, t_name in enumerate(teacher_sorted):
                    # Find corresponding student layer
                    t_depth = self.teacher_info[t_name].depth_level
                    best_match = None
                    best_diff = float('inf')

                    for s_name in student_sorted:
                        if s_name in used_student_layers:
                            continue

                        s_depth = self.student_info[s_name].depth_level
                        diff = abs(t_depth - s_depth)

                        if diff < best_diff and (not match_by_depth or diff <= depth_tolerance):
                            best_diff = diff
                            best_match = s_name

                    if best_match:
                        t_info = self.teacher_info[t_name]
                        s_info = self.student_info[best_match]

                        # Check if projection is needed
                        proj_needed = (
                            t_info.in_features != s_info.in_features or
                            t_info.out_features != s_info.out_features
                        )

                        mapping = LayerMapping(
                            teacher_layer=t_name,
                            student_layer=best_match,
                            teacher_info=t_info,
                            student_info=s_info,
                            projection_needed=proj_needed,
                        )
                        mappings.append(mapping)
                        used_student_layers.add(best_match)

        return mappings

    def find_attention_mappings(self) -> List[LayerMapping]:
        """Find mappings specifically for attention layers."""
        return [
            m for m in self.find_layer_mappings()
            if m.teacher_info.layer_type in [
                LayerType.ATTENTION_SELF,
                LayerType.ATTENTION_CROSS,
                LayerType.ATTENTION_JOINT,
            ]
        ]

    def find_mlp_mappings(self) -> List[LayerMapping]:
        """Find mappings specifically for MLP/FFN layers."""
        return [
            m for m in self.find_layer_mappings()
            if m.teacher_info.layer_type in [
                LayerType.FFN,
                LayerType.MLP,
            ]
        ]


class UNetToDiTMapper:
    """Specialized mapper for UNet to DiT conversion."""

    # Semantic mapping from UNet blocks to DiT blocks
    SEMANTIC_MAP = {
        # Early layers (encoding)
        "input_blocks.0": "double_blocks.0",
        "input_blocks.1": "double_blocks.1",
        "input_blocks.2": "double_blocks.2",
        # Middle layers
        "input_blocks.3": "double_blocks.3",
        "input_blocks.4": "double_blocks.4",
        "middle_block": "double_blocks.mid",
        # Late layers (decoding)
        "output_blocks.0": "double_blocks.-3",
        "output_blocks.1": "double_blocks.-2",
        "output_blocks.2": "double_blocks.-1",
    }

    @classmethod
    def create_mapping(
        cls,
        unet_layers: Dict[str, torch.Tensor],
        dit_layers: Dict[str, torch.Tensor],
    ) -> Dict[str, str]:
        """Create a mapping from UNet layer names to DiT layer names."""
        mapping = {}

        # Get block counts
        unet_blocks = set()
        dit_blocks = set()

        for name in unet_layers:
            for pattern in ["input_blocks", "output_blocks", "middle_block"]:
                if pattern in name:
                    match = re.search(rf"{pattern}\.(\d+)", name)
                    if match:
                        unet_blocks.add(f"{pattern}.{match.group(1)}")
                    elif pattern == "middle_block":
                        unet_blocks.add("middle_block.0")

        for name in dit_layers:
            for pattern in ["double_blocks", "single_blocks", "joint_blocks"]:
                match = re.search(rf"{pattern}\.(\d+)", name)
                if match:
                    dit_blocks.add(f"{pattern}.{match.group(1)}")

        # Create linear mapping based on relative positions
        unet_sorted = sorted(list(unet_blocks), key=lambda x: (
            0 if "input" in x else (1 if "middle" in x else 2),
            int(re.search(r"\.(\d+)", x).group(1)) if re.search(r"\.(\d+)", x) else 0
        ))

        dit_sorted = sorted(list(dit_blocks), key=lambda x: (
            int(re.search(r"\.(\d+)", x).group(1)) if re.search(r"\.(\d+)", x) else 0
        ))

        # Map proportionally
        if unet_sorted and dit_sorted:
            for i, unet_block in enumerate(unet_sorted):
                relative_pos = i / max(len(unet_sorted) - 1, 1)
                dit_idx = int(relative_pos * (len(dit_sorted) - 1))
                mapping[unet_block] = dit_sorted[dit_idx]

        return mapping


class VideoModelMapper:
    """Specialized mapper for video model architectures."""

    @classmethod
    def find_temporal_layers(cls, layers: Dict[str, torch.Tensor]) -> List[str]:
        """Find layers related to temporal processing."""
        temporal_layers = []
        for name in layers:
            if any(x in name.lower() for x in ["temporal", "time_mix", "t_attn", "temp_"]):
                temporal_layers.append(name)
        return temporal_layers

    @classmethod
    def create_temporal_mapping(
        cls,
        teacher_layers: Dict[str, torch.Tensor],
        student_layers: Dict[str, torch.Tensor],
    ) -> Dict[str, str]:
        """Create mapping for temporal layers between video models."""
        teacher_temporal = cls.find_temporal_layers(teacher_layers)
        student_temporal = cls.find_temporal_layers(student_layers)

        mapping = {}

        # Sort by position in name
        teacher_sorted = sorted(teacher_temporal)
        student_sorted = sorted(student_temporal)

        # Proportional mapping
        for i, t_name in enumerate(teacher_sorted):
            if student_sorted:
                relative_pos = i / max(len(teacher_sorted) - 1, 1)
                s_idx = int(relative_pos * (len(student_sorted) - 1))
                mapping[t_name] = student_sorted[s_idx]

        return mapping


def create_layer_mapping(
    teacher_state_dict: Dict[str, torch.Tensor],
    student_state_dict: Dict[str, torch.Tensor],
    teacher_arch: str = "auto",
    student_arch: str = "auto",
) -> Tuple[List[LayerMapping], Dict[str, Any]]:
    """Create a complete layer mapping between two models.

    Returns:
        Tuple of (mappings, stats)
    """
    mapper = LayerMapper(
        teacher_state_dict,
        student_state_dict,
        teacher_arch,
        student_arch,
    )

    all_mappings = mapper.find_layer_mappings()
    attn_mappings = mapper.find_attention_mappings()
    mlp_mappings = mapper.find_mlp_mappings()

    stats = {
        "total_mappings": len(all_mappings),
        "attention_mappings": len(attn_mappings),
        "mlp_mappings": len(mlp_mappings),
        "projection_needed": sum(1 for m in all_mappings if m.projection_needed),
        "teacher_layers": len(teacher_state_dict),
        "student_layers": len(student_state_dict),
    }

    return all_mappings, stats


def get_matching_layer_pairs(
    teacher_state_dict: Dict[str, torch.Tensor],
    student_state_dict: Dict[str, torch.Tensor],
    include_patterns: List[str] = None,
    exclude_patterns: List[str] = None,
) -> List[Tuple[str, str, bool]]:
    """Get pairs of matching layers between teacher and student.

    Returns:
        List of (teacher_key, student_key, needs_projection) tuples
    """
    if include_patterns is None:
        include_patterns = ["attn", "mlp", "ff", "proj", "linear"]
    if exclude_patterns is None:
        exclude_patterns = ["norm", "embed", "time"]

    pairs = []
    mappings, _ = create_layer_mapping(teacher_state_dict, student_state_dict)

    for mapping in mappings:
        t_key = mapping.teacher_layer
        s_key = mapping.student_layer

        # Apply filters
        if not any(p in t_key.lower() for p in include_patterns):
            continue
        if any(p in t_key.lower() for p in exclude_patterns):
            continue

        pairs.append((t_key, s_key, mapping.projection_needed))

    return pairs
