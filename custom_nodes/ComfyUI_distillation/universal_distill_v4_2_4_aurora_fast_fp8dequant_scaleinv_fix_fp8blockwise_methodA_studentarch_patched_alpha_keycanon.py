#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Universal SVD-LoRA Distillation v4.2.4 (Cross-Arch Innovations) — Plug-and-Play Truncated SVD (Low-VRAM)

Key changes vs v3.7.16 (SVD-Optimized):
----------------------
0. Adds AURORA-SVD (Adaptive Unrolled Residual-Order Range Augmentation SVD) as a distillation-centric low-rank backend.
0. Fixes DRF-SVD backend definition/indentation so --svd-mode drf works as intended.
0. Randomized SVD now uses QR-normalized subspace iteration (Halko et al., Alg. 4.4) for better numerical stability.
0. Adaptive rank selection is now measured against ||Δ||_F^2 and grows rank geometrically instead of always computing SVD at max_rank.
1. Adds SVD-LLM style full-covariance whitening (Cholesky) as a calibration mode ("cov")
   to replace diagonal-only RMS scaling when requested.
2. Restores v3.7.15-stable TIES merge semantics by default and applies the TIES Trim
   step to single-delta cases by default (use --no-ties-trim-single to disable).
3. Preserves the low-VRAM streaming/offload design (cpu/disk offload + prefetch).

Output:
-------
Writes a PEFT LoRA adapter (adapter_model.safetensors + adapter_config.json).

License: Apache 2.0
"""

import os
import re
import json
import math
import time
import gc
import argparse
import warnings
import threading
import queue
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set, Union, Iterator
from enum import Enum
from contextlib import contextmanager
from pathlib import Path
import tempfile
import shutil

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

try:
    from safetensors.torch import load_file, save_file, safe_open
except ImportError:
    raise RuntimeError("safetensors is required. pip install safetensors")

try:
    from transformers import AutoConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    warnings.warn("transformers not found. Architecture auto-detection disabled.")


# =============================================================================
#                    FLOAT8 / QUANTIZED WEIGHT SAFETY HELPERS
# =============================================================================

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


_FLOAT8_DTYPES: Tuple[torch.dtype, ...] = _collect_float8_dtypes()


def _is_float8_dtype(dtype: torch.dtype) -> bool:
    """Best-effort float8 dtype check (supports future torch additions)."""
    try:
        if dtype in _FLOAT8_DTYPES:
            return True
    except Exception:
        pass
    return "float8" in str(dtype)


def _resolve_dequant_dtype(config: "DistillConfig", device: torch.device) -> torch.dtype:
    """Resolve target dtype for streaming dequantization / float8 safety casts."""
    mode = str(getattr(config, "dequantize_dtype", "auto") or "auto").lower().strip()
    if mode in ("bf16", "bfloat16"):
        return torch.bfloat16
    if mode in ("fp16", "float16", "half"):
        return torch.float16
    if mode in ("fp32", "float32", "float"):
        return torch.float32

    # auto
    if device.type == "cuda":
        try:
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16

    # CPU / other: prefer bf16 for range; torch will emulate if needed.
    return torch.bfloat16


def _safe_weight_dtype(src_dtype: torch.dtype,
                       config: "DistillConfig",
                       device: torch.device,
                       ref_dtype: Optional[torch.dtype] = None) -> torch.dtype:
    """Choose an output dtype that avoids float8 mixed arithmetic pitfalls."""
    if not _is_float8_dtype(src_dtype):
        return src_dtype
    if ref_dtype is not None and (not _is_float8_dtype(ref_dtype)) and ("float" in str(ref_dtype)):
        return ref_dtype
    return _resolve_dequant_dtype(config, device)

def _fp8_dequantize(weight: torch.Tensor, scale_inv: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    FP8 -> float32 dequantization (pure PyTorch, no Triton) with support for:

      1) DeepSeek / MiniMax per-group block-wise FP8:
         - weight shape: (M, N)
         - scale_inv shape: (ceil(M/bs), ceil(N/bs)) OR flattened (ceil(M/bs) * ceil(N/bs))
         - kernel logic: y = x * scale_inv  (multiply!)

      2) TransformerEngine-style scalar / broadcastable scale_inv:
         - falls back to broadcasting multiplication.

    Returns:
        torch.float32 tensor (caller can cast to BF16/FP16 as needed).
    """
    if weight is None or scale_inv is None:
        raise ValueError("weight and scale_inv must be non-None")

    # Scalar fast-path
    try:
        if scale_inv.numel() == 1:
            return weight.float() * scale_inv.float()
    except Exception:
        # If numel() is not available for some reason, fall back to generic path.
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
        # Some checkpoints store 2D but not in the expected shape; if total elements match, reshape.
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


# =============================================================================
#                              ENUMS & DATA CLASSES
# =============================================================================

class AttentionType(Enum):
    MHA = "multi_head_attention"
    MQA = "multi_query_attention"
    GQA = "grouped_query_attention"

class MLPType(Enum):
    STANDARD = "standard_ffn"
    SWIGLU = "swiglu"
    GEGLU = "geglu"
    MOE = "mixture_of_experts"

class OffloadStrategy(Enum):
    NONE = "none"
    CPU = "cpu"
    DISK = "disk"
    AUTO = "auto"


@dataclass
class ArchitectureInfo:
    """Detected architecture information."""
    attention_type: AttentionType = AttentionType.MHA
    mlp_type: MLPType = MLPType.STANDARD
    num_attention_heads: int = 32
    num_kv_heads: int = 32
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    vocab_size: int = 32000
    num_experts: int = 0
    num_experts_per_tok: int = 0
    layer_prefix: str = "model.layers"
    attn_prefix: str = "self_attn"
    mlp_prefix: str = "mlp"
    q_proj_name: str = "q_proj"
    k_proj_name: str = "k_proj"
    v_proj_name: str = "v_proj"
    o_proj_name: str = "o_proj"
    gate_proj_name: str = "gate_proj"
    up_proj_name: str = "up_proj"
    down_proj_name: str = "down_proj"
    has_fused_qkv: bool = False
    qkv_proj_name: str = "qkv_proj"


@dataclass
class DistillConfig:
    """Configuration for Universal SVD-LoRA Distillation.

    v3.7 goals:
      - Truly low-cost, out-of-the-box SVD distillation via truncated / randomized SVD by default.
      - Preserve v3.2's low-VRAM, shard-wise processing and layer offloading characteristics.
      - Improve cross-architecture robustness via efficient SVD-based projection with bounded rank.
    """

    # Paths
    teacher_path: str
    student_path: str
    output_path: str

    # LoRA rank controls
    rank_default: int = 64
    rank_attn: Optional[int] = None
    rank_mlp: Optional[int] = None
    lora_alpha: Optional[int] = None

    alpha_mode: str = "auto"  # auto | rank | fixed
    use_adaptive_rank: bool = True
    energy_threshold: float = 0.95
    min_rank: int = 8
    max_rank: int = 256

    # Delta regularization / merge helpers
    use_dare: bool = False
    dare_drop_rate: float = 0.3
    use_ties: bool = True
    ties_density: float = 0.3
    ties_trim_single: bool = True

    # Optional MoE blending (kept for backwards-compat; not used by default)
    moe_merge_method: str = "none"
    max_experts_to_blend: int = 2

    # Layer mapping schedule
    map_schedule: str = "linear"     # linear | sigmoid
    sigmoid_k: float = 0.15

    interp_mode: str = "schedule"  # schedule | lsq

    # Runtime / memory controls
    num_gpus: int = 1
    gpu_memory_fraction: float = 0.95
    use_cuda_streams: bool = True
    num_cuda_streams: int = 3
    use_mixed_precision: bool = True
    svd_batch_size: int = 1

    # Quantized-weight handling (streaming dequantization)
    #
    # - float8 weights (e.g., native FP8-trained checkpoints) cannot participate in mixed-dtype
    #   arithmetic (e.g., float8 - bf16) on many PyTorch builds. We resolve this by converting
    #   float8 tensors to a safe floating dtype during streaming load and/or at delta time.
    #
    # Notes:
    # - This is *not* a general-purpose AWQ/GPTQ dequantizer. AWQ/GPTQ models typically store
    #   qweight/qzeros/scales rather than dense ".weight" tensors; this script expects dense
    #   weight matrices for SVD/LoRA extraction.
    dequantize_float8: bool = True
    dequantize_dtype: str = "auto"  # auto|bf16|fp16|fp32

    fp8_block_size: int = 128  # per-group FP8 block size for *_scale_inv (DeepSeek/MiniMax default: 128)
    offload_strategy: str = "cpu"    # cpu | disk | none
    offload_dir: Optional[str] = None
    prefetch_layers: int = 2
    use_pinned_memory: bool = True
    max_cpu_memory_gb: float = 64.0

    # SVD execution / batching
    chunk_threshold_mb: int = 512
    chunk_size_mb: int = 128
    micro_batch_size: int = 8

    # Module selection
    include_pattern: str = "self_attn|mlp"
    exclude_pattern: str = ""
    include_embed_lm_head: bool = False

    # Cross-architecture projection (teacher tensor -> student tensor shape)
    use_svd_projection: bool = True
    projection_rank: int = 256
    projection_min_rank: int = 8
    projection_adaptive_rank: bool = True
    projection_energy_threshold: float = 0.99
    projection_randomized_iter: int = 2
    projection_randomized_oversamples: int = 8

    # Cross-arch subspace-delta controls
    subspace_delta: str = "off"  # off|auto|diag|full
    subspace_offdiag_threshold: float = 0.55
    subspace_denom: str = "core"  # core|full
    subspace_norm_match: bool = True

    # SVD backend selection
    #   - auto: heuristic (full vs low-rank)
    #   - full: torch.linalg.svd
    #   - randomized: Halko Alg. 4.4 (stable subspace iteration)
    #   - krylov: Block Krylov subspace (stores intermediate iterates)
    #   - adaptive: Adaptive range finder (Halko Alg. 4.2) for energy-targeted rank
    svd_mode: str = "auto"  # auto | full | randomized | krylov | adaptive | drf | aurora
    svd_randomized_iter: int = 2
    svd_randomized_oversamples: int = 8
    svd_auto_min_dim: int = 1024
    svd_auto_full_rank_ratio: float = 0.6
    svd_auto_lowrank: str = "randomized"  # randomized | krylov | drf | aurora

    # DRF-SVD (Dual-Residual Folding SVD) knobs
    #   - steps: number of residual-folding refinements (>=0)
    #   - theta: residual scaling exponent in [0,1] (0=none, 1=full 1/σ weighting)
    #   - resid_eps: numerical stabilizer for 1/σ^theta
    svd_drf_steps: int = 1
    svd_drf_theta: float = 0.5
    svd_drf_resid_eps: float = 1e-8

    # AURORA-SVD (Adaptive Unrolled Residual-Order Range Augmentation SVD) knobs
    #   - steps: refinement rounds (>=0)
    #   - order: 1=first-order residual; 2=adds second-order residual response via A/A^T (recommended)
    #   - theta1/theta2: residual scaling exponents in [0,1]
    #   - resid_eps: numerical stabilizer for scaling
    #   - level2_keep: fraction of components kept for the level-2 term in (0,1]; 1.0 keeps all
    svd_aurora_steps: int = 1
    svd_aurora_order: int = 2
    svd_aurora_theta1: float = 0.5
    svd_aurora_theta2: float = 0.25
    svd_aurora_resid_eps: float = 1e-8
    svd_aurora_level2_keep: float = 1.0
    svd_aurora_level1_keep: float = 1.0

    # AURORA speed/quality controls (optional)
    #   - min_steps: always run at least this many refinement rounds when steps>0
    #   - early_stop: relative residual-improvement tolerance (0 disables early stop)
    #   - early_stop_patience: number of consecutive checks below tolerance required to stop
    #   - early_stop_check_every: evaluate early-stop condition every N refinement rounds
    #   - core_solver: 'svd' (torch.linalg.svd) or 'eigh' (Gram-eigh reconstruction)
    #   - svd_driver: CUDA SVD driver hint ('gesvdj' often faster on small cores); ignored if unsupported
    svd_aurora_min_steps: int = 1
    svd_aurora_early_stop: float = 0.005
    svd_aurora_early_stop_patience: int = 2
    svd_aurora_early_stop_check_every: int = 1
    svd_aurora_core_solver: str = "svd"
    svd_aurora_svd_driver: str = "gesvdj"
    # Adaptive range finder (Alg. 4.2) knobs (used when svd_mode=adaptive)
    svd_adaptive_block_size: int = 32  # columns added per iteration
    svd_adaptive_n_test: int = 8       # test vectors for residual estimate

    # Delta stability guard (scales delta if ||delta||/||student|| exceeds this ratio; 0 disables)
    max_delta_ratio: float = 0.35


    # Calibration-aware SVD (optional; default: off)
    #
    # Purpose: collect activation statistics on the *student* model (calibration prompts)
    # and apply data-aware whitening before SVD so the truncated low-rank factors better
    # match data-distribution sensitivity.
    #
    # calibration_mode:
    #   - "none": disable
    #   - "rms":  diagonal RMS whitening using per-channel activation RMS (input/output)
    #   - "cov":  SVD-LLM style full-covariance input whitening via Cholesky factor S
    #             of E[xx^T], using PiSSA on (Δ·S) and then unwhitening with S^{-1}.
    calibration_mode: str = "none"  # choices: none | rms | cov
    calib_data: Optional[str] = None          # .txt (1 prompt/line), .json/.jsonl, or Alpaca JSON (.json/.jsonl)
    calib_format: str = "auto"                # auto | txt | jsonl | json | alpaca
    calib_alpaca_template: str = "classic"     # classic | plain (only used when calib_format=alpaca or auto-detected)
    calib_alpaca_include_output: bool = False # include output tokens in calibration forward (alpaca only)
    calib_max_samples: int = 128              # number of prompts (after filtering empties)
    calib_max_length: int = 256               # max tokens per prompt
    calib_padding: str = "longest"          # longest | max_length (padding strategy during tokenization)
    calib_use_attention_mask: bool = True   # exclude attention_mask==0 tokens from activation stats (padding tokens)
    calib_batch_size: int = 1                 # batch size for forward passes
    calib_device: str = "auto"                # auto | cuda | cpu
    calib_dtype: str = "auto"                 # auto | bf16 | fp16 | fp32
    calib_load: Optional[str] = None          # load precomputed stats (.safetensors)
    calib_save: Optional[str] = None          # save computed stats (.safetensors)
    calib_eps: float = 1e-6                   # numerical stabilizer for whitening
    calib_collect_in: bool = True             # collect input RMS
    calib_collect_out: bool = True            # collect output RMS

    # Full-covariance (SVD-LLM) calibration controls (only used when calib_mode=cov)
    #
    # Notes:
    #   - cov whitening is applied on the *input* side (per SVD-LLM). For very large
    #     input dims (e.g., MLP down_proj), collection can be gated via calib_cov_max_dim.
    #   - calibration can be chunked across layers to bound peak memory.
    calib_cov_max_dim: int = 8192             # skip covariance collection if in_features > this
    calib_cov_chunk_layers: int = 4           # number of layers per calibration pass (>=1)
    calib_cov_groups: str = "qkv,o,mlp"       # comma-list: qkv,o,mlp,down
    calib_cov_store_dtype: str = "fp16"       # fp16|bf16|fp32 (saved chol factor dtype)

    # Misc
    auto_adjust_dare: bool = True
    seed: int = 42
    verbose: bool = True

# =============================================================================
#                           GPU MEMORY MANAGEMENT  
# =============================================================================

class GPUMemoryManager:
    def __init__(self, device: Union[str, torch.device], 
                 memory_fraction: float = 0.85,
                 verbose: bool = True):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.memory_fraction = memory_fraction
        self.verbose = verbose
        
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            self.total_memory = torch.cuda.get_device_properties(self.device).total_memory
            self.max_memory = int(self.total_memory * memory_fraction)
        else:
            self.total_memory = 0
            self.max_memory = 0
    
    def get_stats(self) -> Dict[str, float]:
        if self.device.type != "cuda":
            return {"total_gb": 0, "used_gb": 0, "free_gb": 0}
        
        allocated = torch.cuda.memory_allocated(self.device)
        return {
            "total_gb": self.total_memory / (1024**3),
            "used_gb": allocated / (1024**3),
            "free_gb": (self.max_memory - allocated) / (1024**3),
        }
    
    def can_allocate(self, size_bytes: int) -> bool:
        if self.device.type != "cuda":
            return True
        allocated = torch.cuda.memory_allocated(self.device)
        return (allocated + size_bytes) < self.max_memory
    
    def clear_cache(self):
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


# =============================================================================
#                           CUDA STREAM MANAGER
# =============================================================================

class CUDAStreamManager:
    def __init__(self, device: torch.device, num_streams: int = 3):
        self.device = device
        self.num_streams = num_streams
        self.streams = []
        self.current_stream_idx = 0
        
        if device.type == "cuda":
            for _ in range(num_streams):
                self.streams.append(torch.cuda.Stream(device=device))
    
    @contextmanager
    def stream_context(self, stream_idx: Optional[int] = None):
        if not self.streams:
            yield
            return
        
        idx = stream_idx if stream_idx is not None else self.current_stream_idx
        self.current_stream_idx = (self.current_stream_idx + 1) % self.num_streams
        
        with torch.cuda.stream(self.streams[idx]):
            yield
    
    def synchronize_current(self):
        if self.streams:
            self.streams[self.current_stream_idx].synchronize()
    
    def synchronize_all(self):
        for stream in self.streams:
            stream.synchronize()


# =============================================================================
#                           LAYER OFFLOADER
# =============================================================================

class LayerOffloader:
    def __init__(self,
                 strategy: OffloadStrategy,
                 device: torch.device,
                 offload_dir: Optional[str] = None,
                 use_pinned: bool = True,
                 max_cpu_gb: float = 32.0,
                 prefetch_count: int = 2,
                 verbose: bool = True):
        self.strategy = strategy
        self.device = device
        self.use_pinned = use_pinned and device.type == "cuda"
        self.max_cpu_bytes = int(max_cpu_gb * 1024**3)
        self.prefetch_count = prefetch_count
        self.verbose = verbose
        
        self.gpu_cache: Dict[str, torch.Tensor] = {}
        self.cpu_cache: Dict[str, torch.Tensor] = {}
        self.disk_paths: Dict[str, str] = {}
        self.cpu_bytes_used = 0
        
        if offload_dir:
            self.offload_dir = offload_dir
        else:
            self.offload_dir = tempfile.mkdtemp(prefix="distill_offload_")
        os.makedirs(self.offload_dir, exist_ok=True)
        
        self.prefetch_queue = queue.Queue()
        self.stop_prefetch = threading.Event()
        self.prefetch_thread = None
        self.transfer_stream = None
        
        if device.type == "cuda" and strategy != OffloadStrategy.NONE:
            self.transfer_stream = torch.cuda.Stream(device=device)
            self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
            self.prefetch_thread.start()
    
    def _prefetch_worker(self):
        while not self.stop_prefetch.is_set():
            try:
                key = self.prefetch_queue.get(timeout=0.1)
                if key not in self.gpu_cache:
                    self._load_to_gpu(key)
            except queue.Empty:
                continue
    
    def _to_pinned(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.use_pinned and not tensor.is_pinned():
            pinned = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
            pinned.copy_(tensor)
            return pinned
        return tensor
    
    def store(self, key: str, tensor: torch.Tensor, priority: str = "cpu"):
        if self.strategy == OffloadStrategy.NONE:
            self.gpu_cache[key] = tensor
        else:
            size = tensor.numel() * tensor.element_size()
            if priority == "disk":
                self._store_to_disk(key, tensor)
            elif self.cpu_bytes_used + size <= self.max_cpu_bytes:
                cpu_tensor = tensor.cpu()
                self.cpu_cache[key] = self._to_pinned(cpu_tensor)
                self.cpu_bytes_used += size
            else:
                self._store_to_disk(key, tensor)
    
    def _store_to_disk(self, key: str, tensor: torch.Tensor):
        safe_key = re.sub(r'[^\w\-.]', '_', key)
        path = os.path.join(self.offload_dir, f"{safe_key}.pt")
        torch.save(tensor.cpu(), path)
        self.disk_paths[key] = path
    
    def _load_to_gpu(self, key: str) -> Optional[torch.Tensor]:
        if key in self.gpu_cache:
            return self.gpu_cache[key]
        
        tensor = None
        if key in self.cpu_cache:
            tensor = self.cpu_cache[key]
        elif key in self.disk_paths:
            tensor = torch.load(self.disk_paths[key])
        
        if tensor is not None:
            if self.transfer_stream:
                with torch.cuda.stream(self.transfer_stream):
                    gpu_tensor = tensor.to(self.device, non_blocking=True)
                self.transfer_stream.synchronize()
            else:
                gpu_tensor = tensor.to(self.device)
            
            self.gpu_cache[key] = gpu_tensor
            return gpu_tensor
        return None
    
    def get(self, key: str, prefetch_next: Optional[List[str]] = None) -> Optional[torch.Tensor]:
        if prefetch_next:
            for next_key in prefetch_next[:self.prefetch_count]:
                if next_key not in self.gpu_cache:
                    self.prefetch_queue.put(next_key)
        return self._load_to_gpu(key)
    
    def evict(self, key: str):
        if key in self.gpu_cache:
            del self.gpu_cache[key]
    
    def evict_all_gpu(self):
        self.gpu_cache.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
    
    def cleanup(self):
        self.stop_prefetch.set()
        if self.prefetch_thread:
            self.prefetch_thread.join(timeout=1.0)
        self.gpu_cache.clear()
        self.cpu_cache.clear()
        if os.path.exists(self.offload_dir):
            shutil.rmtree(self.offload_dir, ignore_errors=True)


# =============================================================================
#                        GPU-ACCELERATED SVD
# =============================================================================

class GPUAcceleratedSVD:
    """SVD backend with GPU acceleration and low-memory low-rank fallbacks.

    Implemented low-rank backends:
      - Randomized SVD with numerically-stable subspace iteration (Halko et al., Alg. 4.4)
      - Block Krylov subspace SVD (stores intermediate iterates, improves accuracy per iteration)
      - Adaptive range finder (Halko et al., Alg. 4.2) for energy-targeted rank selection
      - DRF-SVD (Dual-Residual Folding SVD): residual-folding refinement for improved subspace capture
    """

    def __init__(
        self,
        device: torch.device,
        memory_manager: GPUMemoryManager,
        stream_manager: Optional[CUDAStreamManager],
        use_mixed_precision: bool = True,
        chunk_threshold_mb: int = 512,
        chunk_size_mb: int = 128,
        force_cpu: bool = False,
        verbose: bool = False,
        svd_mode: str = "auto",
        randomized_iter: int = 2,
        randomized_oversamples: int = 8,
        auto_min_dim: int = 1024,
        auto_full_rank_ratio: float = 0.6,
        auto_lowrank: str = "randomized",  # randomized|krylov
        adaptive_block_size: int = 32,
        adaptive_n_test: int = 8,
        drf_steps: int = 1,
        drf_theta: float = 0.5,
        drf_resid_eps: float = 1e-8,
        aurora_steps: int = 1,
        aurora_order: int = 2,
        aurora_theta1: float = 0.5,
        aurora_theta2: float = 0.25,
        aurora_resid_eps: float = 1e-8,
        aurora_level2_keep: float = 1.0,
        aurora_level1_keep: float = 1.0,
        aurora_min_steps: int = 1,
        aurora_early_stop: float = 0.005,
        aurora_early_stop_patience: int = 2,
        aurora_early_stop_check_every: int = 1,
        aurora_core_solver: str = "svd",
        aurora_svd_driver: str = "gesvdj",
    ):
        self.device = device
        self.memory_manager = memory_manager
        self.stream_manager = stream_manager
        self.use_mixed_precision = use_mixed_precision
        self.chunk_threshold_bytes = int(chunk_threshold_mb) * 1024 * 1024
        self.chunk_size_bytes = int(chunk_size_mb) * 1024 * 1024
        self.force_cpu = bool(force_cpu)
        self.verbose = bool(verbose)

        self.svd_mode = (svd_mode or "auto").lower()
        self.randomized_iter = int(randomized_iter)
        self.randomized_oversamples = int(randomized_oversamples)
        self.auto_min_dim = int(auto_min_dim)
        self.auto_full_rank_ratio = float(auto_full_rank_ratio)
        self.auto_lowrank = (auto_lowrank or "randomized").lower()

        self.adaptive_block_size = int(adaptive_block_size)
        self.adaptive_n_test = int(adaptive_n_test)
        # DRF-SVD knobs
        self.drf_steps = int(drf_steps)
        self.drf_theta = float(drf_theta)
        self.drf_resid_eps = float(drf_resid_eps)
        # AURORA-SVD knobs
        self.aurora_steps = int(aurora_steps)
        self.aurora_order = int(aurora_order)
        self.aurora_theta1 = float(aurora_theta1)
        self.aurora_theta2 = float(aurora_theta2)
        self.aurora_resid_eps = float(aurora_resid_eps)
        self.aurora_level2_keep = float(aurora_level2_keep)
        self.aurora_level1_keep = float(aurora_level1_keep)
        self.aurora_min_steps = int(aurora_min_steps)
        self.aurora_early_stop = float(aurora_early_stop)
        self.aurora_early_stop_patience = int(aurora_early_stop_patience)
        self.aurora_early_stop_check_every = int(aurora_early_stop_check_every)
        self.aurora_core_solver = (aurora_core_solver or "svd").lower()
        self.aurora_svd_driver = str(aurora_svd_driver or "")

        # Lightweight AURORA telemetry (for speed/quality tuning)
        self._aurora_calls = 0
        self._aurora_steps_used_total = 0

        # Validate
        if self.svd_mode not in ("auto", "full", "randomized", "krylov", "adaptive", "drf", "aurora"):
            raise ValueError(
                f"Invalid svd_mode: {svd_mode} "
                f"(expected auto|full|randomized|krylov|adaptive|drf|aurora)"
            )
        if self.auto_lowrank not in ("randomized", "krylov", "drf", "aurora"):
            raise ValueError("auto_lowrank must be randomized|krylov|drf|aurora")
        if self.randomized_iter < 0:
            raise ValueError("randomized_iter must be >= 0")
        if self.randomized_oversamples < 0:
            raise ValueError("randomized_oversamples must be >= 0")
        if not (0.0 < self.auto_full_rank_ratio <= 1.0):
            raise ValueError("auto_full_rank_ratio must be in (0, 1]")
        if self.adaptive_block_size <= 0:
            raise ValueError("adaptive_block_size must be > 0")
        if self.adaptive_n_test <= 0:
            raise ValueError("adaptive_n_test must be > 0")
        if self.drf_steps < 0:
            raise ValueError("drf_steps must be >= 0")
        if not (0.0 <= self.drf_theta <= 1.0):
            raise ValueError("drf_theta must be in [0, 1]")
        if self.drf_resid_eps <= 0:
            raise ValueError("drf_resid_eps must be > 0")

        # AURORA validation
        if self.aurora_steps < 0:
            raise ValueError("aurora_steps must be >= 0")
        if self.aurora_order not in (1, 2):
            raise ValueError("aurora_order must be 1 or 2")
        if not (0.0 <= self.aurora_theta1 <= 1.0):
            raise ValueError("aurora_theta1 must be in [0, 1]")
        if not (0.0 <= self.aurora_theta2 <= 1.0):
            raise ValueError("aurora_theta2 must be in [0, 1]")
        if self.aurora_resid_eps <= 0:
            raise ValueError("aurora_resid_eps must be > 0")
        if not (0.0 <= self.aurora_level2_keep <= 1.0):
            raise ValueError("aurora_level2_keep must be in [0, 1]")
        if not (0.0 <= self.aurora_level1_keep <= 1.0):
            raise ValueError("aurora_level1_keep must be in [0, 1]")
        if self.aurora_min_steps < 0:
            raise ValueError("aurora_min_steps must be >= 0")
        if self.aurora_early_stop < 0:
            raise ValueError("aurora_early_stop must be >= 0")
        if self.aurora_early_stop_patience <= 0:
            raise ValueError("aurora_early_stop_patience must be > 0")
        if self.aurora_early_stop_check_every <= 0:
            raise ValueError("aurora_early_stop_check_every must be > 0")
        if self.aurora_core_solver not in ("svd", "eigh"):
            raise ValueError("aurora_core_solver must be svd|eigh")


    @contextmanager
    def _stream_context(self):
        if self.stream_manager is None:
            yield
        else:
            with self.stream_manager.stream_context():
                yield

    # -------------------------------------------------------------------------
    # Public APIs
    # -------------------------------------------------------------------------

    def svd(self, tensor: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute truncated SVD (U, S, Vh) with a bounded rank.

        Note:
          - svd_mode=adaptive is an energy-targeted algorithm; for fixed-rank calls it
            dispatches to the configured low-rank backend (auto_lowrank).
          - Use `svd_energy(...)` to run the Adaptive range finder (Alg. 4.2).
        """
        if tensor is None:
            raise ValueError("tensor is None")
        if tensor.dim() != 2:
            raise ValueError(f"SVD expects a 2D tensor, got dim={tensor.dim()}")

        m, n = tensor.shape
        min_dim = min(m, n)
        r = int(rank)
        if r <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")
        r = min(r, min_dim)

        # Estimate size in fp32 (we cast for numerical stability)
        tensor_bytes = int(m) * int(n) * 4

        if self.force_cpu:
            return self._cpu_svd(tensor, r)

        mode = self.svd_mode

        # Heuristics for auto mode: decide full vs low-rank.
        if mode == "auto":
            if tensor_bytes > self.chunk_threshold_bytes:
                mode = self.auto_lowrank
            elif min_dim >= self.auto_min_dim and r < int(min_dim * self.auto_full_rank_ratio):
                mode = self.auto_lowrank
            else:
                mode = "full"

        # svd_mode=adaptive is only meaningful for `svd_energy`; for fixed-rank it becomes low-rank.
        if mode == "adaptive":
            mode = self.auto_lowrank

        if mode in ("randomized", "krylov", "drf", "aurora"):
            return self._lowrank_svd(
                tensor,
                r,
                algo=mode,
                n_oversamples=self.randomized_oversamples,
                n_iter=self.randomized_iter,
            )

        # Full SVD path
        try:
            return self._gpu_svd(tensor, r)
        except RuntimeError as e:
            if self.verbose:
                print(f"[SVD] GPU full SVD failed ({type(e).__name__}: {e}); falling back to low-rank SVD.")
            return self._lowrank_svd(
                tensor,
                r,
                algo=self.auto_lowrank,
                n_oversamples=self.randomized_oversamples,
                n_iter=self.randomized_iter,
            )

    def randomized_svd(
        self,
        tensor: torch.Tensor,
        rank: int,
        n_oversamples: Optional[int] = None,
        n_iter: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Explicit low-rank SVD (randomized/Krylov) with per-call parameters.

        Dispatch:
          - If svd_mode is krylov, uses Block Krylov.
          - If svd_mode is auto and auto_lowrank is krylov, uses Block Krylov.
          - Otherwise uses randomized subspace iteration (Alg. 4.4).
        """
        algo = "randomized"
        if self.svd_mode == "krylov":
            algo = "krylov"
        elif self.svd_mode == "drf":
            algo = "drf"
        elif self.svd_mode == "aurora":
            algo = "aurora"
        elif self.svd_mode == "auto":
            if self.auto_lowrank == "krylov":
                algo = "krylov"
            elif self.auto_lowrank == "drf":
                algo = "drf"
            elif self.auto_lowrank == "aurora":
                algo = "aurora"

        return self._lowrank_svd(
            tensor,
            int(rank),
            algo=algo,
            n_oversamples=self.randomized_oversamples if n_oversamples is None else int(n_oversamples),
            n_iter=self.randomized_iter if n_iter is None else int(n_iter),
        )

    def svd_energy(
        self,
        tensor: torch.Tensor,
        energy_threshold: float,
        min_rank: int,
        max_rank: int,
        block_size: Optional[int] = None,
        n_test: Optional[int] = None,
        n_iter: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Energy-targeted low-rank SVD using Adaptive range finder (Halko et al., Alg. 4.2).

        Parameters
        ----------
        energy_threshold:
            Target retained Frobenius energy fraction in (0,1]. The adaptive range finder
            controls the residual Frobenius energy via eps = sqrt(1 - energy_threshold).
        min_rank / max_rank:
            Hard rank bounds.
        block_size:
            Number of basis vectors added per adaptive iteration.
        n_test:
            Number of random test vectors for residual estimation.
        n_iter:
            Optional subspace-iteration steps applied to each new block before QR.
        """
        if tensor is None:
            raise ValueError("tensor is None")
        if tensor.dim() != 2:
            raise ValueError(f"SVD expects a 2D tensor, got dim={tensor.dim()}")

        with self._stream_context():
            x = tensor.to(self.device, non_blocking=True).float()
            m, n = x.shape
            min_dim = min(m, n)

            max_r = min(int(max_rank), min_dim)
            min_r = max(1, min(int(min_rank), max_r))

            tau = float(energy_threshold)
            tau = float(max(0.0, min(1.0, tau)))

            # Residual Frobenius tolerance: ||(I-QQ^T)A||_F <= eps ||A||_F
            eps = math.sqrt(max(0.0, 1.0 - tau))

            bs = self.adaptive_block_size if block_size is None else int(block_size)
            nt = self.adaptive_n_test if n_test is None else int(n_test)
            q = 0 if n_iter is None else int(n_iter)

            Q = self._adaptive_range_finder(
                x,
                eps=eps,
                max_rank=max_r,
                block_size=bs,
                n_test=nt,
                n_iter=q,
            )

            # Ensure at least min_rank basis vectors.
            if Q.shape[1] < min_r:
                add = min_r - Q.shape[1]
                omega = torch.randn(n, add, device=self.device, dtype=torch.float32)
                Y = x @ omega
                if Q.shape[1] > 0:
                    Y = Y - Q @ (Q.transpose(0, 1) @ Y)
                Qi, _ = torch.linalg.qr(Y, mode="reduced")
                Q = torch.cat([Q, Qi], dim=1)

            # Cap and re-orthonormalize.
            if Q.shape[1] > max_r:
                Q = Q[:, :max_r]
            Q, _ = torch.linalg.qr(Q, mode="reduced")

            B = Q.transpose(0, 1) @ x
            Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)
            U = Q @ Ub
            return U, S, Vh

    # -------------------------------------------------------------------------
    # Full SVD backends
    # -------------------------------------------------------------------------

    def _gpu_svd(self, x: torch.Tensor, r: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full SVD on GPU, then truncate to r."""
        with self._stream_context():
            x = x.to(self.device, non_blocking=True).float()
            U, S, Vh = torch.linalg.svd(x, full_matrices=False)
            return U[:, :r], S[:r], Vh[:r, :]

    def _cpu_svd(self, x: torch.Tensor, r: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full SVD on CPU, then truncate to r."""
        x = x.cpu().float()
        U, S, Vh = torch.linalg.svd(x, full_matrices=False)
        return U[:, :r].to(self.device), S[:r].to(self.device), Vh[:r, :].to(self.device)

    # -------------------------------------------------------------------------
    # Low-rank SVD backends
    # -------------------------------------------------------------------------

    def _lowrank_svd(
        self,
        x: torch.Tensor,
        r: int,
        algo: str,
        n_oversamples: int = 8,
        n_iter: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if algo == "krylov":
            return self._krylov_svd(x, r, n_oversamples=n_oversamples, n_iter=n_iter)
        if algo == "drf":
            return self._drf_svd(x, r, n_oversamples=n_oversamples, n_iter=n_iter)
        if algo == "aurora":
            return self._aurora_svd(x, r, n_oversamples=n_oversamples, n_iter=n_iter)
        return self._randomized_svd(x, r, n_oversamples=n_oversamples, n_iter=n_iter)

    def _randomized_svd(
        self,
        x: torch.Tensor,
        r: int,
        n_oversamples: int = 8,
        n_iter: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomized SVD using numerically-stable subspace iteration (Halko et al., Algorithm 4.4).

        This alternates applications of A and A^T with intermediate QR orthonormalization.
        """
        with self._stream_context():
            x = x.to(self.device, non_blocking=True).float()
            m, n = x.shape

            r = min(int(r), min(m, n))
            p = max(0, int(n_oversamples))
            k = min(min(m, n), r + p)
            if k <= 0:
                raise ValueError("randomized_svd: computed k <= 0")

            omega = torch.randn(n, k, device=self.device, dtype=torch.float32)
            Y = x @ omega
            Q, _ = torch.linalg.qr(Y, mode="reduced")

            q = max(0, int(n_iter))
            for _ in range(q):
                Y_tilde = x.transpose(0, 1) @ Q
                Q_tilde, _ = torch.linalg.qr(Y_tilde, mode="reduced")
                Y = x @ Q_tilde
                Q, _ = torch.linalg.qr(Y, mode="reduced")

            B = Q.transpose(0, 1) @ x
            Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)
            U = Q @ Ub
            return U[:, :r], S[:r], Vh[:r, :]

    def _krylov_svd(
        self,
        x: torch.Tensor,
        r: int,
        n_oversamples: int = 8,
        n_iter: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Block Krylov subspace SVD.

        Builds the Krylov subspace span{AΩ, (AA^T)AΩ, ..., (AA^T)^q AΩ} without forming AA^T,
        using alternating multiplies by A and A^T. Intermediate orthonormal blocks are stored
        and concatenated before solving the projected SVD.
        """
        with self._stream_context():
            x = x.to(self.device, non_blocking=True).float()
            m, n = x.shape

            r = min(int(r), min(m, n))
            p = max(0, int(n_oversamples))
            k = min(min(m, n), r + p)
            if k <= 0:
                raise ValueError("krylov_svd: computed k <= 0")

            omega = torch.randn(n, k, device=self.device, dtype=torch.float32)
            Y = x @ omega
            Q, _ = torch.linalg.qr(Y, mode="reduced")

            blocks = [Q]
            q = max(0, int(n_iter))
            for _ in range(q):
                # One stable subspace iteration step, but keep each block.
                Y_tilde = x.transpose(0, 1) @ Q
                Q_tilde, _ = torch.linalg.qr(Y_tilde, mode="reduced")
                Y = x @ Q_tilde
                Q, _ = torch.linalg.qr(Y, mode="reduced")
                blocks.append(Q)

            Qbar = torch.cat(blocks, dim=1)
            Qbar, _ = torch.linalg.qr(Qbar, mode="reduced")

            B = Qbar.transpose(0, 1) @ x
            Ub, S, Vh = torch.linalg.svd(B, full_matrices=False)
            U = Qbar @ Ub
            return U[:, :r], S[:r], Vh[:r, :]


    def _drf_svd(
        self,
        x: torch.Tensor,
        r: int,
        n_oversamples: int = 8,
        n_iter: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        DRF-SVD (Dual-Residual Folding SVD) — a residual-folding refinement on top of a
        single randomized range capture.
    
        Motivation (distillation-centric):
          For distillation deltas, the spectrum is often moderately ill-conditioned and
          the LoRA rank budget is tight. Standard randomized SVD accuracy can be limited
          by incomplete capture of directions that are *correlated* with the current
          approximate singular space but live in its orthogonal complement. DRF-SVD
          explicitly augments both left and right subspaces with *scaled residual*
          directions from the current approximation and then performs a Rayleigh–Ritz
          compression back to the target rank.
    
        Algorithm sketch (one refinement step):
          1) Compute an initial k0=(r+p) low-rank SVD approximation via randomized SVD.
          2) Form residual blocks:
                R_L = A V - U diag(S)
                R_R = A^T U - V diag(S)
             then scale them by 1/(S^theta) to stabilize small/large singular components.
          3) Fold residual blocks into the subspaces:
                Q = orth([U, R_L]),  P = orth([V, R_R])
          4) Solve the projected SVD of M = Q^T A P and truncate back to k0.
          5) Repeat for `drf_steps` rounds (default 1), then return the top-r slice.
    
        Notes:
          - This is *not* power iteration: residual folding adds directions proportional to
            the current first-order optimality violations for each singular triplet.
          - Numerical stability: QR is used for orthonormalization; residual scaling uses
            an epsilon floor to avoid division by 0.
        """
        with self._stream_context():
            A = x.to(self.device, non_blocking=True).float()
            m, n = A.shape
            min_dim = min(m, n)
            r = min(int(r), min_dim)
            p = max(0, int(n_oversamples))
    
            # Internal working rank: keep (r+p) components for refinement, then truncate to r.
            k0 = min(min_dim, r + p)
            if k0 <= 0:
                raise ValueError("drf_svd: computed k0 <= 0")
    
            # Initial capture (use oversamples=0 here to make k0 the true working rank).
            U, S, Vh = self._randomized_svd(A, k0, n_oversamples=0, n_iter=max(0, int(n_iter)))
            # Ensure contiguous
            U = U.contiguous()
            S = S.contiguous()
            Vh = Vh.contiguous()
    
            steps = max(0, int(getattr(self, "drf_steps", 1)))
            theta = float(getattr(self, "drf_theta", 0.5))
            theta = float(max(0.0, min(1.0, theta)))
            eps = float(getattr(self, "drf_resid_eps", 1e-8))
    
            for _ in range(steps):
                # Current right singular basis
                V = Vh.transpose(0, 1).contiguous()  # (n, k0)
    
                # Residual blocks (m,k0) and (n,k0)
                AV = A @ V
                ATU = A.transpose(0, 1) @ U
    
                # R_L = A V - U diag(S);   R_R = A^T U - V diag(S)
                US = U * S.unsqueeze(0)
                VS = V * S.unsqueeze(0)
                R_L = AV - US
                R_R = ATU - VS
    
                # Scale residuals by 1/(S^theta) to reduce dominance of the largest modes.
                if theta > 0.0:
                    denom = torch.pow(torch.clamp(S, min=eps), theta).unsqueeze(0)
                    R_L = R_L / denom
                    R_R = R_R / denom
    
                # Fold residuals into subspaces and re-orthonormalize.
                Q = torch.cat([U, R_L], dim=1)
                P = torch.cat([V, R_R], dim=1)
    
                Q, _ = torch.linalg.qr(Q, mode="reduced")
                P, _ = torch.linalg.qr(P, mode="reduced")
    
                # Project and solve small SVD
                # M = Q^T A P, computed as (Q^T (A P)) for efficiency.
                AP = A @ P
                M = Q.transpose(0, 1) @ AP
    
                Uc, Sc, Vhc = torch.linalg.svd(M, full_matrices=False)
    
                # Keep k0 components for potential further refinement.
                kk = min(int(Sc.numel()), int(k0))
                if kk <= 0:
                    break
    
                U = (Q @ Uc[:, :kk]).contiguous()
                S = Sc[:kk].contiguous()
                Vh = (Vhc[:kk, :] @ P.transpose(0, 1)).contiguous()
    
                # Defensive re-orthonormalization drift guard (cheap; kk is small).
                U, _ = torch.linalg.qr(U, mode="reduced")
                Vt = Vh.transpose(0, 1).contiguous()
                Vt, _ = torch.linalg.qr(Vt, mode="reduced")
                Vh = Vt.transpose(0, 1).contiguous()
    
            # Return top-r
            U = U[:, :r]
            S = S[:r]
            Vh = Vh[:r, :]
            return U, S, Vh
    
    def _aurora_svd(
        self,
        x: torch.Tensor,
        r: int,
        n_oversamples: int = 8,
        n_iter: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        AURORA-SVD (Adaptive Unrolled Residual-Order Range Augmentation SVD).

        Distillation-centric goal:
          Improve low-rank approximation quality under a tight LoRA rank budget by
          augmenting the approximation subspace with *structured residual orders*,
          then performing a Rayleigh–Ritz compression back to the working rank.

        Core idea (per refinement round):
          - Start from a working-rank approximation A ≈ U diag(S) V^T.
          - Form first-order residual blocks:
                R_L = A V − U diag(S)
                R_R = A^T U − V diag(S)
          - Optionally form a second-order residual response (order=2):
                T_L = A (scaled R_R),   T_R = A^T (scaled R_L)
          - Build enlarged subspaces Q, P and compute the best rank-(r+p) approximation
            inside span(Q) × span(P) by SVD of the projected core Q^T A P.

        Implementation notes (v4.2.0 speed path):
          - Uses *block-anchored orthonormalization*: keep current U/V blocks intact and
            orthonormalize residual blocks against them, avoiding full QR on concatenated
            bases while preserving the augmented subspaces (up to floating-point effects).
          - Avoids redundant re-orthonormalization of U/V: U and V remain orthonormal by
            construction (QR on residual blocks + orthogonalization against anchors; and
            SVD of the projected core).
          - Reuses A@V when constructing A@P (since V is an explicit leading block of P).
        """
        with self._stream_context():
            A = x.to(self.device, non_blocking=True).float()
            m, n = A.shape
            min_dim = min(m, n)

            r = min(int(r), min_dim)
            p = max(0, int(n_oversamples))
            k0 = min(min_dim, r + p)
            if k0 <= 0:
                raise ValueError("aurora_svd: computed k0 <= 0")

            # Initial capture (use oversamples=0 so k0 is the true working rank).
            U, S, Vh = self._randomized_svd(A, k0, n_oversamples=0, n_iter=max(0, int(n_iter)))
            U = U.contiguous()
            S = S.contiguous()
            Vh = Vh.contiguous()

            steps = max(0, int(getattr(self, "aurora_steps", 1)))
            order = int(getattr(self, "aurora_order", 2))
            order = 1 if order <= 1 else 2

            theta1 = float(getattr(self, "aurora_theta1", 0.5))
            theta2 = float(getattr(self, "aurora_theta2", 0.25))
            theta1 = float(max(0.0, min(1.0, theta1)))
            theta2 = float(max(0.0, min(1.0, theta2)))

            eps = float(getattr(self, "aurora_resid_eps", 1e-8))
            eps = float(max(1e-30, eps))

            level2_keep = float(getattr(self, "aurora_level2_keep", 1.0))
            level2_keep = float(max(0.0, min(1.0, level2_keep)))

            level1_keep = float(getattr(self, "aurora_level1_keep", 1.0))
            level1_keep = float(max(0.0, min(1.0, level1_keep)))

            min_steps = max(0, int(getattr(self, "aurora_min_steps", 1)))
            min_steps = min(min_steps, steps)

            early_stop_tol = float(getattr(self, "aurora_early_stop", 0.0))
            early_stop_tol = float(max(0.0, early_stop_tol))
            early_stop_patience = max(1, int(getattr(self, "aurora_early_stop_patience", 2)))
            early_stop_check_every = max(1, int(getattr(self, "aurora_early_stop_check_every", 1)))

            core_solver = str(getattr(self, "aurora_core_solver", "svd") or "svd").lower()
            if core_solver not in ("svd", "eigh"):
                core_solver = "svd"
            svd_driver = str(getattr(self, "aurora_svd_driver", "") or "")

            use_amp = bool(self.use_mixed_precision) and (self.device.type == "cuda")
            if use_amp:
                # Prefer bf16 when supported; otherwise fp16.
                amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                amp_dtype = None  # type: ignore

            def _mm(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
                """Matrix multiply with an optional AMP speed path; returns fp32 tensor."""
                if use_amp:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        return (X @ Y).float()
                return X @ Y

            def _orth_block(block: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
                """Orthonormalize `block` against an existing orthonormal `basis`."""
                if block.numel() == 0:
                    return block
                if basis.numel() != 0 and basis.shape[1] > 0:
                    block = block - basis @ (basis.transpose(0, 1) @ block)
                # If the block is (near) zero after projection, QR will produce garbage; handle safely.
                if torch.linalg.norm(block, ord="fro") < 1e-20:
                    return block[:, :0]
                Qb, _ = torch.linalg.qr(block, mode="reduced")
                return Qb

            def _core_svd(M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                """Solve the small projected SVD for the Rayleigh–Ritz step."""
                if core_solver == "eigh":
                    a, b = M.shape
                    if a <= b:
                        C = M @ M.transpose(0, 1)
                        evals, Uc = torch.linalg.eigh(C)
                        evals = torch.flip(evals, dims=[0])
                        Uc = torch.flip(Uc, dims=[1])
                        Sc = torch.sqrt(torch.clamp(evals, min=0.0))
                        inv = torch.where(Sc > 1e-12, 1.0 / Sc, torch.zeros_like(Sc))
                        V = (M.transpose(0, 1) @ Uc) * inv.unsqueeze(0)
                        Vhc = V.transpose(0, 1)
                        return Uc, Sc, Vhc
                    else:
                        C = M.transpose(0, 1) @ M
                        evals, V = torch.linalg.eigh(C)
                        evals = torch.flip(evals, dims=[0])
                        V = torch.flip(V, dims=[1])
                        Sc = torch.sqrt(torch.clamp(evals, min=0.0))
                        inv = torch.where(Sc > 1e-12, 1.0 / Sc, torch.zeros_like(Sc))
                        Uc = (M @ V) * inv.unsqueeze(0)
                        Vhc = V.transpose(0, 1)
                        return Uc, Sc, Vhc

                # Default: torch.linalg.svd, with an optional CUDA driver hint.
                try:
                    if (self.device.type == "cuda") and svd_driver:
                        return torch.linalg.svd(M, full_matrices=False, driver=svd_driver)
                    return torch.linalg.svd(M, full_matrices=False)
                except TypeError:
                    return torch.linalg.svd(M, full_matrices=False)

            prev_res2: Optional[torch.Tensor] = None
            stall = 0
            steps_used = 0

            for it in range(steps):
                V = Vh.transpose(0, 1).contiguous()  # (n, k0)

                # First-order residuals
                AV = _mm(A, V)
                ATU = _mm(A.transpose(0, 1), U)

                US = U * S.unsqueeze(0)
                VS = V * S.unsqueeze(0)

                R_L = AV - US  # (m, k0)
                R_R = ATU - VS # (n, k0)

                # Early-stop: check improvement in first-order optimality residual.
                # We evaluate at the *start* of the refinement round (before updating U/S/V again),
                # so stopping returns the current approximation.
                if early_stop_tol > 0.0:
                    res2 = (R_L.float() * R_L.float()).sum() + (R_R.float() * R_R.float()).sum()
                    if prev_res2 is not None and (it >= min_steps) and ((it % early_stop_check_every) == 0):
                        rel_impr = ((prev_res2 - res2) / (prev_res2 + eps)).clamp(min=-1.0, max=1.0)
                        if float(rel_impr.item()) < early_stop_tol:
                            stall += 1
                        else:
                            stall = 0
                        if stall >= early_stop_patience:
                            break
                    prev_res2 = res2

                # Residual scaling for stability / tail emphasis
                if theta1 > 0.0:
                    denom1 = torch.pow(torch.clamp(S, min=eps), theta1).unsqueeze(0)
                    R_Ls = R_L / denom1
                    R_Rs = R_R / denom1
                else:
                    R_Ls = R_L
                    R_Rs = R_R

                # Build orthonormal bases with anchored blocks (U / V).
                Q_blocks: List[torch.Tensor] = [U]
                P_blocks: List[torch.Tensor] = [V]

                # Optional level-1 pruning: keep only the most residual-dominant columns.
                # This reduces QR/core sizes without changing the anchored U/V blocks.
                R_Ls1 = R_Ls
                R_Rs1 = R_Rs
                if level1_keep <= 0.0:
                    R_Ls1 = R_Ls[:, :0]
                    R_Rs1 = R_Rs[:, :0]
                elif level1_keep < 1.0 and k0 > 1:
                    eL1 = torch.sum(R_L.float() * R_L.float(), dim=0)
                    eR1 = torch.sum(R_R.float() * R_R.float(), dim=0)
                    e1 = (eL1 + eR1).contiguous()
                    keep1 = max(1, int(round(level1_keep * k0)))
                    keep1 = min(keep1, k0)
                    _, topi1 = torch.topk(e1, k=keep1, largest=True, sorted=False)
                    R_Ls1 = R_Ls[:, topi1]
                    R_Rs1 = R_Rs[:, topi1]

                Q1 = _orth_block(R_Ls1, U)
                if Q1.numel() != 0 and Q1.shape[1] > 0:
                    Q_blocks.append(Q1)

                P1 = _orth_block(R_Rs1, V)
                if P1.numel() != 0 and P1.shape[1] > 0:
                    P_blocks.append(P1)

                if order >= 2:
                    # Select a subset of components for the level-2 term (optional).
                    if level2_keep <= 0.0:
                        idx = None
                    elif level2_keep >= 1.0 or k0 <= 1:
                        idx = None
                    else:
                        # Residual energy per component: ||R_L[:,i]||^2 + ||R_R[:,i]||^2
                        eL = torch.sum(R_L.float() * R_L.float(), dim=0)
                        eR = torch.sum(R_R.float() * R_R.float(), dim=0)
                        e = (eL + eR).contiguous()
                        keep = max(1, int(round(level2_keep * k0)))
                        keep = min(keep, k0)
                        _, topi = torch.topk(e, k=keep, largest=True, sorted=False)
                        idx = topi

                    if idx is None:
                        R_L2 = R_Ls
                        R_R2 = R_Rs
                        S2 = S
                    else:
                        R_L2 = R_Ls[:, idx]
                        R_R2 = R_Rs[:, idx]
                        S2 = S[idx]

                    # Second-order residual response (apply A / A^T)
                    T_L = _mm(A, R_R2)
                    T_R = _mm(A.transpose(0, 1), R_L2)

                    # Orthogonalize level-2 blocks against current bases.
                    Q_basis = torch.cat(Q_blocks, dim=1)
                    P_basis = torch.cat(P_blocks, dim=1)
                    T_L = T_L - Q_basis @ (Q_basis.transpose(0, 1) @ T_L)
                    T_R = T_R - P_basis @ (P_basis.transpose(0, 1) @ T_R)

                    # Optional additional scaling for the level-2 response.
                    if theta2 > 0.0:
                        denom2 = torch.pow(torch.clamp(S2, min=eps), theta2).unsqueeze(0)
                        T_L = T_L / denom2
                        T_R = T_R / denom2

                    Q2 = _orth_block(T_L, Q_basis)
                    if Q2.numel() != 0 and Q2.shape[1] > 0:
                        Q_blocks.append(Q2)

                    P2 = _orth_block(T_R, P_basis)
                    if P2.numel() != 0 and P2.shape[1] > 0:
                        P_blocks.append(P2)

                # Final orthonormal bases (already orthonormal by construction)
                Q = torch.cat(Q_blocks, dim=1)
                P = torch.cat(P_blocks, dim=1)

                # Compute A@P with reuse of A@V (V is the explicit leading block of P).
                if len(P_blocks) == 1:
                    AP = AV
                else:
                    P_rest = torch.cat(P_blocks[1:], dim=1)
                    AP_rest = _mm(A, P_rest)
                    AP = torch.cat([AV, AP_rest], dim=1)

                # Project and solve small SVD (Rayleigh–Ritz)
                M = Q.transpose(0, 1) @ AP
                Uc, Sc, Vhc = _core_svd(M)

                kk = min(int(Sc.numel()), int(k0))
                if kk <= 0:
                    break

                U = (Q @ Uc[:, :kk]).contiguous()
                S = Sc[:kk].contiguous()
                Vh = (Vhc[:kk, :] @ P.transpose(0, 1)).contiguous()

                steps_used += 1

            # AURORA telemetry (helps tuning speed/quality without affecting results)
            self._aurora_calls += 1
            self._aurora_steps_used_total += int(steps_used)

            U = U[:, :r]
            S = S[:r]
            Vh = Vh[:r, :]
            return U, S, Vh
    # -------------------------------------------------------------------------
    # Adaptive range finder (Alg. 4.2)
    # -------------------------------------------------------------------------

    def _adaptive_range_finder(
        self,
        x: torch.Tensor,
        eps: float,
        max_rank: int,
        block_size: int,
        n_test: int,
        n_iter: int = 0,
    ) -> torch.Tensor:
        """
        Adaptive randomized range finder (Halko et al., Alg. 4.2).

        Maintains an estimate of the residual via random test vectors:
          H = (I - QQ^T) A G
        and stops when ||H||_F <= eps ||A||_F or max_rank is reached.
        """
        m, n = x.shape
        max_r = min(int(max_rank), min(m, n))
        if max_r <= 0:
            return torch.zeros((m, 0), device=self.device, dtype=torch.float32)

        bs = max(1, int(block_size))
        nt = max(1, int(n_test))
        q = max(0, int(n_iter))
        eps = float(max(0.0, eps))

        normA = torch.linalg.norm(x, ord="fro")
        if normA < 1e-20:
            return torch.zeros((m, 0), device=self.device, dtype=torch.float32)

        G = torch.randn(n, nt, device=self.device, dtype=torch.float32)
        H = x @ G

        Q: Optional[torch.Tensor] = None
        k = 0

        # Current residual estimate
        res = torch.linalg.norm(H, ord="fro")
        tol = eps * normA

        while (res > tol) and (k < max_r):
            b = min(bs, max_r - k)

            omega = torch.randn(n, b, device=self.device, dtype=torch.float32)
            Y = x @ omega  # (m, b)

            # Optional stabilization / spectrum-boost for each added block.
            if q > 0:
                # Stable subspace iteration restricted to the block.
                Qi, _ = torch.linalg.qr(Y, mode="reduced")
                for _ in range(q):
                    Z = x.transpose(0, 1) @ Qi
                    Qz, _ = torch.linalg.qr(Z, mode="reduced")
                    Y = x @ Qz
                    Qi, _ = torch.linalg.qr(Y, mode="reduced")
                Y = Qi

            # Orthogonalize against current basis.
            if Q is not None and Q.shape[1] > 0:
                Y = Y - Q @ (Q.transpose(0, 1) @ Y)

            Qi, _ = torch.linalg.qr(Y, mode="reduced")

            # Double-orthogonalize for numerical safety.
            if Q is not None and Q.shape[1] > 0:
                Qi = Qi - Q @ (Q.transpose(0, 1) @ Qi)
                Qi, _ = torch.linalg.qr(Qi, mode="reduced")

            if Qi.numel() == 0 or Qi.shape[1] == 0:
                break

            Q = Qi if Q is None else torch.cat([Q, Qi], dim=1)
            k = Q.shape[1]

            # Update residual sketch: H <- (I - QiQi^T) H
            H = H - Qi @ (Qi.transpose(0, 1) @ H)
            res = torch.linalg.norm(H, ord="fro")

        if Q is None:
            return torch.zeros((m, 0), device=self.device, dtype=torch.float32)

        # Final re-orthonormalization (cheap for small k, avoids drift)
        Q, _ = torch.linalg.qr(Q, mode="reduced")
        if Q.shape[1] > max_r:
            Q = Q[:, :max_r]
        return Q



# =============================================================================
#                           HELPER FUNCTIONS
# =============================================================================

NORM_TOKENS = ["layernorm", "rmsnorm", "layer_norm", "rms_norm", "ln_", "norm"]
EMBED_TOKENS = ["embed", "wte", "wpe", "lm_head", "output.weight"]


def seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_index_map(model_path: str) -> Dict[str, str]:
    """Read weight index map from model directory."""
    idx_files = [
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ]
    
    for idx_file in idx_files:
        path = os.path.join(model_path, idx_file)
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                return data.get("weight_map", {})
    
    # Single file models
    for ext in [".safetensors", ".bin"]:
        for name in ["model", "pytorch_model"]:
            path = os.path.join(model_path, f"{name}{ext}")
            if os.path.exists(path):
                if ext == ".safetensors":
                    with safe_open(path, framework="pt") as f:
                        return {k: f"{name}{ext}" for k in f.keys()}
    
    return {}


def scan_layers(keys: List[str]) -> Dict[str, List[int]]:
    """Scan model keys to find layer structure."""
    layers = defaultdict(set)
    
    for key in keys:
        m = re.search(r'(model\.layers|transformer\.h|decoder\.layers)\.(\d+)\.', key)
        if m:
            prefix = m.group(1)
            idx = int(m.group(2))
            layers[prefix].add(idx)
    
    return {k: sorted(list(v)) for k, v in layers.items()}


def split_key(key: str) -> Optional[Tuple[str, str, int, str]]:
    """Split a model key into components."""
    patterns = [
        r'^(model\.layers)\.(\d+)\.(.+)$',
        r'^(transformer\.h)\.(\d+)\.(.+)$',
        r'^(decoder\.layers)\.(\d+)\.(.+)$',
    ]
    
    for pattern in patterns:
        m = re.match(pattern, key)
        if m:
            prefix = m.group(1)
            idx = int(m.group(2))
            rest = m.group(3)
            token_name = rest.split('.')[0]
            return prefix, token_name, idx, rest
    
    return None


# =============================================================================
#                      ARCHITECTURE DETECTION
# =============================================================================

def detect_architecture_from_config(model_path: str) -> ArchitectureInfo:
    """Detect architecture from transformers config."""
    info = ArchitectureInfo()
    
    if not HAS_TRANSFORMERS:
        return info
    
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        info.hidden_size = getattr(config, "hidden_size", 4096)
        info.intermediate_size = getattr(config, "intermediate_size", 11008)
        info.num_hidden_layers = getattr(config, "num_hidden_layers", 32)
        info.num_attention_heads = getattr(config, "num_attention_heads", 32)
        info.vocab_size = getattr(config, "vocab_size", 32000)
        
        num_kv = getattr(config, "num_key_value_heads", None)
        if num_kv is None:
            num_kv = info.num_attention_heads
        info.num_kv_heads = num_kv
        
        if num_kv == info.num_attention_heads:
            info.attention_type = AttentionType.MHA
        elif num_kv == 1:
            info.attention_type = AttentionType.MQA
        else:
            info.attention_type = AttentionType.GQA
        
        info.num_experts = getattr(config, "num_local_experts", 0)
        if info.num_experts == 0:
            info.num_experts = getattr(config, "num_experts", 0)
        info.num_experts_per_tok = getattr(config, "num_experts_per_tok", 2)
        
        if info.num_experts > 0:
            info.mlp_type = MLPType.MOE
        elif hasattr(config, "hidden_act"):
            act = config.hidden_act.lower() if isinstance(config.hidden_act, str) else ""
            if "silu" in act or "swish" in act:
                info.mlp_type = MLPType.SWIGLU
            elif "gelu" in act:
                info.mlp_type = MLPType.GEGLU
    
    except Exception as e:
        warnings.warn(f"Config detection failed: {e}")
    
    return info


def detect_architecture_from_weights(keys: List[str]) -> ArchitectureInfo:
    """Detect architecture from weight keys."""
    info = ArchitectureInfo()
    
    key_str = " ".join(keys[:500])
    
    if ".experts." in key_str or "block_sparse_moe" in key_str:
        info.mlp_type = MLPType.MOE
        expert_indices = set()
        for k in keys:
            m = re.search(r'experts\.(\d+)', k)
            if m:
                expert_indices.add(int(m.group(1)))
        if expert_indices:
            info.num_experts = max(expert_indices) + 1
    
    if "c_attn" in key_str or "query_key_value" in key_str or "qkv_proj" in key_str:
        info.has_fused_qkv = True
    
    layer_indices = set()
    for k in keys:
        m = re.search(r'layers\.(\d+)', k) or re.search(r'\.h\.(\d+)', k)
        if m:
            layer_indices.add(int(m.group(1)))
    if layer_indices:
        info.num_hidden_layers = max(layer_indices) + 1
    
    return info


# =============================================================================
#                           DARE + TIES
# =============================================================================

def apply_dare(delta: torch.Tensor, drop_rate: float = 0.7, 
               rescale: bool = True) -> torch.Tensor:
    """DARE: Drop And REscale for delta parameters."""
    if delta.dim() != 2 or drop_rate <= 0:
        return delta
    
    mask = torch.rand_like(delta.float()) > drop_rate
    sparse_delta = delta * mask
    
    if rescale and drop_rate < 1.0:
        sparse_delta = sparse_delta / (1.0 - drop_rate)
    
    return sparse_delta


def apply_ties_single(delta: torch.Tensor, density: float = 0.3) -> torch.Tensor:
    """
    Apply TIES-style magnitude trimming to a single delta.
    
    This is the "Trim" step of TIES: keep only the top-k parameters by magnitude.
    Useful for sparsification even when there's only one delta.
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


def apply_ties(deltas: List[torch.Tensor], density: float = 0.3) -> torch.Tensor:
    """TIES-Merging: Trim, Elect Sign, Disjoint Merge."""
    if not deltas:
        return torch.zeros(1)

    if len(deltas) == 1:
        return deltas[0]

    device = deltas[0].device
    dtype = deltas[0].dtype
    shape = deltas[0].shape
    deltas = [d if d.shape == shape else torch.zeros(shape, device=device, dtype=dtype) 
              for d in deltas]

    stacked = torch.stack([d.float() for d in deltas], dim=0)

    k = max(1, int(stacked[0].numel() * density))
    abs_flat = stacked.abs().view(len(deltas), -1)

    trimmed = []
    for i in range(len(deltas)):
        threshold = torch.topk(abs_flat[i], k, largest=True).values[-1]
        mask = abs_flat[i] >= threshold
        trimmed_flat = stacked[i].view(-1) * mask.float()
        trimmed.append(trimmed_flat.view(shape))

    stacked_trimmed = torch.stack(trimmed, dim=0)

    signs = torch.sign(stacked_trimmed)
    sign_sum = signs.sum(dim=0)
    elected_sign = torch.sign(sign_sum)
    elected_sign = torch.where(elected_sign == 0, torch.ones_like(elected_sign), elected_sign)

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
    """Compute convex mixing weight w in [0,1] minimizing ||(1-w)Δ_floor + wΔ_ceil||_F.

    Weight-space least-squares mixing rule:

      argmin_w || Δ_floor + w(Δ_ceil - Δ_floor) ||_F^2

      w* = - <Δ_floor, Δ_ceil - Δ_floor> / ||Δ_ceil - Δ_floor||_F^2

    If the denominator is too small or inputs are invalid, returns fallback.
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

        if w < 0.0:
            w = 0.0
        elif w > 1.0:
            w = 1.0
        return float(w)
    except Exception:
        return float(fallback)


# =============================================================================
#                           ADAPTIVE RANK
# =============================================================================

def compute_adaptive_rank(
    singular_values: torch.Tensor,
    energy_threshold: float = 0.95,
    min_rank: int = 8,
    max_rank: int = 256,
    total_energy: Optional[Union[float, torch.Tensor]] = None,
) -> int:
    """
    Compute an adaptive rank based on retained Frobenius-energy.

    Parameters
    ----------
    singular_values:
        1D tensor of (estimated) singular values, typically sorted in descending order.
    energy_threshold:
        Target fraction of Frobenius energy to retain.
    min_rank / max_rank:
        Hard bounds on the returned rank.
    total_energy:
        If provided, this is interpreted as ||A||_F^2 for the *original* matrix A.
        This is the preferred mode when `singular_values` is truncated (e.g. you only
        computed the top-k singular values). If not provided, `total_energy` is
        computed as sum(s_i^2) over the supplied `singular_values` (legacy behavior).

    Notes
    -----
    For truncated spectra, using `total_energy=None` can systematically overestimate the
    retained-energy ratio (because missing tail energy is treated as 0). Passing
    `total_energy=||A||_F^2` avoids this.
    """
    s = singular_values.detach().float()
    if s.numel() == 0:
        return max(0, int(min_rank))

    # Guard against tiny negatives from numerical noise (SVD should be non-negative).
    s = torch.clamp(s, min=0.0)

    if total_energy is None:
        total = torch.sum(s ** 2)
    else:
        if torch.is_tensor(total_energy):
            total = total_energy.to(device=s.device, dtype=s.dtype)
        else:
            total = torch.tensor(float(total_energy), device=s.device, dtype=s.dtype)

    if total < 1e-20:
        # Degenerate matrix: keep at least min_rank (and at least 1 if possible).
        return max(1, int(min_rank))

    cumulative = torch.cumsum(s ** 2, dim=0)
    target = float(energy_threshold) * total

    hit = (cumulative >= target)
    if bool(hit.any()):
        r = int(hit.nonzero(as_tuple=True)[0][0].item() + 1)
    else:
        r = int(s.numel())

    r = max(int(min_rank), min(r, int(max_rank), int(s.numel())))
    return r

# =============================================================================
#                  IMPROVED DIMENSION PROJECTION (FIX #2)
# =============================================================================

def svd_projection(
    src: torch.Tensor,
    target_shape: Tuple[int, ...],
    svd_engine: "GPUAcceleratedSVD",
    config: "DistillConfig",
    device: torch.device,
    ref: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Project a tensor to `target_shape`.

    v3.7.6 fix (student-subspace alignment):
      The legacy (pool/crop/interpolate) projection loses too much information and can
      destabilize merging (gibberish / repetition), especially for cross-architecture
      distillation.

      If `ref` (student tensor) is provided and both `src/ref` are 2D matrices, we:
        1) Compute a truncated SVD of teacher: Wt ≈ Ut diag(St) Vt^T
        2) Compute a truncated SVD of student: Ws ≈ Us diag(Ss) Vs^T
        3) Build teacher low-rank factors Bt, At (PiSSA-style: sqrt split)
        4) Resize factors to the student's row/col sizes (Bt -> rows, At -> cols)
        5) Project the resized teacher into the *student* singular subspaces:
             core = Us^T (Bt_rs @ At_rs) Vs
             Wproj = Us @ core @ Vs^T

      This explicitly converts teacher information into the student's low-rank subspace
      (the "common" space) and avoids the destructive "pool/crop" behavior.

    Notes:
      - For non-2D tensors, falls back to safe interpolation/crop.
      - For same-shape matrices, returns `src` unchanged.
    """
    if tuple(src.shape) == tuple(target_shape):
        return src

    # Output dtype safety: avoid returning float8 tensors (float8 mixed arithmetic is fragile
    # across torch builds). Prefer the student reference dtype when available.
    ref_dtype = None
    try:
        if ref is not None and isinstance(ref, torch.Tensor):
            ref_dtype = ref.dtype
    except Exception:
        ref_dtype = None
    out_dtype = _safe_weight_dtype(src.dtype, config, device, ref_dtype=ref_dtype)

    # -------- 1D vectors: linear interpolation --------
    if src.dim() == 1 and len(target_shape) == 1:
        out_len = int(target_shape[0])
        x = torch.nan_to_num(src.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if x.numel() == out_len:
            return x.to(dtype=out_dtype, device=src.device)
        x3 = x.view(1, 1, -1)
        y3 = F.interpolate(x3, size=out_len, mode="linear", align_corners=False)
        return y3.view(out_len).to(dtype=out_dtype, device=src.device)

    # -------- 2D matrices: student-subspace alignment (preferred) --------
    if src.dim() == 2 and len(target_shape) == 2:
        tm, tn = int(target_shape[0]), int(target_shape[1])
        m, n = int(src.shape[0]), int(src.shape[1])

        # Bound the working rank
        k_cap = int(getattr(config, 'projection_rank', 256))
        k_max = min(m, n, tm, tn, k_cap)
        if k_max <= 0:
            return torch.zeros((tm, tn), device=src.device, dtype=out_dtype)

        # Resize helpers
        def _resize_rows(mat: torch.Tensor, new_rows: int) -> torch.Tensor:
            if mat.shape[0] == new_rows:
                return mat
            # (rows, k) -> (1, k, rows) -> interpolate -> (new_rows, k)
            x = mat.transpose(0, 1).unsqueeze(0)
            y = F.interpolate(x, size=new_rows, mode="linear", align_corners=False)
            return y.squeeze(0).transpose(0, 1)

        def _resize_cols(mat: torch.Tensor, new_cols: int) -> torch.Tensor:
            if mat.shape[1] == new_cols:
                return mat
            # (k, cols) -> (1, k, cols) -> interpolate -> (k, new_cols)
            x = mat.unsqueeze(0)
            y = F.interpolate(x, size=new_cols, mode="linear", align_corners=False)
            return y.squeeze(0)

        # Fast fallback: if no ref matrix is provided
        if ref is None or (not isinstance(ref, torch.Tensor)) or ref.dim() != 2:
            # Teacher-only SVD factor resize (legacy), then reconstruct
            src_f = torch.nan_to_num(src.to(device).float(), nan=0.0, posinf=0.0, neginf=0.0)
            try:
                Ut, St, Vht = svd_engine.randomized_svd(
                    src_f,
                    k_max,
                    n_oversamples=getattr(config, 'projection_randomized_oversamples', 8),
                    n_iter=getattr(config, 'projection_randomized_iter', 2),
                )
                if getattr(config, 'projection_adaptive_rank', True):
                    k_t = compute_adaptive_rank(
                        St,
                        energy_threshold=getattr(config, 'projection_energy_threshold', 0.99),
                        min_rank=getattr(config, 'projection_min_rank', 8),
                        max_rank=k_max,
                    )
                    k = max(1, min(k_t, k_max))
                    Ut, St, Vht = Ut[:, :k], St[:k], Vht[:k, :]
                else:
                    k = k_max

                sqrtS = torch.sqrt(torch.clamp(St, min=0.0)).to(dtype=Ut.dtype)
                Bt = Ut * sqrtS.unsqueeze(0)          # (m, k)
                At = sqrtS.unsqueeze(1) * Vht         # (k, n)

                Bt_rs = _resize_rows(Bt, tm)
                At_rs = _resize_cols(At, tn)

                out = Bt_rs @ At_rs
                return out.to(dtype=out_dtype, device=src.device)
            except Exception:
                # Last resort: bilinear interpolation of the dense matrix
                dense = torch.nan_to_num(src.to(device).float(), nan=0.0, posinf=0.0, neginf=0.0)
                out = F.interpolate(dense.unsqueeze(0).unsqueeze(0), size=(tm, tn), mode="bilinear", align_corners=False)
                return out.squeeze(0).squeeze(0).to(dtype=out_dtype, device=src.device)

        # Ensure ref has the expected target shape
        ref = ref.to(device)
        if tuple(ref.shape) != (tm, tn):
            # We do not attempt to project ref; fallback to teacher-only projection
            return svd_projection(src, (tm, tn), svd_engine, config, device, ref=None)

        # Compute truncated SVDs (teacher + student)
        src_f = torch.nan_to_num(src.to(device).float(), nan=0.0, posinf=0.0, neginf=0.0)
        ref_f = torch.nan_to_num(ref.to(device).float(), nan=0.0, posinf=0.0, neginf=0.0)

        try:
            Ut, St, Vht = svd_engine.randomized_svd(
                src_f,
                k_max,
                n_oversamples=getattr(config, 'projection_randomized_oversamples', 8),
                n_iter=getattr(config, 'projection_randomized_iter', 2),
            )
            Us, Ss, Vhs = svd_engine.randomized_svd(
                ref_f,
                k_max,
                n_oversamples=getattr(config, 'projection_randomized_oversamples', 8),
                n_iter=getattr(config, 'projection_randomized_iter', 2),
            )

            if getattr(config, 'projection_adaptive_rank', True):
                k_t = compute_adaptive_rank(
                    St,
                    energy_threshold=getattr(config, 'projection_energy_threshold', 0.99),
                    min_rank=getattr(config, 'projection_min_rank', 8),
                    max_rank=k_max,
                )
                k_s = compute_adaptive_rank(
                    Ss,
                    energy_threshold=getattr(config, 'projection_energy_threshold', 0.99),
                    min_rank=getattr(config, 'projection_min_rank', 8),
                    max_rank=k_max,
                )
                k = max(1, min(k_t, k_s, k_max))
            else:
                k = k_max

            Ut, St, Vht = Ut[:, :k], St[:k], Vht[:k, :]
            Us, Ss, Vhs = Us[:, :k], Ss[:k], Vhs[:k, :]

            # Teacher low-rank factors (PiSSA sqrt split)
            sqrtSt = torch.sqrt(torch.clamp(St, min=0.0)).to(dtype=Ut.dtype)
            Bt = Ut * sqrtSt.unsqueeze(0)            # (m, k)
            At = sqrtSt.unsqueeze(1) * Vht           # (k, n)

            # Resize teacher factors to student dims
            Bt_rs = _resize_rows(Bt, tm)             # (tm, k)
            At_rs = _resize_cols(At, tn)             # (k, tn)

            # Student right singular vectors
            Vs = Vhs.transpose(0, 1).contiguous()    # (tn, k)

            # core = Us^T (Bt_rs @ At_rs) Vs  (k×k)
            # Compute as (Us^T Bt_rs) @ (At_rs Vs) for efficiency.
            left = Us.transpose(0, 1) @ Bt_rs        # (k, k)
            right = At_rs @ Vs                       # (k, k)
            core = left @ right                      # (k, k)

            out = Us @ core @ Vs.transpose(0, 1)     # (tm, tn)
            return out.to(dtype=out_dtype, device=src.device)

        except Exception:
            dense = torch.nan_to_num(src.to(device).float(), nan=0.0, posinf=0.0, neginf=0.0)
            out = F.interpolate(dense.unsqueeze(0).unsqueeze(0), size=(tm, tn), mode="bilinear", align_corners=False)
            return out.squeeze(0).squeeze(0).to(dtype=out_dtype, device=src.device)

    # -------- Generic fallback for other ranks/shapes --------
    # Fallback: flatten + interpolate length-wise, then reshape/crop.
    tgt_numel = int(torch.tensor(target_shape).prod().item())
    x = torch.nan_to_num(src.float().view(1, 1, -1), nan=0.0, posinf=0.0, neginf=0.0)
    y = F.interpolate(x, size=tgt_numel, mode="linear", align_corners=False)
    out = y.view(*target_shape)
    return out.to(dtype=out_dtype, device=src.device)

# =============================================================================
#           CROSS-ARCH SUBSPACE-DELTA (FIX: preserve student residual space)
# =============================================================================

def _resize_2d_rows_linear(mat: torch.Tensor, target_rows: int) -> torch.Tensor:
    """Resize a 2D matrix along rows using 1D linear interpolation (preserves columns)."""
    if mat.dim() != 2:
        return mat
    if mat.shape[0] == target_rows:
        return mat
    if target_rows <= 0:
        return mat[:0]
    # Treat columns as channels, interpolate over rows
    x = mat.transpose(0, 1).unsqueeze(0)  # (1, cols, rows)
    y = F.interpolate(x, size=target_rows, mode="linear", align_corners=False)
    return y.squeeze(0).transpose(0, 1).contiguous()


def _resize_2d_cols_linear(mat: torch.Tensor, target_cols: int) -> torch.Tensor:
    """Resize a 2D matrix along cols using 1D linear interpolation (preserves rows)."""
    if mat.dim() != 2:
        return mat
    if mat.shape[1] == target_cols:
        return mat
    if target_cols <= 0:
        return mat[:, :0]
    # Treat rows as channels, interpolate over cols
    x = mat.unsqueeze(0)  # (1, rows, cols)
    y = F.interpolate(x, size=target_cols, mode="linear", align_corners=False)
    return y.squeeze(0).contiguous()


def compute_student_svd_basis(
    student_mat: torch.Tensor,
    svd_engine: "GPUAcceleratedSVD",
    config: "DistillConfig",
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the student's truncated SVD basis (U, S, V) used as the common subspace.
    Returns:
      U: (out, k), S: (k,), V: (in, k) where V = Vh^T
    """
    if student_mat.dim() != 2:
        raise ValueError("student_mat must be 2D")
    tm, tn = int(student_mat.shape[0]), int(student_mat.shape[1])
    k_cap = min(int(getattr(config, "projection_rank", 256)), tm, tn)
    if k_cap <= 0:
        raise ValueError("projection_rank too small for matrix")

    ref_f = torch.nan_to_num(student_mat.to(device).float(), nan=0.0, posinf=0.0, neginf=0.0)
    U, S, Vh = svd_engine.randomized_svd(
        ref_f,
        k_cap,
        n_oversamples=int(getattr(config, "projection_randomized_oversamples", 8)),
        n_iter=int(getattr(config, "projection_randomized_iter", 2)),
    )

    k = k_cap
    if getattr(config, "projection_adaptive_rank", True):
        k = compute_adaptive_rank(
            S,
            energy_threshold=float(getattr(config, "projection_energy_threshold", 0.99)),
            min_rank=int(getattr(config, "projection_min_rank", 8)),
            max_rank=k_cap,
        )
    k = max(1, min(int(k), int(S.numel()), k_cap))

    U = U[:, :k].contiguous()
    S = S[:k].contiguous()
    Vh = Vh[:k, :].contiguous()
    V = Vh.transpose(0, 1).contiguous()
    return U, S, V


def teacher_core_in_student_basis(
    teacher_mat: torch.Tensor,
    U_s: torch.Tensor,
    V_s: torch.Tensor,
    svd_engine: "GPUAcceleratedSVD",
    config: "DistillConfig",
    device: torch.device,
) -> torch.Tensor:
    """
    Compute teacher's representation in the student's SVD basis:
        core_t = U_s^T * Wt_resized * V_s
    where Wt_resized is constructed from teacher truncated SVD factors (PiSSA sqrt-split)
    and then row/col-resized to the student's matrix shape.

    Returns:
      core_t: (k, k) in float32 on `device`.
    """
    if teacher_mat.dim() != 2:
        raise ValueError("teacher_mat must be 2D")
    if U_s.dim() != 2 or V_s.dim() != 2:
        raise ValueError("U_s and V_s must be 2D")

    tm, tn = int(U_s.shape[0]), int(V_s.shape[0])
    k_s = int(U_s.shape[1])

    src_f = torch.nan_to_num(teacher_mat.to(device).float(), nan=0.0, posinf=0.0, neginf=0.0)
    r_cap = min(k_s, int(src_f.shape[0]), int(src_f.shape[1]))
    if r_cap <= 0:
        return torch.zeros((k_s, k_s), device=device, dtype=torch.float32)

    U_t, S_t, Vh_t = svd_engine.randomized_svd(
        src_f,
        r_cap,
        n_oversamples=int(getattr(config, "projection_randomized_oversamples", 8)),
        n_iter=int(getattr(config, "projection_randomized_iter", 2)),
    )

    r = r_cap
    if getattr(config, "projection_adaptive_rank", True):
        r = compute_adaptive_rank(
            S_t,
            energy_threshold=float(getattr(config, "projection_energy_threshold", 0.99)),
            min_rank=int(getattr(config, "projection_min_rank", 8)),
            max_rank=r_cap,
        )
    r = max(1, min(int(r), int(S_t.numel()), r_cap))

    U_t = U_t[:, :r]
    S_t = S_t[:r]
    Vh_t = Vh_t[:r, :]

    sqrt_S = torch.sqrt(torch.clamp(S_t.float(), min=1e-10)).to(dtype=U_t.dtype)
    B_t = (U_t * sqrt_S.unsqueeze(0)).contiguous()              # (m_t, r)
    A_t = (sqrt_S.unsqueeze(1) * Vh_t).contiguous()             # (r, n_t)

    B_rs = _resize_2d_rows_linear(B_t, tm)                       # (tm, r)
    A_rs = _resize_2d_cols_linear(A_t, tn)                       # (r, tn)

    # core = U_s^T (B_rs A_rs) V_s = (U_s^T B_rs) (A_rs V_s)
    left = U_s.transpose(0, 1) @ B_rs                            # (k, r)
    right = A_rs @ V_s                                           # (r, k)
    core = left @ right                                          # (k, k)
    return core.to(dtype=torch.float32)


def extract_lora_from_core_delta(
    delta_core: torch.Tensor,
    U_s: torch.Tensor,
    V_s: torch.Tensor,
    rank: int,
    config: DistillConfig,
    out_dtype: torch.dtype,
    svd_engine: "GPUAcceleratedSVD",
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    """
    LoRA extraction that respects student subspace:
        dc = U_s^T * delta_core * V_s   (r x r)
        SVD(dc) -> Uc,S,Vc
        lora_B = U_s * (Uc*sqrt(S))
        lora_A = (sqrt(S)*Vc^T) * V_s^T
    """
    # Project delta into student subspace in float32 for numerical stability
    try:
        U32 = U_s.to(dtype=torch.float32)
        V32 = V_s.to(dtype=torch.float32)
        D32 = delta_core.to(dtype=torch.float32)
        dc = U32.T @ D32 @ V32  # (r, r)
    except Exception as e:
        print(f"    ⚠️  Core delta projection failed: {str(e)}")
        return None, None, 0

    rcap = int(min(rank, dc.shape[0], dc.shape[1]))
    if rcap <= 0:
        return None, None, 0

    Uc = S = Vh = None

    # 1) Use configured SVD engine (handles dtype/device)
    try:
        Uc, S, Vh = svd_engine.svd(dc, rcap)
    except Exception as e:
        print(f"    ⚠️  Core delta SVD engine failed: {str(e)}")
        Uc = S = Vh = None

    # 2) Fallback: CPU float64 full SVD (more stable)
    if Uc is None or S is None or Vh is None:
        try:
            Uc_full, S_full, Vh_full = torch.linalg.svd(
                dc.detach().cpu().double(), full_matrices=False
            )
            Uc = Uc_full.to(device=dc.device, dtype=torch.float32)
            S = S_full.to(device=dc.device, dtype=torch.float32)
            Vh = Vh_full.to(device=dc.device, dtype=torch.float32)
        except Exception as e:
            print(f"    ⚠️  Core delta CPU SVD fallback failed: {str(e)}")
            return None, None, 0

    # Ensure non-negative singular values
    S = torch.clamp(S, min=0.0)

    # Determine actual rank based on energy threshold
    try:
        if config.adaptive_rank and config.energy_threshold < 1.0:
            S2 = S * S
            total_energy = S2.sum()
            if total_energy > 0:
                cumulative_energy = torch.cumsum(S2, dim=0) / total_energy
                actual_rank = int(torch.searchsorted(cumulative_energy, config.energy_threshold).item()) + 1
            else:
                actual_rank = 1
            actual_rank = max(1, min(actual_rank, rcap))
        else:
            actual_rank = rcap
    except Exception as e:
        print(f"    ⚠️  Core delta rank selection failed: {str(e)}")
        actual_rank = rcap

    # Truncate
    Uc = Uc[:, :actual_rank]
    S = S[:actual_rank]
    Vh = Vh[:actual_rank, :]

    # Construct LoRA factors in original space
    try:
        sqrt_S = torch.sqrt(torch.clamp(S, min=0.0)).unsqueeze(0)  # (1, r)
        lora_B = (U_s @ (Uc * sqrt_S)).to(dtype=out_dtype)
        lora_A = ((sqrt_S.T * Vh) @ V_s.T).to(dtype=out_dtype)
        return lora_A, lora_B, actual_rank
    except Exception as e:
        print(f"    ⚠️  Core delta LoRA reconstruction failed: {str(e)}")
        return None, None, 0

def extract_lora_pissa(
    delta: torch.Tensor,
    rank: int,
    svd_engine: GPUAcceleratedSVD,
    config: DistillConfig,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    """
    Extract LoRA matrices using PiSSA (sqrt-split SVD factors).

    SVD improvements (no backward-compat constraints assumed):
      1) Adaptive-rank selection is measured against the *true* Frobenius energy ||Δ||_F^2,
         not just the energy of the (possibly truncated) singular-value vector.
      2) When adaptive rank is enabled, the SVD rank is grown geometrically until the
         energy target is met or max_rank is reached. This avoids always computing SVD at
         max_rank (which can be prohibitively expensive when max_rank is large).
    """
    if delta.dim() != 2:
        return None, None, 0

    out_features, in_features = delta.shape
    min_dim = min(out_features, in_features)

    # Sanitize numerical pathologies early.
    delta = torch.nan_to_num(delta, nan=0.0, posinf=1e6, neginf=-1e6)

    # Frobenius norm / energy (used for exact energy accounting in adaptive rank).
    delta_norm = torch.norm(delta.float())
    if delta_norm < 1e-10:
        return None, None, 0
    total_energy = delta_norm.float() ** 2  # ||Δ||_F^2

    # Respect hard bounds.
    max_rank = int(min(getattr(config, "max_rank", 256), min_dim))
    min_rank = int(max(1, min(getattr(config, "min_rank", 8), max_rank)))

    # If adaptive rank is disabled, do a single SVD at the requested rank (bounded).
    if not bool(getattr(config, "use_adaptive_rank", True)):
        actual_rank = int(max(min_rank, min(int(rank), max_rank)))
        try:
            U, S, Vh = svd_engine.svd(delta, actual_rank)
        except Exception as e:
            print(f"  SVD failed completely, skipping: {type(e).__name__}")
            return None, None, 0

        if S.numel() == 0 or torch.all(S < 1e-10):
            return None, None, 0

        S = torch.clamp(S, min=0.0)
        sqrtS = torch.sqrt(S).to(dtype=U.dtype)
        lora_B = U[:, :actual_rank] * sqrtS.unsqueeze(0)
        lora_A = sqrtS.unsqueeze(1) * Vh[:actual_rank, :]
        return lora_A.float(), lora_B.float(), actual_rank

    # Adaptive-rank path: grow the SVD rank until the retained energy meets the target.
    energy_threshold = float(getattr(config, "energy_threshold", 0.95))
    energy_threshold = float(max(0.0, min(1.0, energy_threshold)))
    target_energy = total_energy * energy_threshold

    # Fast-path: Adaptive range finder (Halko et al., Alg. 4.2) + one projected SVD.
    if str(getattr(config, "svd_mode", "auto")).lower() == "adaptive":
        try:
            U, S, Vh = svd_engine.svd_energy(
                delta,
                energy_threshold=energy_threshold,
                min_rank=min_rank,
                max_rank=max_rank,
                block_size=int(getattr(config, "svd_adaptive_block_size", 32)),
                n_test=int(getattr(config, "svd_adaptive_n_test", 8)),
                n_iter=int(getattr(config, "svd_randomized_iter", 0)),
            )
        except Exception as e:
            if bool(getattr(config, "verbose", False)):
                print(f"  Adaptive SVD failed, falling back to geometric SVD: {type(e).__name__}")
        else:
            if S.numel() == 0 or torch.all(S < 1e-10):
                return None, None, 0

            actual_rank = compute_adaptive_rank(
                S,
                energy_threshold=energy_threshold,
                min_rank=min_rank,
                max_rank=max_rank,
                total_energy=total_energy,
            )

            U = U[:, :actual_rank]
            S = S[:actual_rank]
            Vh = Vh[:actual_rank, :]

            S = torch.clamp(S, min=0.0)
            sqrtS = torch.sqrt(S).to(dtype=U.dtype)
            lora_B = U * sqrtS.unsqueeze(0)
            lora_A = sqrtS.unsqueeze(1) * Vh

            return lora_A.float(), lora_B.float(), actual_rank

    # Start from the user-provided rank (bounded), then grow geometrically as needed.
    r_try = int(max(min_rank, min(int(rank), max_rank)))
    r_try = max(1, min(r_try, max_rank))

    U: Optional[torch.Tensor] = None
    S: Optional[torch.Tensor] = None
    Vh: Optional[torch.Tensor] = None

    while True:
        try:
            U, S, Vh = svd_engine.svd(delta, r_try)
        except Exception as e:
            print(f"  SVD failed at rank {r_try}, skipping: {type(e).__name__}")
            return None, None, 0

        if S.numel() == 0 or torch.all(S < 1e-10):
            return None, None, 0

        # Compute retained energy for prefixes of the singular spectrum we actually computed.
        s = torch.clamp(S.detach().float(), min=0.0)
        cum_energy = torch.cumsum(s ** 2, dim=0)

        hit = (cum_energy >= target_energy)
        if bool(hit.any()):
            r_needed = int(hit.nonzero(as_tuple=True)[0][0].item() + 1)
            actual_rank = int(max(min_rank, min(r_needed, max_rank)))
            break

        # Not enough energy captured by top-r_try singular values.
        if r_try >= max_rank:
            actual_rank = int(max_rank)
            break

        # Grow rank geometrically (fast convergence, bounded by max_rank).
        r_next = int(max(r_try + 1, math.ceil(r_try * 2.0)))
        r_try = int(min(max_rank, r_next))

    # Slice to actual_rank (may be smaller than the last computed r_try).
    U = U[:, :actual_rank]
    S = S[:actual_rank]
    Vh = Vh[:actual_rank, :]

    S = torch.clamp(S, min=0.0)
    sqrtS = torch.sqrt(S).to(dtype=U.dtype)
    lora_B = U * sqrtS.unsqueeze(0)
    lora_A = sqrtS.unsqueeze(1) * Vh

    return lora_A.float(), lora_B.float(), actual_rank

# =============================================================================
#                 CALIBRATION-AWARE (DATA-AWARE) SVD  (OPTIONAL)
# =============================================================================
#
# Goal:
#   Collect activation statistics on the STUDENT model (calibration prompts) and apply
#   data-aware whitening before SVD so the truncated low-rank factors better match the
#   target input distribution.
#
# Modes:
#   - rms: diagonal whitening with per-channel activation RMS (input/output).
#   - cov: SVD-LLM style full-covariance whitening on the input side using a Cholesky
#          factor S of E[xx^T]. PiSSA is applied on (Δ·S) and then unwhitened by S^{-1}.
#
# Notes:
#   - Optional; default disabled ("none").
#   - Calibration only loads the STUDENT model; teacher weights are not needed.
#   - Stats are stored in safetensors (see save/load helpers below).


@dataclass
class CalibStatsEntry:
    """Calibration stats for a module or a deterministic shared-input group."""

    # Diagonal (ASVD-style) stats
    in_rms: Optional[torch.Tensor] = None    # (in_features,)
    out_rms: Optional[torch.Tensor] = None   # (out_features,)

    # Full-covariance (SVD-LLM) stats
    in_chol: Optional[torch.Tensor] = None   # (in_features, in_features) lower-triangular


def _dtype_from_str(dtype_str: str, device: torch.device) -> torch.dtype:
    ds = (dtype_str or "auto").lower()
    if ds == "bf16":
        return torch.bfloat16
    if ds == "fp16":
        return torch.float16
    if ds == "fp32":
        return torch.float32
    # auto
    if device.type == "cuda":
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32


def _infer_calib_device(calib_device: str) -> torch.device:
    cd = (calib_device or "auto").lower()
    if cd == "cuda":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cd == "cpu":
        return torch.device("cpu")
    # auto
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



@contextmanager
def _temporary_default_device(dev: torch.device):
    """Temporarily set PyTorch default device (PyTorch 2.x) for modules that allocate via torch.empty() without device.

    This is a targeted workaround for some model implementations (including some MoE variants) that create very large
    parameters with `torch.empty(...)` on the default device (CPU), which can trigger CPU OOM during initialization.

    If torch.set_default_device is unavailable, this context is a no-op.
    """
    if not hasattr(torch, "set_default_device"):
        yield
        return
    try:
        prev = torch.get_default_device() if hasattr(torch, "get_default_device") else "cpu"
    except Exception:
        prev = "cpu"
    try:
        torch.set_default_device(dev)
        yield
    finally:
        try:
            torch.set_default_device(prev)
        except Exception:
            # Best-effort restore
            try:
                torch.set_default_device("cpu")
            except Exception:
                pass


def _is_cpu_oom_error(e: BaseException) -> bool:
    """Heuristic detector for CPU allocation OOM errors (PyTorch allocator)."""
    msg = str(e)
    return ("DefaultCPUAllocator" in msg and "not enough memory" in msg) or ("alloc_cpu.cpp" in msg and "not enough memory" in msg)


def _from_pretrained_compat(model_cls, pretrained_path: str, dtype: torch.dtype, **kwargs):
    """Transformers compatibility layer: prefer `dtype=...` (new) and fall back to `torch_dtype=...` (old)."""
    try:
        return model_cls.from_pretrained(pretrained_path, dtype=dtype, **kwargs)
    except TypeError:
        # Older transformers
        return model_cls.from_pretrained(pretrained_path, torch_dtype=dtype, **kwargs)


def _calib_max_memory_dict(device: torch.device, max_cpu_memory_gb: float = 64.0, gpu_fraction: float = 0.95) -> Optional[Dict[Union[int, str], str]]:
    """Build a max_memory dict for transformers/accelerate device_map loading.

    Important: use *free* VRAM when available (torch.cuda.mem_get_info) instead of total VRAM.
    This avoids over-committing GPU memory when other processes already occupy VRAM, which can
    otherwise lead to CUDA OOM even for small allocations during model load.
    """
    if device.type != "cuda":
        return None

    gpu_gb: Optional[int] = None
    try:
        free_b, _total_b = torch.cuda.mem_get_info(device)
        free_gb = free_b / (1024 ** 3)
        gpu_gb = max(1, int(free_gb * float(gpu_fraction)))
    except Exception:
        try:
            total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            gpu_gb = max(1, int(total_gb * float(gpu_fraction)))
        except Exception:
            gpu_gb = None

    cpu_gb = max(1, int(float(max_cpu_memory_gb))) if max_cpu_memory_gb and max_cpu_memory_gb > 0 else None

    mm: Dict[Union[int, str], str] = {}
    if gpu_gb is not None:
        mm[int(device.index or 0)] = f"{gpu_gb}GiB"
    if cpu_gb is not None:
        mm["cpu"] = f"{cpu_gb}GiB"
    return mm if mm else None

def _accelerate_meta_load_for_calibration(model_cls,
                                         pretrained_path: str,
                                         dtype: torch.dtype,
                                         device_map: Union[str, Dict[str, Any]],
                                         max_memory: Optional[Dict[Union[int, str], str]],
                                         offload_folder: Optional[str],
                                         trust_remote_code: bool = True):
    """Load a (potentially huge) model for calibration via meta-init + checkpoint dispatch.

    Why:
      Some MoE implementations allocate very large parameter tensors during __init__ via
      `torch.empty(...)` on the default device (often CPU). This can trigger CPU OOM *before*
      transformers' low_cpu_mem_usage/device_map logic gets a chance to shard/offload weights.

    How:
      - Instantiate the model on the meta device (no real allocation) using accelerate.init_empty_weights().
      - Stream-load and dispatch checkpoints using accelerate.load_checkpoint_and_dispatch().

    This path requires `accelerate` to be installed.
    """
    try:
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    except Exception as e:
        raise RuntimeError(
            "Calibration meta-init loading requires `accelerate` (pip install accelerate)."
        ) from e

    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(pretrained_path, trust_remote_code=trust_remote_code)

    with init_empty_weights():
        # AutoModel* classes support from_config; trust_remote_code is required for custom architectures.
        model = model_cls.from_config(cfg, trust_remote_code=trust_remote_code)

    # Some models require tying after init; best-effort.
    try:
        model.tie_weights()
    except Exception:
        pass

    # accelerate signature differs slightly across versions; probe a few variants.
    base_kwargs = dict(
        device_map=device_map,
        max_memory=max_memory,
        offload_folder=offload_folder,
        offload_state_dict=True,
    )

    # Optional kwarg on newer accelerate
    try:
        base_kwargs["offload_buffers"] = True
    except Exception:
        pass

    # Try dtype keyword variants (newer uses dtype; older uses torch_dtype)
    for dtype_key in ("dtype", "torch_dtype"):
        try:
            return load_checkpoint_and_dispatch(model, pretrained_path, **base_kwargs, **{dtype_key: dtype})
        except TypeError:
            continue

    # Retry without offload_buffers if signature mismatch
    base_kwargs.pop("offload_buffers", None)
    for dtype_key in ("dtype", "torch_dtype"):
        try:
            return load_checkpoint_and_dispatch(model, pretrained_path, **base_kwargs, **{dtype_key: dtype})
        except TypeError:
            continue

    # Final attempt: no dtype override
    return load_checkpoint_and_dispatch(model, pretrained_path, **base_kwargs)

def _load_student_model_for_calibration(student_path: str,
                                       device: torch.device,
                                       dtype: torch.dtype,
                                       config: DistillConfig,
                                       model_cls):
    """Robust student model loader for calibration.

    Key failure mode addressed:
      - Some (especially MoE) architectures allocate huge tensors on *CPU* during __init__,
        triggering CPU OOM before transformers' device_map/offload code can run.

    Loader policy:
      1) Try transformers.from_pretrained with device_map="auto" + low_cpu_mem_usage (fast path).
      2) If we detect CPU-OOM during initialization, fall back to accelerate meta-init + dispatch.
      3) Final fallback: try meta default-device (best-effort), then raise the original error.

    Notes:
      - We intentionally do NOT force default-device='cuda' as a fallback because that can
        cause full model initialization allocations on GPU and can OOM immediately for large models.
    """
    base_kwargs = dict(trust_remote_code=True, low_cpu_mem_usage=True)

    # Offload directory for device_map paths.
    offload_folder: Optional[str] = None
    try:
        base = getattr(config, "offload_dir", None) or getattr(config, "output_path", ".")
        offload_folder = os.path.join(base, "calib_offload")
        os.makedirs(offload_folder, exist_ok=True)
    except Exception:
        offload_folder = None

    if device.type == "cuda":
        max_memory = _calib_max_memory_dict(
            device=device,
            max_cpu_memory_gb=float(getattr(config, "max_cpu_memory_gb", 64.0) or 64.0),
            gpu_fraction=float(getattr(config, "gpu_memory_fraction", 0.95) or 0.95),
        )

        # Attempt 1: transformers device_map auto
        try:
            return _from_pretrained_compat(
                model_cls,
                student_path,
                dtype=dtype,
                device_map="auto",
                max_memory=max_memory,
                offload_folder=offload_folder,
                offload_state_dict=True,
                offload_buffers=True,
                **base_kwargs,
            )
        except Exception as e1:
            # Cleanup before retries
            try:
                import gc
                gc.collect()
            except Exception:
                pass
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            if _is_cpu_oom_error(e1):
                # Attempt 2: accelerate meta-init + dispatch
                try:
                    return _accelerate_meta_load_for_calibration(
                        model_cls=model_cls,
                        pretrained_path=student_path,
                        dtype=dtype,
                        device_map="auto",
                        max_memory=max_memory,
                        offload_folder=offload_folder,
                        trust_remote_code=True,
                    )
                except Exception:
                    # Attempt 3: best-effort meta default-device with transformers
                    try:
                        with _temporary_default_device(torch.device("meta")):
                            return _from_pretrained_compat(
                                model_cls,
                                student_path,
                                dtype=dtype,
                                device_map="auto",
                                max_memory=max_memory,
                                offload_folder=offload_folder,
                                offload_state_dict=True,
                                offload_buffers=True,
                                **base_kwargs,
                            )
                    except Exception:
                        pass

            # Not a CPU-OOM init case (or all fallbacks failed): raise original
            raise

    # CPU path
    model = _from_pretrained_compat(model_cls, student_path, dtype=dtype, **base_kwargs)
    return model.to(device)

def _flatten_text_field(x: Any) -> str:
    """Best-effort: normalize a field that may be str / list[str] / other into a single string."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        parts: List[str] = []
        for v in x:
            if v is None:
                continue
            if isinstance(v, str):
                s = v.strip()
            else:
                s = str(v).strip()
            if s:
                parts.append(s)
        return "\n".join(parts)
    try:
        return str(x)
    except Exception:
        return ""


def _looks_like_alpaca_record(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    # Classic Alpaca keys: instruction / input / output (some datasets use lists for input/output)
    if "instruction" in obj and ("input" in obj or "output" in obj):
        return True
    # Some variants add system_prompt or system
    if ("system_prompt" in obj or "system" in obj) and ("instruction" in obj or "input" in obj):
        return True
    return False


def _format_alpaca_prompt(obj: Dict[str, Any],
                          template: str = "classic",
                          include_output: bool = False) -> str:
    """
    Build a calibration prompt from an Alpaca-style record.

    Supported fields (best-effort):
      - system_prompt / system / system_message (optional)
      - instruction / prompt / query (required-ish)
      - input / context (optional)
      - output / response (optional; included only if include_output=True)
    """
    sys_p = _flatten_text_field(obj.get("system_prompt", obj.get("system", obj.get("system_message", "")))).strip()
    instruction = _flatten_text_field(obj.get("instruction", obj.get("prompt", obj.get("query", "")))).strip()
    inp = _flatten_text_field(obj.get("input", obj.get("context", ""))).strip()
    out = _flatten_text_field(obj.get("output", obj.get("response", ""))).strip()

    template = (template or "classic").lower().strip()
    prefix = (sys_p + "\n\n") if sys_p else ""

    if template == "plain":
        parts: List[str] = []
        if sys_p:
            parts.append(sys_p)
        if instruction:
            parts.append(instruction)
        if inp:
            parts.append(inp)
        if include_output and out:
            parts.append(out)
        return "\n\n".join(parts).strip()

    # classic
    if inp:
        prompt = f"{prefix}### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
    else:
        prompt = f"{prefix}### Instruction:\n{instruction}\n\n### Response:\n"
    if include_output and out:
        prompt = prompt + out
    return prompt


def _stream_json_array_first_n(path: str, n: int) -> Optional[List[Any]]:
    """
    Streaming parse for a top-level JSON array file: returns first n items without loading whole file.
    Returns None if the file does not look like a JSON array.
    """
    try:
        decoder = json.JSONDecoder()
        buf = ""
        items: List[Any] = []
        with open(path, "r", encoding="utf-8") as f:
            # Read until we find the opening '['
            while True:
                if not buf:
                    chunk = f.read(65536)
                    if not chunk:
                        return []
                    buf += chunk
                buf = buf.lstrip()
                if buf.startswith("["):
                    buf = buf[1:]
                    break
                # Not an array
                return None

            while len(items) < n:
                buf = buf.lstrip()
                if buf.startswith("]"):
                    break
                # Ensure buffer has enough content for a full JSON object
                while True:
                    try:
                        obj, idx = decoder.raw_decode(buf)
                        items.append(obj)
                        buf = buf[idx:]
                        buf = buf.lstrip()
                        if buf.startswith(","):
                            buf = buf[1:]
                        break
                    except json.JSONDecodeError:
                        chunk = f.read(65536)
                        if not chunk:
                            return items
                        buf += chunk
        return items
    except Exception:
        return None


def _load_calib_texts(path: Optional[str],
                      max_samples: int,
                      fmt: str = "auto",
                      alpaca_template: str = "classic",
                      alpaca_include_output: bool = False) -> List[str]:
    """
    Load calibration texts.

    Supported formats:
      - txt:   1 prompt per line
      - jsonl: 1 JSON per line; supports {"text": "..."} or Alpaca records
      - json:  JSON array/object; supports "text" or Alpaca records (streamed for arrays)
      - alpaca: force Alpaca parsing/formatting for .json/.jsonl
    """
    if not path:
        # Minimal built-in prompts (language-mixed) as a fallback to keep the feature plug-and-play.
        return [
            "你好，請用一句話介紹你自己。",
            "請解釋什麼是知識蒸餾（Knowledge Distillation）。",
            "請把下面句子翻譯成英文：人工智慧正在改變世界。",
            "Write a short paragraph about neural networks.",
            "Summarize the following in one sentence: Large language models learn patterns from data.",
            "List three key points about transformer attention.",
        ][:max_samples]

    fmt = (fmt or "auto").lower().strip()
    p_lower = str(path).lower()
    if fmt == "auto":
        if p_lower.endswith(".jsonl"):
            fmt = "jsonl"
        elif p_lower.endswith(".json"):
            fmt = "json"
        else:
            fmt = "txt"

    # For alpaca, we accept both .json and .jsonl
    force_alpaca = (fmt == "alpaca")
    if force_alpaca:
        # decide underlying file type
        if p_lower.endswith(".jsonl"):
            fmt_under = "jsonl"
        else:
            fmt_under = "json"
    else:
        fmt_under = fmt

    def _extract_text(obj: Any) -> Optional[str]:
        if obj is None:
            return None
        if isinstance(obj, str):
            s = obj.strip()
            return s if s else None
        if isinstance(obj, dict):
            if force_alpaca or _looks_like_alpaca_record(obj):
                t = _format_alpaca_prompt(obj, template=alpaca_template, include_output=alpaca_include_output)
                t = (t or "").strip()
                return t if t else None
            # json/jsonl generic: prefer "text"
            t = obj.get("text", None)
            if isinstance(t, str) and t.strip():
                return t.strip()
            # fallback: first string-ish value
            for v in obj.values():
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return None
        if isinstance(obj, (list, tuple)):
            # list[str] -> join
            s = _flatten_text_field(obj).strip()
            return s if s else None
        # last resort
        try:
            s = str(obj).strip()
            return s if s else None
        except Exception:
            return None

    texts: List[str] = []

    try:
        if fmt_under == "txt":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    texts.append(line)
                    if len(texts) >= max_samples:
                        break
            return texts[:max_samples] if max_samples > 0 else texts

        if fmt_under == "jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    t = _extract_text(obj)
                    if t:
                        texts.append(t)
                    if len(texts) >= max_samples:
                        break
            if texts:
                return texts[:max_samples] if max_samples > 0 else texts
            # If empty, fall through to fallback prompts
            return _load_calib_texts(None, max_samples, "txt")

        if fmt_under == "json":
            # First try jsonl-style streaming (many datasets are .json but actually JSONL)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            # not jsonl
                            texts = []
                            break
                        t = _extract_text(obj)
                        if t:
                            texts.append(t)
                        if len(texts) >= max_samples:
                            break
                if texts:
                    return texts[:max_samples] if max_samples > 0 else texts
            except Exception:
                pass

            # Streaming parse JSON array for first N items
            recs = _stream_json_array_first_n(path, max_samples if max_samples > 0 else 0)
            if recs is not None:
                for obj in recs:
                    t = _extract_text(obj)
                    if t:
                        texts.append(t)
                    if len(texts) >= max_samples:
                        break
                if texts:
                    return texts[:max_samples] if max_samples > 0 else texts

            # Fallback: full json.load (may be heavy for huge files; use jsonl if possible)
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)

            # Unwrap common containers
            records: List[Any] = []
            if isinstance(obj, list):
                records = obj
            elif isinstance(obj, dict):
                # common keys
                for k in ("data", "train", "records", "samples", "items"):
                    v = obj.get(k, None)
                    if isinstance(v, list):
                        records = v
                        break
                if not records:
                    records = [obj]
            else:
                records = [obj]

            for r in records:
                t = _extract_text(r)
                if t:
                    texts.append(t)
                if len(texts) >= max_samples:
                    break

            if texts:
                return texts[:max_samples] if max_samples > 0 else texts
            return _load_calib_texts(None, max_samples, "txt")

    except Exception:
        # If load fails, fall back to built-in prompts (do not hard-fail).
        return _load_calib_texts(None, max_samples, "txt")

    return texts[:max_samples] if max_samples > 0 else texts


class _RMSAccumulator:
    __slots__ = ("in_sumsq", "out_sumsq", "in_count", "out_count")

    def __init__(self, in_features: int, out_features: int, collect_in: bool, collect_out: bool):
        self.in_sumsq = torch.zeros(in_features, dtype=torch.float32) if collect_in else None
        self.out_sumsq = torch.zeros(out_features, dtype=torch.float32) if collect_out else None
        self.in_count = 0
        self.out_count = 0

    def add_in(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        if self.in_sumsq is None:
            return
        # x: (..., in_features)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).float()
        x = x.reshape(-1, x.shape[-1])

        if attention_mask is not None and torch.is_tensor(attention_mask):
            m = attention_mask
            if m.dim() > 1:
                m = m.reshape(-1)
            else:
                m = m.reshape(-1)

            if m.numel() == x.shape[0]:
                m = m.to(device=x.device, dtype=x.dtype)
                v = (x * x * m.unsqueeze(1)).sum(dim=0)
                cnt = int(m.sum().item())
            else:
                v = (x * x).sum(dim=0)
                cnt = int(x.shape[0])
        else:
            v = (x * x).sum(dim=0)
            cnt = int(x.shape[0])

        if cnt <= 0:
            return
        self.in_sumsq += v.detach().cpu()
        self.in_count += cnt
    def add_out(self, y: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        if self.out_sumsq is None:
            return
        # y: (..., out_features)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).float()
        y = y.reshape(-1, y.shape[-1])

        if attention_mask is not None and torch.is_tensor(attention_mask):
            m = attention_mask
            if m.dim() > 1:
                m = m.reshape(-1)
            else:
                m = m.reshape(-1)

            if m.numel() == y.shape[0]:
                m = m.to(device=y.device, dtype=y.dtype)
                v = (y * y * m.unsqueeze(1)).sum(dim=0)
                cnt = int(m.sum().item())
            else:
                v = (y * y).sum(dim=0)
                cnt = int(y.shape[0])
        else:
            v = (y * y).sum(dim=0)
            cnt = int(y.shape[0])

        if cnt <= 0:
            return
        self.out_sumsq += v.detach().cpu()
        self.out_count += cnt
    def finalize(self, eps: float) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        in_rms = None
        out_rms = None
        if self.in_sumsq is not None:
            denom = max(1, self.in_count)
            in_rms = torch.sqrt(self.in_sumsq / float(denom)).clamp(min=eps).contiguous()
        if self.out_sumsq is not None:
            denom = max(1, self.out_count)
            out_rms = torch.sqrt(self.out_sumsq / float(denom)).clamp(min=eps).contiguous()
        return in_rms, out_rms


class _CovAccumulator:
    """Accumulate full input covariance: sum(x^T x) and token count."""

    __slots__ = ("dim", "gram", "count")

    def __init__(self, dim: int, device: torch.device):
        self.dim = int(dim)
        self.gram = torch.zeros((self.dim, self.dim), device=device, dtype=torch.float32)
        self.count = 0

    def add(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> None:
        if x is None:
            return
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        if x.numel() == 0:
            return
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=torch.float32)
        if x.shape[-1] != self.dim:
            return

        if attn_mask is not None:
            m = attn_mask
            if m.dim() > 1:
                m = m.reshape(-1)
            if m.numel() == x.shape[0]:
                m = m.to(device=x.device, dtype=torch.float32)
                x = x * m.unsqueeze(1)
                cnt = int(torch.sum(m).item())
            else:
                cnt = int(x.shape[0])
        else:
            cnt = int(x.shape[0])

        # x^T x
        self.gram += x.transpose(0, 1) @ x
        self.count += max(0, cnt)


def _parse_layer_index(name: str) -> Optional[int]:
    """Return layer index if name matches common HF layer naming."""
    m = re.search(r'(?:model\.layers|transformer\.h|decoder\.layers)\.(\d+)\.', name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _cov_group_key_for_module(module_name: str, student_arch: ArchitectureInfo) -> Optional[str]:
    """Map a module name to a deterministic shared-input covariance key."""
    if not module_name or module_name.count(".") < 2:
        return None
    leaf = module_name.split(".")[-1]
    base = ".".join(module_name.split(".")[:-1])

    qn = getattr(student_arch, "q_proj_name", "q_proj")
    kn = getattr(student_arch, "k_proj_name", "k_proj")
    vn = getattr(student_arch, "v_proj_name", "v_proj")
    on = getattr(student_arch, "o_proj_name", "o_proj")
    gn = getattr(student_arch, "gate_proj_name", "gate_proj")
    un = getattr(student_arch, "up_proj_name", "up_proj")
    dn = getattr(student_arch, "down_proj_name", "down_proj")

    if leaf in (qn, kn, vn):
        return f"{base}.qkv_in"
    if leaf == on:
        return f"{base}.o_in"
    if leaf in (gn, un, "w1", "w3"):
        return f"{base}.up_in"
    if leaf in (dn, "w2"):
        return f"{base}.down_in"
    return None


def _resolve_cov_entry_key(module_name: str, calib_stats: Dict[str, CalibStatsEntry], student_arch: ArchitectureInfo) -> Optional[str]:
    """Resolve the key to use for covariance whitening (exact module or shared-input group)."""
    if module_name in calib_stats and calib_stats[module_name].in_chol is not None:
        return module_name
    gk = _cov_group_key_for_module(module_name, student_arch)
    if gk and gk in calib_stats and calib_stats[gk].in_chol is not None:
        return gk
    return None


def _safe_cholesky(cov: torch.Tensor, eps: float, max_tries: int = 6) -> torch.Tensor:
    """Robust Cholesky with escalating jitter."""
    cov = (cov + cov.transpose(0, 1)) * 0.5
    d = int(cov.shape[0])
    eye = torch.eye(d, device=cov.device, dtype=cov.dtype)
    jitter = float(eps)
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(cov + jitter * eye)
        except Exception:
            jitter *= 10.0
    # Last-resort: attempt cholesky_ex, then jitter more
    L, info = torch.linalg.cholesky_ex(cov + jitter * eye)
    if int(info) == 0:
        return L
    raise RuntimeError("Cholesky failed (covariance not SPD even after jitter).")


def collect_calibration_stats_rms(student_path: str,
                                  student_arch: ArchitectureInfo,
                                  config: DistillConfig) -> Dict[str, CalibStatsEntry]:
    """Collect per-channel activation RMS stats on the STUDENT model."""
    if getattr(config, "calibration_mode", "none") == "none":
        return {}

    if not HAS_TRANSFORMERS:
        raise RuntimeError("Calibration mode requires transformers installed.")

    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = _infer_calib_device(getattr(config, "calib_device", "auto"))
    dtype = _dtype_from_str(getattr(config, "calib_dtype", "auto"), device)

    texts = _load_calib_texts(
        getattr(config, "calib_data", None),
        int(getattr(config, "calib_max_samples", 128)),
        getattr(config, "calib_format", "auto"),
        getattr(config, "calib_alpaca_template", "classic"),
        bool(getattr(config, "calib_alpaca_include_output", False)),
    )
    if not texts:
        return {}

    tok = AutoTokenizer.from_pretrained(student_path, trust_remote_code=True, use_fast=True)
    if tok.pad_token_id is None:
        # Make padding well-defined for batch calibration
        tok.pad_token = tok.eos_token

    # Load model with robust low-memory strategy (handles CPU-OOM init for some MoE implementations)
    try:
        model = _load_student_model_for_calibration(
            student_path=student_path,
            device=device,
            dtype=dtype,
            config=config,
            model_cls=AutoModelForCausalLM,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load student model for calibration: {type(e).__name__}: {e}")

    model.eval()

    # Select target linear modules for stats collection
    include_re = None
    try:
        pat = getattr(config, "include_pattern", "self_attn|mlp") or "self_attn|mlp"
        include_re = re.compile(pat)
    except Exception:
        include_re = re.compile("self_attn|mlp")

    target_leaf_names = set([
        student_arch.q_proj_name, student_arch.k_proj_name, student_arch.v_proj_name, student_arch.o_proj_name,
        student_arch.gate_proj_name, student_arch.up_proj_name, student_arch.down_proj_name,
        # Fallback names commonly seen in SwiGLU variants
        "w1", "w2", "w3",
    ])

    acc: Dict[str, _RMSAccumulator] = {}
    mask_ref: Dict[str, Optional[torch.Tensor]] = {"mask": None}
    hooks: List[Any] = []

    def _should_hook(name: str, module: torch.nn.Module) -> bool:
        if not isinstance(module, torch.nn.Linear):
            return False
        if include_re is not None and (not include_re.search(name)):
            return False
        leaf = name.split(".")[-1]
        return leaf in target_leaf_names

    # Register hooks
    for name, module in model.named_modules():
        if not _should_hook(name, module):
            continue
        try:
            in_f = int(getattr(module, "in_features", 0))
            out_f = int(getattr(module, "out_features", 0))
            if in_f <= 0 or out_f <= 0:
                continue
            acc[name] = _RMSAccumulator(in_f, out_f,
                                        collect_in=bool(getattr(config, "calib_collect_in", True)),
                                        collect_out=bool(getattr(config, "calib_collect_out", True)))
        except Exception:
            continue

        def _pre_hook(mod, inputs, name=name):
            if name not in acc:
                return
            try:
                x = inputs[0]
                if isinstance(x, (tuple, list)):
                    x = x[0]
                if x is None or not torch.is_tensor(x) or x.dim() == 0:
                    return
                acc[name].add_in(x, mask_ref["mask"] if config.calib_use_attention_mask else None)
            except Exception:
                return

        def _fwd_hook(mod, inputs, output, name=name):
            if name not in acc:
                return
            try:
                y = output
                if isinstance(y, (tuple, list)):
                    y = y[0]
                if y is None or not torch.is_tensor(y) or y.dim() == 0:
                    return
                acc[name].add_out(y, mask_ref["mask"] if config.calib_use_attention_mask else None)
            except Exception:
                return

        hooks.append(module.register_forward_pre_hook(_pre_hook))
        hooks.append(module.register_forward_hook(_fwd_hook))

    if not acc:
        # Nothing matched; cleanup
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {}

    # Determine input device
    try:
        input_device = next(model.parameters()).device
    except Exception:
        input_device = device

    bs = max(1, int(getattr(config, "calib_batch_size", 1)))
    max_len = max(8, int(getattr(config, "calib_max_length", 256)))

    with torch.inference_mode():
        for i in range(0, len(texts), bs):
            batch_texts = texts[i:i+bs]
            enc = tok(
                batch_texts,
                return_tensors="pt",
                padding=("max_length" if (config.calib_padding == "max_length" and max_len is not None) else True),
                truncation=True,
                max_length=max_len,
            )
            enc = {k: v.to(input_device) for k, v in enc.items()}
            mask_ref["mask"] = enc.get("attention_mask", None)
            try:
                _ = model(**enc, use_cache=False)
            except Exception:
                # Skip problematic batch
                continue

    # Cleanup hooks
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    stats: Dict[str, CalibStatsEntry] = {}
    eps = float(getattr(config, "calib_eps", 1e-6))
    for name, a in acc.items():
        in_rms, out_rms = a.finalize(eps)
        stats[name] = CalibStatsEntry(in_rms=in_rms, out_rms=out_rms)

    # Release model memory
    del model
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return stats


def collect_calibration_stats_cov(student_path: str,
                                  student_arch: ArchitectureInfo,
                                  config: DistillConfig) -> Dict[str, CalibStatsEntry]:
    """Collect full-covariance (SVD-LLM) input whitening stats on the STUDENT model.

    Produces in_chol factors keyed by deterministic shared-input group keys (see
    _cov_group_key_for_module). These factors are consumed by extract_lora_pissa_calibrated
    when calib_mode=cov.
    """
    if not HAS_TRANSFORMERS:
        raise RuntimeError("Calibration mode requires transformers installed.")

    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = _infer_calib_device(getattr(config, "calib_device", "auto"))
    dtype = _dtype_from_str(getattr(config, "calib_dtype", "auto"), device)

    texts = _load_calib_texts(
        getattr(config, "calib_data", None),
        int(getattr(config, "calib_max_samples", 128)),
        getattr(config, "calib_format", "auto"),
        getattr(config, "calib_alpaca_template", "classic"),
        bool(getattr(config, "calib_alpaca_include_output", False)),
    )
    if not texts:
        return {}

    tok = AutoTokenizer.from_pretrained(student_path, trust_remote_code=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Cov calibration is numerically sensitive: prefer fp32 on CPU.
    calib_dtype = dtype if device.type == "cuda" else torch.float32
    try:
        model = _load_student_model_for_calibration(
            student_path=student_path,
            device=device,
            dtype=calib_dtype,
            config=config,
            model_cls=AutoModelForCausalLM,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load student model for calibration: {type(e).__name__}: {e}")

    model.eval()

    include_pat = getattr(config, "include_pattern", "")
    exclude_pat = getattr(config, "exclude_pattern", "")
    include_re = re.compile(include_pat) if include_pat else None
    exclude_re = re.compile(exclude_pat) if exclude_pat else None

    # Which covariance groups to collect
    groups_raw = str(getattr(config, "calib_cov_groups", "qkv,o,mlp")).split(",")
    want_groups = {g.strip().lower() for g in groups_raw if g.strip()}
    max_dim = int(getattr(config, "calib_cov_max_dim", 8192))
    chunk_layers = max(1, int(getattr(config, "calib_cov_chunk_layers", 4)))
    store_dtype_str = str(getattr(config, "calib_cov_store_dtype", "fp16")).lower()
    store_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(store_dtype_str, torch.float16)
    eps = float(getattr(config, "calib_eps", 1e-6))
    use_attn_mask = bool(getattr(config, "calib_use_attention_mask", True))

    # Build per-layer representative module selection.
    module_dict: Dict[str, torch.nn.Module] = dict(model.named_modules())

    qn = getattr(student_arch, "q_proj_name", "q_proj")
    kn = getattr(student_arch, "k_proj_name", "k_proj")
    vn = getattr(student_arch, "v_proj_name", "v_proj")
    on = getattr(student_arch, "o_proj_name", "o_proj")
    gn = getattr(student_arch, "gate_proj_name", "gate_proj")
    un = getattr(student_arch, "up_proj_name", "up_proj")
    dn = getattr(student_arch, "down_proj_name", "down_proj")

    priorities: Dict[str, List[str]] = {
        "qkv": [qn, kn, vn],
        "o": [on],
        "mlp": [gn, un, "w1", "w3"],
        "down": [dn, "w2"],
    }

    layer_sel: Dict[int, Dict[str, str]] = defaultdict(dict)
    layer_pri: Dict[int, Dict[str, int]] = defaultdict(dict)

    for name, mod in module_dict.items():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if include_re and not include_re.search(name):
            continue
        if exclude_re and exclude_re.search(name):
            continue

        li = _parse_layer_index(name)
        if li is None:
            continue

        leaf = name.split(".")[-1]
        for g in want_groups:
            if g not in priorities:
                continue
            plist = priorities[g]
            if leaf not in plist:
                continue
            pidx = plist.index(leaf)
            if (g not in layer_sel[li]) or (pidx < layer_pri[li].get(g, 10**9)):
                layer_sel[li][g] = name
                layer_pri[li][g] = pidx

    layer_indices = sorted(layer_sel.keys())
    if not layer_indices:
        # Nothing matched; return empty.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {}

    # Determine the device for feeding input tensors.
    try:
        input_device = next(model.parameters()).device
    except Exception:
        input_device = device

    max_len = int(getattr(config, "calib_max_length", 256)) if getattr(config, "calib_max_length", None) else None
    bs = int(getattr(config, "calib_batch_size", 1))

    stats: Dict[str, CalibStatsEntry] = {}

    # Chunked passes to bound peak memory.
    mask_ref: Dict[str, Optional[torch.Tensor]] = {"mask": None}
    for c0 in range(0, len(layer_indices), chunk_layers):
        chunk = layer_indices[c0:c0 + chunk_layers]

        acc: Dict[str, _CovAccumulator] = {}
        hooks: List[Any] = []

        # Register hooks for this chunk.
        for li in chunk:
            for g in want_groups:
                if g not in priorities:
                    continue
                mod_name = layer_sel.get(li, {}).get(g, None)
                if not mod_name:
                    continue
                mod = module_dict.get(mod_name, None)
                if mod is None or not isinstance(mod, torch.nn.Linear):
                    continue

                in_dim = int(getattr(mod, "in_features", 0) or 0)
                if in_dim <= 0 or in_dim > max_dim:
                    continue

                gk = _cov_group_key_for_module(mod_name, student_arch)
                if not gk:
                    continue
                if gk in acc:
                    continue

                try:
                    dev = next(mod.parameters()).device
                except Exception:
                    dev = input_device

                acc[gk] = _CovAccumulator(in_dim, dev)

                def _make_pre_hook(key: str):
                    def _pre_hook(_mod, inputs):
                        if not inputs:
                            return
                        x = inputs[0]
                        if not torch.is_tensor(x):
                            return
                        m = mask_ref.get("mask", None)
                        if not use_attn_mask:
                            m = None
                        acc[key].add(x, m)
                    return _pre_hook

                hooks.append(mod.register_forward_pre_hook(_make_pre_hook(gk)))

        if not acc:
            # Nothing to collect in this chunk.
            for h in hooks:
                try:
                    h.remove()
                except Exception:
                    pass
            continue

        # Run calibration forward passes.
        with torch.inference_mode():
            for i in range(0, len(texts), bs):
                batch_texts = texts[i:i + bs]
                enc = tok(
                    batch_texts,
                    return_tensors="pt",
                    padding=("max_length" if (getattr(config, "calib_padding", "longest") == "max_length" and max_len is not None) else True),
                    truncation=True,
                    max_length=max_len,
                )
                enc = {k: v.to(input_device) for k, v in enc.items()}
                mask_ref["mask"] = enc.get("attention_mask", None)
                try:
                    _ = model(**enc, use_cache=False)
                except Exception:
                    continue

        # Cleanup hooks for this chunk.
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

        # Finalize and store cholesky factors.
        for gk, a in acc.items():
            if a.count <= 0:
                continue
            cov = a.gram / float(max(1, a.count))
            try:
                L = _safe_cholesky(cov, eps=eps)
            except Exception:
                continue

            stats[gk] = CalibStatsEntry(in_chol=L.detach().to(dtype=store_dtype).cpu().contiguous())

        # Release chunk accumulators.
        del acc
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Release model memory
    del model
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return stats


def collect_calibration_stats(student_path: str,
                              student_arch: ArchitectureInfo,
                              config: DistillConfig) -> Dict[str, CalibStatsEntry]:
    """Dispatch calibration collection based on config.calibration_mode."""
    mode = str(getattr(config, "calibration_mode", "none")).lower()
    if mode == "rms":
        return collect_calibration_stats_rms(student_path, student_arch, config)
    if mode == "cov":
        return collect_calibration_stats_cov(student_path, student_arch, config)
    return {}


def save_calibration_stats(stats: Dict[str, CalibStatsEntry], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out: Dict[str, torch.Tensor] = {}
    for name, e in stats.items():
        if e.in_rms is not None:
            out[f"{name}.in_rms"] = e.in_rms.detach().cpu().contiguous()
        if e.out_rms is not None:
            out[f"{name}.out_rms"] = e.out_rms.detach().cpu().contiguous()
        if e.in_chol is not None:
            out[f"{name}.in_chol"] = e.in_chol.detach().cpu().contiguous()
    if not out:
        return
    save_file(out, path, metadata={"format": "ud_calib_stats_v2"})


def load_calibration_stats(path: str) -> Dict[str, CalibStatsEntry]:
    tensors = load_file(path)
    stats: Dict[str, CalibStatsEntry] = {}
    tmp: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
    for k, v in tensors.items():
        if k.endswith(".in_rms"):
            tmp[k[:-len(".in_rms")]]["in"] = v
        elif k.endswith(".out_rms"):
            tmp[k[:-len(".out_rms")]]["out"] = v
        elif k.endswith(".in_chol"):
            tmp[k[:-len(".in_chol")]]["in_chol"] = v
    for name, d in tmp.items():
        stats[name] = CalibStatsEntry(
            in_rms=d.get("in", None),
            out_rms=d.get("out", None),
            in_chol=d.get("in_chol", None),
        )
    return stats


def _delta_whiten_diag(delta: torch.Tensor,
                      in_rms: Optional[torch.Tensor],
                      out_rms: Optional[torch.Tensor],
                      eps: float) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Return whitened delta and the (clamped) scaling vectors used.

    delta: (out, in)
    in_rms:  (in,)
    out_rms: (out,)
    """
    dw = delta
    in_s = None
    out_s = None

    if out_rms is not None:
        out_s = out_rms.to(device=delta.device, dtype=torch.float32).clamp(min=eps)
        dw = dw / out_s.unsqueeze(1)

    if in_rms is not None:
        in_s = in_rms.to(device=delta.device, dtype=torch.float32).clamp(min=eps)
        dw = dw / in_s.unsqueeze(0)

    return dw, in_s, out_s


def extract_lora_pissa_calibrated(
    delta: torch.Tensor,
    rank: int,
    svd_engine: GPUAcceleratedSVD,
    config: DistillConfig,
    module_name: str,
    calib_stats: Dict[str, CalibStatsEntry],
    student_arch: ArchitectureInfo,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
    """Calibration-aware PiSSA.

    - rms: diagonal RMS whitening (ASVD-style), then unwhiten LoRA factors.
    - cov: SVD-LLM style full-covariance input whitening via Cholesky factor S of E[xx^T].
           PiSSA is applied on (Δ·S) and then unwhitened by S^{-1} on the right factor.
    """
    if not calib_stats:
        return extract_lora_pissa(delta, rank, svd_engine, config)

    mode = str(getattr(config, "calibration_mode", "none")).lower()
    eps = float(getattr(config, "calib_eps", 1e-6))

    if mode == "rms":
        entry = calib_stats.get(module_name, None)
        if entry is None:
            return extract_lora_pissa(delta, rank, svd_engine, config)
        if entry.in_rms is None and entry.out_rms is None:
            return extract_lora_pissa(delta, rank, svd_engine, config)

        dw, in_s, out_s = _delta_whiten_diag(delta.float(), entry.in_rms, entry.out_rms, eps)

        # Run PiSSA on whitened delta
        lora_A_w, lora_B_w, actual_rank = extract_lora_pissa(dw.to(delta.dtype), rank, svd_engine, config)
        if lora_A_w is None or lora_B_w is None or actual_rank <= 0:
            return None, None, 0

        # Unwhiten factors back to original space: Δ ≈ (D_out·B_w) (A_w·D_in)
        lora_A = lora_A_w.float()
        lora_B = lora_B_w.float()
        if out_s is not None:
            lora_B = lora_B * out_s.unsqueeze(1)
        if in_s is not None:
            lora_A = lora_A * in_s.unsqueeze(0)

        return lora_A.to(dtype=delta.dtype).contiguous(), lora_B.to(dtype=delta.dtype).contiguous(), actual_rank

    if mode == "cov":
        key = _resolve_cov_entry_key(module_name, calib_stats, student_arch)
        if not key:
            return extract_lora_pissa(delta, rank, svd_engine, config)

        S = calib_stats[key].in_chol
        if S is None:
            return extract_lora_pissa(delta, rank, svd_engine, config)

        # Ensure S is on the same device.
        S = S.to(device=delta.device, dtype=torch.float32)

        # PiSSA on (Δ·S)
        dw = (delta.float() @ S).to(dtype=delta.dtype)
        lora_A_w, lora_B_w, actual_rank = extract_lora_pissa(dw, rank, svd_engine, config)
        if lora_A_w is None or lora_B_w is None or actual_rank <= 0:
            return None, None, 0

        # Unwhiten on the right: A = A_w · S^{-1}
        A_w = lora_A_w.float().contiguous()
        B = lora_B_w.float().contiguous()

        # Solve (S^T) X^T = A_w^T  =>  X = A_w · S^{-1}
        try:
            A = torch.linalg.solve_triangular(S.transpose(0, 1), A_w.transpose(0, 1), upper=True).transpose(0, 1)
        except Exception:
            # Fallback: generic solve (slower, but robust)
            A = torch.linalg.solve(S.transpose(0, 1), A_w.transpose(0, 1)).transpose(0, 1)

        return A.to(dtype=delta.dtype).contiguous(), B.to(dtype=delta.dtype).contiguous(), actual_rank

    return extract_lora_pissa(delta, rank, svd_engine, config)


# =============================================================================
#                        LAYER MAPPING
# =============================================================================

def teacher_idx_from_student_idx(s_idx: int, s_total: int, t_total: int,
                                  schedule: str = "sigmoid",
                                  k: float = 0.15) -> Tuple[int, float]:
    """Map student layer index to teacher layer index with interpolation weight."""
    if t_total <= 1 or s_total <= 1:
        return 0, 0.0
    
    s_norm = s_idx / (s_total - 1)
    
    if schedule == "sigmoid":
        x = 12 * (s_norm - 0.5)
        t_norm = 1 / (1 + math.exp(-k * x))
    else:
        t_norm = s_norm
    
    t_float = t_norm * (t_total - 1)
    t_floor = int(t_float)
    t_floor = max(0, min(t_floor, t_total - 1))
    interp_w = t_float - t_floor
    
    return t_floor, interp_w


def create_layer_mapping(teacher_info: ArchitectureInfo,
                         student_info: ArchitectureInfo) -> Dict[str, str]:
    """Create mapping from student layer names to teacher layer names."""
    mapping = {}
    
    attn_map = [
        (student_info.q_proj_name, teacher_info.q_proj_name),
        (student_info.k_proj_name, teacher_info.k_proj_name),
        (student_info.v_proj_name, teacher_info.v_proj_name),
        (student_info.o_proj_name, teacher_info.o_proj_name),
    ]
    
    mlp_map = [
        (student_info.gate_proj_name, teacher_info.gate_proj_name),
        (student_info.up_proj_name, teacher_info.up_proj_name),
        (student_info.down_proj_name, teacher_info.down_proj_name),
    ]
    
    for s_name, t_name in attn_map + mlp_map:
        if s_name != t_name:
            mapping[s_name] = t_name
    
    return mapping


# =============================================================================
#                          PEFT FORMAT OUTPUT
# =============================================================================

def to_peft_key(param_key: str, student_arch: ArchitectureInfo) -> str:
    """
    Method A: derive the PEFT module key from the *student* parameter key.

    This avoids hard-coding top-level prefixes (e.g. forcing 'model.') and instead
    uses the student's detected layer prefix (student_arch.layer_prefix) as an anchor.

    Examples (student HF state_dict keys):
      - model.layers.0.self_attn.q_proj.weight
      - transformer.h.0.attn.c_attn.weight
      - decoder.layers.0.self_attn.q_proj.weight
      - gpt_neox.layers.0.attention.query_key_value.weight

    Output (PEFT adapter module key):
      - base_model.model.<module_path_without_.weight>
        e.g. base_model.model.model.layers.0.self_attn.q_proj
    """
    key = param_key

    # Strip parameter suffix (.weight / .bias)
    for suf in ('.weight', '.bias'):
        if key.endswith(suf):
            key = key[: -len(suf)]
            break

    base_prefix = 'base_model.model.'
    if key.startswith(base_prefix):
        return key

    # If the student's layer prefix appears inside the key, slice from it to drop wrapper prefixes.
    lp = getattr(student_arch, 'layer_prefix', '') or ''
    if lp:
        needle = lp + '.'
        idx = key.find(needle)
        if idx != -1:
            key = key[idx:]

    return base_prefix + key


def save_peft_adapter(
    lora_state: Dict[str, torch.Tensor],
    output_dir: str,
    config: DistillConfig,
    student_arch: ArchitectureInfo,
    rank_info: Dict[str, int],
) -> None:
    """
    Save PEFT-compatible LoRA adapter artifacts.

    Outputs:
    - adapter_model.safetensors: LoRA weights (standard format: lora_A.weight)
    - adapter_model_with_adapter_name.safetensors: LoRA weights (named adapter format: lora_A.default.weight)
    - adapter_config.json: PEFT config

    Method A key normalization:
      If the produced keys are missing the PEFT wrapper prefix (base_model.model.),
      infer and apply it using the student's architecture anchor (student_arch.layer_prefix).
    """
    os.makedirs(output_dir, exist_ok=True)

    base_prefix = 'base_model.model.'

    # -----------------------------
    # Normalize LoRA state_dict keys (Method A)
    # -----------------------------
    lp = getattr(student_arch, 'layer_prefix', '') or ''

    def _normalize_key(k: str, use_adapter_name: bool = False) -> str:
        kk = k
        # Normalize key format based on use_adapter_name flag
        if use_adapter_name:
            # Convert lora_A.weight -> lora_A.default.weight (named adapter format)
            if re.search(r'\.lora_[AB]\.weight$', kk):
                kk = re.sub(r'\.lora_([AB])\.weight$', r'.lora_\1.default.weight', kk)
        else:
            # Convert lora_A.default.weight -> lora_A.weight (standard format)
            if re.search(r'\.lora_[AB]\.default\.weight$', kk):
                kk = re.sub(r'\.lora_([AB])\.default\.weight$', r'.lora_\1.weight', kk)

        # If already in PEFT wrapper namespace, keep as-is.
        if kk.startswith(base_prefix):
            return kk

        # Drop any wrapper prefixes before the student's layer prefix, if present.
        if lp:
            needle = lp + '.'
            idx = kk.find(needle)
            if idx != -1:
                kk = kk[idx:]

        return base_prefix + kk

    # Build two versions: standard (no adapter name) and named adapter
    normalized_lora_state: Dict[str, torch.Tensor] = {}           # standard: lora_A.weight
    normalized_lora_state_named: Dict[str, torch.Tensor] = {}     # named: lora_A.default.weight
    for k, v in lora_state.items():
        nk = _normalize_key(k, use_adapter_name=False)
        nk_named = _normalize_key(k, use_adapter_name=True)
        normalized_lora_state[nk] = v
        normalized_lora_state_named[nk_named] = v


    # -----------------------------
    # Canonicalize module paths against the *student* model index (Method A+)
    # -----------------------------
    student_module_paths = None
    try:
        student_root = getattr(config, 'student_path', None) or getattr(config, 'student', None) or ''
        if student_root:
            idx_path = os.path.join(student_root, 'model.safetensors.index.json')
            if os.path.exists(idx_path):
                with open(idx_path, 'r', encoding='utf-8') as f:
                    _idx = json.load(f)
                _wmap = _idx.get('weight_map', {}) or {}
                _mods = set()
                for _wk in _wmap.keys():
                    # skip scale metadata
                    if _wk.endswith('_scale_inv'):
                        continue
                    if _wk.endswith('.weight') or _wk.endswith('.bias'):
                        _mods.add(_wk.rsplit('.', 1)[0])
                student_module_paths = _mods if _mods else None
    except Exception:
        student_module_paths = None

    def _canon_module_path(module_path: str) -> str:
        if not student_module_paths:
            return module_path
        if module_path in student_module_paths:
            return module_path

        # Try common root prefixes
        for root in ('model.', 'transformer.', 'decoder.'):
            cand = root + module_path
            if cand in student_module_paths:
                return cand

        # Suffix match (unique)
        cands = [p for p in student_module_paths if p.endswith(module_path)]
        if len(cands) == 1:
            return cands[0]

        # Layer-prefix anchored repair (e.g., layers.0.* -> model.layers.0.*)
        lp2 = getattr(student_arch, 'layer_prefix', '') or ''
        if lp2 and lp2.endswith('layers') and module_path.startswith('layers.'):
            root = lp2.split('.', 1)[0]
            cand = root + '.' + module_path
            if cand in student_module_paths:
                return cand

        return module_path

    _lora_key_re = re.compile(r'^(?:base_model\.model\.)?(?P<module_path>.+?)\.lora_(?P<ab>[AB])(?:\.(?P<adapter>[^.]+))?\.weight$')

    if student_module_paths:
        # Canonicalize standard format (no adapter name)
        canon_state: Dict[str, torch.Tensor] = {}
        for _k, _v in normalized_lora_state.items():
            mm = _lora_key_re.match(_k)
            if not mm:
                continue
            mp = mm.group('module_path')
            ab = mm.group('ab')
            mp2 = _canon_module_path(mp)
            # Standard format: lora_A.weight (no adapter name)
            nk = f"{base_prefix}{mp2}.lora_{ab}.weight"
            canon_state[nk] = _v
        if canon_state:
            normalized_lora_state = canon_state

        # Canonicalize named adapter format (with .default)
        canon_state_named: Dict[str, torch.Tensor] = {}
        for _k, _v in normalized_lora_state_named.items():
            mm = _lora_key_re.match(_k)
            if not mm:
                continue
            mp = mm.group('module_path')
            ab = mm.group('ab')
            ad = mm.group('adapter') or 'default'
            mp2 = _canon_module_path(mp)
            # Named adapter format: lora_A.default.weight
            nk = f"{base_prefix}{mp2}.lora_{ab}.{ad}.weight"
            canon_state_named[nk] = _v
        if canon_state_named:
            normalized_lora_state_named = canon_state_named

        # Canonicalize rank_info keys to match rewritten module paths (best-effort)
        try:
            canon_rank: Dict[str, int] = {}
            for rk, rv in (rank_info or {}).items():
                rrk = rk
                if rrk.startswith(base_prefix):
                    rrk = rrk[len(base_prefix):]
                rrk = _canon_module_path(rrk)
                canon_rank[base_prefix + rrk] = int(rv)
            if canon_rank:
                rank_info = canon_rank
        except Exception:
            pass

    # -----------------------------
    # Infer target_modules from normalized keys
    # -----------------------------
    target_modules = set()
    key_re = re.compile(r'^(?:base_model\.model\.)?(?P<module_path>.+?)\.lora_[AB](?:\.(?P<adapter>[^.]+))?\.weight$')

    for k in normalized_lora_state.keys():
        m = key_re.match(k)
        if not m:
            continue
        module_path = m.group('module_path')

        # Remove wrapper noise before layer prefix (keeps config compatible with base model module names).
        if lp:
            needle = lp + '.'
            idx = module_path.find(needle)
            if idx != -1:
                module_path = module_path[idx:]

        leaf = module_path.split('.')[-1]
        if leaf:
            target_modules.add(leaf)

    target_modules = sorted(target_modules)

    # -----------------------------
    # Effective rank + rank_pattern (keys must be base-model module names, i.e., WITHOUT base_prefix)
    # -----------------------------
    effective_rank = (
        max(1, int(round(sum(rank_info.values()) / max(1, len(rank_info)))))
        if rank_info else int(getattr(config, 'min_rank', 8))
    )

    rank_pattern = {}
    for k, r in rank_info.items():
        kk = k
        # Rank keys may be stored with or without base_prefix; normalize to base-model module names.
        if kk.startswith(base_prefix):
            kk = kk[len(base_prefix):]

        if lp:
            needle = lp + '.'
            idx = kk.find(needle)
            if idx != -1:
                kk = kk[idx:]

        rank_pattern[kk] = int(r)

    # -----------------------------
    # Save adapter_model.safetensors (standard format: lora_A.weight)
    # -----------------------------
    weights_path = os.path.join(output_dir, 'adapter_model.safetensors')
    lora_weights_cpu = {k: v.detach().cpu().contiguous() for k, v in normalized_lora_state.items()}
    save_file(lora_weights_cpu, weights_path)

    # -----------------------------
    # Save adapter_model_with_adapter_name.safetensors (named adapter format: lora_A.default.weight)
    # -----------------------------
    weights_path_named = os.path.join(output_dir, 'adapter_model_with_adapter_name.safetensors')
    lora_weights_cpu_named = {k: v.detach().cpu().contiguous() for k, v in normalized_lora_state_named.items()}
    save_file(lora_weights_cpu_named, weights_path_named)

    print(f"   📁 Saved two adapter weight files:")
    print(f"      - adapter_model.safetensors (standard: lora_A.weight)")
    print(f"      - adapter_model_with_adapter_name.safetensors (named: lora_A.default.weight)")

    # -----------------------------
    # Save adapter_config.json (PEFT)
    # -----------------------------
    _raw_alpha = getattr(config, 'lora_alpha', 0)
    try:
        lora_alpha = int(_raw_alpha) if _raw_alpha is not None else 0
    except (TypeError, ValueError):
        lora_alpha = 0
    if lora_alpha <= 0:
        # Common convention: alpha ~= r
        lora_alpha = int(effective_rank)

    adapter_cfg = {
        'base_model_name_or_path': str(getattr(config, 'student', getattr(config, 'student_path', 'student_model'))),
        'bias': 'none',
        'fan_in_fan_out': False,
        'inference_mode': True,
        'init_lora_weights': True,
        'lora_alpha': int(lora_alpha),
        'lora_dropout': float(getattr(config, 'lora_dropout', 0.0)),
        'modules_to_save': [],
        'peft_type': 'LORA',
        'r': int(effective_rank),
        'target_modules': target_modules,
        'task_type': 'CAUSAL_LM',
    }

    if rank_pattern:
        adapter_cfg['rank_pattern'] = rank_pattern

    config_path = os.path.join(output_dir, 'adapter_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(adapter_cfg, f, indent=2, ensure_ascii=False)

    # Optional: write minimal meta for debugging / reproducibility (does not affect PEFT).
    meta_path = os.path.join(output_dir, 'distill_adapter_meta.json')
    meta = {
        'method': 'A',
        'peft_base_prefix': base_prefix,
        'student_layer_prefix': lp,
        'num_lora_tensors': int(len(normalized_lora_state)),
        'num_target_modules': int(len(target_modules)),
        'weight_files': {
            'standard': 'adapter_model.safetensors',
            'named_adapter': 'adapter_model_with_adapter_name.safetensors',
        },
        'key_formats': {
            'standard': 'base_model.model.{module}.lora_A.weight',
            'named_adapter': 'base_model.model.{module}.lora_A.default.weight',
        },
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

class AcceleratedDistiller:
    """GPU-accelerated distillation with memory management."""
    
    def __init__(self, config: DistillConfig, device: torch.device,
                 teacher_arch: ArchitectureInfo, student_arch: ArchitectureInfo):
        self.config = config
        self.device = device
        self.teacher_arch = teacher_arch
        self.student_arch = student_arch
        
        # Check for cross-architecture distillation
        self.is_cross_arch = (
            teacher_arch.hidden_size != student_arch.hidden_size or
            teacher_arch.num_hidden_layers != student_arch.num_hidden_layers
        )
        
        if self.is_cross_arch and config.verbose:
            print(f"\n⚠️  Cross-architecture distillation detected!")
            print(f"   Teacher: {teacher_arch.num_hidden_layers} layers, {teacher_arch.hidden_size} hidden")
            print(f"   Student: {student_arch.num_hidden_layers} layers, {student_arch.hidden_size} hidden")
        
        # FIX #3: Auto-adjust DARE for cross-architecture
        if self.is_cross_arch and config.auto_adjust_dare and config.use_dare:
            # Reduce drop rate for cross-arch to preserve more info
            old_rate = config.dare_drop_rate
            config.dare_drop_rate = min(0.1, config.dare_drop_rate)
            if config.verbose and old_rate != config.dare_drop_rate:
                print(f"   Auto-adjusted DARE drop_rate: {old_rate} → {config.dare_drop_rate}")
        
        self.memory_manager = GPUMemoryManager(
            device, config.gpu_memory_fraction, config.verbose
        )
        
        self.stream_manager = CUDAStreamManager(
            device, config.num_cuda_streams
        ) if config.use_cuda_streams and device.type == "cuda" else None
        
        if config.offload_strategy == "auto":
            stats = self.memory_manager.get_stats()
            if stats["total_gb"] < 16:
                strategy = OffloadStrategy.CPU
            else:
                strategy = OffloadStrategy.NONE
        else:
            strategy = OffloadStrategy[config.offload_strategy.upper()]
        
        self.offloader = LayerOffloader(
            strategy=strategy,
            device=device,
            offload_dir=config.offload_dir,
            use_pinned=config.use_pinned_memory,
            max_cpu_gb=config.max_cpu_memory_gb,
            prefetch_count=config.prefetch_layers,
            verbose=config.verbose
        )
        
        self.svd_engine = GPUAcceleratedSVD(
            device=device,
            memory_manager=self.memory_manager,
            stream_manager=self.stream_manager,
            use_mixed_precision=config.use_mixed_precision,
            chunk_threshold_mb=config.chunk_threshold_mb,
            chunk_size_mb=config.chunk_size_mb,
            verbose=config.verbose,
            svd_mode=getattr(config, 'svd_mode', 'auto'),
            randomized_iter=getattr(config, 'svd_randomized_iter', 2),
            randomized_oversamples=getattr(config, 'svd_randomized_oversamples', 8),
            auto_min_dim=getattr(config, 'svd_auto_min_dim', 1024),
            auto_full_rank_ratio=getattr(config, 'svd_auto_full_rank_ratio', 0.6),
            auto_lowrank=getattr(config, 'svd_auto_lowrank', 'randomized'),
            adaptive_block_size=getattr(config, 'svd_adaptive_block_size', 32),
            adaptive_n_test=getattr(config, 'svd_adaptive_n_test', 8),
            drf_steps=int(getattr(config, 'svd_drf_steps', 1)),
            drf_theta=float(getattr(config, 'svd_drf_theta', 0.5)),
            drf_resid_eps=float(getattr(config, 'svd_drf_resid_eps', 1e-8)),
            aurora_steps=int(getattr(config, 'svd_aurora_steps', 1)),
            aurora_order=int(getattr(config, 'svd_aurora_order', 2)),
            aurora_theta1=float(getattr(config, 'svd_aurora_theta1', 0.5)),
            aurora_theta2=float(getattr(config, 'svd_aurora_theta2', 0.25)),
            aurora_resid_eps=float(getattr(config, 'svd_aurora_resid_eps', 1e-8)),
            aurora_level2_keep=float(getattr(config, 'svd_aurora_level2_keep', 1.0)),
            aurora_level1_keep=float(getattr(config, 'svd_aurora_level1_keep', 1.0)),
            aurora_min_steps=int(getattr(config, 'svd_aurora_min_steps', 1)),
            aurora_early_stop=float(getattr(config, 'svd_aurora_early_stop', 0.0)),
            aurora_early_stop_patience=int(getattr(config, 'svd_aurora_early_stop_patience', 2)),
            aurora_early_stop_check_every=int(getattr(config, 'svd_aurora_early_stop_check_every', 1)),
            aurora_core_solver=str(getattr(config, 'svd_aurora_core_solver', 'svd')),
            aurora_svd_driver=str(getattr(config, 'svd_aurora_svd_driver', 'gesvdj')),
        )
        
        self.lora_weights: Dict[str, torch.Tensor] = {}
        self.rank_info: Dict[str, int] = {}
        # MoE->Dense synthesis cache (teacher MoE -> student dense)
        self._teacher_moe_style: Optional[str] = None
        self._teacher_has_mlp_experts: bool = False
        self._teacher_has_block_sparse_moe: bool = False
        self._moe_dense_cache: Dict[Tuple[str, int, str], torch.Tensor] = {}
        self.stats = defaultdict(int)
        self.calib_stats = None  # optional: {module_name: (in_rms, out_rms)}
    
    def load_tensors_with_prefetch(self,
                                   keys: List[str],
                                   model_folder: str,
                                   weight_map: Dict[str, str],
                                   next_keys: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """Load tensors with optional prefetching."""
        shards = defaultdict(list)
        for k in keys:
            if k in weight_map:
                shards[weight_map[k]].append(k)
        
        tensors = {}
        
        for shard, shard_keys in shards.items():
            path = os.path.join(model_folder, shard)
            with safe_open(path, framework="pt") as f:
                for k in shard_keys:
                    try:
                        t = f.get_tensor(k)

                        # Record original dtype before any scaling/casting.
                        was_float8 = _is_float8_dtype(t.dtype)

                        # 1) TransformerEngine FP8 convention: "<tensor>_scale_inv"
                        #    Dequant: fp16/bf16/fp32 = fp8 * scale_inv
                        applied_te_scale = False
                        if was_float8:
                            scale_inv_key = f"{k}_scale_inv"
                            scale_inv = None
                            if scale_inv_key in f.keys():
                                scale_inv = f.get_tensor(scale_inv_key)
                            elif scale_inv_key in weight_map:
                                other_shard = weight_map.get(scale_inv_key)
                                if other_shard and other_shard != shard:
                                    other_path = os.path.join(model_folder, other_shard)
                                    try:
                                        with safe_open(other_path, framework="pt") as g:
                                            if scale_inv_key in g.keys():
                                                scale_inv = g.get_tensor(scale_inv_key)
                                    except Exception:
                                        scale_inv = None
                            if scale_inv is not None:
                                # DeepSeek/MiniMax block-wise FP8 uses per-group scale_inv (kernel: y = x * scale_inv).
                                # TransformerEngine-style checkpoints also use a _scale_inv tensor; both are handled here.
                                fp8_bs = int(getattr(self.config, "fp8_block_size", 128))
                                t = _fp8_dequantize(t, scale_inv, block_size=fp8_bs)
                                t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
                                applied_te_scale = True

                        # 2) Generic "<tensor>.scales" convention (keep for other quant formats).
                        if not applied_te_scale:
                            scale_key = f"{k}.scales"
                            if scale_key in f.keys():
                                scale = f.get_tensor(scale_key)
                                t = t.float() * scale.float()
                                # Optional post-scale sanitization
                                t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

                        # Dequantize / promote float8 weights to a stable compute dtype.
                        if bool(getattr(self.config, "dequantize_float8", True)) and was_float8:
                            tgt_dtype = _resolve_dequant_dtype(self.config.dequantize_dtype)
                            t = t.to(dtype=tgt_dtype)
                            if torch.is_floating_point(t) and t.dtype in (torch.float16, torch.bfloat16):
                                t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
                        tensors[k] = t
                    except Exception:
                        continue
        
        if self.stream_manager:
            with self.stream_manager.stream_context():
                for k in tensors:
                    tensors[k] = tensors[k].to(self.device, non_blocking=True)
            self.stream_manager.synchronize_current()
        else:
            for k in tensors:
                tensors[k] = tensors[k].to(self.device)
        
        return tensors
    

    # -----------------------------
    # MoE -> Dense support (teacher MoE, student dense)
    # -----------------------------
    def _init_teacher_moe_style(self, teacher_weight_map: Dict[str, str]) -> None:
        """One-time detection of teacher MoE key layout."""
        if self._teacher_moe_style is not None:
            return
        # Detect common HF MoE layouts
        self._teacher_has_mlp_experts = any(".mlp.experts." in k for k in teacher_weight_map.keys())
        self._teacher_has_block_sparse_moe = any("block_sparse_moe.experts." in k for k in teacher_weight_map.keys())

        if self._teacher_has_mlp_experts and self._teacher_has_block_sparse_moe:
            self._teacher_moe_style = "both"
        elif self._teacher_has_mlp_experts:
            self._teacher_moe_style = "mlp_experts"
        elif self._teacher_has_block_sparse_moe:
            self._teacher_moe_style = "block_sparse_moe"
        else:
            self._teacher_moe_style = "none"

    def _estimate_moe_expert_weights(
        self,
        layer_prefix: str,
        layer_idx: int,
        teacher_weight_map: Dict[str, str],
        num_experts: int,
    ) -> torch.Tensor:
        """Estimate per-expert routing weights using router matrix norms (data-free)."""
        # Default: uniform
        w = torch.full((num_experts,), 1.0 / max(1, num_experts), dtype=torch.float32)

        # Candidate router keys (common HF names)
        router_keys: List[str] = []
        if self._teacher_has_mlp_experts:
            router_keys.append(f"{layer_prefix}.{layer_idx}.mlp.gate.weight")
        if self._teacher_has_block_sparse_moe:
            router_keys.append(f"{layer_prefix}.{layer_idx}.block_sparse_moe.gate.weight")
        # fallback variants
        if self._teacher_has_mlp_experts:
            router_keys.append(f"{layer_prefix}.{layer_idx}.mlp.router.weight")
        if self._teacher_has_block_sparse_moe:
            router_keys.append(f"{layer_prefix}.{layer_idx}.block_sparse_moe.router.weight")

        router_key = next((k for k in router_keys if k in teacher_weight_map), None)
        if router_key is None:
            return w

        try:
            t = self.load_tensors_with_prefetch([router_key], self.config.teacher_path, teacher_weight_map)[router_key]
            t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).float()
            if t.dim() == 2 and t.shape[0] == num_experts:
                scores = torch.linalg.vector_norm(t, ord=2, dim=1)
                # Softmax on scores for stability
                w = torch.softmax(scores, dim=0)
        except Exception:
            # Keep uniform
            pass

        return w

    def _synthesize_moe_dense_part(
        self,
        layer_prefix: str,
        layer_idx: int,
        part: str,
        teacher_weight_map: Dict[str, str],
    ) -> Optional[torch.Tensor]:
        """Synthesize a dense MLP matrix from teacher MoE experts for a single layer and part.

        part is one of: 'gate_proj', 'up_proj', 'down_proj'
        """
        self._init_teacher_moe_style(teacher_weight_map)
        if self._teacher_moe_style in (None, "none"):
            return None

        num_experts = int(getattr(self.teacher_arch, "num_experts", 0) or 0)
        if num_experts <= 0:
            # Try infer from keys (robust fallback)
            max_e = -1
            rx = re.compile(rf"^{re.escape(layer_prefix)}\.{layer_idx}\.(?:mlp\.experts|block_sparse_moe\.experts)\.(\d+)\.")
            for k in teacher_weight_map.keys():
                m = rx.match(k)
                if m:
                    max_e = max(max_e, int(m.group(1)))
            num_experts = max_e + 1
        if num_experts <= 0:
            return None

        # Cache key
        cache_key = (layer_prefix, int(layer_idx), str(part))
        if cache_key in self._moe_dense_cache:
            return self._moe_dense_cache[cache_key]

        # Map dense part -> possible expert param names
        part_candidates: List[str] = []
        if part == "gate_proj":
            part_candidates = ["gate_proj", "w1"]
        elif part == "up_proj":
            part_candidates = ["up_proj", "w3"]
        elif part == "down_proj":
            part_candidates = ["down_proj", "w2"]
        else:
            return None

        # Estimate routing weights (data-free, router norm)
        w = self._estimate_moe_expert_weights(layer_prefix, layer_idx, teacher_weight_map, num_experts)

        # Top-k experts (use existing config knob max_experts_to_blend; if <=0, keep all)
        topk = int(self.config.max_experts_to_blend or 0)
        if topk <= 0:
            topk = num_experts
        topk = min(topk, num_experts)

        # Select experts
        if topk < num_experts:
            vals, idxs = torch.topk(w, k=topk, largest=True)
            sel_experts = idxs.tolist()
            sel_w = (vals / (vals.sum() + 1e-12)).tolist()
        else:
            sel_experts = list(range(num_experts))
            sel_w = (w / (w.sum() + 1e-12)).tolist()

        # Build actual keys to load for selected experts
        expert_keys: List[str] = []
        expert_key_for: Dict[int, str] = {}

        def _first_existing(cands: List[str]) -> Optional[str]:
            for kk in cands:
                if kk in teacher_weight_map:
                    return kk
            return None

        for e in sel_experts:
            cands: List[str] = []
            if self._teacher_has_mlp_experts:
                for pn in part_candidates:
                    cands.append(f"{layer_prefix}.{layer_idx}.mlp.experts.{e}.{pn}.weight")
            if self._teacher_has_block_sparse_moe:
                for pn in part_candidates:
                    cands.append(f"{layer_prefix}.{layer_idx}.block_sparse_moe.experts.{e}.{pn}.weight")
            kk = _first_existing(cands)
            if kk is None:
                continue
            expert_key_for[e] = kk
            expert_keys.append(kk)

        if not expert_keys:
            return None

        # Load expert weights (streamed accumulation)
        tensors = self.load_tensors_with_prefetch(expert_keys, self.config.teacher_path, teacher_weight_map)

        acc: Optional[torch.Tensor] = None
        total_w = 0.0
        for e, we in zip(sel_experts, sel_w):
            kk = expert_key_for.get(e)
            if kk is None:
                continue
            t = tensors.get(kk, None)
            if t is None:
                continue
            # Accumulate in fp32 on CPU (keeps VRAM low, maximizes precision)
            t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0).float()
            if acc is None:
                acc = t.mul(float(we))
            else:
                acc.add_(t, alpha=float(we))
            total_w += float(we)

        if acc is None or total_w <= 0.0:
            return None

        # Normalize in case some experts were missing
        if abs(total_w - 1.0) > 1e-3:
            acc.div_(total_w)

        self._moe_dense_cache[cache_key] = acc
        return acc

    def _maybe_synthesize_teacher_from_moe(
        self,
        layer_prefix: str,
        t_floor: int,
        t_ceil: int,
        mapped_rest: str,
        teacher_weight_map: Dict[str, str],
    ) -> Dict[str, torch.Tensor]:
        """If teacher is MoE and student expects dense MLP weights, synthesize teacher matrices."""
        if self.teacher_arch.mlp_type != MLPType.MOE:
            return {}
        if self.student_arch.mlp_type == MLPType.MOE:
            return {}

        # Only for standard MLP projections
        part = None
        if ".mlp.gate_proj." in f".{mapped_rest}.":
            part = "gate_proj"
        elif ".mlp.up_proj." in f".{mapped_rest}.":
            part = "up_proj"
        elif ".mlp.down_proj." in f".{mapped_rest}.":
            part = "down_proj"
        else:
            return {}

        out: Dict[str, torch.Tensor] = {}
        tf = self._synthesize_moe_dense_part(layer_prefix, int(t_floor), part, teacher_weight_map)
        if tf is not None:
            out[f"{layer_prefix}.{t_floor}.<moe_synth>.mlp.{part}.weight"] = tf
        if int(t_ceil) != int(t_floor):
            tc = self._synthesize_moe_dense_part(layer_prefix, int(t_ceil), part, teacher_weight_map)
            if tc is not None:
                out[f"{layer_prefix}.{t_ceil}.<moe_synth>.mlp.{part}.weight"] = tc

        return out


    def process_layer(
        self,
        student_key: str,
        student_tensor: torch.Tensor,
        teacher_tensors: Dict[str, torch.Tensor],
        interp_w: float,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, int]]:
        """Process a single layer: compute delta, apply pruning, extract LoRA."""
        if not teacher_tensors:
            return None

        teacher_keys = list(teacher_tensors.keys())
        student_on_device = student_tensor.to(self.device, non_blocking=True)
        target_shape = student_tensor.shape

        # ---------------------------------------------------------------------
        # Cross-architecture safe path:
        #   When projection is needed (teacher matrix shape != student matrix shape),
        #   do NOT compute delta = (projected_teacher - student_full).
        #   That subtracts the student's residual space and pushes the weight toward
        #   a low-rank matrix, which is a common cause of degenerate/infinite output.
        #
        #   Instead:
        #     1) compute student's truncated SVD basis (U_s, S_s, V_s)
        #     2) compute teacher core in that basis: core_t = U_s^T Wt_resized V_s
        #     3) compute student core: core_s = diag(S_s)
        #     4) delta_core = core_t - core_s   (only inside the shared subspace)
        #     5) extract LoRA directly from delta_core without forming the full delta
        # ---------------------------------------------------------------------
        try_subspace = (
            self.config.use_svd_projection
            and self.is_cross_arch
            and (getattr(self.config, "subspace_delta", "off") != "off")
            and student_on_device.dim() == 2
            and len(target_shape) == 2
            and any(tuple(t.shape) != tuple(target_shape) for t in teacher_tensors.values())
        )

        if try_subspace:
            try:
                # Student basis (common space)
                U_s, S_s, V_s = compute_student_svd_basis(
                    student_on_device, self.svd_engine, self.config, self.device
                )
                k = int(S_s.numel())
                diag_Ss = torch.diag(S_s).to(device=self.device, dtype=torch.float32)

                # Compute teacher cores and build delta in core space
                if len(teacher_keys) == 1:
                    core_t = teacher_core_in_student_basis(
                        teacher_tensors[teacher_keys[0]],
                        U_s, V_s,
                        self.svd_engine, self.config, self.device
                    )
                    delta_core = core_t - diag_Ss
                    if self.config.use_dare:
                        delta_core = apply_dare(delta_core, self.config.dare_drop_rate)
                    if self.config.use_ties and getattr(self.config, "ties_trim_single", True):
                        delta_core = apply_ties_single(delta_core, self.config.ties_density)
                else:
                    core_floor = teacher_core_in_student_basis(
                        teacher_tensors[teacher_keys[0]],
                        U_s, V_s,
                        self.svd_engine, self.config, self.device
                    )
                    core_ceil = teacher_core_in_student_basis(
                        teacher_tensors[teacher_keys[1]],
                        U_s, V_s,
                        self.svd_engine, self.config, self.device
                    )
                    delta_floor = core_floor - diag_Ss
                    delta_ceil = core_ceil - diag_Ss

                    mix_w = float(interp_w)
                    if getattr(self.config, "interp_mode", "schedule") == "lsq":
                        mix_w = lsq_mix_weight_from_deltas(delta_floor, delta_ceil, mix_w)

                    if self.config.use_ties:
                        deltas = [delta_floor * (1.0 - mix_w), delta_ceil * mix_w]
                        if self.config.use_dare:
                            deltas = [apply_dare(d, self.config.dare_drop_rate) for d in deltas]
                        delta_core = apply_ties(deltas, self.config.ties_density)
                    else:
                        core_t = core_floor * (1.0 - mix_w) + core_ceil * mix_w
                        delta_core = core_t - diag_Ss
                        if self.config.use_dare:
                            delta_core = apply_dare(delta_core, self.config.dare_drop_rate)

                # Subspace-delta stabilization / mode control
                mode = getattr(self.config, "subspace_delta", "off")

                # Optional: match teacher core scale to student core scale
                if getattr(self.config, "subspace_norm_match", True):
                    core_s_norm = torch.norm(diag_Ss.float()) + 1e-8
                    core_t = (delta_core + diag_Ss).to(dtype=torch.float32)
                    core_t_norm = torch.norm(core_t) + 1e-8
                    scale_t = (core_s_norm / core_t_norm).clamp(0.25, 4.0)
                    core_t = core_t * scale_t
                    delta_core = (core_t - diag_Ss).to(dtype=delta_core.dtype)

                # Optional: suppress off-diagonal energy (improves stability)
                if mode in ("diag", "auto"):
                    diag_part = torch.diag(torch.diag(delta_core))
                    off_part = delta_core - diag_part
                    off_frac = float((torch.norm(off_part.float()) / (torch.norm(delta_core.float()) + 1e-8)).item())
                    thr = float(getattr(self.config, "subspace_offdiag_threshold", 0.55))
                    if mode == "diag" or (mode == "auto" and off_frac > thr):
                        delta_core = diag_part

                # Delta ratio clipping (use ||delta_core|| since U/V are orthonormal)
                if getattr(self.config, "max_delta_ratio", 0.0) and self.config.max_delta_ratio > 0:
                    denom_mode = getattr(self.config, "subspace_denom", "core")
                    if denom_mode == "full":
                        denom = torch.norm(student_on_device.float()) + 1e-8
                    else:
                        denom = torch.norm(diag_Ss.float()) + 1e-8
                    ratio = torch.norm(delta_core.float()) / denom
                    if ratio > self.config.max_delta_ratio:
                        scale_t = (self.config.max_delta_ratio / (ratio + 1e-12)).clamp(max=1.0)
                        scale = float(scale_t.item())
                        delta_core = delta_core * scale
                        print(
                            f"    [delta-clip] scaled by {scale:.4f} (ratio {float(ratio.item()):.4f} -> {self.config.max_delta_ratio:.4f})"
                        )

                # Rank selection hint (still respected when adaptive rank is off)
                is_attn = any(proj in student_key for proj in ["q_proj", "k_proj", "v_proj", "o_proj"])
                is_mlp = any(proj in student_key for proj in ["gate_proj", "up_proj", "down_proj"])

                rank = self.config.rank_default
                if is_attn and self.config.rank_attn is not None:
                    rank = self.config.rank_attn
                elif is_mlp and self.config.rank_mlp is not None:
                    rank = self.config.rank_mlp

                # Output dtype: match student for plug-and-play inference
                out_dtype = student_on_device.dtype if self.config.use_mixed_precision else torch.float32

                lora_A, lora_B, actual_rank = extract_lora_from_core_delta(
                    delta_core,
                    U_s,
                    V_s,
                    int(rank),
                    self.config,
                    out_dtype,
                    svd_engine=self.svd_engine,
                )
                if lora_A is None or lora_B is None or actual_rank <= 0:
                    raise RuntimeError("core-delta LoRA extraction failed")

                # Ensure shapes are correct (otherwise fall back to legacy path)
                if lora_A.shape[1] != target_shape[1] or lora_B.shape[0] != target_shape[0]:
                    raise RuntimeError(
                        f"core-delta LoRA shape mismatch: "
                        f"A{tuple(lora_A.shape)} B{tuple(lora_B.shape)} target{tuple(target_shape)}"
                    )

                return lora_A, lora_B, int(actual_rank)

            except Exception:
                # Fall back to legacy behavior
                pass

        # ---------------------------------------------------------------------
        # Legacy path: teacher and student already share shape, or projection is off.
        # ---------------------------------------------------------------------
        deltas = []
        # Float8 safety: always compute deltas in float32 to avoid unsupported float8 mixed-type promotion
        # (e.g., float8 - bf16) on some torch builds.
        student_f32 = student_on_device.float()
        for teacher_key in teacher_keys:
            teacher_tensor = teacher_tensors[teacher_key].to(self.device)

            # Cross-architecture projection (legacy): only used when caller didn't
            # trigger the safer subspace path above.
            if self.config.use_svd_projection and self.is_cross_arch:
                teacher_tensor = svd_projection(
                    teacher_tensor, target_shape, self.svd_engine, self.config, self.device, ref=student_on_device
                )
            else:
                teacher_tensor = self._project_tensor(teacher_tensor, target_shape)

            delta = teacher_tensor.to(self.device).float() - student_f32

            deltas.append(delta)

        # Apply interpolation or TIES between floor/ceil deltas
        mix_w = float(interp_w)
        if len(deltas) == 2 and getattr(self.config, "interp_mode", "schedule") == "lsq":
            mix_w = lsq_mix_weight_from_deltas(deltas[0], deltas[1], mix_w)
        if len(deltas) == 1:
            final_delta = deltas[0]
            if self.config.use_dare:
                final_delta = apply_dare(final_delta, self.config.dare_drop_rate)
            if self.config.use_ties and getattr(self.config, "ties_trim_single", True):
                final_delta = apply_ties_single(final_delta, self.config.ties_density)
        else:
            if self.config.use_ties:
                weighted_deltas = [deltas[0] * (1 - mix_w), deltas[1] * mix_w]
                if self.config.use_dare:
                    weighted_deltas = [apply_dare(d, self.config.dare_drop_rate) for d in weighted_deltas]
                final_delta = apply_ties(weighted_deltas, self.config.ties_density)
            else:
                final_delta = deltas[0] * (1 - mix_w) + deltas[1] * mix_w
                if self.config.use_dare:
                    final_delta = apply_dare(final_delta, self.config.dare_drop_rate)


        # Final delta ratio clipping (after interpolation/TIES/DARE)
        if getattr(self.config, "max_delta_ratio", 0.0) and self.config.max_delta_ratio > 0:
            denom = torch.norm(student_on_device.float()) + 1e-8
            ratio = torch.norm(final_delta.float()) / denom
            if ratio > self.config.max_delta_ratio:
                scale_t = (self.config.max_delta_ratio / (ratio + 1e-12)).clamp(max=1.0)
                scale = float(scale_t.item())
                final_delta = final_delta * scale
                print(
                    f"    [delta-clip] scaled by {scale:.4f} (ratio {float(ratio.item()):.4f} -> {self.config.max_delta_ratio:.4f})"
                )

        # Extract LoRA from final delta
        is_attn = any(proj in student_key for proj in ["q_proj", "k_proj", "v_proj", "o_proj"])
        is_mlp = any(proj in student_key for proj in ["gate_proj", "up_proj", "down_proj"])

        rank = self.config.rank_default
        if is_attn and self.config.rank_attn is not None:
            rank = self.config.rank_attn
        elif is_mlp and self.config.rank_mlp is not None:
            rank = self.config.rank_mlp

        # LoRA extraction (PiSSA). Optionally apply calibration-aware diagonal whitening.
        module_name = student_key[:-len(".weight")] if student_key.endswith(".weight") else student_key
        if getattr(self.config, "calibration_mode", "none") != "none" and self.calib_stats:
            lora_A, lora_B, actual_rank = extract_lora_pissa_calibrated(
                final_delta, rank, self.svd_engine, self.config, module_name, self.calib_stats, self.student_arch
            )
        else:
            lora_A, lora_B, actual_rank = extract_lora_pissa(final_delta, rank, self.svd_engine, self.config)

        if lora_A is None:
            return None

        return lora_A, lora_B, actual_rank
    def _project_tensor(self, src: torch.Tensor, 
                        target_shape: Tuple[int, ...]) -> torch.Tensor:
        """Simple projection (fallback)."""
        if src.shape == target_shape:
            return src

        # Avoid producing float8 tensors in projected outputs.
        out_dtype = _safe_weight_dtype(src.dtype, self.config, src.device)
        if out_dtype != src.dtype:
            src = src.to(dtype=out_dtype)
        
        if src.dim() == 1:
            out = torch.zeros(target_shape, device=src.device, dtype=src.dtype)
            copy_len = min(src.numel(), target_shape[0])
            out[:copy_len] = src[:copy_len]
            return out
        
        if src.dim() == 2:
            out = torch.zeros(target_shape, device=src.device, dtype=src.dtype)
            copy_rows = min(src.shape[0], target_shape[0])
            copy_cols = min(src.shape[1], target_shape[1])
            out[:copy_rows, :copy_cols] = src[:copy_rows, :copy_cols]
            return out
        
        return torch.zeros(target_shape, device=src.device, dtype=src.dtype)
    
    def run(self,
            student_keys: List[str],
            teacher_weight_map: Dict[str, str],
            student_weight_map: Dict[str, str],
            teacher_layers_map: Dict[str, List[int]],
            student_layers_map: Dict[str, List[int]],
            teacher_arch: ArchitectureInfo,
            student_arch: ArchitectureInfo,
            layer_name_mapping: Dict[str, str]) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
        """Run accelerated distillation."""
        s_layers = max(len(v) for v in student_layers_map.values()) if student_layers_map else 32
        t_layers = max(len(v) for v in teacher_layers_map.values()) if teacher_layers_map else 32
        
        # FIX #4: Skip embeddings for cross-arch with different vocab/hidden
        skip_embed_warning_shown = False
        
        processable_keys = []
        for k in student_keys:
            if not k.endswith(".weight"):
                continue
            if any(k.endswith(s) for s in [".bias"]):
                continue
            if any(tok in k.lower() for tok in NORM_TOKENS):
                continue
            
            # Special handling for embeddings in cross-arch
            if any(tok in k for tok in EMBED_TOKENS):
                if not self.config.include_embed_lm_head:
                    continue
                if self.is_cross_arch:
                    if not skip_embed_warning_shown and self.config.verbose:
                        print(f"\n⚠️  Skipping embeddings for cross-architecture distillation")
                        print(f"   (Teacher hidden={teacher_arch.hidden_size}, Student hidden={student_arch.hidden_size})")
                        skip_embed_warning_shown = True
                    continue
            
            if self.config.include_pattern and not re.search(self.config.include_pattern, k):
                continue
            if self.config.exclude_pattern and re.search(self.config.exclude_pattern, k):
                continue
            processable_keys.append(k)
        
        keys_by_shard = defaultdict(list)
        for k in processable_keys:
            if k in student_weight_map:
                keys_by_shard[student_weight_map[k]].append(k)
        
        pbar = tqdm(total=len(processable_keys),
                    desc="Distilling",
                    dynamic_ncols=True,
                    disable=not self.config.verbose)
        
        for shard_file, shard_keys in sorted(keys_by_shard.items()):
            pbar.set_description(f"Processing {os.path.basename(shard_file)}")
            
            for i in range(0, len(shard_keys), self.config.micro_batch_size):
                micro_keys = shard_keys[i:i + self.config.micro_batch_size]
                next_keys = shard_keys[i + self.config.micro_batch_size:i + 2*self.config.micro_batch_size]
                
                student_tensors = self.load_tensors_with_prefetch(
                    micro_keys, self.config.student_path, student_weight_map, next_keys
                )
                
                for sk in micro_keys:
                    pbar.update(1)
                    
                    st = student_tensors.get(sk)
                    if st is None:
                        continue
                    
                    seg = split_key(sk)
                    if seg is None:
                        continue
                    
                    s_prefix, token_name, s_idx, rest = seg
                    
                    t_floor, interp_w = teacher_idx_from_student_idx(
                        s_idx, s_layers, t_layers, 
                        self.config.map_schedule, self.config.sigmoid_k
                    )
                    t_ceil = min(t_floor + 1, t_layers - 1)
                    
                    mapped_rest = rest
                    for s_name, t_name in layer_name_mapping.items():
                        mapped_rest = mapped_rest.replace(s_name, t_name)
                    
                    t_prefix = s_prefix
                    key_floor = f"{t_prefix}.{t_floor}.{mapped_rest}"
                    key_ceil = f"{t_prefix}.{t_ceil}.{mapped_rest}"
                    
                    teacher_keys_to_load = []
                    for tk in [key_floor, key_ceil]:
                        if tk in teacher_weight_map:
                            teacher_keys_to_load.append(tk)
                    
                    teacher_tensors: Dict[str, torch.Tensor] = {}
                    if teacher_keys_to_load:
                        teacher_tensors = self.load_tensors_with_prefetch(
                            teacher_keys_to_load, self.config.teacher_path, teacher_weight_map
                        )
                    else:
                        teacher_tensors = self._maybe_synthesize_teacher_from_moe(
                            s_prefix, int(t_floor), int(t_ceil), mapped_rest, teacher_weight_map
                        )
                        if not teacher_tensors:
                            continue

                    result = self.process_layer(sk, st, teacher_tensors, interp_w)
                    
                    if result is not None:
                        lora_A, lora_B, actual_rank = result
                        
                        peft_key = to_peft_key(sk, self.student_arch)
                        
                        # Use standard format (without .default) - save_peft_adapter will normalize
                        self.lora_weights[f"{peft_key}.lora_A.weight"] = lora_A.detach().cpu().contiguous()
                        self.lora_weights[f"{peft_key}.lora_B.weight"] = lora_B.detach().cpu().contiguous()

                        self.rank_info[peft_key] = actual_rank
                        self.stats["processed"] += 1
                    
                    del teacher_tensors
                
                del student_tensors
                if i % (self.config.micro_batch_size * 4) == 0:
                    self.memory_manager.clear_cache()
        
        pbar.close()
        
        self.memory_manager.clear_cache()
        self.offloader.cleanup()
        
        return self.lora_weights, self.rank_info


# =============================================================================
#                              MAIN
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Universal SVD-LoRA Distillation v4.2.4 (Cross-Arch Innovations) — Plug-and-Play Truncated SVD (Low-VRAM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
v3.7 highlights:
  - Truncated / randomized SVD by default in 'auto' mode (avoids full SVD on large matrices).
  - Efficient SVD-based projection for cross-architecture distillation with bounded projection rank.
  - Keeps shard-wise streaming + offload (low VRAM) behavior from v3.2.

Examples:
  # Cross-architecture distillation (e.g., 72B -> 36B)
  python %(prog)s --teacher /path/to/teacher --student /path/to/student --output /path/to/lora \
      --adaptive-rank --min-rank 256 --max-rank 2048 --energy-threshold 0.95 \
      --svd-mode auto --projection-rank 256 --svd-projection \
      --ties --ties-density 0.3 --no-dare \
      --include "self_attn|mlp" --offload-strategy cpu --mixed-precision
"""
    )

    # Required
    p.add_argument("--teacher", required=True, help="Teacher model folder (HF safetensors)")
    p.add_argument("--student", required=True, help="Student model folder (HF safetensors)")
    p.add_argument("--output", required=True, help="Output directory for PEFT adapter")

    # Rank controls
    p.add_argument("--rank", type=int, default=64, help="Default LoRA rank (if not using rank-attn/mlp)")
    p.add_argument("--rank-attn", type=int, default=None, help="Override LoRA rank for attention projections")
    p.add_argument("--rank-mlp", type=int, default=None, help="Override LoRA rank for MLP projections")
    p.add_argument("--lora-alpha", type=int, default=None, help="LoRA alpha; default=r (scale=1)")
    p.add_argument("--alpha-mode", type=str, default="auto", choices=["auto","rank","fixed"], help="LoRA alpha policy: auto=rank if --lora-alpha not set; rank=alpha_pattern=r (scale~1); fixed=global alpha only")

    p.add_argument("--adaptive-rank", action="store_true", default=True, help="Enable per-layer adaptive rank")
    p.add_argument("--no-adaptive-rank", action="store_false", dest="adaptive_rank")
    p.add_argument("--energy-threshold", type=float, default=0.95, help="Energy threshold for adaptive rank (0-1)")
    p.add_argument("--min-rank", type=int, default=8, help="Minimum rank when adaptive rank is enabled")
    p.add_argument("--max-rank", type=int, default=256, help="Maximum rank when adaptive rank is enabled")

    # Layer mapping
    p.add_argument("--map-schedule", choices=["linear", "sigmoid"], default="linear",
                   help="Teacher->student layer mapping schedule")
    p.add_argument("--sigmoid-k", type=float, default=0.15, help="Sigmoid schedule slope")
    p.add_argument("--interp-mode", choices=["schedule", "lsq"], default="schedule",
                   help="Teacher floor/ceil mix: schedule=use mapping schedule weight; lsq=per-matrix least-squares mix weight in student space (clamped 0..1).")

    # Merge / regularization
    # DARE (off by default in v3.7)
    p.add_argument("--dare", action="store_true", default=False, help="Enable DARE (dropout on delta)")
    p.add_argument("--no-dare", action="store_false", dest="dare")
    p.add_argument("--dare-drop-rate", type=float, default=0.3, help="DARE dropout probability")
    p.add_argument("--auto-adjust-dare", action="store_true", default=True,
                   help="Auto-adjust DARE for cross-architecture (when DARE enabled)")
    p.add_argument("--no-auto-adjust-dare", action="store_false", dest="auto_adjust_dare")

    # TIES (on by default)
    p.add_argument("--ties", action="store_true", default=True, help="Enable TIES merging when combining deltas")
    p.add_argument("--no-ties", action="store_false", dest="ties")
    p.add_argument("--ties-density", type=float, default=0.3, help="Keep ratio for TIES trimming (0-1)")
    p.add_argument("--ties-trim-single", action="store_true", default=True,
                   help="Apply the TIES Trim step to single-delta cases (default: on; matches v3.7.15).")
    p.add_argument("--no-ties-trim-single", action="store_false", dest="ties_trim_single",
                   help="Disable TIES Trim for single-delta cases.")

    # Stability guard
    p.add_argument("--max-delta-ratio", type=float, default=0.35,
                   help="If >0, scales delta when ||delta||/||student|| exceeds this value")

    # SVD backend controls
    svd_group = p.add_argument_group("SVD Options")
    svd_group.add_argument("--svd-mode", choices=["auto", "full", "randomized", "krylov", "adaptive", "drf", "aurora"], default="auto",
                           help="SVD backend: auto|full|randomized|krylov|adaptive|drf|aurora")
    svd_group.add_argument("--svd-rand-iter", type=int, default=2,
                           help="Low-rank iterations: randomized=subspace iters (Alg. 4.4), krylov=depth, adaptive=per-block iters")
    svd_group.add_argument("--svd-rand-oversamples", type=int, default=8,
                           help="Low-rank oversamples (k = rank + oversamples) for randomized/krylov")
    svd_group.add_argument("--svd-auto-min-dim", type=int, default=1024,
                           help="In auto mode, prefer low-rank SVD when min(m,n) >= this")
    svd_group.add_argument("--svd-auto-full-rank-ratio", type=float, default=0.6,
                           help="In auto mode, use full SVD when rank >= ratio*min(m,n)")
    svd_group.add_argument("--svd-auto-lowrank", choices=["randomized", "krylov", "drf", "aurora"], default="randomized",
                           help="In auto mode (when low-rank is selected), choose randomized, krylov, drf, or aurora")
    svd_group.add_argument("--svd-adaptive-block-size", type=int, default=32,
                           help="(adaptive) columns added per adaptive iteration (Alg. 4.2)")
    svd_group.add_argument("--svd-adaptive-n-test", type=int, default=8,
                           help="(adaptive) test vectors for residual estimate (Alg. 4.2)")

    svd_group.add_argument("--svd-drf-steps", type=int, default=1,
                       help="(drf) Number of dual-residual folding refinement rounds (>=0).")
    svd_group.add_argument("--svd-drf-theta", type=float, default=0.5,
                       help="(drf) Residual scaling exponent in [0,1]. 0=no scaling; 1=full 1/sigma scaling.")
    svd_group.add_argument("--svd-drf-resid-eps", type=float, default=1e-8,
                       help="(drf) Numerical floor for sigma in residual scaling.")

    svd_group.add_argument("--svd-aurora-steps", type=int, default=1,
                       help="(aurora) Number of AURORA refinement rounds (>=0).")
    svd_group.add_argument("--svd-aurora-order", type=int, choices=[1, 2], default=2,
                       help="(aurora) Residual order: 1=first-order residual only; 2=adds second-order residual response via A/A^T.")
    svd_group.add_argument("--svd-aurora-theta1", type=float, default=0.5,
                       help="(aurora) Level-1 residual scaling exponent in [0,1]. 0=no scaling; 1=full 1/sigma scaling.")
    svd_group.add_argument("--svd-aurora-theta2", type=float, default=0.25,
                       help="(aurora) Level-2 residual scaling exponent in [0,1].")
    svd_group.add_argument("--svd-aurora-resid-eps", type=float, default=1e-8,
                       help="(aurora) Numerical floor for sigma in residual scaling.")
    svd_group.add_argument("--svd-aurora-level2-keep", type=float, default=1.0,
                       help="(aurora) Fraction (0..1] of components kept for the level-2 term; <1 keeps only the most residual-dominant components.")
    svd_group.add_argument("--svd-aurora-level1-keep", type=float, default=1.0,
                       help="(aurora) Fraction (0..1] of components kept for the level-1 residual term (Q1/P1). <1 prunes residual columns by energy to reduce QR/core sizes.")
    svd_group.add_argument("--svd-aurora-min-steps", type=int, default=1,
                       help="(aurora) Minimum number of refinement rounds to run when svd-aurora-steps>0 (guards against overly aggressive early stop).")
    svd_group.add_argument("--svd-aurora-early-stop", type=float, default=0.005,
                       help="(aurora) Relative improvement tolerance for early stopping (0 disables). Uses first-order residual decrease; lower = more steps (potentially higher quality).")
    svd_group.add_argument("--svd-aurora-early-stop-patience", type=int, default=2,
                       help="(aurora) Number of consecutive below-tolerance checks required to stop.")
    svd_group.add_argument("--svd-aurora-early-stop-check-every", type=int, default=1,
                       help="(aurora) Evaluate early-stop condition every N refinement rounds (reduces sync overhead when steps is large).")
    svd_group.add_argument("--svd-aurora-core-solver", choices=["svd", "eigh"], default="svd",
                       help="(aurora) Projected-core solver: svd=torch.linalg.svd; eigh=reconstruct via eigendecomposition of Gram matrix (may be faster on some GPUs).")
    svd_group.add_argument("--svd-aurora-svd-driver", type=str, default="gesvdj",
                       help="(aurora) CUDA SVD driver hint (e.g., gesvdj, gesvd). Ignored if unsupported by your torch build; set '' to disable.")

    # Cross-architecture projection controls
    cross_group = p.add_argument_group("Cross-Architecture Projection")
    cross_group.add_argument("--svd-projection", action="store_true", default=True,
                             help="Enable SVD-based teacher->student tensor projection")
    cross_group.add_argument("--no-svd-projection", action="store_false", dest="svd_projection")
    cross_group.add_argument("--projection-rank", type=int, default=256,
                             help="Max rank used for SVD projection (bounded for cost control)")
    cross_group.add_argument("--projection-min-rank", type=int, default=256,
                             help="Min rank for projection when projection-adaptive-rank is enabled")
    cross_group.add_argument("--projection-adaptive-rank", action="store_true", default=True,
                             help="Enable energy-based adaptive rank for projection SVD")
    cross_group.add_argument("--no-projection-adaptive-rank", action="store_false", dest="projection_adaptive_rank")
    cross_group.add_argument("--projection-energy-threshold", type=float, default=0.99,
                             help="Energy threshold for projection adaptive rank (0-1)")
    cross_group.add_argument("--projection-rand-iter", type=int, default=2,
                             help="Randomized SVD power iterations for projection")
    cross_group.add_argument("--projection-rand-oversamples", type=int, default=8,
                             help="Randomized SVD oversamples for projection")


    cross_group.add_argument("--subspace-delta", type=str, default="off",
                             choices=["off", "auto", "diag", "full"],
                             help="Cross-arch: subspace-delta mode. off=disable (use full-delta), diag=diagonal core only, full=full core, auto=diag if off-diagonal energy is high")
    cross_group.add_argument("--subspace-offdiag-threshold", type=float, default=0.55,
                             help="subspace-delta auto: if off-diagonal Frobenius fraction > threshold, fall back to diag")
    cross_group.add_argument("--subspace-denom", type=str, default="core", choices=["core", "full"],
                             help="subspace-delta delta-clip denominator: core=||diag(S_student)||, full=||W_student||")
    cross_group.add_argument("--no-subspace-norm-match", action="store_true",
                             help="Disable subspace core norm matching (scale teacher core to student core norm)")
    # Performance / memory
    perf = p.add_argument_group("Performance / Memory")
    perf.add_argument("--num-gpus", type=int, default=1)
    perf.add_argument("--gpu-memory-fraction", type=float, default=0.95)
    perf.add_argument("--no-cuda-streams", action="store_true", help="Disable CUDA streams")
    perf.add_argument("--cuda-streams", type=int, default=3)
    perf.add_argument("--svd-batch-size", type=int, default=1)
    perf.add_argument("--micro-batch", type=int, default=8, help="Tensors per IO micro-batch")
    perf.add_argument("--prefetch-layers", type=int, default=2)

    perf.add_argument("--offload-strategy", choices=["cpu", "disk", "none"], default="cpu")
    perf.add_argument("--offload-dir", type=str, default=None)
    perf.add_argument("--pinned-memory", action="store_true", default=True)
    perf.add_argument("--no-pinned-memory", action="store_false", dest="pinned_memory")
    perf.add_argument("--max-cpu-memory", type=float, default=64.0)

    perf.add_argument("--chunk-threshold", type=int, default=512, help="MB: use randomized SVD above this size")
    perf.add_argument("--chunk-size", type=int, default=128, help="MB: chunk size (reserved for future use)")

    # Precision
    p.add_argument("--mixed-precision", action="store_true", default=True)
    p.add_argument("--no-mixed-precision", action="store_false", dest="mixed_precision")

    # Quantized weights / FP8
    quant = p.add_argument_group("Quantization / FP8")
    quant.add_argument("--dequantize-float8", action="store_true", default=True,
                       help="If enabled, float8 weight tensors (e.g., FP8-trained checkpoints) are cast to a safe dtype during streaming load.")
    quant.add_argument("--no-dequantize-float8", action="store_false", dest="dequantize_float8",
                       help="Disable float8 streaming dequantization (not recommended unless you know your torch build supports your float8 ops).")
    quant.add_argument("--dequantize-dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                       help="Target dtype for float8 dequantization casts. auto=bf16 if supported else fp16.")
    quant.add_argument("--fp8-block-size", type=int, default=128,
                       help="Per-group FP8 block size for *_scale_inv dequantization (DeepSeek/MiniMax default: 128)")

    # Module selection
    p.add_argument("--include", type=str, default="self_attn|mlp",
                   help="Regex include filter for tensor keys")
    p.add_argument("--exclude", type=str, default="",
                   help="Regex exclude filter for tensor keys")
    p.add_argument("--include-embed", action="store_true", default=False,
                   help="Include embedding and lm_head weights (only safe when shapes match)")


    # Calibration-aware SVD (optional)
    calib = p.add_argument_group("Calibration-aware SVD (optional)")
    calib.add_argument("--calib-mode", choices=["none", "rms", "cov"], default="none",
                       help="Enable calibration-aware whitening before SVD. none=disable, rms=diagonal RMS whitening, cov=full-covariance (SVD-LLM-style) whitening.")
    calib.add_argument("--calib-data", type=str, default=None,
                       help="Calibration prompts file: .txt (1 prompt/line) or .jsonl ({\"text\":...}). If omitted, a small built-in prompt set is used.")
    calib.add_argument("--calib-format", choices=["auto", "txt", "jsonl", "json", "alpaca"], default="auto",
                       help="Calibration file format (auto uses extension).")
    calib.add_argument("--calib-alpaca-template", choices=["classic", "plain"], default="classic",
                       help="Alpaca prompt template for calibration (only for calib-format=alpaca/auto).")
    calib.add_argument("--calib-alpaca-include-output", action="store_true",
                       help="Include Alpaca output field in calibration forward (alpaca only).")
    calib.add_argument("--calib-max-samples", type=int, default=128,
                       help="Max number of calibration prompts to use.")
    calib.add_argument("--calib-max-length", type=int, default=256,
                       help="Max token length per prompt (tokenizer truncation).")
    calib.add_argument("--calib-padding", type=str, default="longest", choices=["longest","max_length"],
                       help="Padding strategy for calibration tokenization (longest minimizes padding).")
    calib.add_argument("--calib-no-attention-mask", action="store_true",
                       help="Disable attention_mask-based padding exclusion in calibration stats.")
    calib.add_argument("--calib-batch-size", type=int, default=1,
                       help="Batch size for calibration forward passes.")
    calib.add_argument("--calib-device", choices=["auto", "cuda", "cpu"], default="auto",
                       help="Device placement for calibration model forward (auto=cuda if available).")
    calib.add_argument("--calib-dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                       help="dtype for calibration forward (auto: bf16 if supported on cuda else fp16; cpu->fp32).")
    calib.add_argument("--calib-load", type=str, default=None,
                       help="Load precomputed calibration stats (.safetensors). Skips forward calibration.")
    calib.add_argument("--calib-save", type=str, default=None,
                       help="Save calibration stats (.safetensors) for reuse.")
    calib.add_argument("--calib-eps", type=float, default=1e-6,
                       help="Epsilon for whitening (clamp min).")
    calib.add_argument("--calib-cov-chunk-layers", type=int, default=4,
                       help="(cov mode) Number of layers per calibration pass (controls memory).")
    calib.add_argument("--calib-cov-max-dim", type=int, default=8192,
                       help="(cov mode) Max in_features dimension to collect full covariance (skip larger dims).")
    calib.add_argument("--calib-cov-groups", type=str, default="qkv,mlp,o",
                       help="(cov mode) Comma-separated groups to collect: qkv,o,mlp,down.")
    calib.add_argument("--calib-cov-store-dtype", choices=["fp16", "bf16", "fp32"], default="fp16",
                       help="(cov mode) dtype for storing Cholesky factors in calib-save.")
    calib.add_argument("--calib-no-in", action="store_true",
                       help="Disable input RMS collection (not recommended).")
    calib.add_argument("--calib-no-out", action="store_true",
                       help="Disable output RMS collection (reduces overhead; input-only whitening).")
    # MoE (kept for compatibility)
    p.add_argument("--moe-method", type=str, default="none")
    p.add_argument("--max-experts", type=int, default=8)

    # Misc
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--quiet", action="store_true")

    return p


def main():
    args = build_argparser().parse_args()
    
    config = DistillConfig(
        teacher_path=args.teacher,
        student_path=args.student,
        output_path=args.output,
        rank_default=args.rank,
        rank_attn=args.rank_attn,
        rank_mlp=args.rank_mlp,
        lora_alpha=args.lora_alpha,
        alpha_mode=args.alpha_mode,
        use_adaptive_rank=args.adaptive_rank,
        energy_threshold=args.energy_threshold,
        min_rank=args.min_rank,
        max_rank=args.max_rank,
        use_dare=args.dare,
        dare_drop_rate=args.dare_drop_rate,
        use_ties=args.ties,
        ties_density=args.ties_density,
        ties_trim_single=args.ties_trim_single,
        max_delta_ratio=args.max_delta_ratio,

                # Calibration-aware SVD
        calibration_mode=args.calib_mode,
        calib_data=args.calib_data,
        calib_format=args.calib_format,
        calib_alpaca_template=args.calib_alpaca_template,
        calib_alpaca_include_output=bool(args.calib_alpaca_include_output),
        calib_max_samples=args.calib_max_samples,
        calib_max_length=args.calib_max_length,
        calib_padding=args.calib_padding,
        calib_use_attention_mask=(not args.calib_no_attention_mask),
        calib_batch_size=args.calib_batch_size,
        calib_device=args.calib_device,
        calib_dtype=args.calib_dtype,
        calib_load=args.calib_load,
        calib_save=args.calib_save,
        calib_eps=args.calib_eps,
        calib_cov_chunk_layers=args.calib_cov_chunk_layers,
        calib_cov_max_dim=args.calib_cov_max_dim,
        calib_cov_groups=args.calib_cov_groups,
        calib_cov_store_dtype=args.calib_cov_store_dtype,
        calib_collect_in=(not args.calib_no_in),
        calib_collect_out=(not args.calib_no_out),

moe_merge_method=args.moe_method,
        max_experts_to_blend=args.max_experts,
        map_schedule=args.map_schedule,
        sigmoid_k=args.sigmoid_k,
        interp_mode=args.interp_mode,

        # SVD backend
        svd_mode=args.svd_mode,
        svd_randomized_iter=args.svd_rand_iter,
        svd_randomized_oversamples=args.svd_rand_oversamples,
        svd_auto_min_dim=args.svd_auto_min_dim,
        svd_auto_full_rank_ratio=args.svd_auto_full_rank_ratio,
        svd_auto_lowrank=args.svd_auto_lowrank,
        svd_adaptive_block_size=args.svd_adaptive_block_size,
        svd_adaptive_n_test=args.svd_adaptive_n_test,
        svd_drf_steps=args.svd_drf_steps,
        svd_drf_theta=args.svd_drf_theta,
        svd_drf_resid_eps=args.svd_drf_resid_eps,
        svd_aurora_steps=args.svd_aurora_steps,
        svd_aurora_order=args.svd_aurora_order,
        svd_aurora_theta1=args.svd_aurora_theta1,
        svd_aurora_theta2=args.svd_aurora_theta2,
        svd_aurora_resid_eps=args.svd_aurora_resid_eps,
        svd_aurora_level2_keep=args.svd_aurora_level2_keep,
        svd_aurora_level1_keep=args.svd_aurora_level1_keep,
        svd_aurora_min_steps=args.svd_aurora_min_steps,
        svd_aurora_early_stop=args.svd_aurora_early_stop,
        svd_aurora_early_stop_patience=args.svd_aurora_early_stop_patience,
        svd_aurora_early_stop_check_every=args.svd_aurora_early_stop_check_every,
        svd_aurora_core_solver=args.svd_aurora_core_solver,
        svd_aurora_svd_driver=args.svd_aurora_svd_driver,

        # Cross-arch projection
        use_svd_projection=args.svd_projection,
        projection_rank=args.projection_rank,
        projection_min_rank=args.projection_min_rank,
        projection_adaptive_rank=args.projection_adaptive_rank,
        projection_energy_threshold=args.projection_energy_threshold,
        projection_randomized_iter=args.projection_rand_iter,
        projection_randomized_oversamples=args.projection_rand_oversamples,
        subspace_delta=args.subspace_delta,
        subspace_offdiag_threshold=args.subspace_offdiag_threshold,
        subspace_denom=args.subspace_denom,
        subspace_norm_match=(not args.no_subspace_norm_match),

        num_gpus=args.num_gpus,
        gpu_memory_fraction=args.gpu_memory_fraction,
        use_cuda_streams=not args.no_cuda_streams,
        num_cuda_streams=args.cuda_streams,
        use_mixed_precision=args.mixed_precision,
        dequantize_float8=args.dequantize_float8,
        dequantize_dtype=args.dequantize_dtype,
        fp8_block_size=args.fp8_block_size,
        svd_batch_size=args.svd_batch_size,
        offload_strategy=args.offload_strategy,
        offload_dir=args.offload_dir,
        prefetch_layers=args.prefetch_layers,
        use_pinned_memory=args.pinned_memory,
        max_cpu_memory_gb=args.max_cpu_memory,
        chunk_threshold_mb=args.chunk_threshold,
        chunk_size_mb=args.chunk_size,
        micro_batch_size=args.micro_batch,
        include_pattern=args.include,
        exclude_pattern=args.exclude,
        include_embed_lm_head=args.include_embed,
        auto_adjust_dare=args.auto_adjust_dare,
        seed=args.seed,
        verbose=not args.quiet,
    )
    
    seed_all(config.seed)
    
    print("=" * 80)
    print("Universal SVD-LoRA Distillation v4.2.4 (Cross-Arch Innovations)")
    print("Plug-and-Play Truncated SVD (Low-VRAM) + Low-VRAM Offload")
    print("=" * 80)
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"\n🖥️  GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        try:
            torch.backends.cuda.preferred_linalg_library("cusolver")
        except Exception:
            pass
    else:
        device = torch.device("cpu")
        print("\n⚠️  No GPU available, using CPU")
    
    print("\n📂 Loading model indices...")
    teacher_weight_map = read_index_map(config.teacher_path)
    student_weight_map = read_index_map(config.student_path)
    
    teacher_keys = sorted(teacher_weight_map.keys())
    student_keys = sorted(student_weight_map.keys())
    
    print(f"   Teacher: {len(teacher_keys)} tensors")
    print(f"   Student: {len(student_keys)} tensors")
    
    print("\n🔍 Detecting architectures...")
    if HAS_TRANSFORMERS:
        teacher_arch = detect_architecture_from_config(config.teacher_path)
        student_arch = detect_architecture_from_config(config.student_path)
    else:
        teacher_arch = detect_architecture_from_weights(teacher_keys)
        student_arch = detect_architecture_from_weights(student_keys)
    
    teacher_arch_w = detect_architecture_from_weights(teacher_keys)
    if teacher_arch_w.mlp_type == MLPType.MOE:
        teacher_arch.mlp_type = MLPType.MOE
        teacher_arch.num_experts = teacher_arch_w.num_experts
    
    print(f"\n📊 Teacher: {teacher_arch.attention_type.value}, {teacher_arch.mlp_type.value}")
    print(f"   Layers: {teacher_arch.num_hidden_layers}, Hidden: {teacher_arch.hidden_size}")
    if teacher_arch.num_experts > 0:
        print(f"   Experts: {teacher_arch.num_experts}")
    
    print(f"\n📊 Student: {student_arch.attention_type.value}, {student_arch.mlp_type.value}")
    print(f"   Layers: {student_arch.num_hidden_layers}, Hidden: {student_arch.hidden_size}")
    
    teacher_layers_map = scan_layers(teacher_keys)
    student_layers_map = scan_layers(student_keys)
    
    layer_mapping = create_layer_mapping(teacher_arch, student_arch)
    if layer_mapping and config.verbose:
        print(f"\n📋 Layer Name Mapping: {layer_mapping}")
    
    print(f"\n🚀 Starting GPU-accelerated distillation...")
    print(f"   CUDA Streams: {config.num_cuda_streams if config.use_cuda_streams else 'Disabled'}")
    print(f"   Mixed Precision: {config.use_mixed_precision}")
    print(f"   Offload Strategy: {config.offload_strategy}")
    print(f"   Prefetch Layers: {config.prefetch_layers}")
    print(f"   DARE: {config.use_dare} (drop_rate={config.dare_drop_rate})")
    print(f"   TIES: {config.use_ties} (density={config.ties_density}, trim_single={getattr(config, 'ties_trim_single', True)})")
    print(f"   Adaptive Rank: {config.use_adaptive_rank} (threshold={config.energy_threshold})")
    print(f"   Rank Range: {config.min_rank} - {config.max_rank}")
    print(f"   SVD Projection: {config.use_svd_projection}")

    # Calibration-aware SVD (optional)
    calib_stats: Dict[str, CalibStatsEntry] = {}
    if getattr(config, "calibration_mode", "none") != "none":
        print(f"\n🧪 Calibration-aware SVD: {config.calibration_mode}")
        if getattr(config, "calib_load", None):
            calib_stats = load_calibration_stats(config.calib_load)
            print(f"   Loaded calibration stats: {len(calib_stats)} modules")
            print(f"   From: {config.calib_load}")
        else:
            calib_stats = collect_calibration_stats(config.student_path, student_arch, config)
            print(f"   Collected calibration stats: {len(calib_stats)} modules")
            if getattr(config, "calib_save", None):
                save_calibration_stats(calib_stats, config.calib_save)
                print(f"   Saved calibration stats to: {config.calib_save}")

    start_time = time.time()

    distiller = AcceleratedDistiller(config, device, teacher_arch, student_arch)
    distiller.calib_stats = calib_stats if calib_stats else None
    lora_weights, rank_info = distiller.run(
        student_keys=student_keys,
        teacher_weight_map=teacher_weight_map,
        student_weight_map=student_weight_map,
        teacher_layers_map=teacher_layers_map,
        student_layers_map=student_layers_map,
        teacher_arch=teacher_arch,
        student_arch=student_arch,
        layer_name_mapping=layer_mapping
    )
    
    elapsed = time.time() - start_time
    
    if not lora_weights:
        print("❌ CRITICAL: No LoRA weights were generated!")
        return 1
    
    save_peft_adapter(lora_weights, config.output_path, config, student_arch, rank_info)
    
    print(f"\n✅ Generated {len(lora_weights) // 2} LoRA pairs in {elapsed:.1f}s")
    print(f"   Throughput: {len(lora_weights) // 2 / elapsed:.1f} layers/sec")
    
    if rank_info:
        ranks = list(rank_info.values())
        print(f"   Rank range: {min(ranks)}-{max(ranks)} (avg: {sum(ranks)/len(ranks):.1f})")

    # AURORA telemetry (if used)
    try:
        calls = int(getattr(distiller.svd_engine, "_aurora_calls", 0) or 0)
        if calls > 0:
            total_steps = int(getattr(distiller.svd_engine, "_aurora_steps_used_total", 0) or 0)
            print(f"   AURORA avg refinement steps used: {total_steps / max(1, calls):.2f} (calls={calls})")
    except Exception:
        pass

    print("\n🎉 Distillation Complete!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit(main())