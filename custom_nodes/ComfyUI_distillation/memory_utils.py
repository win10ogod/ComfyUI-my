"""
Memory Management Utilities for ComfyUI Distillation

This module provides memory management utilities for GPU/CPU operations:
- GPU Memory Manager
- CUDA Stream Manager
- Layer Offloader (CPU/Disk offloading with prefetching)
"""

import os
import re
import gc
import queue
import shutil
import tempfile
import threading
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from contextlib import contextmanager
from enum import Enum

import torch


class OffloadStrategy(Enum):
    """Offload strategy options."""
    NONE = "none"
    CPU = "cpu"
    DISK = "disk"
    AUTO = "auto"


@dataclass
class MemoryStats:
    """Memory statistics container."""
    total_gb: float = 0.0
    used_gb: float = 0.0
    free_gb: float = 0.0
    peak_gb: float = 0.0


class GPUMemoryManager:
    """
    GPU memory management utility.

    Provides memory statistics, allocation checks, and cache management.
    """

    def __init__(
        self,
        device: Union[str, torch.device],
        memory_fraction: float = 0.85,
        verbose: bool = True
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.memory_fraction = memory_fraction
        self.verbose = verbose
        self.peak_allocated = 0

        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            self.total_memory = torch.cuda.get_device_properties(self.device).total_memory
            self.max_memory = int(self.total_memory * memory_fraction)
        else:
            self.total_memory = 0
            self.max_memory = 0

    def get_stats(self) -> MemoryStats:
        """Get current GPU memory statistics."""
        if self.device.type != "cuda":
            return MemoryStats()

        allocated = torch.cuda.memory_allocated(self.device)
        self.peak_allocated = max(self.peak_allocated, allocated)

        return MemoryStats(
            total_gb=self.total_memory / (1024**3),
            used_gb=allocated / (1024**3),
            free_gb=(self.max_memory - allocated) / (1024**3),
            peak_gb=self.peak_allocated / (1024**3),
        )

    def can_allocate(self, size_bytes: int) -> bool:
        """Check if we can allocate the given size."""
        if self.device.type != "cuda":
            return True
        allocated = torch.cuda.memory_allocated(self.device)
        return (allocated + size_bytes) < self.max_memory

    def estimate_tensor_size(self, shape: tuple, dtype: torch.dtype = torch.float32) -> int:
        """Estimate tensor size in bytes."""
        numel = 1
        for dim in shape:
            numel *= dim

        dtype_sizes = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.float64: 8,
            torch.int32: 4,
            torch.int64: 8,
            torch.int16: 2,
            torch.int8: 1,
            torch.uint8: 1,
        }
        element_size = dtype_sizes.get(dtype, 4)
        return numel * element_size

    def clear_cache(self):
        """Clear GPU cache and run garbage collection."""
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def synchronize(self):
        """Synchronize CUDA device."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def reset_peak_stats(self):
        """Reset peak memory statistics."""
        self.peak_allocated = 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)


class CUDAStreamManager:
    """
    CUDA stream management for async operations.

    Provides context managers for stream-based parallel operations.
    """

    def __init__(self, device: torch.device, num_streams: int = 3):
        self.device = device
        self.num_streams = num_streams
        self.streams: List[torch.cuda.Stream] = []
        self.current_stream_idx = 0

        if device.type == "cuda":
            for _ in range(num_streams):
                self.streams.append(torch.cuda.Stream(device=device))

    @contextmanager
    def stream_context(self, stream_idx: Optional[int] = None):
        """Context manager for using a CUDA stream."""
        if not self.streams:
            yield
            return

        idx = stream_idx if stream_idx is not None else self.current_stream_idx
        self.current_stream_idx = (self.current_stream_idx + 1) % self.num_streams

        with torch.cuda.stream(self.streams[idx]):
            yield

    def get_stream(self, idx: Optional[int] = None) -> Optional[torch.cuda.Stream]:
        """Get a specific stream."""
        if not self.streams:
            return None
        if idx is None:
            idx = self.current_stream_idx
        return self.streams[idx % self.num_streams]

    def synchronize_current(self):
        """Synchronize current stream."""
        if self.streams:
            self.streams[self.current_stream_idx].synchronize()

    def synchronize_all(self):
        """Synchronize all streams."""
        for stream in self.streams:
            stream.synchronize()

    def wait_for_stream(self, stream_idx: int):
        """Wait for a specific stream to complete."""
        if self.streams and 0 <= stream_idx < len(self.streams):
            self.streams[stream_idx].synchronize()


class LayerOffloader:
    """
    Layer offloading utility for memory-efficient processing.

    Supports:
    - CPU offloading with pinned memory
    - Disk offloading for very large models
    - Async prefetching for efficient loading
    """

    def __init__(
        self,
        strategy: Union[OffloadStrategy, str],
        device: torch.device,
        offload_dir: Optional[str] = None,
        use_pinned: bool = True,
        max_cpu_gb: float = 32.0,
        prefetch_count: int = 2,
        verbose: bool = True
    ):
        if isinstance(strategy, str):
            strategy = OffloadStrategy(strategy.lower())

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

        self.prefetch_queue: queue.Queue = queue.Queue()
        self.stop_prefetch = threading.Event()
        self.prefetch_thread: Optional[threading.Thread] = None
        self.transfer_stream: Optional[torch.cuda.Stream] = None

        if device.type == "cuda" and strategy != OffloadStrategy.NONE:
            self.transfer_stream = torch.cuda.Stream(device=device)
            self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
            self.prefetch_thread.start()

    def _prefetch_worker(self):
        """Background worker for prefetching tensors."""
        while not self.stop_prefetch.is_set():
            try:
                key = self.prefetch_queue.get(timeout=0.1)
                if key not in self.gpu_cache:
                    self._load_to_gpu(key)
            except queue.Empty:
                continue
            except Exception as e:
                if self.verbose:
                    print(f"[LayerOffloader] Prefetch error for {key}: {e}")

    def _to_pinned(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor to pinned memory if possible."""
        if self.use_pinned and not tensor.is_pinned():
            try:
                pinned = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
                pinned.copy_(tensor)
                return pinned
            except Exception:
                return tensor
        return tensor

    def store(self, key: str, tensor: torch.Tensor, priority: str = "cpu"):
        """
        Store a tensor for later retrieval.

        Args:
            key: Unique identifier for the tensor
            tensor: Tensor to store
            priority: Where to store ('cpu', 'disk', 'gpu')
        """
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
        """Store tensor to disk."""
        safe_key = re.sub(r'[^\w\-.]', '_', key)
        path = os.path.join(self.offload_dir, f"{safe_key}.pt")
        torch.save(tensor.cpu(), path)
        self.disk_paths[key] = path

    def _load_to_gpu(self, key: str) -> Optional[torch.Tensor]:
        """Load tensor to GPU from cache or disk."""
        if key in self.gpu_cache:
            return self.gpu_cache[key]

        tensor = None
        if key in self.cpu_cache:
            tensor = self.cpu_cache[key]
        elif key in self.disk_paths:
            try:
                tensor = torch.load(self.disk_paths[key], weights_only=True)
            except TypeError:
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

    def get(
        self,
        key: str,
        prefetch_next: Optional[List[str]] = None
    ) -> Optional[torch.Tensor]:
        """
        Get a tensor, optionally prefetching the next ones.

        Args:
            key: Tensor identifier
            prefetch_next: List of keys to prefetch in background

        Returns:
            The requested tensor or None if not found
        """
        if prefetch_next:
            for next_key in prefetch_next[:self.prefetch_count]:
                if next_key not in self.gpu_cache:
                    self.prefetch_queue.put(next_key)
        return self._load_to_gpu(key)

    def evict(self, key: str):
        """Evict a tensor from GPU cache."""
        if key in self.gpu_cache:
            del self.gpu_cache[key]

    def evict_all_gpu(self):
        """Evict all tensors from GPU cache."""
        self.gpu_cache.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def get_all_keys(self) -> List[str]:
        """Get all stored tensor keys."""
        keys = set(self.gpu_cache.keys())
        keys.update(self.cpu_cache.keys())
        keys.update(self.disk_paths.keys())
        return list(keys)

    def has_key(self, key: str) -> bool:
        """Check if a key exists."""
        return (
            key in self.gpu_cache or
            key in self.cpu_cache or
            key in self.disk_paths
        )

    def get_storage_location(self, key: str) -> Optional[str]:
        """Get where a tensor is stored."""
        if key in self.gpu_cache:
            return "gpu"
        if key in self.cpu_cache:
            return "cpu"
        if key in self.disk_paths:
            return "disk"
        return None

    def cleanup(self):
        """Clean up resources."""
        self.stop_prefetch.set()
        if self.prefetch_thread:
            self.prefetch_thread.join(timeout=1.0)

        self.gpu_cache.clear()
        self.cpu_cache.clear()
        self.cpu_bytes_used = 0

        if os.path.exists(self.offload_dir):
            try:
                shutil.rmtree(self.offload_dir, ignore_errors=True)
            except Exception:
                pass

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False


class TensorPool:
    """
    Tensor pool for efficient memory reuse.

    Provides pre-allocated tensors to reduce allocation overhead.
    """

    def __init__(self, device: torch.device, max_tensors: int = 10):
        self.device = device
        self.max_tensors = max_tensors
        self.pool: Dict[tuple, List[torch.Tensor]] = {}
        self._lock = threading.Lock()

    def get(
        self,
        shape: tuple,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Get a tensor from the pool or allocate a new one."""
        key = (shape, dtype)

        with self._lock:
            if key in self.pool and self.pool[key]:
                tensor = self.pool[key].pop()
                tensor.zero_()
                return tensor

        return torch.zeros(shape, dtype=dtype, device=self.device)

    def release(self, tensor: torch.Tensor):
        """Release a tensor back to the pool."""
        key = (tuple(tensor.shape), tensor.dtype)

        with self._lock:
            if key not in self.pool:
                self.pool[key] = []

            if len(self.pool[key]) < self.max_tensors:
                self.pool[key].append(tensor.detach())

    def clear(self):
        """Clear the pool."""
        with self._lock:
            self.pool.clear()
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


def get_optimal_offload_strategy(
    model_size_gb: float,
    available_gpu_gb: float,
    available_cpu_gb: float = 64.0
) -> OffloadStrategy:
    """
    Determine the optimal offload strategy based on available resources.

    Args:
        model_size_gb: Estimated model size in GB
        available_gpu_gb: Available GPU memory in GB
        available_cpu_gb: Available CPU memory in GB

    Returns:
        Recommended OffloadStrategy
    """
    # If model fits in GPU, no offloading needed
    if model_size_gb * 1.2 < available_gpu_gb:  # 20% margin
        return OffloadStrategy.NONE

    # If model fits in CPU+GPU, use CPU offloading
    if model_size_gb * 1.1 < (available_gpu_gb + available_cpu_gb * 0.8):
        return OffloadStrategy.CPU

    # Otherwise use disk offloading
    return OffloadStrategy.DISK


def estimate_model_memory(
    num_params: int,
    dtype: torch.dtype = torch.float32,
    include_gradients: bool = False,
    include_optimizer: bool = False
) -> float:
    """
    Estimate memory requirements for a model.

    Args:
        num_params: Number of model parameters
        dtype: Parameter data type
        include_gradients: Include gradient memory
        include_optimizer: Include optimizer state memory

    Returns:
        Estimated memory in GB
    """
    dtype_sizes = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float64: 8,
    }
    bytes_per_param = dtype_sizes.get(dtype, 4)

    # Base model size
    memory_bytes = num_params * bytes_per_param

    # Gradients (same size as parameters)
    if include_gradients:
        memory_bytes *= 2

    # Optimizer states (Adam uses 2x for moment estimates)
    if include_optimizer:
        memory_bytes += num_params * 8  # 2 fp32 moments

    return memory_bytes / (1024 ** 3)
