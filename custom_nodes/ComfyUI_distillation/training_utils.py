#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Utilities for Cross-Architecture Distillation

Provides:
- Learning rate schedulers
- Loss balancing strategies
- Gradient accumulation
- Memory-efficient training
- Checkpoint management
"""

import os
import gc
import math
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import json

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
#                         LEARNING RATE SCHEDULERS
# =============================================================================

class LRSchedulerType(Enum):
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WARMUP = "cosine_warmup"
    POLYNOMIAL = "polynomial"
    ONE_CYCLE = "one_cycle"


def get_lr_scheduler(
    scheduler_type: LRSchedulerType,
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    warmup_steps: int = 0,
    min_lr: float = 0.0,
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create a learning rate scheduler."""

    if scheduler_type == LRSchedulerType.CONSTANT:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    elif scheduler_type == LRSchedulerType.LINEAR:
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=min_lr / optimizer.param_groups[0]['lr'],
            total_iters=num_steps
        )

    elif scheduler_type == LRSchedulerType.COSINE:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_steps,
            eta_min=min_lr
        )

    elif scheduler_type == LRSchedulerType.COSINE_WARMUP:
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, num_steps - warmup_steps))
            return max(min_lr / optimizer.param_groups[0]['lr'], 0.5 * (1.0 + math.cos(math.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif scheduler_type == LRSchedulerType.POLYNOMIAL:
        power = kwargs.get("power", 2.0)
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, num_steps - warmup_steps))
            return max(min_lr / optimizer.param_groups[0]['lr'], (1 - progress) ** power)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif scheduler_type == LRSchedulerType.ONE_CYCLE:
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optimizer.param_groups[0]['lr'],
            total_steps=num_steps,
            pct_start=kwargs.get("pct_start", 0.3),
            anneal_strategy=kwargs.get("anneal_strategy", "cos"),
            final_div_factor=optimizer.param_groups[0]['lr'] / max(min_lr, 1e-8),
        )

    else:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)


# =============================================================================
#                         LOSS BALANCING
# =============================================================================

class LossBalancer:
    """Automatically balance multiple loss terms."""

    def __init__(
        self,
        loss_names: List[str],
        initial_weights: Optional[Dict[str, float]] = None,
        balancing_strategy: str = "uncertainty",  # uncertainty, gradnorm, ema
        ema_decay: float = 0.99,
    ):
        self.loss_names = loss_names
        self.balancing_strategy = balancing_strategy
        self.ema_decay = ema_decay

        # Initialize weights
        if initial_weights:
            self.weights = {k: initial_weights.get(k, 1.0) for k in loss_names}
        else:
            self.weights = {k: 1.0 for k in loss_names}

        # For uncertainty weighting (learned log variances)
        if balancing_strategy == "uncertainty":
            self.log_vars = {k: torch.nn.Parameter(torch.zeros(1)) for k in loss_names}

        # For EMA-based balancing
        self.loss_history = {k: [] for k in loss_names}
        self.ema_values = {k: 1.0 for k in loss_names}

    def compute_weighted_loss(
        self,
        losses: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute weighted combination of losses."""
        total_loss = torch.tensor(0.0, device=next(iter(losses.values())).device)
        weight_dict = {}

        if self.balancing_strategy == "uncertainty":
            for name in self.loss_names:
                if name in losses:
                    precision = torch.exp(-self.log_vars[name])
                    loss = precision * losses[name] + self.log_vars[name]
                    total_loss = total_loss + loss
                    weight_dict[name] = precision.item()

        elif self.balancing_strategy == "ema":
            for name in self.loss_names:
                if name in losses:
                    # Update EMA
                    loss_val = losses[name].item()
                    self.ema_values[name] = (
                        self.ema_decay * self.ema_values[name] +
                        (1 - self.ema_decay) * loss_val
                    )
                    # Inverse EMA weighting (down-weight high-loss terms)
                    weight = 1.0 / (self.ema_values[name] + 1e-8)
                    total_loss = total_loss + weight * losses[name]
                    weight_dict[name] = weight

        else:  # Fixed weights
            for name in self.loss_names:
                if name in losses:
                    weight = self.weights[name]
                    total_loss = total_loss + weight * losses[name]
                    weight_dict[name] = weight

        return total_loss, weight_dict

    def get_log_var_params(self) -> List[torch.nn.Parameter]:
        """Get learnable log variance parameters."""
        if self.balancing_strategy == "uncertainty":
            return list(self.log_vars.values())
        return []


# =============================================================================
#                         GRADIENT UTILITIES
# =============================================================================

class GradientAccumulator:
    """Handle gradient accumulation for memory-efficient training."""

    def __init__(
        self,
        accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = 1.0,
    ):
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.step_count = 0

    def should_accumulate(self) -> bool:
        """Check if we should continue accumulating."""
        return (self.step_count + 1) % self.accumulation_steps != 0

    def backward(self, loss: torch.Tensor, scale: bool = True):
        """Perform backward pass with optional scaling."""
        if scale:
            loss = loss / self.accumulation_steps
        loss.backward()
        self.step_count += 1

    def step(
        self,
        optimizer: torch.optim.Optimizer,
        model: Optional[nn.Module] = None,
    ) -> bool:
        """Perform optimizer step if accumulation is complete."""
        if self.should_accumulate():
            return False

        # Clip gradients
        if self.max_grad_norm is not None and model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()
        return True

    def reset(self):
        """Reset accumulation counter."""
        self.step_count = 0


def compute_gradient_stats(model: nn.Module) -> Dict[str, float]:
    """Compute gradient statistics for monitoring."""
    total_norm = 0.0
    param_count = 0
    grad_count = 0
    nan_count = 0
    inf_count = 0

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += p.numel()
            grad_count += 1

            if torch.isnan(p.grad).any():
                nan_count += 1
            if torch.isinf(p.grad).any():
                inf_count += 1

    total_norm = total_norm ** 0.5

    return {
        "grad_norm": total_norm,
        "grad_count": grad_count,
        "param_count": param_count,
        "nan_grads": nan_count,
        "inf_grads": inf_count,
    }


# =============================================================================
#                         MEMORY MANAGEMENT
# =============================================================================

class MemoryTracker:
    """Track GPU memory usage during training."""

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.peak_memory = 0
        self.checkpoints = []

    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        if self.device.type != "cuda":
            return {"allocated": 0, "reserved": 0, "free": 0}

        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
        max_memory = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)

        self.peak_memory = max(self.peak_memory, allocated)

        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "free_gb": max_memory - reserved,
            "peak_gb": self.peak_memory,
            "total_gb": max_memory,
        }

    def checkpoint(self, label: str):
        """Record a memory checkpoint."""
        stats = self.get_memory_stats()
        stats["label"] = label
        self.checkpoints.append(stats)

    def clear_cache(self):
        """Clear GPU cache."""
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def report(self) -> str:
        """Generate memory usage report."""
        lines = ["=== Memory Usage Report ==="]
        for cp in self.checkpoints:
            lines.append(
                f"{cp['label']}: {cp['allocated_gb']:.2f}GB allocated, "
                f"{cp['reserved_gb']:.2f}GB reserved"
            )
        stats = self.get_memory_stats()
        lines.append(f"Peak: {stats['peak_gb']:.2f}GB")
        return "\n".join(lines)


@contextmanager
def memory_efficient_mode():
    """Context manager for memory-efficient operations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
#                         CHECKPOINT MANAGEMENT
# =============================================================================

@dataclass
class TrainingState:
    """State of training for checkpointing."""
    step: int = 0
    epoch: int = 0
    best_loss: float = float('inf')
    losses_history: List[Dict[str, float]] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """Manage training checkpoints."""

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        save_best_only: bool = False,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.checkpoints = []

        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        state: TrainingState,
        model_state: Dict[str, torch.Tensor],
        optimizer_state: Optional[Dict] = None,
        scheduler_state: Optional[Dict] = None,
        is_best: bool = False,
    ):
        """Save a checkpoint."""
        checkpoint = {
            "state": {
                "step": state.step,
                "epoch": state.epoch,
                "best_loss": state.best_loss,
                "config": state.config,
            },
            "model_state": model_state,
        }

        if optimizer_state:
            checkpoint["optimizer_state"] = optimizer_state
        if scheduler_state:
            checkpoint["scheduler_state"] = scheduler_state

        # Determine filename
        if is_best:
            filename = "checkpoint_best.pt"
        else:
            filename = f"checkpoint_step_{state.step}.pt"

        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)

        if not is_best:
            self.checkpoints.append(filepath)

            # Cleanup old checkpoints
            while len(self.checkpoints) > self.max_checkpoints:
                old_checkpoint = self.checkpoints.pop(0)
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)

        logging.info(f"Saved checkpoint to {filepath}")

    def load(
        self,
        checkpoint_path: Optional[str] = None,
    ) -> Optional[Dict]:
        """Load a checkpoint."""
        if checkpoint_path is None:
            # Try to load best checkpoint
            best_path = os.path.join(self.checkpoint_dir, "checkpoint_best.pt")
            if os.path.exists(best_path):
                checkpoint_path = best_path
            elif self.checkpoints:
                checkpoint_path = self.checkpoints[-1]
            else:
                return None

        if not os.path.exists(checkpoint_path):
            logging.warning(f"Checkpoint not found: {checkpoint_path}")
            return None

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        logging.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint


# =============================================================================
#                         TRAINING METRICS
# =============================================================================

class MetricsLogger:
    """Log and track training metrics."""

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = log_dir
        self.metrics: Dict[str, List[float]] = {}
        self.step = 0

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics for a step."""
        if step is not None:
            self.step = step
        else:
            self.step += 1

        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)

    def get_average(self, metric_name: str, last_n: int = 100) -> float:
        """Get average of last N values for a metric."""
        if metric_name not in self.metrics:
            return 0.0
        values = self.metrics[metric_name][-last_n:]
        return sum(values) / len(values) if values else 0.0

    def save(self, filename: str = "metrics.json"):
        """Save metrics to file."""
        if self.log_dir:
            filepath = os.path.join(self.log_dir, filename)
            with open(filepath, "w") as f:
                json.dump(self.metrics, f, indent=2)

    def summary(self) -> str:
        """Generate summary of metrics."""
        lines = ["=== Training Metrics Summary ==="]
        for name, values in self.metrics.items():
            if values:
                avg = sum(values) / len(values)
                min_val = min(values)
                max_val = max(values)
                latest = values[-1]
                lines.append(
                    f"{name}: latest={latest:.4f}, avg={avg:.4f}, "
                    f"min={min_val:.4f}, max={max_val:.4f}"
                )
        return "\n".join(lines)


# =============================================================================
#                         EARLY STOPPING
# =============================================================================

class EarlyStopping:
    """Early stopping based on validation loss."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == "min" else float('-inf')
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """Check if training should stop."""
        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_value = float('inf') if self.mode == "min" else float('-inf')
        self.should_stop = False


# =============================================================================
#                         EXPONENTIAL MOVING AVERAGE
# =============================================================================

class EMAModel:
    """Exponential Moving Average of model parameters."""

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        device: Optional[torch.device] = None,
    ):
        self.decay = decay
        self.device = device

        # Create EMA parameters
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                if device:
                    self.shadow[name] = self.shadow[name].to(device)

    def update(self, model: nn.Module):
        """Update EMA parameters."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                new_average = (
                    self.decay * self.shadow[name] +
                    (1.0 - self.decay) * param.data
                )
                self.shadow[name] = new_average

    @contextmanager
    def apply_ema(self, model: nn.Module):
        """Context manager to temporarily apply EMA weights."""
        # Backup current weights
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

        try:
            yield
        finally:
            # Restore original weights
            for name, param in model.named_parameters():
                if name in self.backup:
                    param.data.copy_(self.backup[name])
            self.backup.clear()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get EMA state dict."""
        return dict(self.shadow)

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load EMA state dict."""
        self.shadow = dict(state_dict)
