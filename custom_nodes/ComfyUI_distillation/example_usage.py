#!/usr/bin/env python3
"""
ComfyUI Distillation - Python Usage Examples

This file demonstrates how to use the distillation modules programmatically.
"""

import torch
from pathlib import Path

# ==============================================================================
# Example 1: Quick Distillation with Factory Function
# ==============================================================================

def example_quick_distillation():
    """
    Simplest way to distill a teacher model to a student model.
    """
    from custom_nodes.ComfyUI_distillation import create_distillation_engine

    # Create engine with default settings
    engine = create_distillation_engine(
        rank=64,                    # LoRA rank
        svd_mode="aurora",          # Use AURORA-SVD (best quality)
        use_ties=True,              # Enable TIES trimming
        device="cuda:0",            # GPU device
    )

    # Load your models
    teacher_state_dict = torch.load("teacher_model.safetensors")
    student_state_dict = torch.load("student_model.safetensors")

    # Run distillation
    result = engine.distill_models(
        teacher_state_dict=teacher_state_dict,
        student_state_dict=student_state_dict,
    )

    # Save as LoRA adapter (PEFT format)
    engine.save_lora_adapter(result, "output_lora")

    # Print statistics
    print(f"Layers processed: {result.statistics['layers_processed']}")
    print(f"LoRA parameters: {result.statistics['lora_params']:,}")

    # Cleanup
    engine.cleanup()


# ==============================================================================
# Example 2: Full Configuration Control
# ==============================================================================

def example_full_config():
    """
    Distillation with full configuration control.
    """
    from custom_nodes.ComfyUI_distillation import (
        DistillationEngine,
        DistillationConfig,
    )

    # Create detailed configuration
    config = DistillationConfig(
        # Paths (optional, can be set later)
        teacher_path="",
        student_path="",
        output_path="./distilled_output",

        # LoRA settings
        rank_default=64,
        rank_attn=64,           # Rank for attention layers
        rank_mlp=32,            # Rank for MLP layers (can be different)
        lora_alpha=64,          # LoRA alpha scaling
        alpha_mode="rank",      # alpha = rank

        # Adaptive rank settings
        use_adaptive_rank=True,
        energy_threshold=0.95,  # Keep 95% of Frobenius energy
        min_rank=8,
        max_rank=256,

        # Delta regularization
        use_dare=False,         # DARE sparsification
        dare_drop_rate=0.3,
        use_ties=True,          # TIES trimming
        ties_density=0.3,       # Keep top 30%
        ties_trim_single=True,

        # Memory settings
        gpu_memory_fraction=0.95,
        use_cuda_streams=True,
        num_cuda_streams=3,
        use_mixed_precision=True,
        offload_strategy="cpu",     # "cpu", "disk", or "none"
        max_cpu_memory_gb=64.0,
        prefetch_layers=2,
        use_pinned_memory=True,

        # SVD settings
        svd_mode="aurora",          # "auto", "full", "randomized", "krylov", "drf", "aurora"
        svd_randomized_iter=2,
        svd_randomized_oversamples=8,

        # AURORA-SVD specific
        aurora_steps=1,
        aurora_order=2,             # 2nd order residual
        aurora_theta1=0.5,
        aurora_theta2=0.25,
        aurora_early_stop=0.005,

        # DRF-SVD specific
        drf_steps=1,
        drf_theta=0.5,

        # Projection settings
        use_svd_projection=True,
        projection_rank=256,
        projection_adaptive_rank=True,
        projection_energy_threshold=0.99,

        # FP8 handling
        dequantize_float8=True,
        fp8_block_size=128,

        # Module selection
        include_pattern="self_attn|mlp",
        exclude_pattern="",
        include_embed_lm_head=False,

        # Misc
        max_delta_ratio=0.35,
        seed=42,
        verbose=True,
    )

    # Create engine
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    engine = DistillationEngine(config, device)

    # Process models
    # ... (load and distill)

    return engine


# ==============================================================================
# Example 3: Using SVD Algorithms Directly
# ==============================================================================

def example_svd_algorithms():
    """
    Use SVD algorithms directly for custom applications.
    """
    from custom_nodes.ComfyUI_distillation import (
        GPUAcceleratedSVD,
        SVDConfig,
        GPUMemoryManager,
        CUDAStreamManager,
        compute_adaptive_rank,
    )

    device = torch.device("cuda:0")

    # Create components
    memory_manager = GPUMemoryManager(device, memory_fraction=0.9)
    stream_manager = CUDAStreamManager(device, num_streams=3)

    # Configure SVD
    svd_config = SVDConfig(
        svd_mode="aurora",
        aurora_steps=2,
        aurora_order=2,
        aurora_theta1=0.5,
        aurora_theta2=0.25,
        use_mixed_precision=True,
    )

    # Create SVD engine
    svd_engine = GPUAcceleratedSVD(
        device=device,
        memory_manager=memory_manager,
        stream_manager=stream_manager,
        config=svd_config,
    )

    # Use SVD on a matrix
    matrix = torch.randn(4096, 4096, device=device)

    # Fixed-rank SVD
    U, S, Vh = svd_engine.svd(matrix, rank=64)
    print(f"SVD result: U={U.shape}, S={S.shape}, Vh={Vh.shape}")

    # Energy-targeted SVD
    U, S, Vh = svd_engine.svd_energy(
        matrix,
        energy_threshold=0.95,
        min_rank=8,
        max_rank=256,
    )
    print(f"Adaptive rank: {S.shape[0]}")

    # Compute adaptive rank from singular values
    rank = compute_adaptive_rank(
        S,
        energy_threshold=0.99,
        min_rank=8,
        max_rank=128,
    )
    print(f"Computed rank for 99% energy: {rank}")


# ==============================================================================
# Example 4: Using Merge Strategies
# ==============================================================================

def example_merge_strategies():
    """
    Use DARE/TIES merge strategies for delta tensors.
    """
    from custom_nodes.ComfyUI_distillation import (
        apply_dare,
        apply_ties,
        apply_ties_single,
        DeltaMerger,
        compute_delta_stats,
        slerp_delta,
    )

    # Create sample delta tensors
    delta1 = torch.randn(4096, 4096)
    delta2 = torch.randn(4096, 4096)

    # DARE: Drop And REscale
    sparse_delta = apply_dare(
        delta1,
        drop_rate=0.7,      # Drop 70% of elements
        rescale=True,       # Rescale remaining
    )
    print(f"DARE sparsity: {(sparse_delta == 0).float().mean():.2%}")

    # TIES: Single delta trimming
    trimmed = apply_ties_single(
        delta1,
        density=0.3,        # Keep top 30% by magnitude
    )

    # TIES: Merge multiple deltas
    merged = apply_ties(
        [delta1, delta2],
        density=0.3,
    )

    # SLERP: Spherical interpolation
    interpolated = slerp_delta(delta1, delta2, t=0.5)

    # Using DeltaMerger class
    merger = DeltaMerger(
        method="ties",
        ties_density=0.3,
    )
    processed = merger.merge([delta1, delta2])

    # Compute statistics
    stats = compute_delta_stats(delta1)
    print(f"Delta stats: {stats}")


# ==============================================================================
# Example 5: Cross-Architecture Projection
# ==============================================================================

def example_projection():
    """
    Project weights between different architectures.
    """
    from custom_nodes.ComfyUI_distillation import (
        CrossArchProjector,
        ProjectionConfig,
        svd_projection,
        extract_lora_pissa,
    )

    device = torch.device("cuda:0")

    # Configure projection
    proj_config = ProjectionConfig(
        projection_rank=256,
        projection_min_rank=8,
        projection_adaptive_rank=True,
        projection_energy_threshold=0.99,
        use_adaptive_rank=True,
        energy_threshold=0.95,
        min_rank=8,
        max_rank=256,
    )

    # Create projector
    projector = CrossArchProjector(device, proj_config)

    # Project weight from different shape
    teacher_weight = torch.randn(8192, 4096, device=device)  # Teacher shape
    student_weight = torch.randn(4096, 4096, device=device)  # Student shape

    # Project teacher to student shape
    projected = projector.project_weight(
        teacher_weight,
        target_shape=(4096, 4096),
        ref=student_weight,  # Reference for subspace alignment
    )
    print(f"Projected shape: {projected.shape}")

    # Extract LoRA from delta
    delta = projected - student_weight
    lora_A, lora_B, rank = projector.extract_lora(delta, rank=64)
    print(f"Extracted LoRA: A={lora_A.shape}, B={lora_B.shape}, rank={rank}")


# ==============================================================================
# Example 6: Memory-Efficient Processing with Offloading
# ==============================================================================

def example_memory_management():
    """
    Use memory management utilities for large models.
    """
    from custom_nodes.ComfyUI_distillation import (
        LayerOffloader,
        OffloadStrategy,
        TensorPool,
        get_optimal_offload_strategy,
        estimate_model_memory,
    )

    device = torch.device("cuda:0")

    # Estimate memory requirements
    num_params = 7_000_000_000  # 7B parameters
    memory_gb = estimate_model_memory(
        num_params,
        dtype=torch.bfloat16,
        include_gradients=False,
        include_optimizer=False,
    )
    print(f"Estimated model memory: {memory_gb:.2f} GB")

    # Get optimal offload strategy
    available_gpu = 24.0  # GB
    strategy = get_optimal_offload_strategy(
        model_size_gb=memory_gb,
        available_gpu_gb=available_gpu,
        available_cpu_gb=64.0,
    )
    print(f"Recommended strategy: {strategy.value}")

    # Create offloader
    offloader = LayerOffloader(
        strategy=OffloadStrategy.CPU,
        device=device,
        use_pinned=True,
        max_cpu_gb=64.0,
        prefetch_count=2,
    )

    # Store and retrieve tensors
    tensor = torch.randn(4096, 4096, device=device)
    offloader.store("layer_0_weight", tensor)

    # Later: retrieve with prefetching
    retrieved = offloader.get(
        "layer_0_weight",
        prefetch_next=["layer_1_weight", "layer_2_weight"],
    )

    # Cleanup
    offloader.cleanup()

    # Tensor pool for reuse
    pool = TensorPool(device, max_tensors=10)

    temp = pool.get((4096, 4096), dtype=torch.float32)
    # ... use tensor ...
    pool.release(temp)  # Return to pool for reuse


# ==============================================================================
# Example 7: FP8 Weight Handling
# ==============================================================================

def example_fp8_handling():
    """
    Handle FP8/quantized weights.
    """
    from custom_nodes.ComfyUI_distillation import (
        is_float8_dtype,
        fp8_dequantize,
        FP8WeightHandler,
        resolve_dequant_dtype,
        check_fp8_compatibility,
    )

    device = torch.device("cuda:0")

    # Check FP8 compatibility
    compat = check_fp8_compatibility(device)
    print(f"FP8 compatibility: {compat}")

    # Create FP8 handler
    handler = FP8WeightHandler(
        block_size=128,
        target_dtype="auto",
        device=device,
    )

    # Example: dequantize FP8 weight with block-wise scale
    if compat.get("native_fp8_support"):
        # Simulate FP8 weight (in practice, loaded from checkpoint)
        # weight_fp8 = ...  # shape: (M, N) in float8_e4m3fn
        # scale_inv = ...   # shape: (ceil(M/128), ceil(N/128))

        # Dequantize
        # weight_fp32 = fp8_dequantize(weight_fp8, scale_inv, block_size=128)
        pass

    # Resolve dtype for safe computation
    dtype = resolve_dequant_dtype("auto", device)
    print(f"Resolved dtype: {dtype}")


# ==============================================================================
# Example 8: Complete Workflow
# ==============================================================================

def example_complete_workflow():
    """
    Complete distillation workflow from start to finish.
    """
    from custom_nodes.ComfyUI_distillation import (
        create_distillation_engine,
        DistillationConfig,
        check_fp8_compatibility,
    )
    from safetensors.torch import load_file, save_file

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Check FP8 support
    fp8_compat = check_fp8_compatibility(device)
    print(f"Device: {device}, FP8 support: {fp8_compat.get('native_fp8_support', False)}")

    # Paths
    teacher_path = "path/to/teacher/model.safetensors"
    student_path = "path/to/student/model.safetensors"
    output_path = "path/to/output/lora"

    # Create distillation engine
    engine = create_distillation_engine(
        teacher_path=teacher_path,
        student_path=student_path,
        output_path=output_path,
        rank=64,
        svd_mode="aurora",
        use_ties=True,
        device=str(device),
        verbose=True,
    )

    # Load models
    print("Loading teacher model...")
    teacher_state_dict = load_file(teacher_path)

    print("Loading student model...")
    student_state_dict = load_file(student_path)

    # Run distillation
    print("Running distillation...")
    result = engine.distill_models(
        teacher_state_dict=teacher_state_dict,
        student_state_dict=student_state_dict,
    )

    # Print results
    print("\n=== Distillation Complete ===")
    print(f"Layers processed: {result.statistics['layers_processed']}")
    print(f"Layers skipped: {result.statistics['layers_skipped']}")
    print(f"Total parameters: {result.statistics['total_params']:,}")
    print(f"LoRA parameters: {result.statistics['lora_params']:,}")
    compression = 1 - result.statistics['lora_params'] / result.statistics['total_params']
    print(f"Compression: {compression:.2%}")

    # Save LoRA adapter
    print(f"\nSaving LoRA to {output_path}...")
    engine.save_lora_adapter(
        result,
        output_path,
        base_model_name="student_model",
    )

    # Cleanup
    engine.cleanup()
    print("Done!")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print("ComfyUI Distillation - Usage Examples")
    print("=" * 50)
    print("\nAvailable examples:")
    print("1. example_quick_distillation()")
    print("2. example_full_config()")
    print("3. example_svd_algorithms()")
    print("4. example_merge_strategies()")
    print("5. example_projection()")
    print("6. example_memory_management()")
    print("7. example_fp8_handling()")
    print("8. example_complete_workflow()")
    print("\nRun any function to see the example in action.")
