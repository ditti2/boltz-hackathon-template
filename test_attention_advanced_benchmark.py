#!/usr/bin/env python3
"""
Advanced benchmark script to compare different attention implementations:
1. Original Boltz AttentionPairBias
2. cuEquivariance-based implementation
3. Flash Attention-based implementation
4. Optimized hybrid implementation

This script focuses on realistic protein structure prediction workloads
with larger sequence lengths and better profiling.
"""

import torch
import torch.nn as nn
import time
import math
from typing import Optional, Dict
import numpy as np
import argparse

# Try to import specialized attention implementations
AVAILABLE_IMPLEMENTATIONS = ["boltz"]

try:
    from cuequivariance_torch import attention_pair_bias as cueq_attention_pair_bias
    HAS_CUEQUIVARIANCE = True
    AVAILABLE_IMPLEMENTATIONS.append("cuequivariance")
    print("✓ cuEquivariance is available")
except ImportError:
    HAS_CUEQUIVARIANCE = False
    print("✗ cuEquivariance is not available")

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
    AVAILABLE_IMPLEMENTATIONS.append("flash")
    print("✓ Flash Attention is available")
except ImportError:
    HAS_FLASH_ATTN = False
    print("✗ Flash Attention is not available")

# Import Boltz implementations
import sys
import os
sys.path.append('./src')

from boltz.model.layers.attention import AttentionPairBias as BoltzAttentionPairBias

# Import optimized implementations if available
if HAS_CUEQUIVARIANCE:
    from boltz.model.layers.enhanced_attention import EnhancedAttentionPairBias
    AVAILABLE_IMPLEMENTATIONS.append("enhanced")

if HAS_FLASH_ATTN and HAS_CUEQUIVARIANCE:
    from boltz.model.layers.flash_attention import FlashAttentionPairBias
    AVAILABLE_IMPLEMENTATIONS.append("flash_enhanced")


def memory_stats():
    """Get current GPU memory usage stats."""
    if not torch.cuda.is_available():
        return {"allocated_gb": 0, "reserved_gb": 0, "free_gb": 0}
    
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1e9
    
    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "free_gb": free
    }


def create_attention_model(name, c_s, c_z, num_heads, device):
    """Create an attention model of the specified type."""
    if name == "boltz":
        return BoltzAttentionPairBias(c_s, c_z, num_heads).to(device)
    elif name == "cuequivariance":
        from test_attention_pair_bias_integration import CuEquivAttentionPairBias
        return CuEquivAttentionPairBias(c_s, c_z, num_heads).to(device)
    elif name == "enhanced":
        return EnhancedAttentionPairBias(c_s, c_z, num_heads).to(device)
    elif name == "flash_enhanced":
        return FlashAttentionPairBias(c_s, c_z, num_heads).to(device)
    else:
        raise ValueError(f"Unknown attention implementation: {name}")


def copy_weights(target_model, source_model):
    """Copy weights from source model to target model."""
    # This function handles different model architectures with compatible weights
    
    # For QKV projections
    if hasattr(target_model, "proj_qkv") and hasattr(source_model, "proj_q"):
        # Flash attention has combined QKV
        q_weight = source_model.proj_q.weight
        k_weight = source_model.proj_k.weight
        v_weight = source_model.proj_v.weight
        
        # Concatenate along output dimension
        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        target_model.proj_qkv.weight.data.copy_(qkv_weight)
        
        if source_model.proj_q.bias is not None:
            q_bias = source_model.proj_q.bias
            # Assuming k and v don't have bias as per standard implementation
            qkv_bias = torch.cat([q_bias, torch.zeros_like(q_bias), torch.zeros_like(q_bias)])
            target_model.proj_qkv.bias.data.copy_(qkv_bias)
    elif hasattr(target_model, "proj_q") and hasattr(source_model, "proj_q"):
        # Standard separate Q, K, V projections
        target_model.proj_q.weight.data.copy_(source_model.proj_q.weight.data)
        target_model.proj_k.weight.data.copy_(source_model.proj_k.weight.data)
        target_model.proj_v.weight.data.copy_(source_model.proj_v.weight.data)
        
        if hasattr(target_model.proj_q, "bias") and target_model.proj_q.bias is not None:
            target_model.proj_q.bias.data.copy_(source_model.proj_q.bias.data)
    
    # Gating projection
    if hasattr(target_model, "proj_g") and hasattr(source_model, "proj_g"):
        target_model.proj_g.weight.data.copy_(source_model.proj_g.weight.data)
        
        if hasattr(target_model.proj_g, "bias") and target_model.proj_g.bias is not None:
            target_model.proj_g.bias.data.copy_(source_model.proj_g.bias.data)
    
    # Output projection
    if hasattr(target_model, "proj_o") and hasattr(source_model, "proj_o"):
        target_model.proj_o.weight.data.copy_(source_model.proj_o.weight.data)
        
        if hasattr(target_model.proj_o, "bias") and target_model.proj_o.bias is not None:
            target_model.proj_o.bias.data.copy_(source_model.proj_o.bias.data)
    
    # Z projection
    if hasattr(target_model, "norm_z") and hasattr(source_model, "proj_z"):
        if isinstance(source_model.proj_z, nn.Sequential):
            # Handle Boltz's Sequential module
            target_model.norm_z.weight.data.copy_(source_model.proj_z[0].weight.data)
            target_model.norm_z.bias.data.copy_(source_model.proj_z[0].bias.data)
            target_model.proj_z.weight.data.copy_(source_model.proj_z[1].weight.data)
        elif hasattr(source_model, "norm_z"):
            # Handle optimized implementation
            target_model.norm_z.weight.data.copy_(source_model.norm_z.weight.data)
            target_model.norm_z.bias.data.copy_(source_model.norm_z.bias.data)
            target_model.proj_z.weight.data.copy_(source_model.proj_z.weight.data)


def benchmark_model(
    model_name,
    batch_size,
    seq_len,
    c_s=128,
    c_z=64,
    num_heads=8,
    iterations=10,
    warmup=3,
    fp16=False,
    reference_output=None,
    dtype=torch.float32
):
    """Benchmark a specific model implementation."""
    multiplicity = 1
    
    # Create test data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate input data in the specified precision
    s = torch.randn(batch_size, seq_len, c_s, device=device, dtype=dtype)
    z = torch.randn(batch_size, seq_len, seq_len, c_z, device=device, dtype=dtype)
    mask = torch.ones(batch_size, seq_len, device=device)
    
    # Apply some masking to simulate realistic use case
    mask[:, seq_len//2:] = 0
    
    # Create model
    model = create_attention_model(model_name, c_s, c_z, num_heads, device)
    
    # If reference model exists, copy weights to ensure fair comparison
    if reference_output is not None and "reference_model" in reference_output:
        copy_weights(model, reference_output["reference_model"])
    
    # Record memory before and after model creation
    mem_before = memory_stats()
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(s, z, mask, multiplicity)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    torch.cuda.reset_peak_memory_stats() if device.type == 'cuda' else None
    mem_after_warmup = memory_stats()
    
    # Benchmark runs
    run_times = []
    
    for i in range(iterations):
        # Clear CUDA cache to ensure consistent timing
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        start_time = time.time()
        with torch.no_grad():
            if fp16:
                with torch.cuda.amp.autocast():
                    output = model(s, z, mask, multiplicity)
            else:
                output = model(s, z, mask, multiplicity)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        run_times.append(time.time() - start_time)
    
    # Calculate statistics
    mean_time = np.mean(run_times)
    std_time = np.std(run_times)
    
    # Record peak memory
    mem_peak = memory_stats()
    mem_used = mem_peak["allocated_gb"] - mem_before["allocated_gb"]
    
    # Check correctness against reference if available
    max_diff = None
    relative_diff = None
    if reference_output is not None and "output" in reference_output:
        ref_output = reference_output["output"]
        max_diff = torch.max(torch.abs(output - ref_output)).item()
        relative_diff = max_diff / (torch.max(torch.abs(ref_output)).item() + 1e-9)
    
    results = {
        "model": model_name,
        "time_ms": mean_time * 1000,
        "time_std_ms": std_time * 1000,
        "mem_used_gb": mem_used,
        "output": output,
        "reference_model": model,  # Save model for weight copying
        "max_diff": max_diff,
        "relative_diff": relative_diff
    }
    
    return results


def run_full_benchmark(
    seq_lengths,
    batch_sizes,
    models=None,
    c_s=128,
    c_z=64,
    num_heads=8,
    iterations=10,
    warmup=3,
    fp16=False,
):
    """Run benchmarks for multiple models across different configurations."""
    if models is None:
        models = AVAILABLE_IMPLEMENTATIONS
    
    print(f"\n{'='*80}")
    print(f"Running benchmarks for models: {', '.join(models)}")
    print(f"{'='*80}")
    
    all_results = []
    reference_cache = {}
    
    # Use the same dtype for all tests to ensure consistency
    dtype = torch.float16 if fp16 else torch.float32
    
    for seq_len in seq_lengths:
        for batch_size in batch_sizes:
            print(f"\n{'-'*80}")
            print(f"Benchmarking: seq_len={seq_len}, batch_size={batch_size}, dtype={dtype}")
            print(f"{'-'*80}")
            
            # Initialize reference for this configuration
            reference = None
            
            # Run benchmark for each model type
            for model_name in models:
                try:
                    print(f"Testing {model_name}...")
                    result = benchmark_model(
                        model_name=model_name,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        c_s=c_s,
                        c_z=c_z,
                        num_heads=num_heads,
                        iterations=iterations,
                        warmup=warmup,
                        fp16=fp16,
                        reference_output=reference,
                        dtype=dtype
                    )
                    
                    # First model becomes the reference for output comparison
                    if reference is None:
                        reference = result
                    
                    # Calculate speedup relative to Boltz baseline
                    if "boltz" in result:
                        baseline_time = result["boltz"]["time_ms"]
                        speedup = baseline_time / result["time_ms"]
                    else:
                        speedup = None
                    
                    # Print individual result
                    print(f"  ✓ {model_name}: {result['time_ms']:.2f} ± {result['time_std_ms']:.2f} ms")
                    
                    if result["max_diff"] is not None:
                        print(f"    - Max difference: {result['max_diff']:.6f}")
                        print(f"    - Relative difference: {result['relative_diff']:.6f}")
                        print(f"    - Memory used: {result['mem_used_gb']:.2f} GB")
                        
                        if speedup is not None:
                            print(f"    - Speedup vs Boltz: {speedup:.2f}x")
                    
                    # Store the result
                    result_entry = {
                        "seq_len": seq_len,
                        "batch_size": batch_size,
                        "model": model_name,
                        "time_ms": result["time_ms"],
                        "time_std_ms": result["time_std_ms"],
                        "mem_used_gb": result["mem_used_gb"]
                    }
                    
                    if result["max_diff"] is not None:
                        result_entry["max_diff"] = result["max_diff"]
                        result_entry["relative_diff"] = result["relative_diff"]
                    
                    if speedup is not None:
                        result_entry["speedup"] = speedup
                    
                    all_results.append(result_entry)
                    
                except Exception as e:
                    print(f"  ✗ Error benchmarking {model_name}: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Print summary table
    print(f"\n{'='*100}")
    print("SUMMARY TABLE")
    print(f"{'='*100}")
    header = f"{'Seq Len':8} {'Batch':8} {'Model':15} {'Time (ms)':12} {'Memory (GB)':12} {'Max Diff':12} {'Speedup':10}"
    print(header)
    print(f"{'-'*8} {'-'*8} {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
    
    # Group results by seq_len and batch_size
    configs = {}
    for r in all_results:
        key = (r["seq_len"], r["batch_size"])
        if key not in configs:
            configs[key] = []
        configs[key].append(r)
    
    # For each configuration, find the boltz baseline and calculate speedups
    for key, results in sorted(configs.items()):
        boltz_time = None
        for r in results:
            if r["model"] == "boltz":
                boltz_time = r["time_ms"]
                break
        
        for r in results:
            if boltz_time is not None:
                speedup = boltz_time / r["time_ms"]
            else:
                speedup = float('nan')
            
            max_diff = r.get("max_diff", float('nan'))
            
            print(f"{r['seq_len']:8d} {r['batch_size']:8d} {r['model']:15} "
                  f"{r['time_ms']:12.2f} {r['mem_used_gb']:12.2f} {max_diff:12.6f} {speedup:10.2f}x")
    
    return all_results


def main():
    """Main function to run benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark attention implementations")
    parser.add_argument("--seq_lengths", type=int, nargs="+", default=[64, 128, 256, 512, 1024],
                        help="Sequence lengths to benchmark")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2, 4],
                        help="Batch sizes to benchmark")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help=f"Models to benchmark (available: {', '.join(AVAILABLE_IMPLEMENTATIONS)})")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup iterations")
    parser.add_argument("--fp16", action="store_true",
                        help="Use half precision (fp16)")
    
    args = parser.parse_args()
    
    print("Advanced AttentionPairBias Implementations Benchmark")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, running on CPU (will be slow)")
    else:
        device = torch.cuda.current_device()
        print(f"✓ Using CUDA device: {torch.cuda.get_device_name(device)}")
        print(f"✓ CUDA capability: {torch.cuda.get_device_capability(device)}")
        print(f"✓ CUDA memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    print(f"✓ Available implementations: {', '.join(AVAILABLE_IMPLEMENTATIONS)}")
    print(f"✓ Running with precision: {('FP16' if args.fp16 else 'FP32')}")
    
    # Validate models list
    if args.models:
        invalid_models = set(args.models) - set(AVAILABLE_IMPLEMENTATIONS)
        if invalid_models:
            print(f"⚠ Invalid models: {', '.join(invalid_models)}")
            print(f"⚠ Available models: {', '.join(AVAILABLE_IMPLEMENTATIONS)}")
            return
    
    # Run benchmarks
    results = run_full_benchmark(
        seq_lengths=args.seq_lengths,
        batch_sizes=args.batch_sizes,
        models=args.models,
        iterations=args.iterations,
        warmup=args.warmup,
        fp16=args.fp16
    )
    
    print(f"\n{'='*60}")
    print("Optimization Recommendations")
    print(f"{'='*60}")
    
    # Check if Flash Attention is available
    if not HAS_FLASH_ATTN:
        print("1. Install Flash Attention for maximum performance:")
        print("   pip install flash-attn --no-build-isolation")
        print("   - Provides much faster attention computation")
        print("   - Significantly reduces memory usage")
        print("   - Enables better scaling for large sequences")
    
    # Check if cuEquivariance is properly optimized
    cuequivariance_slower = False
    for r in results:
        if r.get("model") == "cuequivariance" and r.get("speedup", 0) < 0.9:
            cuequivariance_slower = True
            break
    
    if cuequivariance_slower:
        print("\n2. Optimize cuEquivariance integration:")
        print("   - The current kernel is slower than PyTorch's implementation")
        print("   - Consider profiling the kernel with Nsight Compute")
        print("   - Check for unnecessary memory operations or synchronizations")
        print("   - Ensure tensor layouts are optimized for GPU memory access patterns")
    
    print("\n3. Additional optimizations:")
    print("   - Use fused kernels where possible (QKV projection, attention+bias)")
    print("   - Implement custom CUDA kernels for specific operations")
    print("   - Consider using CUTLASS or other optimized libraries")
    print("   - Experiment with different memory layouts (e.g., interleaved vs. planar)")
    
    if args.fp16:
        print("\n4. Mixed precision optimizations:")
        print("   - Already using FP16 for maximum throughput")
        print("   - Consider using bfloat16 for better numerical stability")
    else:
        print("\n4. Enable mixed precision:")
        print("   - Run with --fp16 to enable half precision")
        print("   - Can provide significant speedup with minimal accuracy loss")


if __name__ == "__main__":
    main()