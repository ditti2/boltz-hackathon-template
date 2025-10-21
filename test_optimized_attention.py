#!/usr/bin/env python3
"""
Benchmark script for optimized AttentionPairBias implementations.
This script compares:
1. Original Boltz implementation
2. cuEquivariance implementation
3. Memory-optimized implementation with QKV fusion and LayerNorm optimization
"""

import torch
import torch.nn as nn
import time
import math
import numpy as np
import argparse
from typing import Dict, Optional

# Try to import cuEquivariance
try:
    from cuequivariance_torch import attention_pair_bias as cueq_attention_pair_bias
    HAS_CUEQUIVARIANCE = True
    print("✓ cuEquivariance is available")
except ImportError:
    HAS_CUEQUIVARIANCE = False
    print("✗ cuEquivariance is not available")

# Try to import APEX for optimized LayerNorm
try:
    from apex.normalization import FusedLayerNorm
    HAS_APEX = True
    print("✓ APEX FusedLayerNorm is available")
except ImportError:
    HAS_APEX = False
    print("✗ APEX FusedLayerNorm is not available")

# Import Boltz implementations
import sys
import os
sys.path.append('./src')

# Import attention implementations
from boltz.model.layers.attention import AttentionPairBias as BoltzAttentionPairBias

try:
    from boltz.model.layers.optimized_attention import OptimizedAttentionPairBias
    HAS_OPTIMIZED = True
    print("✓ OptimizedAttentionPairBias is available")
except ImportError:
    HAS_OPTIMIZED = False
    print("✗ OptimizedAttentionPairBias is not available")

if HAS_CUEQUIVARIANCE:
    try:
        from test_attention_pair_bias_integration import CuEquivAttentionPairBias
        HAS_CUEQ_TEST = True
        print("✓ CuEquivAttentionPairBias test is available")
    except ImportError:
        HAS_CUEQ_TEST = False
        print("✗ CuEquivAttentionPairBias test is not available")


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


def copy_weights_optimized(target_model, source_model):
    """Copy weights from Boltz model to optimized model."""
    # Handle QKV projection weights
    if hasattr(target_model, "proj_qkv") and hasattr(source_model, "proj_q"):
        q_weight = source_model.proj_q.weight
        k_weight = source_model.proj_k.weight
        v_weight = source_model.proj_v.weight
        
        # Concatenate along output dimension for QKV fusion
        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        target_model.proj_qkv.weight.data.copy_(qkv_weight)
        
        if source_model.proj_q.bias is not None:
            q_bias = source_model.proj_q.bias
            # Assuming k and v don't have bias in Boltz implementation
            qkv_bias = torch.cat([q_bias, torch.zeros_like(q_bias), torch.zeros_like(q_bias)])
            target_model.proj_qkv.bias.data.copy_(qkv_bias)
    
    # Gating projection
    if hasattr(target_model, "proj_g") and hasattr(source_model, "proj_g"):
        target_model.proj_g.weight.data.copy_(source_model.proj_g.weight.data)
        
        if hasattr(target_model.proj_g, "bias") and target_model.proj_g.bias is not None:
            if hasattr(source_model.proj_g, "bias") and source_model.proj_g.bias is not None:
                target_model.proj_g.bias.data.copy_(source_model.proj_g.bias.data)
    
    # Output projection
    if hasattr(target_model, "proj_o") and hasattr(source_model, "proj_o"):
        target_model.proj_o.weight.data.copy_(source_model.proj_o.weight.data)
        
        if hasattr(target_model.proj_o, "bias") and target_model.proj_o.bias is not None:
            if hasattr(source_model.proj_o, "bias") and source_model.proj_o.bias is not None:
                target_model.proj_o.bias.data.copy_(source_model.proj_o.bias.data)
    
    # Z projection (normalize and project)
    if hasattr(target_model, "norm_z") and hasattr(source_model, "proj_z"):
        if isinstance(source_model.proj_z, nn.Sequential):
            # Handle Boltz's Sequential module for Z projection
            if hasattr(target_model.norm_z, "norm"):
                # For optimized layernorm wrapper
                target_model.norm_z.norm.weight.data.copy_(source_model.proj_z[0].weight.data)
                target_model.norm_z.norm.bias.data.copy_(source_model.proj_z[0].bias.data)
            else:
                # Direct layernorm
                target_model.norm_z.weight.data.copy_(source_model.proj_z[0].weight.data)
                target_model.norm_z.bias.data.copy_(source_model.proj_z[0].bias.data)
                
            target_model.proj_z.weight.data.copy_(source_model.proj_z[1].weight.data)


def copy_weights_cuequiv(target_model, source_model):
    """Copy weights from Boltz model to cuEquivariance model."""
    # Standard projections
    target_model.proj_q.weight.data = source_model.proj_q.weight.data.clone()
    target_model.proj_k.weight.data = source_model.proj_k.weight.data.clone()
    target_model.proj_v.weight.data = source_model.proj_v.weight.data.clone()
    target_model.proj_g.weight.data = source_model.proj_g.weight.data.clone()
    target_model.proj_o.weight.data = source_model.proj_o.weight.data.clone()
    
    # Z projection and normalization
    target_model.norm_z.weight.data = source_model.proj_z[0].weight.data.clone()
    target_model.norm_z.bias.data = source_model.proj_z[0].bias.data.clone()
    target_model.proj_z.weight.data = source_model.proj_z[1].weight.data.clone()


def benchmark_model(
    model_name,
    model,
    s,
    z,
    mask,
    seq_len,
    batch_size,
    iterations=10,
    warmup=3,
    fp16=False,
    reference_output=None
):
    """
    Benchmark a specific model implementation.
    
    Parameters
    ----------
    model_name : str
        Name of the model for reporting
    model : nn.Module
        Model to benchmark
    s, z, mask : torch.Tensor
        Input tensors
    seq_len : int
        Sequence length for tracking
    batch_size : int
        Batch size for tracking
    iterations : int
        Number of benchmark iterations
    warmup : int
        Number of warmup iterations
    fp16 : bool
        Whether to use FP16 precision
    reference_output : torch.Tensor
        Reference output for correctness checking
        
    Returns
    -------
    dict
        Benchmark results
    """
    device = s.device
    multiplicity = 1
    
    # Record memory before
    mem_before = memory_stats()
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(s, z, mask, multiplicity)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    torch.cuda.reset_peak_memory_stats() if device.type == 'cuda' else None
    
    # Benchmark runs
    run_times = []
    
    for i in range(iterations):
        # Clear CUDA cache for consistent timing
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
    
    # Check correctness if reference is provided
    max_diff = None
    relative_diff = None
    
    if reference_output is not None:
        max_diff = torch.max(torch.abs(output - reference_output)).item()
        relative_diff = max_diff / (torch.max(torch.abs(reference_output)).item() + 1e-9)
    
    return {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "model": model_name,
        "time_ms": mean_time * 1000,
        "time_std_ms": std_time * 1000,
        "mem_used_gb": mem_used,
        "output": output,
        "max_diff": max_diff,
        "relative_diff": relative_diff
    }


def run_benchmark(
    seq_len,
    batch_size,
    c_s=128,
    c_z=64,
    num_heads=8,
    iterations=10,
    warmup=3,
    fp16=False
):
    """
    Run benchmark comparing different attention implementations.
    
    Parameters
    ----------
    seq_len : int
        Sequence length
    batch_size : int
        Batch size
    c_s : int
        Sequence feature dimension
    c_z : int
        Pairwise feature dimension
    num_heads : int
        Number of attention heads
    iterations : int
        Number of benchmark iterations
    warmup : int
        Number of warmup iterations
    fp16 : bool
        Whether to use FP16 precision
    
    Returns
    -------
    list
        Benchmark results for all models
    """
    print(f"\n{'-'*80}")
    print(f"Benchmarking: seq_len={seq_len}, batch_size={batch_size}, fp16={fp16}")
    print(f"{'-'*80}")
    
    # Create test data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if fp16 else torch.float32
    
    # Input tensors
    s = torch.randn(batch_size, seq_len, c_s, device=device, dtype=dtype)
    z = torch.randn(batch_size, seq_len, seq_len, c_z, device=device, dtype=dtype)
    mask = torch.ones(batch_size, seq_len, device=device)
    
    # Apply some masking to simulate realistic use case
    mask[:, seq_len//2:] = 0
    
    results = []
    reference_output = None
    
    # 1. Benchmark Boltz implementation
    print(f"Testing Boltz AttentionPairBias...")
    try:
        boltz_model = BoltzAttentionPairBias(c_s, c_z, num_heads).to(device)
        
        boltz_result = benchmark_model(
            model_name="boltz",
            model=boltz_model,
            s=s,
            z=z,
            mask=mask,
            seq_len=seq_len,
            batch_size=batch_size,
            iterations=iterations,
            warmup=warmup,
            fp16=fp16
        )
        
        reference_output = boltz_result["output"]
        results.append(boltz_result)
        
        print(f"  ✓ Boltz: {boltz_result['time_ms']:.2f} ± {boltz_result['time_std_ms']:.2f} ms")
        print(f"    - Memory used: {boltz_result['mem_used_gb']:.2f} GB")
        
    except Exception as e:
        print(f"  ✗ Error testing Boltz implementation: {e}")
    
    # 2. Benchmark cuEquivariance implementation (if available)
    if HAS_CUEQUIVARIANCE and HAS_CUEQ_TEST:
        print(f"Testing cuEquivariance AttentionPairBias...")
        try:
            cueq_model = CuEquivAttentionPairBias(c_s, c_z, num_heads).to(device)
            copy_weights_cuequiv(cueq_model, boltz_model)
            
            cueq_result = benchmark_model(
                model_name="cuequivariance",
                model=cueq_model,
                s=s,
                z=z,
                mask=mask,
                seq_len=seq_len,
                batch_size=batch_size,
                iterations=iterations,
                warmup=warmup,
                fp16=fp16,
                reference_output=reference_output
            )
            
            results.append(cueq_result)
            
            print(f"  ✓ cuEquivariance: {cueq_result['time_ms']:.2f} ± {cueq_result['time_std_ms']:.2f} ms")
            print(f"    - Max difference: {cueq_result['max_diff']:.6f}")
            print(f"    - Memory used: {cueq_result['mem_used_gb']:.2f} GB")
            if reference_output is not None:
                print(f"    - Speedup vs Boltz: {boltz_result['time_ms']/cueq_result['time_ms']:.2f}x")
                
        except Exception as e:
            print(f"  ✗ Error testing cuEquivariance implementation: {e}")
            import traceback
            traceback.print_exc()
    
    # 3. Benchmark optimized implementation
    if HAS_OPTIMIZED:
        print(f"Testing Optimized AttentionPairBias...")
        try:
            opt_model = OptimizedAttentionPairBias(c_s, c_z, num_heads).to(device)
            copy_weights_optimized(opt_model, boltz_model)
            
            opt_result = benchmark_model(
                model_name="optimized",
                model=opt_model,
                s=s,
                z=z,
                mask=mask,
                seq_len=seq_len,
                batch_size=batch_size,
                iterations=iterations,
                warmup=warmup,
                fp16=fp16,
                reference_output=reference_output
            )
            
            results.append(opt_result)
            
            print(f"  ✓ Optimized: {opt_result['time_ms']:.2f} ± {opt_result['time_std_ms']:.2f} ms")
            print(f"    - Max difference: {opt_result['max_diff']:.6f}")
            print(f"    - Memory used: {opt_result['mem_used_gb']:.2f} GB")
            if reference_output is not None:
                print(f"    - Speedup vs Boltz: {boltz_result['time_ms']/opt_result['time_ms']:.2f}x")
            
            if 'cueq_result' in locals():
                print(f"    - Speedup vs cuEquivariance: {cueq_result['time_ms']/opt_result['time_ms']:.2f}x")
                
        except Exception as e:
            print(f"  ✗ Error testing optimized implementation: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def run_multiple_benchmarks(
    seq_lengths,
    batch_sizes,
    c_s=128,
    c_z=64,
    num_heads=8,
    iterations=10,
    warmup=3,
    fp16=False
):
    """
    Run benchmarks for multiple configurations.
    
    Parameters
    ----------
    seq_lengths : list
        List of sequence lengths
    batch_sizes : list
        List of batch sizes
    c_s, c_z, num_heads : int
        Model parameters
    iterations, warmup : int
        Benchmark parameters
    fp16 : bool
        Whether to use FP16 precision
        
    Returns
    -------
    list
        All benchmark results
    """
    all_results = []
    
    for seq_len in seq_lengths:
        for batch_size in batch_sizes:
            try:
                results = run_benchmark(
                    seq_len=seq_len,
                    batch_size=batch_size,
                    c_s=c_s,
                    c_z=c_z,
                    num_heads=num_heads,
                    iterations=iterations,
                    warmup=warmup,
                    fp16=fp16
                )
                
                all_results.extend(results)
                
            except Exception as e:
                print(f"Error benchmarking seq_len={seq_len}, batch_size={batch_size}: {e}")
                import traceback
                traceback.print_exc()
    
    # Print summary table
    print(f"\n{'='*100}")
    print("SUMMARY TABLE")
    print(f"{'='*100}")
    print(f"{'Seq Len':8} {'Batch':8} {'Model':15} {'Time (ms)':12} {'Memory (GB)':12} {'Max Diff':12} {'Speedup':10}")
    print(f"{'-'*8} {'-'*8} {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
    
    # Group results by configuration
    configs = {}
    for r in all_results:
        key = (r['seq_len'], r['batch_size'])
        if key not in configs:
            configs[key] = {}
        configs[key][r['model']] = r
    
    # Sort by sequence length and batch size
    for key in sorted(configs.keys()):
        seq_len, batch_size = key
        models = configs[key]
        
        # Get Boltz baseline
        boltz_time = models['boltz']['time_ms'] if 'boltz' in models else None
        
        for model_name, result in sorted(models.items()):
            speedup = boltz_time / result['time_ms'] if boltz_time else 1.0
            max_diff = result.get('max_diff', float('nan'))
            
            print(f"{seq_len:8d} {batch_size:8d} {model_name:15} "
                  f"{result['time_ms']:12.2f} {result['mem_used_gb']:12.2f} "
                  f"{max_diff:12.6f} {speedup:10.2f}x")
    
    return all_results


def main():
    """Main function to run benchmarks."""
    parser = argparse.ArgumentParser(description="Benchmark optimized attention implementations")
    parser.add_argument("--seq_lengths", type=int, nargs="+", default=[64, 128, 256, 512, 1024],
                        help="Sequence lengths to benchmark")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 2, 4, 8],
                        help="Batch sizes to benchmark")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warmup iterations")
    parser.add_argument("--fp16", action="store_true",
                        help="Use half precision (fp16)")
    
    args = parser.parse_args()
    
    print("Optimized AttentionPairBias Benchmark")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, running on CPU (will be slow)")
    else:
        device = torch.cuda.current_device()
        print(f"✓ Using CUDA device: {torch.cuda.get_device_name(device)}")
        print(f"✓ CUDA capability: {torch.cuda.get_device_capability(device)}")
        print(f"✓ CUDA memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    print(f"✓ cuEquivariance available: {HAS_CUEQUIVARIANCE}")
    print(f"✓ APEX FusedLayerNorm available: {HAS_APEX}")
    print(f"✓ OptimizedAttentionPairBias available: {HAS_OPTIMIZED}")
    print(f"✓ Running with precision: {('FP16' if args.fp16 else 'FP32')}")
    
    # Run benchmarks
    results = run_multiple_benchmarks(
        seq_lengths=args.seq_lengths,
        batch_sizes=args.batch_sizes,
        iterations=args.iterations,
        warmup=args.warmup,
        fp16=args.fp16
    )
    
    print(f"\n{'='*60}")
    print("Optimization Recommendations")
    print(f"{'='*60}")
    print("1. Memory Layout Optimizations:")
    print("   - Use fused QKV projection for better memory access patterns")
    print("   - Minimize tensor reshapes and transpositions")
    print("   - Keep tensors in contiguous memory where possible")
    
    print("\n2. Kernel Optimizations:")
    print("   - Use cuEquivariance for large sequence lengths (>256)")
    print("   - Consider custom CUDA kernels for maximum performance")
    print("   - Install APEX for optimized LayerNorm operations")
    
    if not args.fp16:
        print("\n3. Enable mixed precision:")
        print("   - Run with --fp16 to enable half precision")
        print("   - Can provide significant speedup with minimal accuracy loss")


if __name__ == "__main__":
    main()
