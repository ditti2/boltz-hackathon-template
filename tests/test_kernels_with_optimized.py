import os
import time

import pandas as pd
import torch
import triton
from profiling import clear_memory, current_memory, memory_measure

from boltz.model.layers.pairformer import PairformerLayer

# Disable auto-tuning (same as test_kernels.py)
os.environ["CUEQ_DEFAULT_CONFIG"] = "1"
os.environ["CUEQ_DISABLE_AOT_TUNING"] = "1"

# Set hyperparameters
C_S = 384
C_Z = 128
BATCH_SIZE = 1
INFERENCE = False
SEQ_LEN = [64, 128, 256, 512]
PRECISION = torch.bfloat16
device = "cuda:0"
torch.set_grad_enabled(not INFERENCE)

# Direct imports without fallbacks
from boltz.model.layers.optimized_attention import HyperOptimizedAttentionPairBias, TurboOptimizedAttentionPairBias
from boltz.model.layers.layernorm_optimized_attention import LayerNormOptimizedAttentionPairBias, FusedAttentionPairBias
from boltz.model.layers.ultra_optimized_attention import UltraFusedAttentionPairBias, CuEquivarianceHybridAttention, MinimalOverheadAttention

HAS_OPTIMIZED = True
HAS_TURBO = True
HAS_LAYERNORM_OPT = True
HAS_ULTRA = True

# Preload modules
model = PairformerLayer(C_S, C_Z, v2=True)
model.cuda()
if INFERENCE:
    model.eval()

# Create optimized model with replaced attention if available
opt_model = None
turbo_model = None
layernorm_model = None
fused_model = None
ultra_model = None
hybrid_model = None
minimal_model = None

if HAS_OPTIMIZED:
    opt_model = PairformerLayer(C_S, C_Z, v2=True)
    opt_model.attention_pair_bias = HyperOptimizedAttentionPairBias(C_S, C_Z, opt_model.attention_pair_bias.num_heads)
    opt_model.cuda()
    if INFERENCE:
        opt_model.eval()

if HAS_TURBO:
    turbo_model = PairformerLayer(C_S, C_Z, v2=True)
    turbo_model.attention_pair_bias = TurboOptimizedAttentionPairBias(C_S, C_Z, turbo_model.attention_pair_bias.num_heads)
    turbo_model.cuda()
    if INFERENCE:
        turbo_model.eval()

if HAS_LAYERNORM_OPT:
    layernorm_model = PairformerLayer(C_S, C_Z, v2=True)
    layernorm_model.attention_pair_bias = LayerNormOptimizedAttentionPairBias(C_S, C_Z, layernorm_model.attention_pair_bias.num_heads)
    layernorm_model.cuda()
    if INFERENCE:
        layernorm_model.eval()
    
    fused_model = PairformerLayer(C_S, C_Z, v2=True)
    fused_model.attention_pair_bias = FusedAttentionPairBias(C_S, C_Z, fused_model.attention_pair_bias.num_heads)
    fused_model.cuda()
    if INFERENCE:
        fused_model.eval()

if HAS_ULTRA:
    ultra_model = PairformerLayer(C_S, C_Z, v2=True)
    ultra_model.attention_pair_bias = UltraFusedAttentionPairBias(C_S, C_Z, ultra_model.attention_pair_bias.num_heads)
    ultra_model.cuda()
    if INFERENCE:
        ultra_model.eval()
        
    hybrid_model = PairformerLayer(C_S, C_Z, v2=True)
    hybrid_model.attention_pair_bias = CuEquivarianceHybridAttention(C_S, C_Z, hybrid_model.attention_pair_bias.num_heads)
    hybrid_model.cuda()
    if INFERENCE:
        hybrid_model.eval()
        
    minimal_model = PairformerLayer(C_S, C_Z, v2=True)
    minimal_model.attention_pair_bias = MinimalOverheadAttention(C_S, C_Z, minimal_model.attention_pair_bias.num_heads)
    minimal_model.cuda()
    if INFERENCE:
        minimal_model.eval()
def fwd(model, s, z, mask, pair_mask, use_cuequiv_mul=False, use_cuequiv_attn=False, use_opt_attn=False, use_turbo_attn=False, use_layernorm_attn=False, use_fused_attn=False):
    if use_fused_attn and fused_model is not None:
        fused_model(s, z, mask, pair_mask, use_cuequiv_mul=use_cuequiv_mul, use_cuequiv_attn=use_cuequiv_attn)
    elif use_layernorm_attn and layernorm_model is not None:
        layernorm_model(s, z, mask, pair_mask, use_cuequiv_mul=use_cuequiv_mul, use_cuequiv_attn=use_cuequiv_attn)
    elif use_turbo_attn and turbo_model is not None:
        turbo_model(s, z, mask, pair_mask, use_cuequiv_mul=use_cuequiv_mul, use_cuequiv_attn=use_cuequiv_attn)
    elif use_opt_attn and opt_model is not None:
        opt_model(s, z, mask, pair_mask, use_cuequiv_mul=use_cuequiv_mul, use_cuequiv_attn=use_cuequiv_attn)
    else:
        model(s, z, mask, pair_mask, use_cuequiv_mul=use_cuequiv_mul, use_cuequiv_attn=use_cuequiv_attn)


def backward(model, s, z, mask, pair_mask, use_cuequiv_mul=False, use_cuequiv_attn=False, use_opt_attn=False, use_turbo_attn=False, use_layernorm_attn=False, use_fused_attn=False):
    if use_fused_attn and fused_model is not None:
        s, z = fused_model(s, z, mask, pair_mask, use_cuequiv_mul=use_cuequiv_mul, use_cuequiv_attn=use_cuequiv_attn)
    elif use_layernorm_attn and layernorm_model is not None:
        s, z = layernorm_model(s, z, mask, pair_mask, use_cuequiv_mul=use_cuequiv_mul, use_cuequiv_attn=use_cuequiv_attn)
    elif use_turbo_attn and turbo_model is not None:
        s, z = turbo_model(s, z, mask, pair_mask, use_cuequiv_mul=use_cuequiv_mul, use_cuequiv_attn=use_cuequiv_attn)
    elif use_opt_attn and opt_model is not None:
        s, z = opt_model(s, z, mask, pair_mask, use_cuequiv_mul=use_cuequiv_mul, use_cuequiv_attn=use_cuequiv_attn)
    else:
        s, z = model(s, z, mask, pair_mask, use_cuequiv_mul=use_cuequiv_mul, use_cuequiv_attn=use_cuequiv_attn)
    (s.sum() + z.sum()).backward()


def speed(func, its=10, warmup=10):
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(its):
        func()
    torch.cuda.synchronize()
    time_a = time.time() - start
    time_a /= its
    return time_a


def analyze_performance_vs_trimul():
    """
    Run manual benchmark to capture results and analyze winners vs Trimul.
    """
    print("\nDetailed Performance Analysis vs Trimul")
    print("=" * 80)
    
    results = []
    
    for size in SEQ_LEN:
        clear_memory(device)
        
        # Create test data
        s = torch.randn((BATCH_SIZE, size, C_S), device=device, dtype=PRECISION, requires_grad=False)
        z = torch.randn((BATCH_SIZE, size, size, C_Z), device=device, dtype=PRECISION, requires_grad=False)
        mask = torch.ones((BATCH_SIZE, size), device=device, dtype=PRECISION, requires_grad=False).float()
        pair_mask = torch.ones((BATCH_SIZE, size, size), device=device, dtype=PRECISION, requires_grad=False).float()

        with torch.autocast("cuda", dtype=PRECISION):
            fn = fwd if INFERENCE else backward
            
            # Measure all configurations
            times = {}
            
            # Default
            times["Default"] = speed(lambda: fn(model, s, z, mask, pair_mask, use_cuequiv_mul=False, use_cuequiv_attn=False, use_opt_attn=False, use_turbo_attn=False, use_layernorm_attn=False, use_fused_attn=False)) / BATCH_SIZE
            
            # Trimul (our baseline for comparison)
            times["Trimul"] = speed(lambda: fn(model, s, z, mask, pair_mask, use_cuequiv_mul=True, use_cuequiv_attn=False, use_opt_attn=False, use_turbo_attn=False, use_layernorm_attn=False, use_fused_attn=False)) / BATCH_SIZE
            
            # TriAttn+Trimul
            times["TriAttn+Trimul"] = speed(lambda: fn(model, s, z, mask, pair_mask, use_cuequiv_mul=True, use_cuequiv_attn=True, use_opt_attn=False, use_turbo_attn=False, use_layernorm_attn=False, use_fused_attn=False)) / BATCH_SIZE
            
            # Our optimizations
            if HAS_OPTIMIZED:
                times["HyperOptAttn"] = speed(lambda: fn(model, s, z, mask, pair_mask, use_cuequiv_mul=False, use_cuequiv_attn=False, use_opt_attn=True, use_turbo_attn=False, use_layernorm_attn=False, use_fused_attn=False)) / BATCH_SIZE
            
            if HAS_TURBO:
                times["TurboOptAttn"] = speed(lambda: fn(model, s, z, mask, pair_mask, use_cuequiv_mul=False, use_cuequiv_attn=False, use_opt_attn=False, use_turbo_attn=True, use_layernorm_attn=False, use_fused_attn=False)) / BATCH_SIZE
            
            if HAS_LAYERNORM_OPT:
                times["LayerNormOpt"] = speed(lambda: fn(model, s, z, mask, pair_mask, use_cuequiv_mul=False, use_cuequiv_attn=False, use_opt_attn=False, use_turbo_attn=False, use_layernorm_attn=True, use_fused_attn=False)) / BATCH_SIZE
                times["FusedAttn"] = speed(lambda: fn(model, s, z, mask, pair_mask, use_cuequiv_mul=False, use_cuequiv_attn=False, use_opt_attn=False, use_turbo_attn=False, use_layernorm_attn=False, use_fused_attn=True)) / BATCH_SIZE
                times["FusedAttn+Trimul"] = speed(lambda: fn(model, s, z, mask, pair_mask, use_cuequiv_mul=True, use_cuequiv_attn=False, use_opt_attn=False, use_turbo_attn=False, use_layernorm_attn=False, use_fused_attn=True)) / BATCH_SIZE
        
        # Find the best performer and compare vs Trimul
        trimul_time = times["Trimul"]
        best_method = min(times.keys(), key=lambda k: times[k])
        best_time = times[best_method]
        
        # Calculate speedup vs Trimul
        speedup_vs_trimul = trimul_time / best_time
        speedup_vs_default = times["Default"] / best_time
        
        result = {
            "Size": size,
            "Trimul_ms": f"{trimul_time * 1000:.2f}",
            "Best_Method": best_method,
            "Best_ms": f"{best_time * 1000:.2f}",
            "Speedup_vs_Trimul": f"{speedup_vs_trimul:.2f}x",
            "Speedup_vs_Default": f"{speedup_vs_default:.2f}x",
            "Winner": "🏆" if best_method != "Trimul" else "Trimul (baseline)"
        }
        
        # Add all timing data
        for method, time_val in times.items():
            result[f"{method}_ms"] = f"{time_val * 1000:.2f}"
        
        results.append(result)
        
        print(f"Size {size}: Best = {best_method} ({best_time*1000:.2f}ms), {speedup_vs_trimul:.2f}x vs Trimul")
    
    # Create detailed DataFrame
    df = pd.DataFrame(results)
    
    print(f"\nDetailed Results:")
    print(f"{'Size':<6} {'Winner':<15} {'Best Time':<12} {'vs Trimul':<12} {'vs Default':<12}")
    print("-" * 70)
    
    for _, row in df.iterrows():
        print(f"{row['Size']:<6} {row['Best_Method']:<15} {row['Best_ms']:<12} {row['Speedup_vs_Trimul']:<12} {row['Speedup_vs_Default']:<12}")
    
    return df
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(its):
        func()
    torch.cuda.synchronize()
    time_a = time.time() - start
    time_a /= its
    return time_a


# Benchmark with Triton performance reporting
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=SEQ_LEN,
        line_arg="provider",
        line_vals=[
            "Default",
            "Trimul",
            "TriAttn+Trimul", 
            "HyperOptAttn",
            "TurboOptAttn",
            "LayerNormOpt",
            "FusedAttn",
            "FusedAttn+Trimul",
            "UltraFused",
            "CuEqHybrid", 
            "MinimalOH",
        ],
        line_names=[
            "Default",
            "Trimul",
            "TriAttn+Trimul",
            "HyperOptAttn", 
            "TurboOptAttn",
            "LayerNormOpt",
            "FusedAttn",
            "FusedAttn+Trimul",
            "UltraFused",
            "CuEqHybrid",
            "MinimalOH",
        ],
        plot_name="optimized_vs_trimul",
        args={},
    )
)
def benchmark(size, provider):
    clear_memory(device)

    # Create test data (same structure as test_kernels.py)
    s = torch.randn(
        (BATCH_SIZE, size, C_S),
        device=device,
        dtype=PRECISION,
        requires_grad=False,
    )
    z = torch.randn(
        (BATCH_SIZE, size, size, C_Z),
        device=device,
        dtype=PRECISION,
        requires_grad=False,
    )
    mask = torch.ones(
        (BATCH_SIZE, size),
        device=device,
        dtype=PRECISION,
        requires_grad=False,
    ).float()
    pair_mask = torch.ones(
        (BATCH_SIZE, size, size),
        device=device,
        dtype=PRECISION,
        requires_grad=False,
    ).float()

    with torch.autocast("cuda", dtype=PRECISION):
        fn = fwd if INFERENCE else backward
        if provider == "Default":
            ms = speed(
                lambda: fn(
                    model, s, z, mask, pair_mask,
                    use_cuequiv_mul=False,
                    use_cuequiv_attn=False,
                    use_opt_attn=False,
                )
            )
        elif provider == "Trimul":
            ms = speed(
                lambda: fn(
                    model, s, z, mask, pair_mask,
                    use_cuequiv_attn=False,
                    use_cuequiv_mul=True,
                    use_opt_attn=False,
                    use_turbo_attn=False,
                    use_layernorm_attn=False,
                    use_fused_attn=False,
                )
            )
        elif provider == "TriAttn+Trimul":
            ms = speed(
                lambda: fn(
                    model, s, z, mask, pair_mask,
                    use_cuequiv_attn=True,
                    use_cuequiv_mul=True,
                    use_opt_attn=False,
                    use_turbo_attn=False,
                    use_layernorm_attn=False,
                    use_fused_attn=False,
                )
            )
        elif provider == "HyperOptAttn":
            if not HAS_OPTIMIZED:
                return float('nan')
            ms = speed(
                lambda: fn(
                    model, s, z, mask, pair_mask,
                    use_cuequiv_attn=False,
                    use_cuequiv_mul=False,
                    use_opt_attn=True,
                    use_turbo_attn=False,
                    use_layernorm_attn=False,
                    use_fused_attn=False,
                )
            )
        elif provider == "TurboOptAttn":
            if not HAS_TURBO:
                return float('nan')
            ms = speed(
                lambda: fn(
                    model, s, z, mask, pair_mask,
                    use_cuequiv_attn=False,
                    use_cuequiv_mul=False,
                    use_opt_attn=False,
                    use_turbo_attn=True,
                    use_layernorm_attn=False,
                    use_fused_attn=False,
                )
            )
        elif provider == "LayerNormOpt":
            if not HAS_LAYERNORM_OPT:
                return float('nan')
            ms = speed(
                lambda: fn(
                    model, s, z, mask, pair_mask,
                    use_cuequiv_attn=False,
                    use_cuequiv_mul=False,
                    use_opt_attn=False,
                    use_turbo_attn=False,
                    use_layernorm_attn=True,
                    use_fused_attn=False,
                )
            )
        elif provider == "FusedAttn":
            if not HAS_LAYERNORM_OPT:
                return float('nan')
            ms = speed(
                lambda: fn(
                    model, s, z, mask, pair_mask,
                    use_cuequiv_attn=False,
                    use_cuequiv_mul=False,
                    use_opt_attn=False,
                    use_turbo_attn=False,
                    use_layernorm_attn=False,
                    use_fused_attn=True,
                )
            )
        elif provider == "FusedAttn+Trimul":
            if not HAS_LAYERNORM_OPT:
                return float('nan')
            ms = speed(
                lambda: fn(
                    model, s, z, mask, pair_mask,
                    use_cuequiv_attn=False,
                    use_cuequiv_mul=True,
                    use_opt_attn=False,
                    use_turbo_attn=False,
                    use_layernorm_attn=False,
                    use_fused_attn=True,
                )
            )
        elif provider == "UltraFused":
            if not HAS_ULTRA:
                return float('nan')
            ms = speed(
                lambda: fn(
                    ultra_model, s, z, mask, pair_mask,
                    use_cuequiv_attn=False,
                    use_cuequiv_mul=False,
                    use_opt_attn=False,
                    use_turbo_attn=False,
                    use_layernorm_attn=False,
                    use_fused_attn=False,
                )
            )
        elif provider == "CuEqHybrid":
            if not HAS_ULTRA:
                return float('nan')
            ms = speed(
                lambda: fn(
                    hybrid_model, s, z, mask, pair_mask,
                    use_cuequiv_attn=False,
                    use_cuequiv_mul=False,
                    use_opt_attn=False,
                    use_turbo_attn=False,
                    use_layernorm_attn=False,
                    use_fused_attn=False,
                )
            )
        elif provider == "MinimalOH":
            if not HAS_ULTRA:
                return float('nan')
            ms = speed(
                lambda: fn(
                    minimal_model, s, z, mask, pair_mask,
                    use_cuequiv_attn=False,
                    use_cuequiv_mul=False,
                    use_opt_attn=False,
                    use_turbo_attn=False,
                    use_layernorm_attn=False,
                    use_fused_attn=False,
                )
            )

    return ms / BATCH_SIZE


if __name__ == "__main__":
    print("Speed comparison: LayerNorm + Fused Attention Optimizations vs Trimul")
    benchmark.run(print_data=True, show_plots=False)
    
    # Detailed performance analysis with winners
    performance_df = analyze_performance_vs_trimul()
    
    # Memory benchmark
    print("\nMemory Benchmark")
    print("=" * 80)
    
    start_mem = current_memory(device)
    memory_results = []
    
    for size in SEQ_LEN:
        print(f"Testing memory for sequence length {size}")
        
        # Create test data
        s = torch.randn(
            (BATCH_SIZE, size, C_S),
            device=device,
            dtype=PRECISION,
            requires_grad=False,
        )
        z = torch.randn(
            (BATCH_SIZE, size, size, C_Z),
            device=device,
            dtype=PRECISION,
            requires_grad=False,
        )
        mask = torch.ones(
            (BATCH_SIZE, size),
            device=device,
            dtype=PRECISION,
            requires_grad=False,
        ).float()
        pair_mask = torch.ones(
            (BATCH_SIZE, size, size),
            device=device,
            dtype=PRECISION,
            requires_grad=False,
        ).float()

        with torch.autocast("cuda", dtype=PRECISION):
            # Memory measurements for each configuration
            memory_default = memory_measure(
                lambda: fwd(model, s, z, mask, pair_mask, use_cuequiv_mul=False, use_cuequiv_attn=False, use_opt_attn=False, use_turbo_attn=False, use_layernorm_attn=False, use_fused_attn=False),
                device=device,
            )
            
            memory_trimul = memory_measure(
                lambda: fwd(model, s, z, mask, pair_mask, use_cuequiv_mul=True, use_cuequiv_attn=False, use_opt_attn=False, use_turbo_attn=False, use_layernorm_attn=False, use_fused_attn=False),
                device=device,
            )
            
            memory_triattn_trimul = memory_measure(
                lambda: fwd(model, s, z, mask, pair_mask, use_cuequiv_mul=True, use_cuequiv_attn=True, use_opt_attn=False, use_turbo_attn=False, use_layernorm_attn=False, use_fused_attn=False),
                device=device,
            )
            
            if HAS_OPTIMIZED:
                memory_hyper = memory_measure(
                    lambda: fwd(model, s, z, mask, pair_mask, use_cuequiv_mul=False, use_cuequiv_attn=False, use_opt_attn=True, use_turbo_attn=False, use_layernorm_attn=False, use_fused_attn=False),
                    device=device,
                )
            else:
                memory_hyper = start_mem
                
            if HAS_TURBO:
                memory_turbo = memory_measure(
                    lambda: fwd(model, s, z, mask, pair_mask, use_cuequiv_mul=False, use_cuequiv_attn=False, use_opt_attn=False, use_turbo_attn=True, use_layernorm_attn=False, use_fused_attn=False),
                    device=device,
                )
            else:
                memory_turbo = start_mem
                
            if HAS_LAYERNORM_OPT:
                memory_layernorm = memory_measure(
                    lambda: fwd(model, s, z, mask, pair_mask, use_cuequiv_mul=False, use_cuequiv_attn=False, use_opt_attn=False, use_turbo_attn=False, use_layernorm_attn=True, use_fused_attn=False),
                    device=device,
                )
                
                memory_fused = memory_measure(
                    lambda: fwd(model, s, z, mask, pair_mask, use_cuequiv_mul=False, use_cuequiv_attn=False, use_opt_attn=False, use_turbo_attn=False, use_layernorm_attn=False, use_fused_attn=True),
                    device=device,
                )
                
                memory_fused_trimul = memory_measure(
                    lambda: fwd(model, s, z, mask, pair_mask, use_cuequiv_mul=True, use_cuequiv_attn=False, use_opt_attn=False, use_turbo_attn=False, use_layernorm_attn=False, use_fused_attn=True),
                    device=device,
                )
            else:
                memory_layernorm = start_mem
                memory_fused = start_mem
                memory_fused_trimul = start_mem
                
            if HAS_ULTRA:
                memory_ultra = memory_measure(
                    lambda: fwd(ultra_model, s, z, mask, pair_mask, use_cuequiv_mul=False, use_cuequiv_attn=False, use_opt_attn=False, use_turbo_attn=False, use_layernorm_attn=False, use_fused_attn=False),
                    device=device,
                )
                memory_hybrid = memory_measure(
                    lambda: fwd(hybrid_model, s, z, mask, pair_mask, use_cuequiv_mul=False, use_cuequiv_attn=False, use_opt_attn=False, use_turbo_attn=False, use_layernorm_attn=False, use_fused_attn=False),
                    device=device,
                )
                memory_minimal = memory_measure(
                    lambda: fwd(minimal_model, s, z, mask, pair_mask, use_cuequiv_mul=False, use_cuequiv_attn=False, use_opt_attn=False, use_turbo_attn=False, use_layernorm_attn=False, use_fused_attn=False),
                    device=device,
                )
            else:
                memory_ultra = start_mem
                memory_hybrid = start_mem
                memory_minimal = start_mem
            
            memory_results.append({
                "size": size,
                "Default": memory_default - start_mem,
                "Trimul": memory_trimul - start_mem,
                "TriAttn+Trimul": memory_triattn_trimul - start_mem,
                "HyperOptAttn": memory_hyper - start_mem,
                "TurboOptAttn": memory_turbo - start_mem,
                "LayerNormOpt": memory_layernorm - start_mem,
                "FusedAttn": memory_fused - start_mem,
                "FusedAttn+Trimul": memory_fused_trimul - start_mem,
                "UltraFused": memory_ultra - start_mem,
                "CuEqHybrid": memory_hybrid - start_mem,
                "MinimalOH": memory_minimal - start_mem,
            })

    memory_df = pd.DataFrame(memory_results)
    print("\nMemory Usage (MB above baseline):")
    print(memory_df)
    
    # Summary analysis
    print("\nSUMMARY ANALYSIS")
    print("=" * 80)
    
    # Show which methods beat Trimul at each size
    trimul_winners = []
    for _, row in performance_df.iterrows():
        if row['Best_Method'] != 'Trimul':
            trimul_winners.append(f"Size {row['Size']}: {row['Best_Method']} beats Trimul by {row['Speedup_vs_Trimul']}")
        else:
            trimul_winners.append(f"Size {row['Size']}: Trimul is still the best")
    
    print("Trimul Comparison Results:")
    for result in trimul_winners:
        print(f"  • {result}")
    
    # Show overall best methods
    best_overall = performance_df.groupby('Best_Method').size().sort_values(ascending=False)
    print(f"\nMost frequent winners:")
    for method, count in best_overall.items():
        print(f"  • {method}: {count}/{len(SEQ_LEN)} sizes")
    
    print("\nNote: Higher speedup values are better (e.g., 1.5x = 50% faster)")
    print("=" * 80)
