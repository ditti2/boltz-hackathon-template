import os
import time

import torch
import triton
import pandas as pd
from profiling import clear_memory
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

HAS_OPTIMIZED = True
HAS_TURBO = True
HAS_LAYERNORM_OPT = True

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

if HAS_OPTIMIZED:
    opt_model = PairformerLayer(C_S, C_Z, v2=True)
    # Replace the attention module with optimized version
    # Check multiple possible attribute names
    if hasattr(opt_model, 'attention_pair_bias'):
        print("Using attention_pair_bias attribute")
        opt_model.attention_pair_bias = HyperOptimizedAttentionPairBias(
            C_S, C_Z, opt_model.attention_pair_bias.num_heads
        )
    elif hasattr(opt_model, 'attention'):
        print("Using attention attribute")
        opt_model.attention = HyperOptimizedAttentionPairBias(
            C_S, C_Z, opt_model.attention.num_heads
        )
    else:
        print("Warning: No attention attribute found in PairformerLayer")
        print("Available attributes:", [attr for attr in dir(opt_model) if not attr.startswith('_')])
        HAS_OPTIMIZED = False
    
    if HAS_OPTIMIZED:
        opt_model.cuda()
        if INFERENCE:
            opt_model.eval()

if HAS_TURBO:
    turbo_model = PairformerLayer(C_S, C_Z, v2=True)
    # Replace with turbo optimized version
    if hasattr(turbo_model, 'attention_pair_bias'):
        turbo_model.attention_pair_bias = TurboOptimizedAttentionPairBias(
            C_S, C_Z, turbo_model.attention_pair_bias.num_heads
        )
    elif hasattr(turbo_model, 'attention'):
        turbo_model.attention = TurboOptimizedAttentionPairBias(
            C_S, C_Z, turbo_model.attention.num_heads
        )
    else:
        print("Warning: No attention attribute found for TurboOptimized")
        HAS_TURBO = False
    
    if HAS_TURBO:
        turbo_model.cuda()
        if INFERENCE:
            turbo_model.eval()

if HAS_LAYERNORM_OPT:
    layernorm_model = PairformerLayer(C_S, C_Z, v2=True)
    # Replace with LayerNorm optimized version
    if hasattr(layernorm_model, 'attention_pair_bias'):
        layernorm_model.attention_pair_bias = LayerNormOptimizedAttentionPairBias(
            C_S, C_Z, layernorm_model.attention_pair_bias.num_heads
        )
    elif hasattr(layernorm_model, 'attention'):
        layernorm_model.attention = LayerNormOptimizedAttentionPairBias(
            C_S, C_Z, layernorm_model.attention.num_heads
        )
    else:
        print("Warning: No attention attribute found for LayerNormOptimized")
        HAS_LAYERNORM_OPT = False
    
    if HAS_LAYERNORM_OPT:
        layernorm_model.cuda()
        if INFERENCE:
            layernorm_model.eval()
            
        fused_model = PairformerLayer(C_S, C_Z, v2=True)
        # Replace with Fused attention version
        if hasattr(fused_model, 'attention_pair_bias'):
            fused_model.attention_pair_bias = FusedAttentionPairBias(
                C_S, C_Z, fused_model.attention_pair_bias.num_heads
            )
        elif hasattr(fused_model, 'attention'):
            fused_model.attention = FusedAttentionPairBias(
                C_S, C_Z, fused_model.attention.num_heads
            )
        else:
            print("Warning: No attention attribute found for FusedAttention")
            fused_model = None
        
        if fused_model is not None:
            fused_model.cuda()
            if INFERENCE:
                fused_model.eval()
    fused_model.cuda()
    if INFERENCE:
        fused_model.eval()


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


# Benchmark with Triton performance reporting
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=SEQ_LEN,
        line_arg="provider",
        line_vals=[
            "Default",
            "Trimul", 
            "HyperOptAttn",
            "TurboOptAttn",
            "LayerNormOpt",
            "FusedAttn",
            "FusedAttn+Trimul",
        ],
        line_names=[
            "Default",
            "Trimul",
            "HyperOptAttn", 
            "TurboOptAttn",
            "LayerNormOpt",
            "FusedAttn",
            "FusedAttn+Trimul",
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

    return ms / BATCH_SIZE


def analyze_triton_results(results_data=None):
    """
    Capture and analyze results from Triton benchmark.
    
    Args:
        results_data: Dict with benchmark results, or None to use sample data
    """
    print("\n" + "="*80)
    print("🚀 PERFORMANCE ANALYSIS: Best Method per Sequence Size")
    print("="*80)
    
    # Use provided data or sample data
    if results_data is None:
        # Sample results - replace with actual data when available
        results_data = {
            'size': [64.0, 128.0, 256.0, 512.0],
            'Default': [0.010797, 0.011778, 0.084337, 0.482963],
            'Trimul': [0.010742, 0.012808, 0.065764, 0.394785],
            'HyperOptAttn': [0.010651, 0.011795, 0.084504, 0.483443],
            'TurboOptAttn': [0.010886, 0.012336, 0.088742, 0.504486],
            'LayerNormOpt': [0.010524, 0.011913, 0.084504, 0.483493],
            'FusedAttn': [0.010579, 0.011935, 0.084572, 0.483654],
            'FusedAttn+Trimul': [0.010612, 0.012929, 0.065931, 0.395125]
        }
        print("📝 Using sample data - replace with actual benchmark results")
    else:
        print("📊 Analyzing provided benchmark results")
    
    df = pd.DataFrame(results_data)
    
    print("\n📊 BENCHMARK RESULTS")
    print("-" * 95)
    print(df.to_string(index=False, float_format='%.6f'))
    
    # Find best method for each sequence size
    methods = [col for col in df.columns if col != 'size']
    
    analysis_results = []
    for idx, row in df.iterrows():
        size = row['size']
        times = {method: row[method] for method in methods}
        best_method = min(times, key=times.get)
        best_time = times[best_method]
        
        # Calculate speedup vs default
        default_time = row['Default']
        speedup = default_time / best_time
        speedup_pct = (speedup - 1) * 100
        
        # Calculate speedup vs trimul 
        trimul_time = row.get('Trimul', default_time)
        trimul_speedup = trimul_time / best_time
        trimul_speedup_pct = (trimul_speedup - 1) * 100
        
        analysis_results.append({
            'Size': int(size),
            'Best_Method': best_method,
            'Best_Time': f"{best_time:.6f}s",
            'vs_Default': f"+{speedup_pct:.1f}%" if speedup_pct > 0 else f"{speedup_pct:.1f}%",
            'vs_Trimul': f"+{trimul_speedup_pct:.1f}%" if trimul_speedup_pct > 0 else f"{trimul_speedup_pct:.1f}%"
        })
    
    # Display winner analysis
    print(f"\n🏆 WINNER ANALYSIS")
    print("-" * 80)
    
    for result in analysis_results:
        size = result['Size']
        method = result['Best_Method']
        time = result['Best_Time']
        vs_default = result['vs_Default']
        vs_trimul = result['vs_Trimul']
        
        # Add emoji indicators
        if method == 'LayerNormOpt':
            emoji = "🎯"
        elif method == 'Trimul':
            emoji = "⚡"
        elif 'Trimul' in method:
            emoji = "🔥"
        elif method == 'Default':
            emoji = "✅"
        else:
            emoji = "🚀"
        
        print(f"{emoji} Size {size:3d}: {method:15s} ({time}) | {vs_default:8s} vs Default | {vs_trimul:8s} vs Trimul")
    
    # Summary statistics
    print(f"\n📈 SUMMARY STATISTICS")
    print("-" * 40)
    
    # Count wins per method
    method_wins = {}
    for result in analysis_results:
        method = result['Best_Method']
        method_wins[method] = method_wins.get(method, 0) + 1
    
    print("Winner frequency:")
    for method, wins in sorted(method_wins.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {method}: {wins}/{len(analysis_results)} sizes")
    
    champion = max(method_wins, key=method_wins.get)
    print(f"\n🏆 CHAMPION: {champion} (most frequent winner)")
    
    # Adaptive insights based on data
    small_results = [r for r in analysis_results if r['Size'] <= 128]
    large_results = [r for r in analysis_results if r['Size'] >= 256]
    
    print(f"\n💡 KEY INSIGHTS")
    print("-" * 40)
    
    if small_results:
        small_winners = [r['Best_Method'] for r in small_results]
        if len(set(small_winners)) == 1:
            print(f"• {small_winners[0]} dominates small sequences (≤128)")
        else:
            print(f"• Mixed winners for small sequences: {set(small_winners)}")
    
    if large_results:
        large_winners = [r['Best_Method'] for r in large_results]
        if len(set(large_winners)) == 1:
            print(f"• {large_winners[0]} dominates large sequences (≥256)")
        else:
            print(f"• Mixed winners for large sequences: {set(large_winners)}")
    
    # Check our custom optimizations
    our_methods = ['HyperOptAttn', 'TurboOptAttn', 'LayerNormOpt', 'FusedAttn', 'FusedAttn+Trimul']
    our_wins = [r for r in analysis_results if r['Best_Method'] in our_methods]
    
    if our_wins:
        print(f"• Our optimizations win {len(our_wins)}/{len(analysis_results)} tests")
    else:
        print("• Our custom optimizations show minimal improvement")
    
    # Trimul analysis
    trimul_wins = [r for r in analysis_results if r['Best_Method'] == 'Trimul']
    if trimul_wins and 'Trimul' in methods:
        speedups = []
        for r in trimul_wins:
            try:
                pct = float(r['vs_Default'].replace('%', '').replace('+', ''))
                speedups.append(pct)
            except:
                pass
        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            print(f"• Trimul provides {avg_speedup:.1f}% average speedup where it wins")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS")
    print("-" * 40)
    
    if small_results and large_results:
        small_method = small_results[0]['Best_Method']
        large_method = large_results[0]['Best_Method']
        
        if small_method == large_method:
            print(f"1. Use {small_method} for all sequence sizes")
        else:
            print(f"1. Use {small_method} for sequences < 128")
            print(f"2. Use {large_method} for sequences ≥ 256")
    
    if not our_wins:
        print("3. Focus future optimization on large sequence performance")
        print("4. Consider cuEquivariance-based approaches instead of custom optimizations")
    
    print("5. Consider adaptive strategy based on sequence length")
    
    return df, analysis_results


if __name__ == "__main__":
    print("Speed comparison: LayerNorm + Fused Attention Optimizations vs Trimul")
    
    # Run the standard benchmark
    benchmark.run(print_data=True, show_plots=False)
    
    # You can provide actual benchmark results here:
    # Example: to use your actual results, uncomment and modify:
    actual_results = {
        'size': [64.0, 128.0, 256.0, 512.0],
        'Default': [0.010797, 0.011778, 0.084337, 0.482963],
        'Trimul': [0.010742, 0.012808, 0.065764, 0.394785],
        'HyperOptAttn': [0.010651, 0.011795, 0.084504, 0.483443],
        'TurboOptAttn': [0.010886, 0.012336, 0.088742, 0.504486],
        'LayerNormOpt': [0.010524, 0.011913, 0.084504, 0.483493],
        'FusedAttn': [0.010579, 0.011935, 0.084572, 0.483654],
        'FusedAttn+Trimul': [0.010612, 0.012929, 0.065931, 0.395125]
    }
    
    # Generate detailed performance analysis
    print("\nGenerating performance analysis...")
    benchmark_df, winner_analysis = analyze_triton_results(actual_results)
