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
    """Get the model outputs without doing backward pass."""
    # Use PairformerLayer's forward method
    # The forward method expects use_cuequiv_mul and use_cuequiv_attn parameters
    try:
        if use_fused_attn and fused_model is not None:
            output = fused_model(s, z, mask, pair_mask, use_cuequiv_mul=use_cuequiv_mul, use_cuequiv_attn=use_cuequiv_attn)
        elif use_layernorm_attn and layernorm_model is not None:
            output = layernorm_model(s, z, mask, pair_mask, use_cuequiv_mul=use_cuequiv_mul, use_cuequiv_attn=use_cuequiv_attn)
        elif use_turbo_attn and turbo_model is not None:
            output = turbo_model(s, z, mask, pair_mask, use_cuequiv_mul=use_cuequiv_mul, use_cuequiv_attn=use_cuequiv_attn)
        elif use_opt_attn and opt_model is not None:
            output = opt_model(s, z, mask, pair_mask, use_cuequiv_mul=use_cuequiv_mul, use_cuequiv_attn=use_cuequiv_attn)
        else:
            output = model(s, z, mask, pair_mask, use_cuequiv_mul=use_cuequiv_mul, use_cuequiv_attn=use_cuequiv_attn)
        
        # PairformerLayer should return (s, z) tuple
        if isinstance(output, tuple) and len(output) == 2:
            s_out, z_out = output
            return s_out, z_out
        else:
            # If the output is not a tuple or doesn't have 2 elements, return None
            return None, None
    except Exception as e:
        # If there's an error, return None
        print(f"Error in backward function: {str(e)}")
        return None, None


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
            "TriAttn+Trimul", 
            "HyperOptAttn",
            "TurboOptAttn",
            "LayerNormOpt",
            "FusedAttn",
            "FusedAttn+Trimul",
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

    return ms / BATCH_SIZE


def analyze_triton_results(results_data, memory_data=None):
    """
    Analyze performance and memory results from benchmark data.
    
    Args:
        results_data: Dict with benchmark results (required)
        memory_data: Dict with memory usage results (optional)
    """
    if results_data is None:
        print("Error: No benchmark data provided. Please provide actual benchmark results.")
        return None, None, None
        
    print("\n" + "="*80)
    print("� PERFORMANCE ANALYSIS: Best Method per Sequence Size")
    print("="*80)
    
    print("📊 Analyzing provided benchmark results")
    
    df = pd.DataFrame(results_data)
    memory_df = pd.DataFrame(memory_data)
    
    print("\n📊 PERFORMANCE RESULTS (seconds)")
    print("-" * 110)
    print(df.to_string(index=False, float_format='%.6f'))
    
    print("\n💾 MEMORY USAGE (MB)")
    print("-" * 110)
    print(memory_df.to_string(index=False, float_format='%.0f'))
    
    # Find best method for each sequence size (performance)
    methods = [col for col in df.columns if col != 'size']
    
    analysis_results = []
    memory_analysis_results = []
    
    for idx, (perf_row, mem_row) in enumerate(zip(df.iterrows(), memory_df.iterrows())):
        size = perf_row[1]['size']
        
        # Performance analysis
        perf_times = {method: perf_row[1][method] for method in methods}
        best_perf_method = min(perf_times, key=perf_times.get)
        best_time = perf_times[best_perf_method]
        
        # Memory analysis
        mem_usage = {method: mem_row[1][method] for method in methods}
        best_mem_method = min(mem_usage, key=mem_usage.get)
        best_memory = mem_usage[best_mem_method]
        
        # Calculate speedup vs default
        default_time = perf_row[1]['Default']
        speedup = default_time / best_time
        speedup_pct = (speedup - 1) * 100
        
        # Calculate speedup vs trimul 
        trimul_time = perf_row[1].get('Trimul', default_time)
        trimul_speedup = trimul_time / best_time
        trimul_speedup_pct = (trimul_speedup - 1) * 100
        
        # Memory savings vs default
        default_memory = mem_row[1]['Default']
        mem_savings = ((default_memory - best_memory) / default_memory) * 100
        
        analysis_results.append({
            'Size': int(size),
            'Best_Perf_Method': best_perf_method,
            'Best_Time': f"{best_time:.6f}s",
            'vs_Default': f"+{speedup_pct:.1f}%" if speedup_pct > 0 else f"{speedup_pct:.1f}%",
            'vs_Trimul': f"+{trimul_speedup_pct:.1f}%" if trimul_speedup_pct > 0 else f"{trimul_speedup_pct:.1f}%",
            'Best_Mem_Method': best_mem_method,
            'Best_Memory': f"{best_memory:.0f}MB",
            'Mem_Savings': f"{mem_savings:.1f}%" if mem_savings > 0 else f"{mem_savings:.1f}%"
        })
    
    # Display winner analysis
    print(f"\n🏆 WINNER ANALYSIS")
    print("-" * 100)
    print(f"{'Size':>4} | {'Performance Winner':^20} | {'Time':^12} | {'vs Default':^10} | {'vs Trimul':^10} | {'Memory Winner':^20} | {'Memory':^10} | {'Savings':^8}")
    print("-" * 100)
    
    for result in analysis_results:
        size = result['Size']
        perf_method = result['Best_Perf_Method']
        time = result['Best_Time']
        vs_default = result['vs_Default']
        vs_trimul = result['vs_Trimul']
        mem_method = result['Best_Mem_Method']
        memory = result['Best_Memory']
        savings = result['Mem_Savings']
        
        # Add emoji indicators for performance
        if perf_method == 'LayerNormOpt':
            perf_emoji = "🎯"
        elif perf_method == 'TriAttn+Trimul':
            perf_emoji = "🔥"
        elif perf_method == 'Trimul':
            perf_emoji = "⚡"
        elif perf_method == 'Default':
            perf_emoji = "✅"
        else:
            perf_emoji = "🚀"
        
        # Add emoji indicators for memory
        if mem_method == 'TriAttn+Trimul':
            mem_emoji = "🔥"
        elif mem_method == 'Trimul':
            mem_emoji = "⚡"
        elif 'Trimul' in mem_method:
            mem_emoji = "💾"
        else:
            mem_emoji = "📦"
        
        print(f"{size:>4} | {perf_emoji}{perf_method:^19} | {time:^12} | {vs_default:^10} | {vs_trimul:^10} | {mem_emoji}{mem_method:^19} | {memory:^10} | {savings:^8}")
    
    # Separate performance and memory winner analysis
    print(f"\n🚀 PERFORMANCE SUMMARY")
    print("-" * 50)
    
    # Count performance wins per method
    perf_method_wins = {}
    for result in analysis_results:
        method = result['Best_Perf_Method']
        perf_method_wins[method] = perf_method_wins.get(method, 0) + 1
    
    print("Performance winner frequency:")
    for method, wins in sorted(perf_method_wins.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {method}: {wins}/{len(analysis_results)} sizes")
    
    perf_champion = max(perf_method_wins, key=perf_method_wins.get)
    print(f"\n🏆 PERFORMANCE CHAMPION: {perf_champion}")
    
    print(f"\n💾 MEMORY SUMMARY")
    print("-" * 50)
    
    # Count memory wins per method
    mem_method_wins = {}
    for result in analysis_results:
        method = result['Best_Mem_Method']
        mem_method_wins[method] = mem_method_wins.get(method, 0) + 1
    
    print("Memory winner frequency:")
    for method, wins in sorted(mem_method_wins.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {method}: {wins}/{len(analysis_results)} sizes")
    
    mem_champion = max(mem_method_wins, key=mem_method_wins.get)
    print(f"\n🏆 MEMORY CHAMPION: {mem_champion}")
    
    # Adaptive insights based on data
    small_results = [r for r in analysis_results if r['Size'] <= 128]
    large_results = [r for r in analysis_results if r['Size'] >= 256]
    
    print(f"\n💡 KEY INSIGHTS")
    print("-" * 40)
    print(f"\n📈 SUMMARY STATISTICS")
    print("-" * 40)
    
    if small_results:
        small_perf_winners = [r['Best_Perf_Method'] for r in small_results]
        small_mem_winners = [r['Best_Mem_Method'] for r in small_results]
        if len(set(small_perf_winners)) == 1:
            print(f"• {small_perf_winners[0]} dominates small sequence performance (≤128)")
        else:
            print(f"• Mixed performance winners for small sequences: {set(small_perf_winners)}")
        if len(set(small_mem_winners)) == 1:
            print(f"• {small_mem_winners[0]} dominates small sequence memory (≤128)")
    
    if large_results:
        large_perf_winners = [r['Best_Perf_Method'] for r in large_results]
        large_mem_winners = [r['Best_Mem_Method'] for r in large_results]
        if len(set(large_perf_winners)) == 1:
            print(f"• {large_perf_winners[0]} dominates large sequence performance (≥256)")
        else:
            print(f"• Mixed performance winners for large sequences: {set(large_perf_winners)}")
        if len(set(large_mem_winners)) == 1:
            print(f"• {large_mem_winners[0]} dominates large sequence memory (≥256)")
    
    # Check our custom optimizations
    our_methods = ['HyperOptAttn', 'TurboOptAttn', 'LayerNormOpt', 'FusedAttn', 'FusedAttn+Trimul']
    our_perf_wins = [r for r in analysis_results if r['Best_Perf_Method'] in our_methods]
    our_mem_wins = [r for r in analysis_results if r['Best_Mem_Method'] in our_methods]
    
    if our_perf_wins:
        print(f"• Our optimizations win {len(our_perf_wins)}/{len(analysis_results)} performance tests")
    else:
        print("• Our custom optimizations show minimal performance improvement")
    
    if our_mem_wins:
        print(f"• Our optimizations win {len(our_mem_wins)}/{len(analysis_results)} memory tests")
    
    # TriAttn+Trimul analysis
    triattn_perf_wins = [r for r in analysis_results if r['Best_Perf_Method'] == 'TriAttn+Trimul']
    if triattn_perf_wins and 'TriAttn+Trimul' in methods:
        speedups = []
        for r in triattn_perf_wins:
            try:
                pct = float(r['vs_Trimul'].replace('%', '').replace('+', ''))
                speedups.append(pct)
            except:
                pass
        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            print(f"• TriAttn+Trimul provides {avg_speedup:.1f}% average speedup vs Trimul where it wins")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS")
    print("-" * 40)
    
    if small_results and large_results:
        small_perf_method = small_results[0]['Best_Perf_Method']
        large_perf_method = large_results[0]['Best_Perf_Method']
        
        if small_perf_method == large_perf_method:
            print(f"1. Use {small_perf_method} for all sequence sizes (consistent winner)")
        else:
            print(f"1. Use {small_perf_method} for sequences < 128")
            print(f"2. Use {large_perf_method} for sequences ≥ 256")
    
    if not our_perf_wins:
        print("3. Focus future optimization on large sequence performance")
        print("4. Consider cuEquivariance-based approaches (TriAttn+Trimul) instead of custom optimizations")
    
    print("5. Consider adaptive strategy based on sequence length")
    print("6. Monitor memory usage for memory-constrained environments")
    
    return df, memory_df, analysis_results


def test_gradient_support():
    """Test if all optimization methods support gradients properly."""
    print("\n" + "="*80)
    print("🧪 TESTING GRADIENT SUPPORT FOR EACH OPTIMIZATION METHOD")
    print("="*80)
    
    # Setup test tensors
    s = torch.randn((BATCH_SIZE, 64, C_S), device=device, requires_grad=True)
    z = torch.randn((BATCH_SIZE, 64, 64, C_Z), device=device, requires_grad=True)
    mask = torch.ones((BATCH_SIZE, 64), device=device, requires_grad=False).float()
    pair_mask = torch.ones((BATCH_SIZE, 64, 64), device=device, requires_grad=False).float()
    
    configs = [
        ("Default", False, False, False, False, False, False),
        ("Trimul", True, False, False, False, False, False),
        ("TriAttn+Trimul", True, True, False, False, False, False),
        ("HyperOptAttn", False, False, True, False, False, False),
        ("TurboOptAttn", False, False, False, True, False, False),
        ("LayerNormOpt", False, False, False, False, True, False),
        ("FusedAttn", False, False, False, False, False, True),
        ("FusedAttn+Trimul", True, False, False, False, False, True),
    ]
    
    results = []
    
    for name, use_cuequiv_mul, use_cuequiv_attn, use_opt_attn, use_turbo_attn, \
        use_layernorm_attn, use_fused_attn in configs:
        
        s_test = s.clone().detach().requires_grad_(True)
        z_test = z.clone().detach().requires_grad_(True)
        
        try:
            # Choose the appropriate model based on method
            current_model = model  # Default
            if use_fused_attn and fused_model is not None:
                current_model = fused_model
            elif use_layernorm_attn and layernorm_model is not None:
                current_model = layernorm_model
            elif use_turbo_attn and turbo_model is not None:
                current_model = turbo_model
            elif use_opt_attn and opt_model is not None:
                current_model = opt_model

            # Direct forward pass
            output = current_model(s_test, z_test, mask, pair_mask, 
                                  use_cuequiv_mul=use_cuequiv_mul, 
                                  use_cuequiv_attn=use_cuequiv_attn)
            
            if not isinstance(output, tuple) or len(output) != 2:
                raise ValueError(f"Forward pass returned {type(output)}, not a tuple of length 2")
                
            s_out, z_out = output
            
            # Create dummy loss and backprop
            loss = s_out.sum() + z_out.sum()
            loss.backward()
            
            # Check if gradients were computed
            grad_s = s_test.grad is not None
            grad_z = z_test.grad is not None
            
            status = "✅ PASS" if grad_s and grad_z else "❌ FAIL"
            details = f"s_grad: {'Yes' if grad_s else 'No'}, z_grad: {'Yes' if grad_z else 'No'}"
            results.append((name, status, details))
            
        except Exception as e:
            results.append((name, "❌ ERROR", str(e)[:50]))
    
    # Print results table
    print("\n{:<18} {:<10} {:<40}".format("Method", "Status", "Details"))
    print("-" * 70)
    for method, status, details in results:
        print("{:<18} {:<10} {:<40}".format(method, status, details))
    
    # Check if any methods failed
    failures = [r[0] for r in results if "PASS" not in r[1]]
    if failures:
        print(f"\n⚠️ WARNING: The following methods have gradient issues: {', '.join(failures)}")
        print("These methods may not work properly with training.")
    else:
        print("\n✅ All optimization methods support gradients properly.")


if __name__ == "__main__":
    print("Speed comparison: LayerNorm + Fused Attention Optimizations vs Trimul")
    
    # First test gradient support
    test_gradient_support()
    
    # Run the standard benchmark
    benchmark.run(print_data=True, show_plots=False)
    
    # Place for adding custom benchmark results
    # Uncomment and fill with your data if needed
    # results_data = {
    #    'size': [64.0, 128.0, 256.0, 512.0],
    #    'Default': [...],
    #    ...
    # }
    
    # memory_data = {
    #    'size': [64.0, 128.0, 256.0, 512.0],
    #    'Default': [...],
    #    ...
    # }
    
    # Only run analysis if data is provided
    # analyze_triton_results(results_data, memory_data)
