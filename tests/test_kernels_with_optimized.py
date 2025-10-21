import os
import time

import pandas as pd
import torch
import triton
from profiling import clear_memory, current_memory, memory_measure

from boltz.model.layers.pairformer import PairformerLayer

# Disable auto-tuning
os.environ["CUEQ_DEFAULT_CONFIG"] = "1"
os.environ["CUEQ_DISABLE_AOT_TUNING"] = "1"

# Set hyperparameters
C_S = 384
C_Z = 128
BATCH_SIZE = 1
INFERENCE = False
SEQ_LEN = [128, 256, 384, 512] #, 768]
PRECISION = torch.bfloat16
COMPILE = False
device = "cuda:0"
torch.set_grad_enabled(not INFERENCE)

# Preload modules
model = PairformerLayer(C_S, C_Z, v2=True)
model.cuda()
if COMPILE:
    model = torch.compile(model, fullgraph=True, dynamic=False)

if INFERENCE:
    model.eval()


def fwd(
    model,
    s,
    z,
    mask,
    pair_mask,
    use_cuequiv_mul=False,
    use_cuequiv_attn=False,
):
    model(
        s,
        z,
        mask,
        pair_mask,
        use_cuequiv_mul=use_cuequiv_mul,
        use_cuequiv_attn=use_cuequiv_attn,
    )


def backward(
    model,
    s,
    z,
    mask,
    pair_mask,
    use_cuequiv_mul=False,
    use_cuequiv_attn=False,
):
    s, z = model(
        s,
        z,
        mask,
        pair_mask,
        use_cuequiv_mul=use_cuequiv_mul,
        use_cuequiv_attn=use_cuequiv_attn,
    )
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


# Full model
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=SEQ_LEN,
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=[
            "Default",
            "TriAttn",
            "Trimul",
            "TriAttn+Trimul",
        ],  # Possible values for `line_arg`.
        line_names=[
            "Default",
            "TriAttn",
            "Trimul",
            "TriAttn+Trimul",
        ],  # Label name for the lines.
        plot_name="performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(size, provider):
    clear_memory(device)

    # Now run the benchmark
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
                    model,
                    s,
                    z,
                    mask,
                    pair_mask,
                    use_cuequiv_mul=False,
                    use_cuequiv_attn=False,
                )
            )
        elif provider == "TriAttn":
            ms = speed(
                lambda: fn(
                    model,
                    s,
                    z,
                    mask,
                    pair_mask,
                    use_cuequiv_attn=True,
                    use_cuequiv_mul=False,
                )
            )
        elif provider == "Trimul":
            ms = speed(
                lambda: fn(
                    model,
                    s,
                    z,
                    mask,
                    pair_mask,
                    use_cuequiv_attn=False,
                    use_cuequiv_mul=True,
                )
            )
        elif provider == "TriAttn+Trimul":
            ms = speed(
                lambda: fn(
                    model,
                    s,
                    z,
                    mask,
                    pair_mask,
                    use_cuequiv_attn=True,
                    use_cuequiv_mul=True,
                )
            )

    # Compute throughput in sequences per second
    return ms / BATCH_SIZE


print("Speed")
benchmark.run(print_data=True, show_plots=False)

start_mem = current_memory(device)

df = []
for size in SEQ_LEN:
    print(size)
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
        memory_default = memory_measure(
            lambda: fwd(
                model,
                s,
                z,
                mask,
                pair_mask,
                use_cuequiv_mul=False,
                use_cuequiv_attn=False,
            ),
            device=device,
        )
        memory_attn = memory_measure(
            lambda: fwd(
                model,
                s,
                z,
                mask,
                pair_mask,
                use_cuequiv_mul=False,
                use_cuequiv_attn=True,
            ),
            device=device,
        )
        memory_mul = memory_measure(
            lambda: fwd(
                model,
                s,
                z,
                mask,
                pair_mask,
                use_cuequiv_mul=True,
                use_cuequiv_attn=False,
            ),
            device=device,
        )
        memory_flash = memory_measure(
            lambda: fwd(
                model,
                s,
                z,
                mask,
                pair_mask,
                use_cuequiv_mul=True,
                use_cuequiv_attn=True,
            ),
            device=device,
        )
        df.append(
            {
                "size": size,
                "Default": memory_default - start_mem,
                "TriAttn": memory_attn - start_mem,
                "Trimul": memory_mul - start_mem,
                "TriAttn+Trimul": memory_flash - start_mem,
            }
        )

df = pd.DataFrame(df)
print("Memory")
print(df)


# Attention-only benchmark based on optimization results
def benchmark_attention_only():
    """
    Test optimized attention implementations based on performance results.
    """
    print("\n" + "="*60)
    print("ATTENTION OPTIMIZATION BENCHMARK")
    print("="*60)
    
    try:
        from boltz.model.layers.attention import AttentionPairBias as BoltzAttention
        from boltz.model.layers.optimized_attention import OptimizedAttentionPairBias as OptAttention
    except ImportError as e:
        print(f"Cannot import attention modules: {e}")
        return
    
    # Best performing configurations from results
    configs = [
        (64, 1),   # 1.06x speedup
        (64, 2),   # 1.35x speedup  
        (64, 4),   # 1.51x speedup (best)
        (128, 2),  # 1.14x speedup
        (256, 2),  # 1.28x speedup
        (512, 2),  # 1.04x speedup
    ]
    
    c_s = 128
    c_z = 64
    num_heads = 8
    
    print(f"{'Seq Len':8} {'Batch':8} {'Model':15} {'Time (ms)':12} {'Speedup':10}")
    print(f"{'-'*8} {'-'*8} {'-'*15} {'-'*12} {'-'*10}")
    
    for seq_len, batch_size in configs:
        clear_memory(device)
        
        # Create test data
        s = torch.randn((batch_size, seq_len, c_s), device=device, dtype=PRECISION)
        z = torch.randn((batch_size, seq_len, seq_len, c_z), device=device, dtype=PRECISION)
        mask = torch.ones((batch_size, seq_len), device=device, dtype=PRECISION).float()
        
        # Test Boltz attention
        boltz_model = BoltzAttention(c_s, c_z, num_heads).to(device)
        with torch.autocast("cuda", dtype=PRECISION):
            boltz_time = speed(lambda: boltz_model(s, z, mask)) * 1000
        
        # Test optimized attention
        opt_model = OptAttention(c_s, c_z, num_heads).to(device)
        with torch.autocast("cuda", dtype=PRECISION):
            opt_time = speed(lambda: opt_model(s, z, mask)) * 1000
        
        speedup = boltz_time / opt_time
        
        print(f"{seq_len:8d} {batch_size:8d} {'boltz':15} {boltz_time:12.2f} {'1.00x':10}")
        print(f"{seq_len:8d} {batch_size:8d} {'optimized':15} {opt_time:12.2f} {speedup:10.2f}x")


if __name__ == "__main__":
    # Run original speed benchmark
    print("Speed")
    benchmark.run(print_data=True, show_plots=False)
    
    # Run memory tests (inline)
    print("\nMemory")
    print(df)
    
    # Run attention optimization benchmark
    benchmark_attention_only()
