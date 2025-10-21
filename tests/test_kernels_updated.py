import os
import time

import pandas as pd
import torch
import triton
from profiling import clear_memory, current_memory, memory_measure

from boltz.model.layers.pairformer import PairformerLayer

# Try to import optimized attention
try:
    from boltz.model.layers.optimized_attention import OptimizedAttentionPairBias
    HAS_OPTIMIZED = True
except ImportError:
    HAS_OPTIMIZED = False
    print("OptimizedAttentionPairBias not available")

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

# Create optimized model with replaced attention if available
opt_model = None
if HAS_OPTIMIZED:
    opt_model = PairformerLayer(C_S, C_Z, v2=True)
    # Replace the attention module with optimized version
    if hasattr(opt_model, 'attention'):
        opt_model.attention = OptimizedAttentionPairBias(
            opt_model.attention.c_s, 
            opt_model.attention.c_z, 
            opt_model.attention.num_heads
        )
    opt_model.cuda()
    if COMPILE:
        opt_model = torch.compile(opt_model, fullgraph=True, dynamic=False)

if INFERENCE:
    model.eval()
    if opt_model is not None:
        opt_model.eval()


def fwd(
    model,
    s,
    z,
    mask,
    pair_mask,
    use_cuequiv_mul=False,
    use_cuequiv_attn=False,
    use_opt_attn=False,
):
    if use_opt_attn and opt_model is not None:
        opt_model(
            s,
            z,
            mask,
            pair_mask,
            use_cuequiv_mul=use_cuequiv_mul,
            use_cuequiv_attn=use_cuequiv_attn,
        )
    else:
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
    use_opt_attn=False,
):
    if use_opt_attn and opt_model is not None:
        s, z = opt_model(
            s,
            z,
            mask,
            pair_mask,
            use_cuequiv_mul=use_cuequiv_mul,
            use_cuequiv_attn=use_cuequiv_attn,
        )
    else:
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
            "OptAttn",
            "OptAttn+Trimul",
        ],  # Possible values for `line_arg`.
        line_names=[
            "Default",
            "TriAttn", 
            "Trimul",
            "TriAttn+Trimul",
            "OptAttn",
            "OptAttn+Trimul",
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
                    use_opt_attn=False,
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
                    use_opt_attn=False,
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
                    use_opt_attn=False,
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
                    use_opt_attn=False,
                )
            )
        elif provider == "OptAttn":
            if not HAS_OPTIMIZED:
                return float('nan')  # Skip if optimized attention not available
            ms = speed(
                lambda: fn(
                    model,
                    s,
                    z,
                    mask,
                    pair_mask,
                    use_cuequiv_attn=False,
                    use_cuequiv_mul=False,
                    use_opt_attn=True,
                )
            )
        elif provider == "OptAttn+Trimul":
            if not HAS_OPTIMIZED:
                return float('nan')  # Skip if optimized attention not available
            ms = speed(
                lambda: fn(
                    model,
                    s,
                    z,
                    mask,
                    pair_mask,
                    use_cuequiv_attn=False,
                    use_cuequiv_mul=True,
                    use_opt_attn=True,
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
                use_opt_attn=False,
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
                use_opt_attn=False,
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
                use_opt_attn=False,
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
                use_opt_attn=False,
            ),
            device=device,
        )
        
        # Add optimized attention measurements if available
        if HAS_OPTIMIZED:
            memory_opt = memory_measure(
                lambda: fwd(
                    model,
                    s,
                    z,
                    mask,
                    pair_mask,
                    use_cuequiv_mul=False,
                    use_cuequiv_attn=False,
                    use_opt_attn=True,
                ),
                device=device,
            )
            memory_opt_mul = memory_measure(
                lambda: fwd(
                    model,
                    s,
                    z,
                    mask,
                    pair_mask,
                    use_cuequiv_mul=True,
                    use_cuequiv_attn=False,
                    use_opt_attn=True,
                ),
                device=device,
            )
        else:
            memory_opt = start_mem
            memory_opt_mul = start_mem
            
        df.append(
            {
                "size": size,
                "Default": memory_default - start_mem,
                "TriAttn": memory_attn - start_mem,
                "Trimul": memory_mul - start_mem,
                "TriAttn+Trimul": memory_flash - start_mem,
                "OptAttn": memory_opt - start_mem,
                "OptAttn+Trimul": memory_opt_mul - start_mem,
            }
        )

df = pd.DataFrame(df)
print("Memory")
print(df)
