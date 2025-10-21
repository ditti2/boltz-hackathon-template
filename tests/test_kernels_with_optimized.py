import os
import time

import torch
import triton
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

# Try to import optimized attention
try:
    from boltz.model.layers.optimized_attention import HyperOptimizedAttentionPairBias, TurboOptimizedAttentionPairBias
    HAS_OPTIMIZED = True
    HAS_TURBO = True
except ImportError:
    try:
        from boltz.model.layers.optimized_attention import HyperOptimizedAttentionPairBias
        HAS_OPTIMIZED = True
        HAS_TURBO = False
        print("TurboOptimizedAttentionPairBias not available")
    except ImportError:
        HAS_OPTIMIZED = False
        HAS_TURBO = False
        print("Optimized attention not available")

# Preload modules
model = PairformerLayer(C_S, C_Z, v2=True)
model.cuda()
if INFERENCE:
    model.eval()

# Create optimized model with replaced attention if available
opt_model = None
turbo_model = None
if HAS_OPTIMIZED:
    opt_model = PairformerLayer(C_S, C_Z, v2=True)
    # Replace the attention module with optimized version
    if hasattr(opt_model, 'attention'):
        opt_model.attention = HyperOptimizedAttentionPairBias(
            C_S, C_Z, opt_model.attention.num_heads
        )
    opt_model.cuda()
    if INFERENCE:
        opt_model.eval()

if HAS_TURBO:
    turbo_model = PairformerLayer(C_S, C_Z, v2=True)
    # Replace with turbo optimized version
    if hasattr(turbo_model, 'attention'):
        turbo_model.attention = TurboOptimizedAttentionPairBias(
            C_S, C_Z, turbo_model.attention.num_heads
        )
    turbo_model.cuda()
    if INFERENCE:
        turbo_model.eval()


def fwd(model, s, z, mask, pair_mask, use_cuequiv_mul=False, use_cuequiv_attn=False, use_opt_attn=False, use_turbo_attn=False):
    if use_turbo_attn and turbo_model is not None:
        turbo_model(s, z, mask, pair_mask, use_cuequiv_mul=use_cuequiv_mul, use_cuequiv_attn=use_cuequiv_attn)
    elif use_opt_attn and opt_model is not None:
        opt_model(s, z, mask, pair_mask, use_cuequiv_mul=use_cuequiv_mul, use_cuequiv_attn=use_cuequiv_attn)
    else:
        model(s, z, mask, pair_mask, use_cuequiv_mul=use_cuequiv_mul, use_cuequiv_attn=use_cuequiv_attn)


def backward(model, s, z, mask, pair_mask, use_cuequiv_mul=False, use_cuequiv_attn=False, use_opt_attn=False, use_turbo_attn=False):
    if use_turbo_attn and turbo_model is not None:
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
            "HyperOptAttn+Trimul",
            "TurboOptAttn",
            "TurboOptAttn+Trimul",
        ],
        line_names=[
            "Default",
            "Trimul",
            "HyperOptAttn", 
            "HyperOptAttn+Trimul",
            "TurboOptAttn",
            "TurboOptAttn+Trimul", 
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
                )
            )
        elif provider == "HyperOptAttn+Trimul":
            if not HAS_OPTIMIZED:
                return float('nan')
            ms = speed(
                lambda: fn(
                    model, s, z, mask, pair_mask,
                    use_cuequiv_attn=False,
                    use_cuequiv_mul=True,
                    use_opt_attn=True,
                    use_turbo_attn=False,
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
                )
            )
        elif provider == "TurboOptAttn+Trimul":
            if not HAS_TURBO:
                return float('nan')
            ms = speed(
                lambda: fn(
                    model, s, z, mask, pair_mask,
                    use_cuequiv_attn=False,
                    use_cuequiv_mul=True,
                    use_opt_attn=False,
                    use_turbo_attn=True,
                )
            )

    return ms / BATCH_SIZE


if __name__ == "__main__":
    print("Speed comparison: Turbo & Hyper-Optimized Attention vs Trimul")
    benchmark.run(print_data=True, show_plots=False)
