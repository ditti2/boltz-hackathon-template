#!/usr/bin/env python3
"""
Test script to demonstrate how to use cuEquivariance AttentionPairBias kernel 
in the Boltz model.

This script shows:
1. How the current Boltz AttentionPairBias implementation works
2. How to use the cuEquivariance attention_pair_bias kernel
3. Performance comparison between the two approaches
"""

import torch
import torch.nn as nn
import time
import math
from typing import Optional

# Try to import cuEquivariance
try:
    from cuequivariance_torch import attention_pair_bias as cueq_attention_pair_bias
    HAS_CUEQUIVARIANCE = True
    print("✓ cuEquivariance is available")
except ImportError:
    HAS_CUEQUIVARIANCE = False
    print("✗ cuEquivariance is not available")

# Import the current Boltz implementation
import sys
import os
sys.path.append('./src')

from boltz.model.layers.attention import AttentionPairBias


class CuEquivAttentionPairBias(nn.Module):
    """
    AttentionPairBias layer using cuEquivariance kernel.
    
    This demonstrates how to use the cuEquivariance attention_pair_bias 
    kernel as a drop-in replacement for the current Boltz implementation.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        num_heads: int,
        inf: float = 1e6,
    ) -> None:
        super().__init__()
        
        assert c_s % num_heads == 0
        
        self.c_s = c_s
        self.c_z = c_z  
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.inf = inf

        # Projection layers - same as Boltz implementation
        self.proj_q = nn.Linear(c_s, c_s)
        self.proj_k = nn.Linear(c_s, c_s, bias=False)
        self.proj_v = nn.Linear(c_s, c_s, bias=False)
        self.proj_g = nn.Linear(c_s, c_s, bias=False)
        self.proj_o = nn.Linear(c_s, c_s, bias=False)
        
        # Layer norm and projection for z (pairwise features)
        self.norm_z = nn.LayerNorm(c_z)
        self.proj_z = nn.Linear(c_z, num_heads, bias=False)

    def forward(
        self,
        s: torch.Tensor,  # (B, S, D)
        z: torch.Tensor,  # (B, N, N, D_z)
        mask: torch.Tensor,  # (B, N)
        multiplicity: int = 1,
    ) -> torch.Tensor:
        """
        Forward pass using cuEquivariance attention_pair_bias kernel.
        
        Args:
            s: Input sequence tensor (B, S, D)
            z: Input pairwise tensor (B, N, N, D_z) 
            mask: Mask tensor (B, N)
            multiplicity: Diffusion batch size multiplier
            
        Returns:
            Output sequence tensor (B, S, D)
        """
        if not HAS_CUEQUIVARIANCE:
            raise RuntimeError("cuEquivariance is not available")
            
        B, S, D = s.shape
        
        # Compute q, k, v projections
        q = self.proj_q(s).view(B, S, self.num_heads, self.head_dim)
        k = self.proj_k(s).view(B, S, self.num_heads, self.head_dim) 
        v = self.proj_v(s).view(B, S, self.num_heads, self.head_dim)
        
        # Reshape for cuEquivariance format: (B, H, S, head_dim)
        q = q.transpose(1, 2)  # (B, H, S, head_dim)
        k = k.transpose(1, 2)  # (B, H, S, head_dim)
        v = v.transpose(1, 2)  # (B, H, S, head_dim)
        
        # Handle multiplicity by repeating tensors
        if multiplicity > 1:
            q = q.repeat_interleave(multiplicity, 0)
            k = k.repeat_interleave(multiplicity, 0)
            v = v.repeat_interleave(multiplicity, 0)
            s = s.repeat_interleave(multiplicity, 0)
            mask = mask.repeat_interleave(multiplicity, 0)
        
        # Prepare z tensor (pairwise features) - normalize and project
        z_norm = self.norm_z(z)  # (B, N, N, D_z)
        
        # Project to head dimension and prepare weights for cuEquivariance
        w_proj_z = self.proj_z.weight  # (num_heads, c_z)
        w_proj_g = self.proj_g.weight  # (c_s, c_s)
        w_proj_o = self.proj_o.weight  # (c_s, c_s)
        w_ln_z = self.norm_z.weight    # (c_z,)
        b_ln_z = self.norm_z.bias      # (c_z,)
        
        # Compute attention scale
        attn_scale = 1.0 / math.sqrt(self.head_dim)
        
        # Call cuEquivariance attention_pair_bias kernel
        output, proj_z = cueq_attention_pair_bias(
            s=s.reshape(B * multiplicity, S, D),  # Flatten batch dimension
            q=q,  # (B*M, H, S, head_dim)
            k=k,  # (B*M, H, S, head_dim) 
            v=v,  # (B*M, H, S, head_dim)
            z=z,  # (B, N, N, c_z)
            mask=mask,  # (B*M, S)
            num_heads=self.num_heads,
            w_proj_z=w_proj_z,  # (num_heads, c_z)
            w_proj_g=w_proj_g,  # (c_s, c_s)
            w_proj_o=w_proj_o,  # (c_s, c_s)
            w_ln_z=w_ln_z,      # (c_z,)
            b_ln_z=b_ln_z,      # (c_z,)
            attn_scale=attn_scale,
            inf=self.inf,
            return_z_proj=True,
        )
        
        # Reshape output back to original batch dimensions
        output = output.view(B, multiplicity, S, D)
        if multiplicity > 1:
            output = output[:, 0]  # Take first multiplicity dimension
            
        return output


def compare_implementations():
    """Compare Boltz vs cuEquivariance implementations."""
    
    # Test parameters
    batch_size = 2
    seq_len = 64
    c_s = 128  # sequence feature dimension
    c_z = 64   # pairwise feature dimension
    num_heads = 8
    multiplicity = 1
    
    print(f"\nTest configuration:")
    print(f"- Batch size: {batch_size}")
    print(f"- Sequence length: {seq_len}")
    print(f"- Sequence features (c_s): {c_s}")
    print(f"- Pairwise features (c_z): {c_z}")
    print(f"- Number of heads: {num_heads}")
    print(f"- Multiplicity: {multiplicity}")
    
    # Create test data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"- Device: {device}")
    
    # Input tensors
    s = torch.randn(batch_size, seq_len, c_s, device=device, dtype=torch.float32)
    z = torch.randn(batch_size, seq_len, seq_len, c_z, device=device, dtype=torch.float32)
    mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.float32)
    
    # Apply some masking
    mask[:, seq_len//2:] = 0  # Mask second half
    
    print(f"\nInput shapes:")
    print(f"- s (sequence): {s.shape}")
    print(f"- z (pairwise): {z.shape}")
    print(f"- mask: {mask.shape}")
    
    # Test 1: Original Boltz implementation
    print(f"\n{'='*50}")
    print("Testing Original Boltz Implementation")
    print(f"{'='*50}")
    
    boltz_model = AttentionPairBias(c_s, c_z, num_heads).to(device)
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = boltz_model(s, z, mask, multiplicity)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    for _ in range(10):
        with torch.no_grad():
            boltz_output = boltz_model(s, z, mask, multiplicity)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    boltz_time = (time.time() - start_time) / 10
    
    print(f"✓ Boltz output shape: {boltz_output.shape}")
    print(f"✓ Boltz average time: {boltz_time*1000:.2f} ms")
    
    # Test 2: cuEquivariance implementation (if available)
    if HAS_CUEQUIVARIANCE:
        print(f"\n{'='*50}")
        print("Testing cuEquivariance Implementation")
        print(f"{'='*50}")
        
        try:
            cueq_model = CuEquivAttentionPairBias(c_s, c_z, num_heads).to(device)
            
            # Copy weights from Boltz model for fair comparison
            cueq_model.proj_q.weight.data = boltz_model.proj_q.weight.data.clone()
            cueq_model.proj_k.weight.data = boltz_model.proj_k.weight.data.clone()
            cueq_model.proj_v.weight.data = boltz_model.proj_v.weight.data.clone()
            cueq_model.proj_g.weight.data = boltz_model.proj_g.weight.data.clone()
            cueq_model.proj_o.weight.data = boltz_model.proj_o.weight.data.clone()
            
            # Copy normalization parameters
            cueq_model.norm_z.weight.data = boltz_model.proj_z[0].weight.data.clone()
            cueq_model.norm_z.bias.data = boltz_model.proj_z[0].bias.data.clone()
            cueq_model.proj_z.weight.data = boltz_model.proj_z[1].weight.data.clone()
            
            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = cueq_model(s, z, mask, multiplicity)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = time.time()
            
            for _ in range(10):
                with torch.no_grad():
                    cueq_output = cueq_model(s, z, mask, multiplicity)
            
            torch.cuda.synchronize() if device.type == 'cuda' else None
            cueq_time = (time.time() - start_time) / 10
            
            print(f"✓ cuEquivariance output shape: {cueq_output.shape}")
            print(f"✓ cuEquivariance average time: {cueq_time*1000:.2f} ms")
            
            # Compare outputs
            max_diff = torch.max(torch.abs(boltz_output - cueq_output))
            rel_diff = max_diff / torch.max(torch.abs(boltz_output))
            
            print(f"\n{'='*50}")
            print("Comparison Results")
            print(f"{'='*50}")
            print(f"✓ Max absolute difference: {max_diff:.6f}")
            print(f"✓ Max relative difference: {rel_diff:.6f}")
            print(f"✓ Speedup: {boltz_time/cueq_time:.2f}x")
            
            if max_diff < 1e-3:
                print("✓ Outputs are very similar!")
            elif max_diff < 1e-2:
                print("⚠ Outputs have small differences (expected due to implementation)")
            else:
                print("✗ Outputs have significant differences")
                
        except Exception as e:
            print(f"✗ cuEquivariance test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n{'='*50}")
        print("cuEquivariance Not Available")
        print(f"{'='*50}")
        print("To test cuEquivariance, install with:")
        print("pip install cuequivariance-torch cuequivariance-ops-torch-cu12")


def main():
    """Main function to run the comparison."""
    print("AttentionPairBias Kernel Comparison")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, running on CPU (will be slow)")
    
    compare_implementations()
    
    print(f"\n{'='*50}")
    print("Integration Recommendations")
    print(f"{'='*50}")
    print("1. Install cuEquivariance dependencies:")
    print("   pip install cuequivariance-torch cuequivariance-ops-torch-cu12")
    print("")
    print("2. Add cuEquivariance support to AttentionPairBias:")
    print("   - Add use_kernels parameter to forward method")
    print("   - Implement cuEquivariance code path similar to triangular attention")
    print("")
    print("3. Expected benefits:")
    print("   - Faster computation for larger sequences")
    print("   - Better memory efficiency")
    print("   - Optimized CUDA kernels")
    print("")
    print("4. Integration points in Boltz:")
    print("   - src/boltz/model/layers/attention.py (AttentionPairBias)")
    print("   - src/boltz/model/layers/attentionv2.py (AttentionPairBias)")
    print("   - Add use_kernels support in PairformerLayer")


if __name__ == "__main__":
    main()