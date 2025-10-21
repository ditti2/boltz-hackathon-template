"""
Optimized implementation of AttentionPairBias with focus on:
1. Memory-efficient operations
2. Optimized LayerNorm kernels 
3. Minimizing tensor reshapes
4. Better CUDA utilization without flash-attn dependency
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional

# Try to import cuEquivariance
try:
    from cuequivariance_torch import attention_pair_bias as cueq_attention_pair_bias
    HAS_CUEQUIVARIANCE = True
except ImportError:
    HAS_CUEQUIVARIANCE = False

# Try to import APEX for optimized LayerNorm
try:
    from apex.normalization import FusedLayerNorm
    HAS_APEX = True
except ImportError:
    HAS_APEX = False


class OptimizedLayerNorm(nn.Module):
    """
    LayerNorm implementation that selects the optimal backend
    based on what's available (APEX FusedLayerNorm or PyTorch LayerNorm).
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        
        if HAS_APEX:
            self.norm = FusedLayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        else:
            self.norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x):
        return self.norm(x)


class OptimizedAttentionPairBias(nn.Module):
    """
    Optimized attention pair bias layer with focus on memory efficiency
    and better CUDA kernel utilization.
    
    This implementation minimizes tensor reshapes and uses optimal memory
    layouts for better performance without flash-attn dependency.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        num_heads: int,
        inf: float = 1e6,
        initial_norm: bool = True,
    ) -> None:
        super().__init__()
        
        assert c_s % num_heads == 0
        
        self.c_s = c_s
        self.c_z = c_z  
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.inf = inf
        self.initial_norm = initial_norm
        
        # Use optimized layer norm if available
        if self.initial_norm:
            self.norm_s = OptimizedLayerNorm(c_s)
        
        # Pre-compute scaling factor
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Use separate projections to handle k_in parameter correctly
        # Q comes from s, K and V come from k_in (which might be different from s)
        self.proj_q = nn.Linear(c_s, c_s)
        self.proj_k = nn.Linear(c_s, c_s, bias=False)
        self.proj_v = nn.Linear(c_s, c_s, bias=False)
        self.proj_g = nn.Linear(c_s, c_s, bias=False)
        
        # Use optimized layernorm for z
        self.norm_z = OptimizedLayerNorm(c_z)
        self.proj_z = nn.Linear(c_z, num_heads, bias=False)
        
        # Output projection
        self.proj_o = nn.Linear(c_s, c_s, bias=False)

    def _process_z(
        self, 
        z: torch.Tensor, 
        model_cache: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Process the pairwise tensor z efficiently, with caching if needed.
        
        Parameters
        ----------
        z : torch.Tensor
            Pairwise tensor (B, N, N, D_z)
        model_cache : dict, optional
            Cache for model state
            
        Returns
        -------
        torch.Tensor
            Processed pairwise bias tensor (B, H, N, N)
        """
        if model_cache is None or "z" not in model_cache:
            # Normalize and project in a single kernel execution (where possible)
            z_norm = self.norm_z(z)
            z_proj = self.proj_z(z_norm)
            
            # Reshape to (B, H, N, N) - one transposition instead of multiple
            z_proj = z_proj.permute(0, 3, 1, 2)
            
            if model_cache is not None:
                model_cache["z"] = z_proj
                
            return z_proj
        else:
            return model_cache["z"]

    def _memory_efficient_attention(
        self,
        q: torch.Tensor,  # (B, S, H, D_h)
        k: torch.Tensor,  # (B, S, H, D_h)
        v: torch.Tensor,  # (B, S, H, D_h)
        z_bias: torch.Tensor,  # (B, H, S, S)
        mask: torch.Tensor,  # (B, S)
    ) -> torch.Tensor:
        """
        Memory-efficient attention computation that minimizes
        intermediate tensor allocations.
        
        Parameters
        ----------
        q, k, v : torch.Tensor
            Query, key, and value tensors (B, S, H, D_h)
        z_bias : torch.Tensor
            Pairwise bias tensor (B, H, S, S)
        mask : torch.Tensor
            Mask tensor (B, S)
            
        Returns
        -------
        torch.Tensor
            Output tensor (B, S, H*D_h)
        """
        B, S, H, D_h = q.shape
        
        # Reshape query and key for efficient batch matmul
        # (B, S, H, D_h) -> (B, H, S, D_h)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores efficiently
        # (B, H, S, D_h) @ (B, H, D_h, S) -> (B, H, S, S)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        
        # Scale and add bias in one operation
        attn_scores = attn_scores * self.scale + z_bias
        
        # Create and apply mask efficiently
        # (B, S) -> (B, 1, 1, S) for broadcasting
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(1)
            attn_scores = attn_scores + (1 - mask_expanded) * -self.inf
        
        # Apply softmax (keeping the operation in place where possible)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        # (B, H, S, S) @ (B, H, S, D_h) -> (B, H, S, D_h)
        attn_output = torch.matmul(attn_probs, v)
        
        # Transpose to original format and reshape
        # (B, H, S, D_h) -> (B, S, H*D_h)
        return attn_output.transpose(1, 2).reshape(B, S, H * D_h)

    def forward(
        self,
        s: torch.Tensor,  # (B, S, D)
        z: torch.Tensor,  # (B, N, N, D_z)
        mask: torch.Tensor,  # (B, N)
        multiplicity: int = 1,
        to_keys=None,
        model_cache: Optional[Dict] = None,
        k_in: Optional[torch.Tensor] = None,  # Added k_in parameter
    ) -> torch.Tensor:
        """
        Forward pass with optimized memory patterns.
        
        Parameters
        ----------
        s : torch.Tensor
            Input sequence tensor (B, S, D)
        z : torch.Tensor
            Input pairwise tensor (B, N, N, D_z)
        mask : torch.Tensor
            Mask tensor (B, N)
        multiplicity : int, optional
            Diffusion batch size multiplier, by default 1
        to_keys : callable, optional
            Function to transform keys
        model_cache : dict, optional
            Cache for model state
            
        Returns
        -------
        torch.Tensor
            Output sequence tensor (B, S, D)
        """
        B, S, D = s.shape
        
        # Apply layer norm if configured
        if self.initial_norm:
            s = self.norm_s(s)
            
        # Handle key input - prioritize explicit k_in parameter
        if k_in is not None:
            # k_in was explicitly provided (e.g., from PairformerLayer)
            pass  # Use the provided k_in
        elif to_keys is not None:
            # Transform keys using to_keys function
            k_in = to_keys(s)
            mask = to_keys(mask.unsqueeze(-1)).squeeze(-1)
        else:
            # Default: use the input s as keys
            k_in = s
            
        # If cuEquivariance is available and the tensors are large enough
        # to benefit from it, use that implementation
        if HAS_CUEQUIVARIANCE and S >= 128:
            return self._forward_cuequivariance(s, k_in, z, mask, multiplicity, model_cache)
        
        # Otherwise use our optimized PyTorch implementation
        
        # 1. Project Q, K, V using correct input tensors
        q = self.proj_q(s)  # Query from s
        k = self.proj_k(k_in)  # Key from k_in  
        v = self.proj_v(k_in)  # Value from k_in
        
        # 2. Reshape for attention computation
        q = q.view(B, S, self.num_heads, self.head_dim)
        k = k.view(B, S, self.num_heads, self.head_dim)
        v = v.view(B, S, self.num_heads, self.head_dim)
        
        # 3. Process z efficiently and with caching
        z_bias = self._process_z(z, model_cache)
        z_bias = z_bias.repeat_interleave(multiplicity, 0) if multiplicity > 1 else z_bias
        
        # 4. Compute gating factor
        g = self.proj_g(s).sigmoid()
        
        # 5. Handle multiplicity (diffusion models)
        if multiplicity > 1:
            q = q.repeat_interleave(multiplicity, 0)
            k = k.repeat_interleave(multiplicity, 0)
            v = v.repeat_interleave(multiplicity, 0)
            mask = mask.repeat_interleave(multiplicity, 0)
            g = g.repeat_interleave(multiplicity, 0)
        
        # 6. Run memory-efficient attention
        attn_output = self._memory_efficient_attention(q, k, v, z_bias, mask)
        
        # 7. Apply gating and output projection
        output = self.proj_o(g * attn_output)
        
        # 8. Handle multiplicity reshaping for output
        if multiplicity > 1:
            output = output.view(B, multiplicity, S, D)
            output = output[:, 0]  # Take first multiplicity dimension
            
        return output

    def _forward_cuequivariance(
        self,
        s: torch.Tensor,
        k_in: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        multiplicity: int = 1,
        model_cache: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Forward pass using cuEquivariance kernel with optimized memory layout.
        
        Parameters
        ----------
        s : torch.Tensor
            Input sequence tensor (B, S, D)
        k_in : torch.Tensor
            Key input tensor (B, S, D)
        z : torch.Tensor
            Input pairwise tensor (B, N, N, D_z)
        mask : torch.Tensor
            Mask tensor (B, N)
        multiplicity : int, optional
            Diffusion batch size multiplier, by default 1
        model_cache : dict, optional
            Cache for model state
            
        Returns
        -------
        torch.Tensor
            Output sequence tensor (B, S, D)
        """
        B, S, D = s.shape
        
        # Get QKV projections directly in the format needed by cuEquivariance
        # This avoids unnecessary reshaping
        qkv = self.proj_qkv(s)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for attention
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, S, D_h)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, S, D_h)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, S, D_h)
        
        # Compute gating
        g = self.proj_g(s).sigmoid()
        
        # Handle multiplicity
        if multiplicity > 1:
            q = q.repeat_interleave(multiplicity, 0)
            k = k.repeat_interleave(multiplicity, 0)
            v = v.repeat_interleave(multiplicity, 0)
            s = s.repeat_interleave(multiplicity, 0)
            mask = mask.repeat_interleave(multiplicity, 0)
            g = g.repeat_interleave(multiplicity, 0)
        
        # Get weights needed for kernel
        w_proj_z = self.proj_z.weight  # (num_heads, c_z)
        w_proj_g = self.proj_g.weight  # (c_s, c_s)
        w_proj_o = self.proj_o.weight  # (c_s, c_s)
        w_ln_z = self.norm_z.norm.weight  # (c_z,)
        b_ln_z = self.norm_z.norm.bias   # (c_z,)
        
        # Call cuEquivariance kernel with optimized tensor format
        output, _ = cueq_attention_pair_bias(
            s=s,  # (B*M, S, D)
            q=q,  # (B*M, H, S, D_h)
            k=k,  # (B*M, H, S, D_h) 
            v=v,  # (B*M, H, S, D_h)
            z=z,  # (B, N, N, D_z)
            mask=mask,  # (B*M, S)
            num_heads=self.num_heads,
            w_proj_z=w_proj_z,  # (num_heads, c_z)
            w_proj_g=w_proj_g,  # (c_s, c_s)
            w_proj_o=w_proj_o,  # (c_s, c_s)
            w_ln_z=w_ln_z,      # (c_z,)
            b_ln_z=b_ln_z,      # (c_z,)
            attn_scale=self.scale,
            inf=self.inf,
            return_z_proj=False,
        )
        
        # Apply gating
        output = output * g
        
        # Handle multiplicity for output
        if multiplicity > 1:
            output = output.view(B, multiplicity, S, D)
            output = output[:, 0]  # Take first multiplicity dimension
            
        return output


# Utility function to create the most optimal implementation
def create_optimized_attention(
    c_s: int,
    c_z: int,
    num_heads: int,
    inf: float = 1e6,
    initial_norm: bool = True,
):
    """
    Create the optimal attention implementation based on available libraries.
    
    Parameters
    ----------
    c_s : int
        Sequence feature dimension
    c_z : int
        Pairwise feature dimension
    num_heads : int
        Number of attention heads
    inf : float
        Value to use for masking
    initial_norm : bool
        Whether to use layer norm on input
        
    Returns
    -------
    nn.Module
        Optimized implementation of attention
    """
    # Always use our optimized implementation as it includes fallback
    return OptimizedAttentionPairBias(c_s, c_z, num_heads, inf, initial_norm)