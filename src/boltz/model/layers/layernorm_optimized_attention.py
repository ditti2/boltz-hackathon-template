"""
Advanced LayerNorm and Attention Optimizations
Focus on the actual computational bottlenecks revealed by benchmarking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional

# Direct imports - no fallbacks
from apex.normalization import FusedLayerNorm


class OptimalLayerNorm(nn.Module):
    """
    LayerNorm that uses APEX FusedLayerNorm for better performance.
    """
    
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Use APEX FusedLayerNorm for optimized performance
        self.norm = FusedLayerNorm(normalized_shape, eps=eps)
        self.backend = "APEX"
    
    def forward(self, x):
        return self.norm(x)


class LayerNormOptimizedAttentionPairBias(nn.Module):
    """
    Attention implementation with focus on LayerNorm optimization and
    addressing the actual computational bottlenecks.
    """
    
    def __init__(self, c_s, c_z, num_heads):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.scale = (self.head_dim ** -0.5)
        
        # Use optimized LayerNorm implementations
        self.norm_s = OptimalLayerNorm(c_s)
        self.norm_z = OptimalLayerNorm(c_z)
        
        # Projections with optimized initialization
        self.proj_q = nn.Linear(c_s, c_s, bias=False)
        self.proj_k = nn.Linear(c_s, c_s, bias=False)
        self.proj_v = nn.Linear(c_s, c_s, bias=False)
        self.proj_g = nn.Linear(c_s, c_s)
        self.proj_o = nn.Linear(c_s, c_s)
        self.proj_z = nn.Linear(c_z, num_heads, bias=False)
        
        # Optimize weight initialization for better numerical stability
        self._init_weights()
        
        self.initial_norm = True
    
    def _init_weights(self):
        """Optimized weight initialization for better convergence."""
        # Xavier initialization for attention weights
        for module in [self.proj_q, self.proj_k, self.proj_v]:
            nn.init.xavier_uniform_(module.weight)
        
        # He initialization for other projections
        for module in [self.proj_g, self.proj_o, self.proj_z]:
            if hasattr(module, 'weight'):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
    
    def forward(self, s, z, mask, multiplicity=1, to_keys=None, 
               model_cache=None, k_in=None):
        B, S, D = s.shape
        
        # Optimized LayerNorm - this is often a bottleneck
        if self.initial_norm:
            s = self.norm_s(s)
            
        # Handle key input efficiently
        if k_in is not None:
            k_input = k_in
        elif to_keys is not None:
            k_input = to_keys(s)
            mask = to_keys(mask.unsqueeze(-1)).squeeze(-1)
        else:
            k_input = s
        
        # Efficient projections with proper memory layout
        q = self.proj_q(s).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.proj_k(k_input).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.proj_v(k_input).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Optimized z processing with fast LayerNorm
        z_norm = self.norm_z(z)
        z_proj = self.proj_z(z_norm)
        z_bias = z_proj.permute(0, 3, 1, 2).contiguous()
        
        # Handle multiplicity
        if multiplicity > 1:
            q = q.repeat_interleave(multiplicity, 0)
            k = k.repeat_interleave(multiplicity, 0)
            v = v.repeat_interleave(multiplicity, 0)
            z_bias = z_bias.repeat_interleave(multiplicity, 0)
            mask = mask.repeat_interleave(multiplicity, 0)
        
        # Optimized attention computation
        attn_output = self._optimized_attention(q, k, v, z_bias, mask)
        
        # Efficient output processing
        g = self.proj_g(s).sigmoid()
        if multiplicity > 1:
            g = g.repeat_interleave(multiplicity, 0)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B * multiplicity, S, -1)
        output = self.proj_o(g * attn_output)
        
        if multiplicity > 1:
            output = output.view(multiplicity, B, S, -1).mean(0)
            
        return output
    
    def _optimized_attention(self, q, k, v, z_bias, mask):
        """
        Optimized attention focusing on actual bottlenecks.
        """
        B, H, S, D = q.shape
        
        # Use optimized scaled dot product when available (PyTorch 2.0+)
        try:
            # Try to use PyTorch's optimized implementation
            if hasattr(F, 'scaled_dot_product_attention') and z_bias is None:
                if mask is not None:
                    attn_mask = mask.bool().unsqueeze(1).unsqueeze(1)
                    attn_mask = attn_mask.expand(B, H, S, S)
                else:
                    attn_mask = None
                
                return F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
                )
        except:
            pass
        
        # Manual implementation with optimizations
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if z_bias is not None:
            scores = scores + z_bias
        
        if mask is not None:
            mask_bool = mask.bool()
            mask_2d = mask_bool.unsqueeze(1).unsqueeze(1) & mask_bool.unsqueeze(1).unsqueeze(-1)
            scores = scores.masked_fill(~mask_2d, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        return output


class FusedAttentionPairBias(nn.Module):
    """
    Attention with aggressive kernel fusion and LayerNorm optimization.
    Targets the computational patterns that make Trimul effective.
    """
    
    def __init__(self, c_s, c_z, num_heads):
        super().__init__()
        self.c_s = c_s
        self.c_z = c_z
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.scale = (self.head_dim ** -0.5)
        
        # Use the fastest available LayerNorm
        self.norm_s = OptimalLayerNorm(c_s)
        self.norm_z = OptimalLayerNorm(c_z)
        
        # Fused weight matrix for Q, K, V
        self.qkv_weight = nn.Parameter(torch.empty(c_s, 3 * c_s))
        self.proj_g = nn.Linear(c_s, c_s)
        self.proj_o = nn.Linear(c_s, c_s)
        self.proj_z = nn.Linear(c_z, num_heads, bias=False)
        
        self._init_weights()
        self.initial_norm = True
    
    def _init_weights(self):
        """Initialize weights optimally."""
        nn.init.xavier_uniform_(self.qkv_weight)
        nn.init.kaiming_uniform_(self.proj_g.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.proj_o.weight, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.proj_z.weight)
    
    def forward(self, s, z, mask, multiplicity=1, to_keys=None, 
               model_cache=None, k_in=None):
        B, S, D = s.shape
        
        # Fast LayerNorm
        if self.initial_norm:
            s = self.norm_s(s)
            
        # Handle key input
        if k_in is not None:
            k_input = k_in
        elif to_keys is not None:
            k_input = to_keys(s)
            mask = to_keys(mask.unsqueeze(-1)).squeeze(-1)
        else:
            k_input = s
        
        # Fused QKV computation - single GEMM
        qkv = F.linear(s, self.qkv_weight)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Handle different k_input
        if k_input is not s:
            kv = F.linear(k_input, self.qkv_weight[:, D:])  # Only K,V part
            k, v = kv.chunk(2, dim=-1)
        
        # Efficient reshape and transpose
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Fast z processing
        z_norm = self.norm_z(z)
        z_bias = self.proj_z(z_norm).permute(0, 3, 1, 2)
        
        # Handle multiplicity
        if multiplicity > 1:
            q = q.repeat_interleave(multiplicity, 0)
            k = k.repeat_interleave(multiplicity, 0)
            v = v.repeat_interleave(multiplicity, 0)
            z_bias = z_bias.repeat_interleave(multiplicity, 0)
            mask = mask.repeat_interleave(multiplicity, 0)
        
        # Optimized attention
        attn_output = self._fused_attention(q, k, v, z_bias, mask)
        
        # Efficient output
        g = self.proj_g(s).sigmoid()
        if multiplicity > 1:
            g = g.repeat_interleave(multiplicity, 0)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(-1, D)
        g_flat = g.view(-1, D)
        output = self.proj_o(g_flat * attn_output).view(B * multiplicity, S, D)
        
        if multiplicity > 1:
            output = output.view(multiplicity, B, S, D).mean(0)
            
        return output
    
    def _fused_attention(self, q, k, v, z_bias, mask):
        """Highly optimized attention computation."""
        # Use the most efficient available method
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale + z_bias
        
        if mask is not None:
            mask_bool = mask.bool()
            mask_2d = mask_bool.unsqueeze(1).unsqueeze(1) & mask_bool.unsqueeze(1).unsqueeze(-1)
            scores = scores.masked_fill(~mask_2d, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)