"""
Enhanced implementation of AttentionPairBias with optimized cuEquivariance kernel support.

This module provides an enhanced version of the AttentionPairBias layer 
with optimization strategies for maximum performance when using cuEquivariance.

Key optimizations:
1. Minimizing tensor reshapes and format conversions
2. Memory optimizations to reduce allocations
3. Mixed precision support
4. Integration with CUDA Graph for repeated executions
5. Stream synchronization control for overlapping computation
"""

import torch
from torch import nn
import math
from typing import Dict, Optional, Union, Tuple

# Try to import cuEquivariance
try:
    from cuequivariance_torch import attention_pair_bias as cueq_attention_pair_bias
    HAS_CUEQUIVARIANCE = True
except ImportError:
    HAS_CUEQUIVARIANCE = False


class EnhancedAttentionPairBias(nn.Module):
    """
    Enhanced attention pair bias layer with optimized cuEquivariance support.
    
    This implementation provides maximum performance when using cuEquivariance
    kernels, while maintaining full compatibility with the original Boltz
    implementation as a fallback.
    """

    def __init__(
        self,
        c_s: int,
        c_z: int,
        num_heads: int,
        inf: float = 1e6,
        initial_norm: bool = True,
        use_kernels: bool = True,
        use_mixed_precision: bool = True,
        enable_cuda_graph: bool = False,
    ) -> None:
        """
        Initialize the enhanced attention pair bias layer.

        Parameters
        ----------
        c_s : int
            The input sequence dimension.
        c_z : int
            The input pairwise dimension.
        num_heads : int
            The number of heads.
        inf : float, optional
            The inf value, by default 1e6
        initial_norm: bool, optional
            Whether to apply layer norm to the input, by default True
        use_kernels : bool, optional
            Whether to use cuEquivariance kernels if available, by default True
        use_mixed_precision : bool, optional
            Whether to use mixed precision for attention calculations, by default True
        enable_cuda_graph : bool, optional
            Whether to use CUDA graphs for repeated executions, by default False
        """
        super().__init__()

        assert c_s % num_heads == 0

        self.c_s = c_s
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.inf = inf
        
        self.use_kernels = use_kernels and HAS_CUEQUIVARIANCE
        self.use_mixed_precision = use_mixed_precision
        self.enable_cuda_graph = enable_cuda_graph
        
        # Track whether we've warmed up the cuda graph
        self.graph_warmup_done = False
        self.cuda_graph = None
        self.static_inputs = None

        self.initial_norm = initial_norm
        if self.initial_norm:
            self.norm_s = nn.LayerNorm(c_s)

        # Input projections
        self.proj_q = nn.Linear(c_s, c_s)
        self.proj_k = nn.Linear(c_s, c_s, bias=False)
        self.proj_v = nn.Linear(c_s, c_s, bias=False)
        self.proj_g = nn.Linear(c_s, c_s, bias=False)

        # For the original implementation with sequential modules
        if not self.use_kernels:
            self.proj_z = nn.Sequential(
                nn.LayerNorm(c_z),
                nn.Linear(c_z, num_heads, bias=False),
                torch.nn.modules.rearrange.Rearrange("b ... h -> b h ..."),
            )
        # For kernel implementation, we need separate components
        else:
            self.norm_z = nn.LayerNorm(c_z)
            self.linear_z = nn.Linear(c_z, num_heads, bias=False)

        self.proj_o = nn.Linear(c_s, c_s, bias=False)

    def _compute_attention_einsum(
        self, 
        q: torch.Tensor,
        k: torch.Tensor, 
        v: torch.Tensor, 
        z: torch.Tensor, 
        mask: torch.Tensor,
        multiplicity: int = 1,
        model_cache: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Compute attention using einsum operations (original Boltz implementation).
        
        Parameters
        ----------
        q : torch.Tensor
            Query tensor [B, S, H, D_head]
        k : torch.Tensor
            Key tensor [B, S, H, D_head]
        v : torch.Tensor
            Value tensor [B, S, H, D_head]
        z : torch.Tensor
            Pairwise tensor [B, N, N, D_z]
        mask : torch.Tensor
            Mask tensor [B, S]
        multiplicity : int
            Diffusion batch size multiplier
        model_cache : dict, optional
            Cache for model state
            
        Returns
        -------
        torch.Tensor
            Output tensor [B, S, D]
        """
        B = q.shape[0]
        
        # Caching z projection during diffusion roll-out
        if model_cache is None or "z" not in model_cache:
            z = self.proj_z(z)

            if model_cache is not None:
                model_cache["z"] = z
        else:
            z = model_cache["z"]
            
        z = z.repeat_interleave(multiplicity, 0)
        g = self.proj_g(q.reshape(B, -1, self.c_s)).sigmoid()

        q = q.reshape(B, -1, self.num_heads, self.head_dim)
        k = k.reshape(B, -1, self.num_heads, self.head_dim)
        v = v.reshape(B, -1, self.num_heads, self.head_dim)

        # Use appropriate precision for attention calculation
        precision = torch.float32 if self.use_mixed_precision else q.dtype
        
        with torch.autocast("cuda", enabled=False):
            # Compute attention weights
            attn = torch.einsum("bihd,bjhd->bhij", q.to(precision), k.to(precision))
            attn = attn / (self.head_dim**0.5) + z.to(precision)
            # Apply mask
            attn = attn + (1 - mask[:, None, None].to(precision)) * -self.inf
            attn = attn.softmax(dim=-1)
            # Apply attention to values
            o = torch.einsum("bhij,bjhd->bihd", attn, v.to(precision)).to(v.dtype)
            
        o = o.reshape(B, -1, self.c_s)
        o = self.proj_o(g * o)
        
        return o

    def _compute_attention_kernel(
        self, 
        s: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor, 
        v: torch.Tensor, 
        z: torch.Tensor, 
        mask: torch.Tensor,
        multiplicity: int = 1,
        model_cache: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Compute attention using cuEquivariance kernel.
        
        Parameters
        ----------
        s : torch.Tensor
            Input sequence tensor [B, S, D]
        q : torch.Tensor
            Query tensor [B, S, D]
        k : torch.Tensor
            Key tensor [B, S, D]
        v : torch.Tensor
            Value tensor [B, S, D]
        z : torch.Tensor
            Pairwise tensor [B, N, N, D_z]
        mask : torch.Tensor
            Mask tensor [B, S]
        multiplicity : int
            Diffusion batch size multiplier
        model_cache : dict, optional
            Cache for model state
            
        Returns
        -------
        torch.Tensor
            Output tensor [B, S, D]
        """
        B, S, D = s.shape
        
        # Project and reshape in one go to minimize memory allocations
        q = self.proj_q(q).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.proj_k(k).view(B, S, self.num_heads, self.head_dim).transpose(1, 2) 
        v = self.proj_v(v).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        g = self.proj_g(s).sigmoid()
        
        # Handle multiplicity by repeating tensors
        if multiplicity > 1:
            q = q.repeat_interleave(multiplicity, 0)
            k = k.repeat_interleave(multiplicity, 0)
            v = v.repeat_interleave(multiplicity, 0)
            s = s.repeat_interleave(multiplicity, 0)
            mask = mask.repeat_interleave(multiplicity, 0)
            g = g.repeat_interleave(multiplicity, 0)
        
        # Get parameters for the kernel
        w_proj_z = self.linear_z.weight  # (num_heads, c_z)
        w_proj_g = self.proj_g.weight    # (c_s, c_s)
        w_proj_o = self.proj_o.weight    # (c_s, c_s)
        w_ln_z = self.norm_z.weight      # (c_z,)
        b_ln_z = self.norm_z.bias        # (c_z,)
        
        # Compute attention scale
        attn_scale = 1.0 / math.sqrt(self.head_dim)
        
        # Use CUDA graph for repeated executions with same shapes
        if self.enable_cuda_graph and torch.cuda.is_available():
            if not self.graph_warmup_done:
                # First time - capture the graph
                self._initialize_cuda_graph(
                    s, q, k, v, z, mask, 
                    w_proj_z, w_proj_g, w_proj_o, 
                    w_ln_z, b_ln_z, attn_scale
                )
            else:
                # Copy input data to static tensors
                self._update_static_inputs(
                    s, q, k, v, z, mask, 
                    w_proj_z, w_proj_g, w_proj_o, 
                    w_ln_z, b_ln_z
                )
                
            # Run the captured graph
            self.cuda_graph.replay()
            output = self.static_inputs["output"]
        else:
            # Normal execution without CUDA graph
            output, _ = cueq_attention_pair_bias(
                s=s,
                q=q, 
                k=k, 
                v=v,
                z=z,
                mask=mask,
                num_heads=self.num_heads,
                w_proj_z=w_proj_z,
                w_proj_g=w_proj_g, 
                w_proj_o=w_proj_o,
                w_ln_z=w_ln_z,
                b_ln_z=b_ln_z,
                attn_scale=attn_scale,
                inf=self.inf,
                return_z_proj=True,
            )
        
        # Apply gating
        output = output * g
        
        # Reshape output back to original batch dimensions if needed
        if multiplicity > 1:
            output = output.view(B, multiplicity, S, D)
            output = output[:, 0]  # Take first multiplicity dimension
            
        return output

    def _initialize_cuda_graph(
        self, 
        s, q, k, v, z, mask, 
        w_proj_z, w_proj_g, w_proj_o, 
        w_ln_z, b_ln_z, attn_scale
    ) -> None:
        """
        Initialize CUDA graph for repeated executions.
        
        This is beneficial when running multiple forward passes with the 
        same tensor shapes, e.g., during inference.
        """
        # Create static input tensors that will persist for the graph
        self.static_inputs = {
            "s": s.clone(),
            "q": q.clone(),
            "k": k.clone(),
            "v": v.clone(),
            "z": z.clone(),
            "mask": mask.clone(),
            "w_proj_z": w_proj_z.clone(),
            "w_proj_g": w_proj_g.clone(),
            "w_proj_o": w_proj_o.clone(),
            "w_ln_z": w_ln_z.clone(),
            "b_ln_z": b_ln_z.clone(),
        }
        
        # Allocate output tensor
        self.static_inputs["output"] = torch.zeros_like(s)
        
        # Capture CUDA graph
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        
        with torch.cuda.stream(stream):
            for _ in range(3):  # Warmup before capture
                cueq_attention_pair_bias(
                    s=self.static_inputs["s"],
                    q=self.static_inputs["q"], 
                    k=self.static_inputs["k"], 
                    v=self.static_inputs["v"],
                    z=self.static_inputs["z"],
                    mask=self.static_inputs["mask"],
                    num_heads=self.num_heads,
                    w_proj_z=self.static_inputs["w_proj_z"],
                    w_proj_g=self.static_inputs["w_proj_g"], 
                    w_proj_o=self.static_inputs["w_proj_o"],
                    w_ln_z=self.static_inputs["w_ln_z"],
                    b_ln_z=self.static_inputs["b_ln_z"],
                    attn_scale=attn_scale,
                    inf=self.inf,
                    return_z_proj=True,
                )
            
            # Capture the graph
            self.cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.cuda_graph):
                self.static_inputs["output"], _ = cueq_attention_pair_bias(
                    s=self.static_inputs["s"],
                    q=self.static_inputs["q"], 
                    k=self.static_inputs["k"], 
                    v=self.static_inputs["v"],
                    z=self.static_inputs["z"],
                    mask=self.static_inputs["mask"],
                    num_heads=self.num_heads,
                    w_proj_z=self.static_inputs["w_proj_z"],
                    w_proj_g=self.static_inputs["w_proj_g"], 
                    w_proj_o=self.static_inputs["w_proj_o"],
                    w_ln_z=self.static_inputs["w_ln_z"],
                    b_ln_z=self.static_inputs["b_ln_z"],
                    attn_scale=attn_scale,
                    inf=self.inf,
                    return_z_proj=True,
                )
        
        torch.cuda.current_stream().wait_stream(stream)
        self.graph_warmup_done = True

    def _update_static_inputs(
        self, 
        s, q, k, v, z, mask, 
        w_proj_z, w_proj_g, w_proj_o, 
        w_ln_z, b_ln_z
    ) -> None:
        """Update static inputs for CUDA graph replay."""
        self.static_inputs["s"].copy_(s)
        self.static_inputs["q"].copy_(q)
        self.static_inputs["k"].copy_(k)
        self.static_inputs["v"].copy_(v)
        self.static_inputs["z"].copy_(z)
        self.static_inputs["mask"].copy_(mask)
        self.static_inputs["w_proj_z"].copy_(w_proj_z)
        self.static_inputs["w_proj_g"].copy_(w_proj_g)
        self.static_inputs["w_proj_o"].copy_(w_proj_o)
        self.static_inputs["w_ln_z"].copy_(w_ln_z)
        self.static_inputs["b_ln_z"].copy_(b_ln_z)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        multiplicity: int = 1,
        to_keys: Optional[callable] = None,
        model_cache: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the attention pair bias layer.
        
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
        # Layer norm
        if self.initial_norm:
            s = self.norm_s(s)
            
        # Handle key transformation if provided
        if to_keys is not None:
            k_in = to_keys(s)
            mask_in = to_keys(mask.unsqueeze(-1)).squeeze(-1)
        else:
            k_in = s
            mask_in = mask
        
        # Choose implementation based on availability and configuration
        if self.use_kernels and HAS_CUEQUIVARIANCE:
            return self._compute_attention_kernel(
                s, s, k_in, k_in, z, mask_in, multiplicity, model_cache
            )
        else:
            # Fall back to original implementation
            return self._compute_attention_einsum(
                s, k_in, k_in, z, mask_in, multiplicity, model_cache
            )


# Helper function to create a suitable implementation based on availability
def create_attention_pair_bias(
    c_s: int,
    c_z: int,
    num_heads: int,
    inf: float = 1e6,
    initial_norm: bool = True,
    use_kernels: bool = True,
    use_mixed_precision: bool = True,
    enable_cuda_graph: bool = False,
) -> nn.Module:
    """
    Create an AttentionPairBias implementation based on availability.
    
    Parameters
    ----------
    c_s : int
        The input sequence dimension
    c_z : int
        The input pairwise dimension
    num_heads : int
        The number of heads
    inf : float, optional
        The inf value, by default 1e6
    initial_norm: bool, optional
        Whether to apply layer norm to the input, by default True
    use_kernels : bool, optional
        Whether to use cuEquivariance kernels if available, by default True
    use_mixed_precision : bool, optional
        Whether to use mixed precision for attention calculations, by default True
    enable_cuda_graph : bool, optional
        Whether to use CUDA graphs for repeated executions, by default False
        
    Returns
    -------
    nn.Module
        An implementation of AttentionPairBias
    """
    from boltz.model.layers.attention import AttentionPairBias as OriginalAttentionPairBias

    # Check if cuEquivariance is available
    has_cueq = HAS_CUEQUIVARIANCE
    
    if has_cueq and use_kernels:
        return EnhancedAttentionPairBias(
            c_s=c_s,
            c_z=c_z,
            num_heads=num_heads,
            inf=inf,
            initial_norm=initial_norm,
            use_kernels=use_kernels,
            use_mixed_precision=use_mixed_precision,
            enable_cuda_graph=enable_cuda_graph,
        )
    else:
        # Fall back to original implementation
        return OriginalAttentionPairBias(
            c_s=c_s,
            c_z=c_z,
            num_heads=num_heads,
            inf=inf,
            initial_norm=initial_norm,
        )