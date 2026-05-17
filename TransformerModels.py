import torch
from torch import nn
from torch.nn import functional as F
import math

# ============================================================================
# ORIGINAL (UNOPTIMIZED) MODULES - preserved for reference/comparison
# ============================================================================

class OptimizedDepthwiseConv(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=5,
        stride=1,
        padding=None,
        res=True,
        dropout=0.0,
        activation='silu',
        bias=True,
        use_alpha=True,
        alpha_init=1.0
    ):
        super().__init__()
        self.res = res
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation
        
        padding = padding or (kernel_size // 2)
        
        self.conv = nn.Conv2d(
            dim, dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=dim,
            bias=bias
        )
        
        self.use_alpha = use_alpha and res
        if self.use_alpha:
            self.alpha = nn.Parameter(torch.ones(1, 1, 1, 1) * alpha_init)
        
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        
        if activation == 'silu' or activation == 'swish':
            self.act = nn.SiLU()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation:
            self.act = nn.SiLU()
        else:
            self.act = None
    
    def forward(self, x):
        out = self.conv(x)
        
        if self.act is not None:
            out = self.act(out)
        
        if self.dropout is not None and self.training:
            out = self.dropout(out)
        
        if self.res:
            if self.use_alpha:
                return out * self.alpha + x
            return out + x
        return out


class OptimizedPointwiseConv(nn.Module):
    def __init__(
        self,
        dim,
        outdim=None,
        res=True,
        dropout=0.0,
        activation='silu',
        bias=True,
        use_alpha=True,
        alpha_init=1.0
    ):
        super().__init__()
        self.res = res
        outdim = outdim or dim
        
        self.conv = nn.Conv2d(dim, outdim, 1, 1, bias=bias)
        
        self.use_alpha = use_alpha and res and (outdim == dim)
        if self.use_alpha:
            self.alpha = nn.Parameter(torch.ones(1, 1, 1, 1) * alpha_init)
        
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        
        if activation == 'silu' or activation == 'swish':
            self.act = nn.SiLU()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'prelu':
            self.act = nn.PReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation:
            self.act = nn.SiLU()
        else:
            self.act = None
        
        self.outdim = outdim
    
    def forward(self, x):
        out = self.conv(x)
        
        if self.act is not None:
            out = self.act(out)
        
        if self.dropout is not None and self.training:
            out = self.dropout(out)
        
        if self.res:
            if self.use_alpha:
                return out * self.alpha + x
            return out + x
        return out


class FusedDepthwisePointwise(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim=None,
        kernel_size=3,
        dropout=0.0,
        activation='silu',
        bias=True,
        res=True,
        use_alpha=True,
        alpha_init=1.0
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim
        
        self.depthwise = OptimizedDepthwiseConv(
            dim, kernel_size, 1, None,
            res=False, dropout=dropout,
            activation=activation, bias=bias,
            use_alpha=False
        )
        
        self.pointwise = OptimizedPointwiseConv(
            dim, hidden_dim,
            res=False, dropout=dropout,
            activation=activation, bias=bias,
            use_alpha=False
        )
        
        self.use_alpha = use_alpha and res
        if self.use_alpha:
            self.alpha = nn.Parameter(torch.ones(1, 1, 1, 1) * alpha_init)
        
        self.res = res
    
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        
        if self.res and self.use_alpha:
            return out * self.alpha + x
        elif self.res:
            return out + x
        return out