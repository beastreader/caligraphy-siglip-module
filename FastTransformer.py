import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.rn = nn.RMSNorm(dim)
    def forward(self, x):
        return self.rn(x)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim=None,
        bias=True,
        glu=True,
        swish=True,
        dropout=0.0
    ):
        super().__init__()
        self.glu = glu
        self.swish = swish
        hidden_dim = hidden_dim or dim * 4

        if glu:
            self.w1 = nn.Linear(dim, hidden_dim * 2, bias=bias)
            self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        elif swish:
            self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
            self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
            self.act = nn.SiLU()
        else:
            self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
            self.w2 = nn.Linear(hidden_dim, dim, bias=bias)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        if self.glu:
            w1_out = self.w1(x)
            gate, up = w1_out.chunk(2, dim=-1)
            return self.dropout(self.w2(F.silu(gate) * up))
        elif self.swish:
            return self.dropout(self.w2(self.act(self.w1(x))))
        else:
            return self.dropout(self.w2(self.w1(x)))


class RotaryEmbedding1D(nn.Module):
    def __init__(self, dim_head, base=10000):
        super().__init__()
        self.dim_head = dim_head
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim_head, 2).float() / dim_head))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = t.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        return freqs  # [seq_len, dim_head // 2]


class RotaryEmbedding2D(nn.Module):
    def __init__(self, dim_head, base=10000):
        super().__init__()
        self.dim_head = dim_head
        self.base = base
        half_dim = dim_head // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len_x, seq_len_y, device):
        t_x = torch.arange(seq_len_x, device=device).type_as(self.inv_freq)
        t_y = torch.arange(seq_len_y, device=device).type_as(self.inv_freq)
        
        freqs_x = t_x.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        freqs_y = t_y.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        
        freqs_x = freqs_x.unsqueeze(1).expand(seq_len_x, seq_len_y, -1)
        freqs_y = freqs_y.unsqueeze(0).expand(seq_len_x, seq_len_y, -1)
        
        freqs = torch.cat([freqs_x, freqs_y], dim=-1)
        return freqs  # [seq_len_x * seq_len_y, dim_head // 2]


def apply_rotary_pos_emb(x, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_out = torch.empty_like(x)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    q_out[..., ::2] = x1 * cos - x2 * sin
    q_out[..., 1::2] = x1 * sin + x2 * cos
    return q_out

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        cross_attend=False,
        flash=True,
        kv_heads=None,
        local_window=None,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.cross_attend = cross_attend
        self.flash = flash
        self.dropout = dropout
        self.kv_heads = kv_heads if kv_heads else heads
        self.local_window = local_window if local_window else -1

        inner_dim = dim_head * heads
        kv_inner = dim_head * self.kv_heads
        
        # Corrected QKV projection size for GQA
        if not cross_attend:
            self.to_qkv = nn.Linear(dim, inner_dim + (2 * kv_inner), bias=False)
        else:
            self.to_q = nn.Linear(dim, inner_dim, bias=False)
            self.to_kv = nn.Linear(dim, 2 * kv_inner, bias=False)

        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.scale = dim_head ** -0.5
        self._mask_cache = {}

    def forward(self, x, context=None, is_causal=False, rope_emb=None, text_mask=None):
        b, n, _ = x.shape
        h, kv_h, d = self.heads, self.kv_heads, self.dim_head
        
        # 1. QKV Projection & Split
        if not self.cross_attend:
            qkv = self.to_qkv(x) # [B, N, (H + 2*KV_H) * D]
            q, k, v = qkv.split([h * d, kv_h * d, kv_h * d], dim=-1)
        else:
            q = self.to_q(x)
            kv = self.to_kv(context if context is not None else x)
            k, v = kv.chunk(2, dim=-1)

        # Reshape to [B, Heads, Seq, Dim_Head]
        q = q.view(b, n, h, d).transpose(1, 2)
        k = k.view(b, -1, kv_h, d).transpose(1, 2)
        v = v.view(b, -1, kv_h, d).transpose(1, 2)

        # 2. Apply RoPE
        if rope_emb is not None:
            # Note: apply_rotary_pos_emb needs to handle GQA (broadcasting K)
            q = apply_rotary_pos_emb(q, *rope_emb)
            k = apply_rotary_pos_emb(k, *rope_emb)

        # 3. Handle Grouped Query Attention (GQA)
        if kv_h != h:
            k = torch.repeat_interleave(k, h // kv_h, dim=1)
            v = torch.repeat_interleave(v, h // kv_h, dim=1)

        # 4. Master Mask Construction (CRITICAL)
        final_mask = None
        
        # Start with Padding Mask (Text Mask)
        if text_mask is not None:
            # text_mask is [B, S_kv], convert to [B, 1, 1, S_kv] for broadcasting
            final_mask = text_mask.view(b, 1, 1, -1).bool()

        # Add Local Window Mask
        if self.local_window > 0:
            q_n, k_n = q.shape[2], k.shape[2]
            cache_key = (q_n, k_n, self.local_window)
            if cache_key not in self._mask_cache:
                # Create causal window mask
                grid_q = torch.arange(q_n).unsqueeze(1)
                grid_k = torch.arange(k_n).unsqueeze(0)
                mask_win = (grid_k <= grid_q) & ((grid_q - grid_k) <= self.local_window)
                self._mask_cache[cache_key] = mask_win.to(q.device)
            
            window_mask = self._mask_cache[cache_key]
            final_mask = window_mask if final_mask is None else final_mask & window_mask

        # 5. Attention Execution
        # SDPA is faster but can be picky about mask dtypes
        if self.flash:
            # SDPA expects float mask for 'attn_mask' or bool for newer versions
            # If is_causal is True, SDPA handles the diagonal, but we merged it into our window mask
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=final_mask,
                is_causal=is_causal if final_mask is None else False, 
                dropout_p=self.dropout if self.training else 0.0,
                scale=self.scale
            )
        else:
            # Manual fallback
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if final_mask is not None:
                # Masked fill expects -inf for zeros
                fill_value = torch.finfo(attn.dtype).min
                attn = attn.masked_fill(~final_mask, fill_value)
            
            if is_causal and final_mask is None:
                # Basic causal if no complex mask exists
                causal_mask = torch.ones(n, n, device=q.device).triu(1).bool()
                attn = attn.masked_fill(causal_mask, torch.finfo(attn.dtype).min)

            attn = attn.softmax(dim=-1)
            out = attn @ v

        # 6. Output Projection
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)
class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        mlp_hidden_dim=None,
        dropout=0.0,
        ff_dropout=0.0,
        cross_attend=False,
        flash=True,
        glu=True,
        swish=True,
        ff_bias=True,
        pre_norm=True,
        kv_heads=None,
        local_window=None
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.cross_attend = cross_attend

        self.norm1 = SimpleRMSNorm(dim)
        self.norm2 = SimpleRMSNorm(dim)

        self.attn = Attention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            cross_attend=cross_attend,
            flash=flash,
            kv_heads=kv_heads,
            local_window=local_window
        )

        self.ff = FeedForward(
            dim=dim,
            hidden_dim=mlp_hidden_dim,
            bias=ff_bias,
            glu=glu,
            swish=swish,
            dropout=ff_dropout
        )

        if cross_attend:
            self.norm3 = SimpleRMSNorm(dim)
            self.cross_attn = Attention(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
                cross_attend=True,
                flash=flash
            )

    def forward(self, x, context=None, is_causal=False, rope_emb=None, text_mask=None):
        x_len = x.shape[1]
        ctx_len = context.shape[1] if context is not None else x_len
        rope_emb_self = None
        if rope_emb is not None and context is not None and context.shape[1] != x.shape[1]:
            rope_emb_self = None
        elif rope_emb is not None:
            rope_emb_self = rope_emb

        if self.pre_norm:
            attn_out = self.attn(self.norm1(x), context, is_causal, rope_emb_self, text_mask=text_mask)
            x = x + attn_out

            if self.cross_attend and context is not None:
                # Cross attention: pass None rope_emb since context has different seq len
                x = x + self.cross_attn(self.norm2(x), context,  is_causal, rope_emb=None, text_mask=text_mask)

            ff_in = self.norm2(x)
            x = x + self.ff(ff_in)
        else:
            x = self.norm1(x + self.attn(x, context, is_causal, rope_emb_self, text_mask=text_mask))
            if self.cross_attend and context is not None:
                x = x + self.cross_attn(self.norm2(x), context, is_causal, rope_emb=None, text_mask=text_mask)
            x = self.norm2(x + self.ff(x))
        return x


class FastEncoder(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads=8,
        dim_head=None,
        mlp_hidden_dim=None,
        dropout=0.0,
        ff_dropout=0.0,
        cross_attend=False,
        flash=True,
        glu=True,
        swish=True,
        ff_bias=True,
        pre_norm=True,
        rope_base=10000,
        rope_type=None,
        kv_heads=None,
        local_window=None,
        use_cuda_graph=False
    ):
        super().__init__()
        self.cross_attend = cross_attend
        self.rope_type = rope_type
        self.use_cuda_graph = use_cuda_graph and torch.cuda.is_available()
        
        dim_head = dim_head or dim // heads
        
        if rope_type == '1d':
            self.rope = RotaryEmbedding1D(dim_head, base=rope_base)
        elif rope_type == '2d':
            self.rope = RotaryEmbedding2D(dim_head, base=rope_base)
        else:
            self.rope = None
        
        self.rope_cache = {}
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                mlp_hidden_dim=mlp_hidden_dim,
                dropout=dropout,
                ff_dropout=ff_dropout,
                cross_attend=cross_attend,
                flash=flash,
                glu=glu,
                swish=swish,
                ff_bias=ff_bias,
                pre_norm=pre_norm,
                kv_heads=kv_heads,
                local_window=local_window
            )
            for _ in range(depth)
        ])

        self.norm = SimpleRMSNorm(dim)
        
        if self.use_cuda_graph:
            self._graph = None
            self._static_inputs = None

    def forward(self, x, context=None, mask=None, is_causal=False, text_mask=None):
        rope_emb = None
        
        if self.rope is not None:
            # Use x's seq len for self-attention RoPE (context can differ in cross attend)
            x_len = x.shape[1]
            if x_len in self.rope_cache:
                rope_emb = self.rope_cache[x_len]
            else:
                cos, sin = self._get_rope_cos_sin(x_len, x.device)
                cos = cos
                sin = sin
                rope_emb = (cos, sin)
                self.rope_cache[x_len] = rope_emb
        
        for layer in self.layers:
            x = layer(x, context, is_causal, rope_emb, text_mask=text_mask)
        return self.norm(x)
    
    def forward_cuda_graph(self, x, context=None, mask=None, is_causal=False):
        if self._graph is None:
            self._static_inputs = (x, context, mask, is_causal)
            self._graph = torch.cuda.CUDAGraph()
            static_out = self.forward(*self._static_inputs)
            self._graph.capture_begin()
            static_out = self.forward(*self._static_inputs)
            self._graph.capture_end()
            self._graph.replay()
        else:
            self._static_inputs = (x, context, mask, is_causal)
            self._graph.replay()
        return self._graph.outputs
    
    def _get_rope_cos_sin(self, seq_len, device):
        if self.rope_type == '1d':
            emb = self.rope(seq_len, device)
        else:
            seq_len_x = seq_len_y = int(math.sqrt(seq_len))
            emb = self.rope(seq_len_x, seq_len_y, device)
            emb = emb.view(-1, emb.shape[-1])
        
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin