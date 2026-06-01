import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PreNormTransformerLayer(nn.Module):
    """Pre-LN transformer layer (more stable training than Post-LN)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class CrossAttentionLayer(nn.Module):
    """Cross-attention between query sequence and context sequence."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.norm_q(query)
        kv = self.norm_kv(context)
        out, _ = self.attn(q, kv, kv, key_padding_mask=key_padding_mask)
        return query + out


class TransformerStack(nn.Module):
    """A stack of pre-LN transformer layers with optional input projection."""

    def __init__(
        self,
        in_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model) if in_dim != d_model else nn.Identity()
        self.layers = nn.ModuleList([
            PreNormTransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return self.output_norm(x)
