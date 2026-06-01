import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MultiHeadAttentionWithRelPos(nn.Module):
    """Multi-head attention with relative positional encoding for GRN edges."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, max_rel_pos: int = 50):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.rel_pos_emb = nn.Embedding(2 * max_rel_pos + 1, self.d_k)
        self.max_rel_pos = max_rel_pos
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        edge_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        Q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        positions = torch.arange(N, device=x.device)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)
        rel_pos = rel_pos.clamp(-self.max_rel_pos, self.max_rel_pos) + self.max_rel_pos
        rel_emb = self.rel_pos_emb(rel_pos)
        rel_bias = torch.einsum("bhid,ijd->bhij", Q, rel_emb) * self.scale
        scores = scores + rel_bias

        if edge_bias is not None:
            scores = scores + edge_bias.unsqueeze(1)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        return self.W_o(out)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttentionWithRelPos(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        edge_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), mask, edge_bias))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class GRNTransformer(nn.Module):
    """Transformer for GRN edge prediction and regulatory strength estimation."""

    def __init__(
        self,
        n_genes: int,
        d_model: int = 768,
        n_heads: int = 8,
        n_layers: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
        n_expression_bins: int = 50,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_genes = n_genes
        self.gradient_checkpointing = gradient_checkpointing

        self.gene_emb = nn.Embedding(n_genes, d_model)
        self.expr_proj = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
        )
        self.input_norm = nn.LayerNorm(d_model)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.edge_predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

    def forward(
        self,
        gene_ids: torch.Tensor,
        expression: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gene_h = self.gene_emb(gene_ids)
        expr_h = self.expr_proj(expression.unsqueeze(-1))
        h = self.input_norm(gene_h + expr_h)

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                import torch.utils.checkpoint as ckpt
                h = ckpt.checkpoint(layer, h, mask, None, use_reentrant=False)
            else:
                h = layer(h, mask)

        B, N, D = h.shape
        hi = h.unsqueeze(2).expand(B, N, N, D)
        hj = h.unsqueeze(1).expand(B, N, N, D)
        edge_input = torch.cat([hi, hj], dim=-1)
        edge_logits = self.edge_predictor(edge_input)

        edge_exists = torch.sigmoid(edge_logits[..., 0])
        edge_sign = torch.tanh(edge_logits[..., 1])
        edge_weight = torch.sigmoid(edge_logits[..., 2])

        return edge_exists, torch.stack([edge_sign, edge_weight], dim=-1)

    def predict_grn(
        self,
        gene_ids: torch.Tensor,
        expression: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            edge_prob, edge_attrs = self.forward(gene_ids, expression)
            edge_mask = edge_prob > threshold
            return edge_mask, edge_prob, edge_attrs
