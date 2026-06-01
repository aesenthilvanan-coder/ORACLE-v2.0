import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GATLayer(nn.Module):
    """Single GAT layer with optional edge features."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_heads: int = 4,
        edge_dim: Optional[int] = None,
        dropout: float = 0.1,
        concat: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.concat = concat
        head_dim = out_dim

        self.W = nn.Linear(in_dim, n_heads * head_dim, bias=False)
        self.a_src = nn.Parameter(torch.randn(1, n_heads, head_dim))
        self.a_dst = nn.Parameter(torch.randn(1, n_heads, head_dim))

        if edge_dim is not None:
            self.edge_proj = nn.Linear(edge_dim, n_heads * head_dim, bias=False)
            self.a_edge = nn.Parameter(torch.randn(1, n_heads, head_dim))
        else:
            self.edge_proj = None
            self.a_edge = None

        self.dropout = nn.Dropout(dropout)
        final_dim = n_heads * head_dim if concat else head_dim
        self.norm = nn.LayerNorm(final_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        N = h.shape[0]
        head_dim = self.out_dim
        Wh = self.W(h).view(N, self.n_heads, head_dim)

        src, dst = edge_index
        attn = (Wh[src] * self.a_src).sum(-1) + (Wh[dst] * self.a_dst).sum(-1)

        if self.edge_proj is not None and edge_attr is not None:
            e_h = self.edge_proj(edge_attr).view(-1, self.n_heads, head_dim)
            attn = attn + (e_h * self.a_edge).sum(-1)

        attn = F.leaky_relu(attn, 0.2)

        attn_max = torch.zeros(N, self.n_heads, device=h.device)
        attn_max.scatter_reduce_(0, dst.unsqueeze(-1).expand(-1, self.n_heads), attn, reduce="amax", include_self=True)
        attn = torch.exp(attn - attn_max[dst])
        attn_sum = torch.zeros(N, self.n_heads, device=h.device)
        attn_sum.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.n_heads), attn)
        attn_norm = attn / (attn_sum[dst] + 1e-8)
        attn_norm = self.dropout(attn_norm)

        msgs = Wh[src] * attn_norm.unsqueeze(-1)
        agg = torch.zeros(N, self.n_heads, head_dim, device=h.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1).expand_as(msgs), msgs)

        if self.concat:
            out = agg.view(N, self.n_heads * head_dim)
        else:
            out = agg.mean(dim=1)

        return self.norm(out)


class MultiLayerGAT(nn.Module):
    """Stacked GAT layers with residual connections."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 4,
        n_heads: int = 4,
        edge_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GATLayer(hidden_dim, hidden_dim // n_heads, n_heads, edge_dim, dropout)
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.input_proj(h)
        for layer in self.layers:
            h = h + layer(h, edge_index, edge_attr)
        return self.output_proj(h)
