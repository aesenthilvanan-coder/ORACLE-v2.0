import torch
import torch.nn as nn
from typing import Optional


class AttentionPooling(nn.Module):
    """Attention-weighted pooling over node representations."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        scores = self.gate(h)
        n_graphs = int(batch.max().item()) + 1
        out = torch.zeros(n_graphs, h.shape[-1], device=h.device)

        for g in range(n_graphs):
            mask = batch == g
            h_g = h[mask]
            s_g = torch.softmax(scores[mask], dim=0)
            out[g] = (s_g * h_g).sum(dim=0)

        return out


class SetTransformerPooling(nn.Module):
    """Set2Set / ISAB-style pooling via cross-attention with learned seed vectors."""

    def __init__(self, hidden_dim: int, n_seeds: int = 4, n_heads: int = 4):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(n_seeds, hidden_dim))
        self.cross_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.proj = nn.Linear(n_seeds * hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        n_graphs = int(batch.max().item()) + 1
        out = []
        seeds = self.seeds.unsqueeze(0)

        for g in range(n_graphs):
            mask = batch == g
            h_g = h[mask].unsqueeze(0)
            pooled, _ = self.cross_attn(seeds, h_g, h_g)
            out.append(pooled.flatten(-2))

        stacked = torch.cat(out, dim=0)
        return self.proj(stacked)


class GlobalMeanAddPool(nn.Module):
    """Concatenation of global mean and global add pooling."""

    def forward(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        try:
            from torch_geometric.nn import global_mean_pool, global_add_pool
            return torch.cat([global_mean_pool(h, batch), global_add_pool(h, batch)], dim=-1)
        except ImportError:
            n_graphs = int(batch.max().item()) + 1
            mean_pool = torch.zeros(n_graphs, h.shape[-1], device=h.device)
            add_pool = torch.zeros(n_graphs, h.shape[-1], device=h.device)
            counts = torch.zeros(n_graphs, 1, device=h.device)
            idx = batch.unsqueeze(-1).expand_as(h)
            add_pool.scatter_add_(0, idx, h)
            counts.scatter_add_(0, batch.unsqueeze(-1), torch.ones(h.shape[0], 1, device=h.device))
            mean_pool = add_pool / counts.clamp(min=1)
            return torch.cat([mean_pool, add_pool], dim=-1)
