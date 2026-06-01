"""
graph_layers.py
---------------
Graph attention layers with edge-feature support and a ready-to-use
molecular graph encoder.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import global_mean_pool
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    Data = None  # type: ignore

try:
    from torch_scatter import scatter_softmax, scatter_sum
except ImportError:
    def scatter_softmax(src: torch.Tensor, index: torch.Tensor,
                        dim: int = 0) -> torch.Tensor:
        """Pure-PyTorch scatter softmax fallback."""
        max_vals = src.new_zeros(int(index.max().item()) + 1)
        for _ in range(src.dim() - 1):
            pass  # only handles 1-D index for now
        max_vals.scatter_reduce_(0, index, src, reduce="amax", include_self=True)
        src_shifted = src - max_vals[index]
        exp = torch.exp(src_shifted)
        sum_exp = src.new_zeros(int(index.max().item()) + 1)
        sum_exp.scatter_add_(0, index, exp)
        return exp / (sum_exp[index] + 1e-9)

    def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                    dim_size: int | None = None) -> torch.Tensor:
        if dim_size is None:
            dim_size = int(index.max().item()) + 1
        shape = list(src.shape)
        shape[dim] = dim_size
        out = src.new_zeros(shape)
        idx = index
        for _ in range(src.dim() - 1):
            idx = idx.unsqueeze(-1)
        idx = idx.expand_as(src)
        out.scatter_add_(dim, idx, src)
        return out


# ---------------------------------------------------------------------------
# GATConvWithEdgeFeatures
# ---------------------------------------------------------------------------


class GATConvWithEdgeFeatures(nn.Module):
    """Graph Attention Network layer with edge-feature incorporation.

    Computes multi-head attention scores jointly from node and edge features,
    then aggregates neighbour messages weighted by those scores.

    Parameters
    ----------
    in_channels : int
        Input node-feature dimensionality.
    out_channels : int
        Output node-feature dimensionality per attention head.
    heads : int
        Number of parallel attention heads.
    edge_dim : int
        Dimensionality of edge features.
    dropout : float
        Attention-weight dropout probability.
    concat : bool
        If ``True`` (default), concatenate head outputs so the output is
        ``heads * out_channels``; if ``False``, average them.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int,
        edge_dim: int,
        dropout: float = 0.1,
        concat: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.concat = concat

        # Project node features into key/query/value spaces for all heads
        self.W_q = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.W_k = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.W_v = nn.Linear(in_channels, heads * out_channels, bias=False)

        # Project edge features into the same per-head space (key and bias)
        self.W_e_k = nn.Linear(edge_dim, heads * out_channels, bias=False)
        self.W_e_bias = nn.Linear(edge_dim, heads, bias=False)

        # Output projection after concatenation / averaging
        out_dim = heads * out_channels if concat else out_channels
        self.out_proj = nn.Linear(out_dim, out_dim)

        self.scale = math.sqrt(out_channels)
        self.attn_dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for lin in [self.W_q, self.W_k, self.W_v, self.W_e_k,
                    self.W_e_bias, self.out_proj]:
            nn.init.xavier_uniform_(lin.weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Run one GAT step with edge features.

        Parameters
        ----------
        x : torch.Tensor
            Node features ``(n_nodes, in_channels)``.
        edge_index : torch.Tensor
            Edge connectivity ``(2, n_edges)``.
        edge_attr : torch.Tensor
            Edge features ``(n_edges, edge_dim)``.

        Returns
        -------
        torch.Tensor
            Updated node features.
            Shape ``(n_nodes, heads * out_channels)`` when concat=True,
            ``(n_nodes, out_channels)`` otherwise.
        """
        n_nodes = x.size(0)
        src, dst = edge_index
        H, D = self.heads, self.out_channels

        # Project to multi-head representations
        q = self.W_q(x).view(n_nodes, H, D)          # (N, H, D)
        k = self.W_k(x).view(n_nodes, H, D)          # (N, H, D)
        v = self.W_v(x).view(n_nodes, H, D)          # (N, H, D)

        e_k = self.W_e_k(edge_attr).view(-1, H, D)   # (E, H, D)
        e_bias = self.W_e_bias(edge_attr)             # (E, H)

        # Attention score: dot(query_dst, key_src + edge_key) / scale + edge_bias
        key_src = k[src] + e_k                        # (E, H, D)
        attn_score = (q[dst] * key_src).sum(-1) / self.scale + e_bias  # (E, H)

        # Per-destination softmax
        # scatter_softmax operates element-wise; we handle per-head via reshape
        attn_flat = attn_score.view(-1)                # (E*H,)
        dst_rep = dst.unsqueeze(-1).expand_as(attn_score).reshape(-1)
        # Offset destination indices per head to keep heads independent
        n_edges = src.size(0)
        head_offset = torch.arange(H, device=x.device).unsqueeze(0) * n_nodes
        dst_offset = dst.unsqueeze(-1) + head_offset   # (E, H)
        dst_flat = dst_offset.view(-1)

        exp_attn = torch.exp(
            attn_flat - attn_flat.new_zeros(H * n_nodes).scatter_reduce_(
                0, dst_flat, attn_flat, reduce="amax", include_self=True
            )[dst_flat]
        )
        sum_exp = attn_flat.new_zeros(H * n_nodes).scatter_add_(
            0, dst_flat, exp_attn
        )
        attn_w = exp_attn / (sum_exp[dst_flat] + 1e-9)  # (E*H,)
        attn_w = attn_w.view(n_edges, H)                 # (E, H)
        attn_w = self.attn_dropout(attn_w)

        # Weighted message aggregation
        val_src = v[src]                               # (E, H, D)
        weighted_msg = val_src * attn_w.unsqueeze(-1)  # (E, H, D)
        weighted_msg_flat = weighted_msg.view(n_edges, H * D)

        agg = scatter_sum(weighted_msg_flat, dst, dim=0, dim_size=n_nodes)  # (N, H*D)
        agg = agg.view(n_nodes, H, D)

        if self.concat:
            out = agg.view(n_nodes, H * D)
        else:
            out = agg.mean(dim=1)                      # (N, D)

        return self.out_proj(out)


# ---------------------------------------------------------------------------
# MolecularGraphEncoder
# ---------------------------------------------------------------------------


class MolecularGraphEncoder(nn.Module):
    """Encodes molecular graphs to fixed-size graph-level embeddings.

    Expects graphs where ``data.x`` contains integer atom-type indices
    (0..9 for H, C, N, O, S, F, Cl, Br, I, P) and ``data.edge_attr``
    contains integer bond-type indices (0..3 for single, double, triple,
    aromatic).

    Parameters
    ----------
    hidden_dim : int
        Embedding dimensionality throughout the network.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # Discrete embeddings
        self.atom_embedder = nn.Embedding(10, hidden_dim)   # 10 atom types
        self.bond_embedder = nn.Embedding(4, hidden_dim)    # 4 bond types

        # Three GAT layers (using GATConvWithEdgeFeatures)
        n_heads = 4
        head_dim = hidden_dim // n_heads
        self.gat_layers = nn.ModuleList([
            GATConvWithEdgeFeatures(
                in_channels=hidden_dim,
                out_channels=head_dim,
                heads=n_heads,
                edge_dim=hidden_dim,
                dropout=0.1,
                concat=True,
            )
            for _ in range(3)
        ])
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(3)]
        )
        self.activations = nn.ModuleList(
            [nn.GELU() for _ in range(3)]
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Encode a molecular graph to a single embedding vector.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Requires:
            - ``x``          : atom type indices ``(n_atoms,)`` or
                               atom-type one-hot / feature matrix
            - ``edge_index`` : ``(2, n_edges)``
            - ``edge_attr``  : bond type indices ``(n_edges,)``
            - ``batch``      : batch assignment ``(n_atoms,)``

        Returns
        -------
        torch.Tensor
            Graph-level embedding ``(batch_size, hidden_dim)``.
        """
        # Handle atom features — either integer indices or float matrix
        if data.x.dim() == 1 or (data.x.dim() == 2 and data.x.size(1) == 1):
            atom_idx = data.x.view(-1).long().clamp(0, 9)
            h = self.atom_embedder(atom_idx)
        else:
            # Float feature matrix: project to hidden_dim via first atom embedder weight
            h = data.x.float() @ self.atom_embedder.weight[:data.x.size(1)].T
            h = h + self.atom_embedder.weight[0].detach() * 0  # shape guard

        # Handle edge features — either integer indices or float matrix
        if data.edge_attr is None:
            edge_h = self.bond_embedder(
                torch.zeros(data.edge_index.size(1), dtype=torch.long,
                            device=data.x.device)
            )
        elif data.edge_attr.dim() == 1 or (
            data.edge_attr.dim() == 2 and data.edge_attr.size(1) == 1
        ):
            bond_idx = data.edge_attr.view(-1).long().clamp(0, 3)
            edge_h = self.bond_embedder(bond_idx)
        else:
            edge_h = data.edge_attr.float() @ self.bond_embedder.weight[:data.edge_attr.size(1)].T

        batch = data.batch if data.batch is not None else torch.zeros(
            h.size(0), dtype=torch.long, device=h.device
        )

        for gat, norm, act in zip(self.gat_layers, self.layer_norms,
                                   self.activations):
            h_new = gat(h, data.edge_index, edge_h)
            h = act(norm(h + h_new))

        # Global mean pooling -> graph embedding
        graph_emb = global_mean_pool(h, batch)   # (batch_size, hidden_dim)
        return graph_emb
