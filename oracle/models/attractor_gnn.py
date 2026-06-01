"""
attractor_gnn.py
----------------
Graph Neural Network for classifying GRN attractor states.

Given the GRN as a graph with node-level gene-expression features and the
current expression state, the model predicts one of three attractor classes:

  0 — Normal      (healthy phenotype)
  1 — Cancer      (tumour attractor)
  2 — Transitional (between-attractor transient state)

Architecture
------------
  - 3 GAT message-passing layers
  - Global mean pooling
  - 3-class MLP prediction head
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv, global_mean_pool
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    Data = None  # type: ignore


# ---------------------------------------------------------------------------
# AttractorGNN
# ---------------------------------------------------------------------------


class AttractorGNN(nn.Module):
    """Classifies the attractor state of a gene regulatory network.

    Parameters
    ----------
    n_genes : int
        Number of genes (nodes) in the GRN.  Used to set node input dimension
        when node features are scalar expression values (n_genes-length vector
        is treated as a graph-level feature; alternatively each node carries
        its own scalar expression value — 1-D per node).
    hidden_dim : int
        Hidden dimensionality throughout the network (default 128).
    """

    N_CLASSES: int = 3  # normal / cancer / transitional

    def __init__(self, n_genes: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim

        # Node encoder: assumes each node has a scalar expression value (1-D)
        # plus an optional one-hot gene-type feature.  We default to 1 feature.
        self.node_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 3 GAT message-passing layers
        n_heads = 4
        head_dim = hidden_dim // n_heads

        self.gat1 = GATConv(hidden_dim, head_dim, heads=n_heads, concat=True,
                             dropout=0.1)
        self.gat2 = GATConv(hidden_dim, head_dim, heads=n_heads, concat=True,
                             dropout=0.1)
        self.gat3 = GATConv(hidden_dim, head_dim, heads=n_heads, concat=True,
                             dropout=0.1)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.act = nn.GELU()

        # Classification head: global embedding -> 3 logits
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, self.N_CLASSES),
        )

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------

    def forward(self, data: Data) -> torch.Tensor:
        """Predict attractor class logits.

        Parameters
        ----------
        data : torch_geometric.data.Data
            GRN graph.  Required fields:

            * ``x``          – node features ``(n_nodes, in_dim)``
              If ``in_dim > 1``, only the first column (expression value) is
              used for the node encoder; the rest is ignored unless you modify
              the ``node_encoder`` input size.
            * ``edge_index`` – ``(2, n_edges)``
            * ``batch``      – (optional) batch assignment ``(n_nodes,)``

        Returns
        -------
        torch.Tensor
            Class logits of shape ``(batch_size, 3)`` or ``(3,)`` for a single
            unbatched graph (caller should apply softmax / argmax as needed).
        """
        x_raw = data.x.float()
        edge_index = data.edge_index
        batch = data.batch if data.batch is not None else \
            torch.zeros(x_raw.size(0), dtype=torch.long, device=x_raw.device)

        # Use first feature column (expression) for the linear encoder
        x_expr = x_raw[:, :1] if x_raw.dim() == 2 else x_raw.unsqueeze(-1)
        h = self.node_encoder(x_expr)                    # (N, hidden_dim)

        # GAT layer 1
        h1 = self.act(self.norm1(h + self.gat1(h, edge_index)))

        # GAT layer 2
        h2 = self.act(self.norm2(h1 + self.gat2(h1, edge_index)))

        # GAT layer 3
        h3 = self.act(self.norm3(h2 + self.gat3(h2, edge_index)))

        # Global pooling
        graph_emb = global_mean_pool(h3, batch)          # (B, hidden_dim)

        # Classification
        logits = self.head(graph_emb)                    # (B, 3)

        # Squeeze batch dimension for single-graph input
        if logits.size(0) == 1 and (data.batch is None):
            logits = logits.squeeze(0)                   # (3,)

        return logits

    # ------------------------------------------------------------------

    def predict_class(self, data: Data) -> int:
        """Return the predicted attractor class (0, 1, or 2) for a single graph."""
        with torch.no_grad():
            logits = self.forward(data)
            return int(logits.argmax(-1).item())

    # ------------------------------------------------------------------

    @staticmethod
    def class_name(idx: int) -> str:
        """Map class index to human-readable label."""
        return {0: "normal", 1: "cancer", 2: "transitional"}.get(idx, "unknown")
