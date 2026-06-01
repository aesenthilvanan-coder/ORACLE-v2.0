"""
embeddings.py
-------------
Atom-type and gene-expression embedding modules used across ORACLE models.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# Canonical atom ordering (index 0..9)
ATOM_TYPES = ["H", "C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]
N_ATOM_TYPES = len(ATOM_TYPES)  # 10


# ---------------------------------------------------------------------------
# AtomEmbedder
# ---------------------------------------------------------------------------


class AtomEmbedder(nn.Module):
    """Maps discrete atom-type indices to dense embeddings.

    Atom types are encoded as integer indices according to the canonical
    ordering: H=0, C=1, N=2, O=3, S=4, F=5, Cl=6, Br=7, I=8, P=9.

    Parameters
    ----------
    hidden_dim : int
        Embedding dimensionality.
    """

    ATOM_TO_IDX: dict[str, int] = {sym: i for i, sym in enumerate(ATOM_TYPES)}

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(N_ATOM_TYPES, hidden_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, atom_types: torch.Tensor) -> torch.Tensor:
        """Embed a batch of atom-type indices.

        Parameters
        ----------
        atom_types : torch.Tensor
            Integer tensor of shape ``(n_atoms,)`` with values in ``[0, 9]``.

        Returns
        -------
        torch.Tensor
            Embedding matrix ``(n_atoms, hidden_dim)``.
        """
        return self.embedding(atom_types.long())

    @classmethod
    def symbol_to_index(cls, symbol: str) -> int:
        """Convert an element symbol string to its canonical index."""
        return cls.ATOM_TO_IDX.get(symbol, 1)  # default to Carbon (1)


# ---------------------------------------------------------------------------
# GeneEmbedder
# ---------------------------------------------------------------------------


class GeneEmbedder(nn.Module):
    """Embeds gene expression values together with learnable positional (gene) encodings.

    Each gene *i* at expression level *v* is represented as:

        embed(i, v) = W_expr * v  +  position_embedding[i]

    where ``W_expr`` maps the scalar expression value to ``hidden_dim``
    dimensions and ``position_embedding`` provides a gene-identity embedding
    analogous to positional encodings in a Transformer.

    Parameters
    ----------
    n_genes : int
        Total number of genes in the vocabulary / panel.
    hidden_dim : int
        Embedding dimensionality.
    """

    def __init__(self, n_genes: int, hidden_dim: int) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.hidden_dim = hidden_dim

        # Scalar expression value -> hidden_dim
        self.expr_proj = nn.Linear(1, hidden_dim)

        # Learnable gene-identity / positional embeddings
        self.position_embedding = nn.Embedding(n_genes, hidden_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.expr_proj.weight, nonlinearity="relu")
        nn.init.zeros_(self.expr_proj.bias)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        expr: torch.Tensor,
        gene_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gene embeddings.

        Parameters
        ----------
        expr : torch.Tensor
            Expression values.  Shapes supported:

            * ``(batch, n_genes)``   — full expression matrix
            * ``(batch, k)``         — subset of k genes (paired with gene_idx)
            * ``(k,)``               — single sample, k genes

        gene_idx : torch.Tensor
            Integer gene indices corresponding to the last dimension of
            *expr*.  Shape ``(k,)`` or ``(n_genes,)``.

        Returns
        -------
        torch.Tensor
            Embeddings with the same leading dimensions as *expr*, plus
            ``hidden_dim`` appended.  E.g. ``(batch, k, hidden_dim)``.
        """
        # Ensure expr has a trailing feature dimension for the linear layer
        if expr.dim() == 1:
            expr = expr.unsqueeze(0)     # (1, k)
        expr_3d = expr.unsqueeze(-1)     # (..., k, 1)

        expr_emb = self.expr_proj(expr_3d)                   # (..., k, hidden_dim)
        pos_emb = self.position_embedding(gene_idx.long())   # (k, hidden_dim)

        # Broadcast position embedding over batch dimensions
        return expr_emb + pos_emb
