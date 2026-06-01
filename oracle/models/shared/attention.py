"""
attention.py
------------
Standard multi-head self-attention module with optional causal masking.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product self-attention.

    Implements the attention mechanism described in "Attention Is All You
    Need" (Vaswani et al., 2017).  Supports optional causal (autoregressive)
    masking and key-padding masking.

    Parameters
    ----------
    d_model : int
        Model embedding dimensionality.  Must be divisible by *n_heads*.
    n_heads : int
        Number of parallel attention heads.
    dropout : float
        Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Projection matrices
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self._reset_parameters()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _reset_parameters(self) -> None:
        """Xavier-uniform initialisation for projection weights."""
        for lin in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(self.W_o.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute multi-head self-attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, seq_len, d_model)``.
        mask : torch.Tensor | None
            Optional attention mask.  Two interpretations are supported:

            * **Causal / additive mask** — float tensor of shape
              ``(seq_len, seq_len)`` or ``(batch, seq_len, seq_len)`` where
              ``-inf`` entries prevent certain positions from attending.
            * **Boolean key-padding mask** — bool tensor of shape
              ``(batch, seq_len)`` where ``True`` marks positions to ignore.

        Returns
        -------
        torch.Tensor
            Attended output of shape ``(batch, seq_len, d_model)``.
        """
        batch, seq_len, _ = x.shape
        H, D = self.n_heads, self.d_k

        # --- Linear projections ------------------------------------------
        Q = self.W_q(x).view(batch, seq_len, H, D).transpose(1, 2)  # (B,H,S,D)
        K = self.W_k(x).view(batch, seq_len, H, D).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_len, H, D).transpose(1, 2)

        # --- Scaled dot-product attention --------------------------------
        scale = math.sqrt(D)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B,H,S,S)

        # Apply mask
        if mask is not None:
            if mask.dtype == torch.bool:
                # Key-padding mask: (B, S) -> (B, 1, 1, S)
                mask = mask.unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(mask, float("-inf"))
            else:
                # Additive mask: broadcast to (B, H, S, S)
                scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)           # (B, H, S, S)
        attn_weights = self.attn_dropout(attn_weights)

        # --- Aggregate values --------------------------------------------
        context = torch.matmul(attn_weights, V)            # (B, H, S, D)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, H * D)

        out = self.out_dropout(self.W_o(context))
        return out

    # ------------------------------------------------------------------
    # Convenience: causal mask factory
    # ------------------------------------------------------------------

    @staticmethod
    def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Return an additive causal mask of shape ``(seq_len, seq_len)``.

        Upper-triangular entries (excluding the diagonal) are set to ``-inf``
        so that each position can only attend to itself and previous positions.
        """
        mask = torch.zeros(seq_len, seq_len, device=device)
        mask = mask.masked_fill(
            torch.triu(torch.ones(seq_len, seq_len, device=device,
                                  dtype=torch.bool), diagonal=1),
            float("-inf"),
        )
        return mask
