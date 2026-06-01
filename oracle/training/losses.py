"""
Loss function classes for the three ORACLE training pipelines.

CancerScoreLoss    – CAM module: classification + monotonicity + smoothness
SwitchPredictorLoss – RSP module: score regression + reversion BCE + sparsity
DiffusionLoss      – TCD module: DDPM coordinate MSE + atom-type cross-entropy
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CancerScoreLoss
# ---------------------------------------------------------------------------

class CancerScoreLoss(nn.Module):
    """Combined loss for training the CancerScoreFunction.

    The total loss is::

        L = cls_loss + mono_weight * mono_loss + smooth_weight * smooth_loss

    where:

    - **cls_loss**    – binary cross-entropy between predicted scores and
                         cancer / normal labels.
    - **mono_loss**   – pseudotime monotonicity penalty:
                         ``relu(early_scores - late_scores).mean()``
    - **smooth_loss** – gradient-magnitude smoothness penalty (L2 norm of
                         finite differences along the batch dimension).

    Parameters
    ----------
    mono_weight:
        Weight on the monotonicity loss term.  Default ``0.1``.
    smooth_weight:
        Weight on the smoothness loss term.  Default ``0.01``.
    """

    def __init__(
        self,
        mono_weight: float = 0.1,
        smooth_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.mono_weight = mono_weight
        self.smooth_weight = smooth_weight

    def forward(
        self,
        cancer_scores: torch.Tensor,
        normal_scores: torch.Tensor,
        pseudotime_pairs: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute the combined cancer-score loss.

        Parameters
        ----------
        cancer_scores:
            Predicted scores for cancer states, shape ``(N_cancer,)``.
        normal_scores:
            Predicted scores for normal states, shape ``(N_normal,)``.
        pseudotime_pairs:
            Optional tensor of shape ``(P, 2)`` where each row ``[i, j]``
            means the *i*-th cancer state is earlier in pseudotime than
            the *j*-th cancer state.
        states:
            Optional tensor of states (unused in base implementation; kept
            for subclass extensibility).

        Returns
        -------
        dict
            Keys: ``"cls_loss"``, ``"mono_loss"``, ``"smooth_loss"``,
            ``"total"``.
        """
        device = cancer_scores.device

        # --- Classification loss ---
        all_scores = torch.cat([cancer_scores, normal_scores], dim=0)
        labels = torch.cat(
            [
                torch.ones(len(cancer_scores), device=device),
                torch.zeros(len(normal_scores), device=device),
            ]
        )
        cls_loss = F.binary_cross_entropy_with_logits(all_scores, labels)

        # --- Monotonicity loss ---
        if pseudotime_pairs is not None and len(pseudotime_pairs) > 0:
            # pairs[i] = [early_idx, late_idx] into cancer_scores
            early_idx = pseudotime_pairs[:, 0].long()
            late_idx = pseudotime_pairs[:, 1].long()
            early_scores = cancer_scores[early_idx]
            late_scores = cancer_scores[late_idx]
            mono_loss = F.relu(early_scores - late_scores).mean()
        else:
            mono_loss = torch.tensor(0.0, device=device)

        # --- Smoothness loss ---
        # Penalise large differences between adjacent scores in the batch
        if len(all_scores) > 1:
            diffs = all_scores[1:] - all_scores[:-1]
            smooth_loss = (diffs ** 2).mean()
        else:
            smooth_loss = torch.tensor(0.0, device=device)

        total = (
            cls_loss
            + self.mono_weight * mono_loss
            + self.smooth_weight * smooth_loss
        )

        return {
            "cls_loss": cls_loss,
            "mono_loss": mono_loss,
            "smooth_loss": smooth_loss,
            "total": total,
        }


# ---------------------------------------------------------------------------
# SwitchPredictorLoss
# ---------------------------------------------------------------------------

class SwitchPredictorLoss(nn.Module):
    """Combined loss for training the SwitchPredictorGNN.

    The total loss is::

        L = score_loss + rev_loss + importance_weight * importance_loss

    where:

    - **score_loss**      – MSE on predicted cancer score vs. target score.
    - **rev_loss**        – BCE on predicted reversion probability vs. label.
    - **importance_loss** – L1 sparsity on gene importance scores.

    Parameters
    ----------
    importance_weight:
        Weight on the L1 sparsity term.  Default ``0.001``.
    """

    def __init__(self, importance_weight: float = 0.001) -> None:
        super().__init__()
        self.importance_weight = importance_weight

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute the combined switch-predictor loss.

        Parameters
        ----------
        outputs:
            Dict from the SwitchPredictorGNN with keys:

            - ``"cancer_score"`` – predicted cancer score, shape ``(B,)``
            - ``"reversion_prob"`` – predicted reversion probability, shape ``(B,)``
            - ``"gene_importance"`` – per-gene importance scores, shape ``(B, G)``

        batch:
            Dict from the DataLoader with keys:

            - ``"target_score"`` – ground-truth cancer score, shape ``(B,)``
            - ``"reversion_label"`` – binary reversion label, shape ``(B,)``

        Returns
        -------
        dict
            Keys: ``"score_loss"``, ``"rev_loss"``, ``"importance_loss"``,
            ``"total"``.
        """
        device = outputs.get("cancer_score", torch.tensor(0.0)).device

        # --- Score MSE loss ---
        if "cancer_score" in outputs and "target_score" in batch:
            score_loss = F.mse_loss(
                outputs["cancer_score"].squeeze(),
                batch["target_score"].squeeze().to(device),
            )
        else:
            score_loss = torch.tensor(0.0, device=device)

        # --- Reversion BCE loss ---
        if "reversion_prob" in outputs and "reversion_label" in batch:
            rev_loss = F.binary_cross_entropy_with_logits(
                outputs["reversion_prob"].squeeze(),
                batch["reversion_label"].squeeze().float().to(device),
            )
        else:
            rev_loss = torch.tensor(0.0, device=device)

        # --- Gene importance L1 sparsity ---
        if "gene_importance" in outputs:
            importance_loss = outputs["gene_importance"].abs().mean()
        else:
            importance_loss = torch.tensor(0.0, device=device)

        total = (
            score_loss
            + rev_loss
            + self.importance_weight * importance_loss
        )

        return {
            "score_loss": score_loss,
            "rev_loss": rev_loss,
            "importance_loss": importance_loss,
            "total": total,
        }


# ---------------------------------------------------------------------------
# DiffusionLoss
# ---------------------------------------------------------------------------

class DiffusionLoss(nn.Module):
    """DDPM loss for training the TCIPDiffusionModel.

    The total loss is::

        L = coord_loss + type_loss

    where:

    - **coord_loss** – MSE between predicted denoised coordinates and the
                        true noise.
    - **type_loss**  – cross-entropy between predicted atom-type logits and
                        true atom types.

    Parameters
    ----------
    coord_weight:
        Weight multiplier on the coordinate loss.  Default ``1.0``.
    type_weight:
        Weight multiplier on the atom-type loss.  Default ``1.0``.
    """

    def __init__(
        self,
        coord_weight: float = 1.0,
        type_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.coord_weight = coord_weight
        self.type_weight = type_weight

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        noise_coords: torch.Tensor,
        atom_types: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute the combined diffusion loss.

        Parameters
        ----------
        output:
            Dict from the TCIPDiffusionModel with keys:

            - ``"pred_noise"`` – predicted noise on coordinates, shape
              ``(B, N_atoms, 3)``.
            - ``"atom_type_logits"`` – predicted atom-type logits, shape
              ``(B, N_atoms, vocab_size)``.

        noise_coords:
            Ground-truth noise added to coordinates at diffusion step *t*,
            shape ``(B, N_atoms, 3)``.
        atom_types:
            Ground-truth atom-type indices, shape ``(B, N_atoms)``.

        Returns
        -------
        dict
            Keys: ``"coord_loss"``, ``"type_loss"``, ``"total"``.
        """
        device = noise_coords.device

        # --- Coordinate MSE loss (predict the noise) ---
        if "pred_noise" in output:
            coord_loss = F.mse_loss(output["pred_noise"], noise_coords)
        else:
            coord_loss = torch.tensor(0.0, device=device)

        # --- Atom-type cross-entropy loss ---
        if "atom_type_logits" in output and atom_types is not None:
            # logits: (B, N_atoms, vocab) → (B*N_atoms, vocab)
            logits = output["atom_type_logits"]
            B, N, V = logits.shape
            type_loss = F.cross_entropy(
                logits.reshape(B * N, V),
                atom_types.reshape(B * N).long(),
            )
        else:
            type_loss = torch.tensor(0.0, device=device)

        total = self.coord_weight * coord_loss + self.type_weight * type_loss

        return {
            "coord_loss": coord_loss,
            "type_loss": type_loss,
            "total": total,
        }
