"""
trajectory_viz.py
-----------------
Visualization tools for cell state trajectories through the attractor landscape.

Class
-----
TrajectoryVisualizer : Plots cancer score trajectories, gene expression
    heatmaps, and before/after perturbation comparisons.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False
    Figure = object  # type: ignore
    logger.warning("matplotlib not available; TrajectoryVisualizer plots disabled.")


class TrajectoryVisualizer:
    """
    Visualizes cell state trajectories through the attractor landscape.

    Parameters
    ----------
    figsize : tuple
        Default figure size (width, height) in inches.
    dpi : int
        Figure resolution.
    """

    def __init__(self, figsize: tuple = (10, 5), dpi: int = 120) -> None:
        self.figsize = figsize
        self.dpi = dpi

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot_cancer_score_trajectory(
        self,
        trajectory: np.ndarray,
        cancer_score_fn: Callable,
        title: str = "",
    ) -> Figure:
        """
        Line plot of cancer score over pseudotime.

        Parameters
        ----------
        trajectory : np.ndarray
            Cell state trajectory, shape (n_steps, n_genes).
        cancer_score_fn : callable
            Callable that maps a (1, n_genes) array to a scalar cancer score.
        title : str
            Plot title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not _MPL_AVAILABLE:
            logger.error("matplotlib required for plot_cancer_score_trajectory.")
            return None  # type: ignore

        import torch

        n_steps = trajectory.shape[0]
        scores = []
        for i in range(n_steps):
            x = torch.tensor(trajectory[i], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                score = float(cancer_score_fn(x).item())
            scores.append(score)

        pseudotime = np.linspace(0, 1, n_steps)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.plot(pseudotime, scores, color="#e74c3c", lw=2.2, label="Cancer Score")
        ax.axhline(0.5, color="gray", linestyle="--", lw=1, alpha=0.7,
                   label="Decision threshold (0.5)")
        ax.fill_between(pseudotime, scores, 0.5,
                        where=[s > 0.5 for s in scores],
                        alpha=0.15, color="#e74c3c", label="Cancer region")
        ax.fill_between(pseudotime, scores, 0.5,
                        where=[s <= 0.5 for s in scores],
                        alpha=0.15, color="#2ecc71", label="Normal region")

        ax.set_xlabel("Pseudotime", fontsize=12)
        ax.set_ylabel("Cancer Score", fontsize=12)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, 1)
        ax.set_title(title or "Cancer Score Trajectory", fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=9, framealpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        return fig

    def plot_gene_trajectory(
        self,
        trajectory: np.ndarray,
        gene_names: List[str],
        genes_to_plot: Optional[List[str]] = None,
    ) -> Figure:
        """
        Heatmap of gene expression over the trajectory (pseudotime).

        Parameters
        ----------
        trajectory : np.ndarray
            Cell state trajectory, shape (n_steps, n_genes).
        gene_names : List[str]
            Ordered list of gene names corresponding to trajectory columns.
        genes_to_plot : List[str], optional
            Subset of genes to visualize.  If None, uses top 50 by variance.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not _MPL_AVAILABLE:
            logger.error("matplotlib required for plot_gene_trajectory.")
            return None  # type: ignore

        n_steps, n_genes = trajectory.shape

        # Select genes
        if genes_to_plot is not None:
            indices = [
                gene_names.index(g) for g in genes_to_plot
                if g in gene_names
            ]
        else:
            # Top 50 by variance across trajectory
            variances = trajectory.var(axis=0)
            indices = list(np.argsort(variances)[::-1][:50])

        if not indices:
            indices = list(range(min(50, n_genes)))

        sub_traj = trajectory[:, indices].T  # (n_selected, n_steps)
        sub_names = [gene_names[i] for i in indices]
        pseudotime = np.linspace(0, 1, n_steps)

        fig_height = max(6, len(indices) * 0.22)
        fig, ax = plt.subplots(figsize=(12, fig_height), dpi=self.dpi)

        im = ax.imshow(
            sub_traj,
            aspect="auto",
            cmap="RdBu_r",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_yticks(range(len(sub_names)))
        ax.set_yticklabels(sub_names, fontsize=max(5, 8 - len(sub_names) // 20))

        # X-axis: pseudotime ticks
        n_xticks = min(10, n_steps)
        xtick_pos = np.linspace(0, n_steps - 1, n_xticks, dtype=int)
        ax.set_xticks(xtick_pos)
        ax.set_xticklabels([f"{pseudotime[i]:.2f}" for i in xtick_pos], fontsize=9)

        ax.set_xlabel("Pseudotime", fontsize=12)
        ax.set_ylabel("Genes", fontsize=12)
        ax.set_title("Gene Expression Trajectory Heatmap", fontsize=14, fontweight="bold")

        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Expression", fontsize=10)

        fig.tight_layout()
        return fig

    def plot_perturbation_effect(
        self,
        before_trajectory: np.ndarray,
        after_trajectory: np.ndarray,
        cancer_score_fn: Callable,
    ) -> Figure:
        """
        Before/after comparison of cancer scores along two trajectories.

        Parameters
        ----------
        before_trajectory : np.ndarray
            Unperturbed trajectory, shape (n_steps, n_genes).
        after_trajectory : np.ndarray
            Perturbed trajectory, shape (n_steps, n_genes).
        cancer_score_fn : callable
            Differentiable cancer score function.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not _MPL_AVAILABLE:
            logger.error("matplotlib required for plot_perturbation_effect.")
            return None  # type: ignore

        import torch

        def _score_trajectory(traj: np.ndarray) -> List[float]:
            scores = []
            for step in traj:
                x = torch.tensor(step, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    scores.append(float(cancer_score_fn(x).item()))
            return scores

        before_scores = _score_trajectory(before_trajectory)
        after_scores = _score_trajectory(after_trajectory)

        n_steps = max(len(before_scores), len(after_scores))
        pt_before = np.linspace(0, 1, len(before_scores))
        pt_after = np.linspace(0, 1, len(after_scores))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=self.dpi,
                                 sharey=True)

        ax_before, ax_after = axes

        # Before
        ax_before.plot(pt_before, before_scores, color="#e74c3c", lw=2.2)
        ax_before.fill_between(pt_before, before_scores, 0, alpha=0.2, color="#e74c3c")
        ax_before.axhline(0.5, color="gray", linestyle="--", lw=1, alpha=0.7)
        ax_before.set_title("Before Perturbation", fontsize=13, fontweight="bold")
        ax_before.set_xlabel("Pseudotime", fontsize=11)
        ax_before.set_ylabel("Cancer Score", fontsize=11)
        ax_before.set_ylim(-0.05, 1.05)
        ax_before.spines["top"].set_visible(False)
        ax_before.spines["right"].set_visible(False)

        # After
        ax_after.plot(pt_after, after_scores, color="#2ecc71", lw=2.2)
        ax_after.fill_between(pt_after, after_scores, 0, alpha=0.2, color="#2ecc71")
        ax_after.axhline(0.5, color="gray", linestyle="--", lw=1, alpha=0.7)
        ax_after.set_title("After Perturbation", fontsize=13, fontweight="bold")
        ax_after.set_xlabel("Pseudotime", fontsize=11)
        ax_after.set_ylim(-0.05, 1.05)
        ax_after.spines["top"].set_visible(False)
        ax_after.spines["right"].set_visible(False)

        # Delta annotation
        delta = np.mean(after_scores) - np.mean(before_scores)
        delta_str = f"ΔScore = {delta:+.3f}"
        fig.suptitle(
            f"Perturbation Effect — {delta_str}",
            fontsize=14,
            fontweight="bold",
        )

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig
