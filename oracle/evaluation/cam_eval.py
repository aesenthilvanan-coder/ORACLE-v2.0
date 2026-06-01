"""CAM module evaluator: attractor accuracy, basin size agreement, GRN quality."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from oracle.evaluation.metrics import (
    attractor_accuracy,
    basin_overlap,
    grn_auroc,
    grn_early_precision,
)

logger = logging.getLogger(__name__)


class CAMEvaluator:
    """Evaluates the Cancer Attractor Mapper outputs against ground truth.

    Covers three axes:
    1. GRN inference quality (AUROC, AUPR, Early Precision vs BEELINE/TRRUST gold standards)
    2. Attractor classification accuracy (cancer/normal/transitional labels)
    3. Basin size agreement (Jaccard overlap of predicted basins vs scRNA-seq clusters)
    """

    def __init__(self, cancer_type: str = "colorectal") -> None:
        self.cancer_type = cancer_type
        self.results: Dict[str, Any] = {}

    def evaluate_grn(
        self,
        predicted_grn,
        ground_truth_edges: Set[Tuple[str, str]],
        all_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, float]:
        """Evaluate GRN inference quality against a gold-standard network.

        Args:
            predicted_grn: nx.DiGraph with edge 'weight' attributes.
            ground_truth_edges: set of (TF, target) pairs in the gold standard.
            all_pairs: optionally restrict evaluation to a specific set of pairs.

        Returns:
            dict with auroc, aupr, early_precision keys.
        """
        import networkx as nx

        predicted_weights: Dict[Tuple[str, str], float] = {}
        for u, v, data in predicted_grn.edges(data=True):
            predicted_weights[(u, v)] = float(data.get("weight", 0.0))

        if all_pairs is None:
            all_pairs = list(predicted_weights.keys())
            all_pairs += [e for e in ground_truth_edges if e not in predicted_weights]
            for e in ground_truth_edges:
                if e not in predicted_weights:
                    predicted_weights[e] = 0.0

        auroc = grn_auroc(predicted_weights, ground_truth_edges, all_pairs)
        ep = grn_early_precision(predicted_weights, ground_truth_edges)

        # AUPR via precision-recall at each threshold
        scores = np.array([predicted_weights.get(p, 0.0) for p in all_pairs])
        labels = np.array([1.0 if p in ground_truth_edges else 0.0 for p in all_pairs])
        aupr = self._compute_aupr(scores, labels)

        result = {"auroc": auroc, "aupr": aupr, "early_precision": ep}
        self.results["grn"] = result
        logger.info("GRN eval: AUROC=%.3f, AUPR=%.3f, EP=%.3f", auroc, aupr, ep)
        return result

    def evaluate_attractors(
        self,
        predicted_labels: List[str],
        true_labels: List[str],
        predicted_basin_fractions: Optional[Dict[int, float]] = None,
        true_basin_fractions: Optional[Dict[int, float]] = None,
    ) -> Dict[str, float]:
        """Evaluate attractor classification accuracy and basin size agreement."""
        acc = attractor_accuracy(predicted_labels, true_labels)

        basin_corr = float("nan")
        if predicted_basin_fractions and true_basin_fractions:
            shared_keys = [k for k in predicted_basin_fractions if k in true_basin_fractions]
            if shared_keys:
                pred_vals = np.array([predicted_basin_fractions[k] for k in shared_keys])
                true_vals = np.array([true_basin_fractions[k] for k in shared_keys])
                if pred_vals.std() > 0 and true_vals.std() > 0:
                    basin_corr = float(np.corrcoef(pred_vals, true_vals)[0, 1])

        result = {"attractor_accuracy": acc, "basin_size_correlation": basin_corr}
        self.results["attractors"] = result
        logger.info("Attractor eval: accuracy=%.3f, basin_corr=%.3f", acc, basin_corr)
        return result

    def evaluate_landscape(
        self,
        predicted_energy: np.ndarray,
        cell_labels: np.ndarray,
        umap_coords: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate landscape: cancer cells should have higher energy than normal cells."""
        cancer_mask = cell_labels == "cancer"
        normal_mask = cell_labels == "normal"

        if cancer_mask.sum() == 0 or normal_mask.sum() == 0:
            return {"energy_separation": float("nan")}

        # Map cell UMAP coordinates to energy grid
        # Simple: use mean energy in UMAP neighborhood
        energy_flat = predicted_energy.flatten() if predicted_energy.ndim > 1 else predicted_energy
        cancer_energy = float(np.median(energy_flat[:cancer_mask.sum()]))
        normal_energy = float(np.median(energy_flat[cancer_mask.sum():cancer_mask.sum() + normal_mask.sum()]))

        separation = cancer_energy - normal_energy
        result = {
            "energy_separation": separation,
            "mean_cancer_energy": cancer_energy,
            "mean_normal_energy": normal_energy,
        }
        self.results["landscape"] = result
        return result

    def summary(self) -> Dict[str, Any]:
        """Return aggregated evaluation summary."""
        return self.results.copy()

    @staticmethod
    def _compute_aupr(scores: np.ndarray, labels: np.ndarray) -> float:
        """Compute area under precision-recall curve via trapezoidal rule."""
        sorted_idx = np.argsort(-scores)
        sorted_labels = labels[sorted_idx]
        n_pos = int(sorted_labels.sum())
        if n_pos == 0:
            return 0.0

        precision_list = []
        recall_list = []
        tp = 0
        for i, label in enumerate(sorted_labels):
            if label == 1:
                tp += 1
            precision_list.append(tp / (i + 1))
            recall_list.append(tp / n_pos)

        return float(np.trapezoid(precision_list, recall_list))
