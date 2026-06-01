"""RSP module evaluator: switch prediction accuracy, reversion trajectory quality."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from oracle.evaluation.metrics import switch_f1, reversion_auc

logger = logging.getLogger(__name__)


class RSPEvaluator:
    """Evaluates RSP module outputs against ground-truth perturbation sets.

    Primary benchmarks:
    - KAIST REVERT: CDX2 activate + SNAI2 repress for colorectal → normal
    - AML ATRA: CEBPA/IRF8/SPI1 activation for AML → normal differentiation
    """

    # Ground-truth perturbation sets per benchmark
    BENCHMARKS: Dict[str, Dict[str, str]] = {
        "kaist_colorectal": {"CDX2": "Activation", "SNAI2": "Repression"},
        "aml_atra": {
            "CEBPA": "Activation",
            "IRF8": "Activation",
            "SPI1": "Activation",
        },
        "breast_erpos": {"ESR1": "Activation", "FOXA1": "Activation"},
        "lung_luad": {"NKX2-1": "Activation", "FOXA2": "Activation"},
    }

    def __init__(self, benchmark: str = "kaist_colorectal") -> None:
        self.benchmark = benchmark
        self.ground_truth = self.BENCHMARKS.get(benchmark, {})
        self.results: Dict[str, Any] = {}

    def evaluate_switch_prediction(
        self,
        predicted_perturbations: Dict[str, str],
        ground_truth_perturbations: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """Compute precision, recall, F1 for perturbation gene + type prediction."""
        gt = ground_truth_perturbations or self.ground_truth
        metrics = switch_f1(predicted_perturbations, gt, match_type=True)
        gene_only = switch_f1(predicted_perturbations, gt, match_type=False)

        result = {
            "gene_and_type_f1": metrics["f1"],
            "gene_and_type_precision": metrics["precision"],
            "gene_and_type_recall": metrics["recall"],
            "gene_only_f1": gene_only["f1"],
            "n_predicted": len(predicted_perturbations),
            "n_ground_truth": len(gt),
        }
        self.results["switch"] = result
        logger.info(
            "Switch eval [%s]: F1=%.3f (gene+type), F1=%.3f (gene only)",
            self.benchmark, metrics["f1"], gene_only["f1"]
        )
        return result

    def evaluate_reversion_trajectory(
        self,
        cancer_scores_before: np.ndarray,
        cancer_scores_after: np.ndarray,
        reversion_threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Evaluate how well the perturbation reduces cancer score.

        Args:
            cancer_scores_before: (N,) cancer scores at t=0 (before perturbation).
            cancer_scores_after: (N,) cancer scores at t=end (after perturbation).
            reversion_threshold: cancer score below which a cell is considered reverted.
        """
        delta = cancer_scores_before - cancer_scores_after
        mean_delta = float(np.mean(delta))
        reversion_frac = float((cancer_scores_after < reversion_threshold).mean())

        result = {
            "mean_cancer_score_reduction": mean_delta,
            "reversion_fraction": reversion_frac,
            "mean_initial_score": float(np.mean(cancer_scores_before)),
            "mean_final_score": float(np.mean(cancer_scores_after)),
        }
        self.results["trajectory"] = result
        logger.info(
            "Trajectory eval: mean_delta=%.3f, reversion_frac=%.3f",
            mean_delta, reversion_frac
        )
        return result

    def evaluate_cancer_score_model(
        self,
        predicted_scores: np.ndarray,
        true_labels: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate CancerScoreFunction as binary classifier (cancer=1, normal=0)."""
        auroc = reversion_auc(predicted_scores, true_labels)
        threshold = 0.5
        predicted_labels = (predicted_scores >= threshold).astype(int)
        accuracy = float((predicted_labels == true_labels).mean())
        tp = int(((predicted_labels == 1) & (true_labels == 1)).sum())
        fp = int(((predicted_labels == 1) & (true_labels == 0)).sum())
        fn = int(((predicted_labels == 0) & (true_labels == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        result = {
            "auroc": auroc,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        self.results["cancer_score"] = result
        logger.info("Cancer score model: AUROC=%.3f, F1=%.3f", auroc, f1)
        return result

    def evaluate_switch_size(
        self,
        predicted_sizes: List[int],
        optimal_size: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate whether the predicted perturbation set size is minimal."""
        if optimal_size is None:
            optimal_size = len(self.ground_truth)

        mean_size = float(np.mean(predicted_sizes))
        fraction_minimal = float(sum(1 for s in predicted_sizes if s <= optimal_size) / max(len(predicted_sizes), 1))

        result = {
            "mean_switch_size": mean_size,
            "optimal_size": optimal_size,
            "fraction_minimal_or_smaller": fraction_minimal,
        }
        self.results["switch_size"] = result
        return result

    def summary(self) -> Dict[str, Any]:
        return self.results.copy()
