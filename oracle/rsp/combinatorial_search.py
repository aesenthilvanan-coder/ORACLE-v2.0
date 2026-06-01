"""
combinatorial_search.py
-----------------------
Exhaustive combinatorial search for small perturbation sets.

This module provides a fallback strategy for identifying cancer-reversion
perturbation sets when the GNN predictor is unavailable or when an
exhaustive ground-truth search is desired for benchmarking.

For each combination size k from 1 up to max_k the searcher:

1. Enumerates all combinations of k genes from the candidate pool.
2. Tries both activation and repression for each gene in the combination
   (all 2^k perturbation-type assignments).
3. Simulates each combination with the ODE PerturbationSimulator.
4. Returns results sorted by ascending mean_cancer_score.

The exponential scaling (2^k * C(n,k) simulations) makes this practical
only for small k (≤ 3) and moderate gene lists.  For larger problems use
MinimalSwitchOptimizer.

RSPOutput dataclass
-------------------
The top-level output type for the RSP module, aggregating the SwitchSet
and all intermediate results needed by downstream modules (Module 3 / TCD).
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from oracle.rsp.cancer_score import RSPConfig
from oracle.rsp.perturbation_sim import PerturbationResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RSPOutput – top-level output dataclass for the RSP module
# ---------------------------------------------------------------------------


@dataclass
class RSPOutput:
    """Aggregated output from the Reversion Switch Predictor pipeline.

    This object is passed to downstream modules (e.g. Module 3 TCD) and
    contains everything produced by the RSP: the optimal switch set, all
    trajectory data, importance scores, and the upstream CAM output.

    Attributes
    ----------
    switch_set : SwitchSet
        The optimal minimal perturbation set found by the optimizer.
    genes_to_activate : List[str]
        Gene names to force toward high expression.
    genes_to_repress : List[str]
        Gene names to force toward low expression.
    n_perturbations : int
        Total number of perturbations (len(activate) + len(repress)).
    predicted_cancer_score_before : float
        Baseline (unperturbed) cancer score.
    predicted_cancer_score_after : float
        GNN-predicted cancer score after applying the switch set.
    predicted_reversion_probability : float
        GNN-predicted probability that the cell will revert to normal.
    validated_reversion_fraction : float
        Empirical reversion fraction from ODE simulation.
    perturbation_trajectories : List[Any]
        Raw trajectory arrays for all validation runs.
    cancer_score_trajectory : List[float]
        Cancer score at each ODE integration step (representative trajectory).
    gene_importance : Dict[str, float]
        Per-gene importance scores from the GNN.
    perturbation_type : Dict[str, str]
        Mapping ``{gene_name: 'activate'|'repress'}``.
    cam_output : Any
        Pass-through of the CAM (Module 1) output object for downstream use.
    """

    switch_set: Any  # SwitchSet
    genes_to_activate: List[str]
    genes_to_repress: List[str]
    n_perturbations: int
    predicted_cancer_score_before: float
    predicted_cancer_score_after: float
    predicted_reversion_probability: float
    validated_reversion_fraction: float
    perturbation_trajectories: List[Any] = field(default_factory=list)
    cancer_score_trajectory: List[float] = field(default_factory=list)
    gene_importance: Dict[str, float] = field(default_factory=dict)
    perturbation_type: Dict[str, str] = field(default_factory=dict)
    cam_output: Any = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# CombinatorialSearcher
# ---------------------------------------------------------------------------


class CombinatorialSearcher:
    """Exhaustive combinatorial search over all k-gene perturbation sets.

    This class is intended as a high-quality fallback / benchmark tool.
    For production use with large gene sets, prefer MinimalSwitchOptimizer.

    Parameters
    ----------
    simulator : PerturbationSimulator
        ODE-based simulator used to evaluate each combination.
    genes : List[str]
        Ordered list of gene names corresponding to expression vector indices.
    config : RSPConfig
        Shared RSP configuration.
    candidate_indices : List[int] or None
        Optional pre-filtered list of gene indices to search over.  If None,
        all genes are considered.  Restricting this to druggable TF indices
        drastically reduces search space.
    """

    def __init__(
        self,
        config_or_simulator=None,
        genes: Optional[List[str]] = None,
        config: Optional[RSPConfig] = None,
        candidate_indices: Optional[List[int]] = None,
    ) -> None:
        # Accept (config) or (simulator, genes, config, ...)
        if isinstance(config_or_simulator, RSPConfig):
            self.config = config_or_simulator
            self.simulator = None
            self.genes = genes or []
        else:
            self.simulator = config_or_simulator
            self.genes = genes or []
            self.config = config or RSPConfig()

        self._candidates: List[int] = (
            candidate_indices
            if candidate_indices is not None
            else list(range(len(self.genes)))
        )

        logger.info(
            "CombinatorialSearcher: %d candidate genes, max_k=%d.",
            len(self._candidates),
            self.config.max_perturbations,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        cancer_attractor: np.ndarray,
        normal_attractor: Optional[np.ndarray] = None,
        grn=None,
        cancer_score_fn=None,
        genes: Optional[List[str]] = None,
        max_k: int = 3,
    ) -> List[PerturbationResult]:
        """Run exhaustive search over all combinations up to size max_k.

        Parameters
        ----------
        cancer_attractor : np.ndarray
            Starting attractor state for simulations.
        normal_attractor : np.ndarray, optional
            Reference normal attractor (informational).
        grn : nx.DiGraph, optional
            Gene regulatory network for context.
        cancer_score_fn : optional
            Cancer score function for perturbation evaluation.
        genes : list of str, optional
            Gene names; replaces self.genes if provided.
        max_k : int
            Maximum combination size.

        Returns
        -------
        List[PerturbationResult]
        """
        if genes is not None:
            self.genes = genes
            self._candidates = list(range(len(genes)))
        if cancer_score_fn is not None:
            self._cancer_score_fn = cancer_score_fn

        # Create a minimal simulator if none provided
        if self.simulator is None:
            from oracle.rsp.perturbation_sim import PerturbationSimulator
            self.simulator = PerturbationSimulator(self.config)
            if cancer_score_fn is not None:
                self.simulator.cancer_score_fn = cancer_score_fn

        cancer_attractor = np.array(cancer_attractor, dtype=np.float32)
        all_results: List[PerturbationResult] = []

        for k in range(1, max_k + 1):
            logger.info("Searching k=%d combinations ...", k)
            combos = self._generate_combinations(k)
            logger.info("  k=%d: %d combinations to evaluate.", k, len(combos))

            for activate_idx, repress_idx in combos:
                try:
                    result = self.simulator.simulate_perturbation(
                        genes_to_activate=list(activate_idx),
                        genes_to_repress=list(repress_idx),
                        n_trajectories=self.config.validation_trajectories,
                    )
                    all_results.append(result)
                except Exception as exc:
                    logger.debug(
                        "Simulation failed for act=%s rep=%s: %s",
                        activate_idx,
                        repress_idx,
                        exc,
                    )

        # Sort by ascending mean cancer score (best first)
        all_results.sort(key=lambda r: r.mean_cancer_score)
        logger.info(
            "Combinatorial search complete: %d results.  "
            "Best score=%.4f (rev_frac=%.3f).",
            len(all_results),
            all_results[0].mean_cancer_score if all_results else float("nan"),
            all_results[0].reversion_fraction if all_results else 0.0,
        )
        return all_results

    # ------------------------------------------------------------------
    # Combination generation
    # ------------------------------------------------------------------

    def _generate_combinations(
        self, k: int
    ) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        """Generate all k-gene perturbation combinations.

        For each k-subset of candidate genes, all 2^k assignments of
        perturbation type (activate / repress) are returned.  Each element
        of the returned list is a tuple:

            ``(activate_indices, repress_indices)``

        where both elements are tuples of gene indices.

        Parameters
        ----------
        k : int
            Combination size.

        Returns
        -------
        List[Tuple[Tuple[int, ...], Tuple[int, ...]]]
        """
        result: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []

        # Choose k genes from candidate pool
        for gene_combo in itertools.combinations(self._candidates, k):
            # All 2^k assignments of activate (0) / repress (1) per gene
            for assignment in itertools.product((0, 1), repeat=k):
                activate = tuple(
                    gene_combo[i] for i in range(k) if assignment[i] == 0
                )
                repress = tuple(
                    gene_combo[i] for i in range(k) if assignment[i] == 1
                )
                result.append((activate, repress))

        return result

    # ------------------------------------------------------------------
    # Convenience: top results with reversion above threshold
    # ------------------------------------------------------------------

    def top_reverting(
        self,
        results: List[PerturbationResult],
        reversion_threshold: float = 0.5,
        top_n: int = 10,
    ) -> List[PerturbationResult]:
        """Filter and return top reverting results.

        Parameters
        ----------
        results : List[PerturbationResult]
            Full result list (typically from ``search``).
        reversion_threshold : float
            Minimum reversion fraction required.
        top_n : int
            Maximum number of results to return.

        Returns
        -------
        List[PerturbationResult]
        """
        filtered = [
            r for r in results if r.reversion_fraction >= reversion_threshold
        ]
        return filtered[:top_n]

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"CombinatorialSearcher("
            f"n_candidates={len(self._candidates)}, "
            f"max_perturbations={self.config.max_perturbations})"
        )
