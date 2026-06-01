import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
import logging

from oracle.interfaces import SwitchSet
from oracle.models.switch_predictor_gnn import SwitchPredictorGNN, build_grn_graph_data
from oracle.rsp.perturbation_sim import PerturbationSimulator

logger = logging.getLogger(__name__)


@dataclass
class PartialSolution:
    activate: List[int]
    repress: List[int]
    predicted_score: float
    predicted_reversion: float


class MinimalSwitchOptimizer:
    """Finds minimal TF perturbation set (≤5) for maximum cancer score reduction.

    Algorithm:
    1. Gradient-based initialization: rank candidates by ∂score/∂x
    2. GNN-guided beam search: maintain top-K partial solutions
    3. ODE validation: verify with full trajectory simulation
    """

    def __init__(
        self,
        gnn: SwitchPredictorGNN,
        simulator: PerturbationSimulator,
        grn,
        genes: List[str],
        cancer_attractor: torch.Tensor,
        normal_attractor: torch.Tensor,
        druggable_genes: Optional[Set[int]] = None,
        max_perturbations: int = 5,
        beam_width: int = 10,
        target_cancer_score: float = 0.25,
        validation_trajectories: int = 100,
        device: torch.device = torch.device("cpu"),
    ):
        self.gnn = gnn.to(device) if gnn is not None else None
        if self.gnn is not None:
            self.gnn.eval()
        self.simulator = simulator
        self.grn = grn
        self.genes = genes
        self.n_genes = len(genes)
        self.cancer_attractor = cancer_attractor.to(device)
        self.normal_attractor = normal_attractor.to(device)
        self.druggable = druggable_genes or set(range(self.n_genes))
        self.max_perturbations = max_perturbations
        self.beam_width = beam_width
        self.target_cancer_score = target_cancer_score
        self.validation_trajectories = validation_trajectories
        self.device = device

        from oracle.cam.grn_inference import load_human_tfs
        self.tf_set = load_human_tfs()
        self.tf_indices = {
            i for i, g in enumerate(genes) if g in self.tf_set
        }

        self.candidates = [
            (i, t)
            for i in range(self.n_genes)
            if i in self.tf_indices and i in self.druggable
            for t in [0, 1]
        ]
        logger.info(f"Optimization candidates: {len(self.candidates)} (TF, type) pairs")

    def optimize(self, cancer_score_fn) -> SwitchSet:
        logger.info("Starting switch set optimization...")

        initial_ranking = self._gradient_based_ranking(cancer_score_fn)
        logger.info(f"Top gradient candidates: {[(self.genes[i], t) for i, t in initial_ranking[:10]]}")

        best_solution = self._beam_search(initial_ranking)

        logger.info("Validating best solution with ODE simulation...")
        validation = self.simulator.simulate_perturbation(
            genes_to_activate=best_solution.activate,
            genes_to_repress=best_solution.repress,
            n_trajectories=self.validation_trajectories,
        )

        logger.info(f"Validated reversion fraction: {validation.reversion_fraction:.1%}")
        logger.info(f"Validated mean cancer score: {validation.mean_cancer_score:.3f}")
        logger.info(f"Genes to activate: {[self.genes[i] for i in best_solution.activate]}")
        logger.info(f"Genes to repress: {[self.genes[i] for i in best_solution.repress]}")

        importance = self._compute_importance_scores(best_solution)

        return SwitchSet(
            genes_to_activate=[self.genes[i] for i in best_solution.activate],
            genes_to_repress=[self.genes[i] for i in best_solution.repress],
            predicted_reversion_probability=best_solution.predicted_reversion,
            validated_reversion_fraction=validation.reversion_fraction,
            predicted_cancer_score_after=validation.mean_cancer_score,
            gene_importance_scores=importance,
            perturbation_types={
                **{self.genes[i]: "activate" for i in best_solution.activate},
                **{self.genes[i]: "repress" for i in best_solution.repress},
            },
        )

    def _gradient_based_ranking(self, cancer_score_fn) -> List[Tuple[int, int]]:
        grad = cancer_score_fn.gradient_wrt_input(self.cancer_attractor)
        scores = []
        for i, t in self.candidates:
            g = grad[i].item()
            expected_decrease = -g if t == 0 else g
            scores.append((expected_decrease, i, t))
        scores.sort(reverse=True)
        return [(i, t) for _, i, t in scores]

    def _beam_search(self, initial_ranking: List[Tuple[int, int]]) -> PartialSolution:
        beam: List[PartialSolution] = []
        for i, t in initial_ranking[:self.beam_width * 2]:
            score, rev = self._gnn_predict(
                activate=[i] if t == 0 else [],
                repress=[i] if t == 1 else [],
            )
            beam.append(PartialSolution(
                activate=[i] if t == 0 else [],
                repress=[i] if t == 1 else [],
                predicted_score=score,
                predicted_reversion=rev,
            ))

        beam.sort(key=lambda s: s.predicted_score)
        beam = beam[:self.beam_width]

        for depth in range(1, self.max_perturbations):
            candidates_this_depth = []
            for solution in beam:
                current_genes = set(solution.activate + solution.repress)
                for i, t in initial_ranking:
                    if i in current_genes:
                        continue
                    if len(solution.activate) + len(solution.repress) >= self.max_perturbations:
                        continue
                    new_activate = solution.activate + ([i] if t == 0 else [])
                    new_repress = solution.repress + ([i] if t == 1 else [])
                    score, rev = self._gnn_predict(new_activate, new_repress)
                    candidates_this_depth.append(PartialSolution(
                        activate=new_activate,
                        repress=new_repress,
                        predicted_score=score,
                        predicted_reversion=rev,
                    ))

            if not candidates_this_depth:
                break

            candidates_this_depth.sort(key=lambda s: s.predicted_score)
            beam = candidates_this_depth[:self.beam_width]

            best_score = beam[0].predicted_score
            logger.info(
                f"Beam search depth {depth+1}: best score = {best_score:.3f}, "
                f"reversion = {beam[0].predicted_reversion:.1%}"
            )
            if best_score < self.target_cancer_score:
                logger.info("Target cancer score reached. Stopping beam search.")
                break

        return beam[0]

    def _gnn_predict(self, activate: List[int], repress: List[int]) -> Tuple[float, float]:
        if self.gnn is None:
            # Heuristic fallback: estimate score reduction from perturbation direction
            x = self.cancer_attractor.clone()
            for i in activate:
                x[i] = 1.0
            for i in repress:
                x[i] = 0.0
            dist_to_normal = torch.norm(x - self.normal_attractor).item()
            dist_cancer_normal = torch.norm(self.cancer_attractor - self.normal_attractor).item() + 1e-8
            progress = 1.0 - dist_to_normal / dist_cancer_normal
            score = max(0.0, 1.0 - progress)
            rev = max(0.0, min(1.0, progress))
            return score, rev
        with torch.no_grad():
            data = build_grn_graph_data(
                self.grn, self.genes,
                self.cancer_attractor, self.normal_attractor,
                activate, repress,
            ).to(self.device)
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=self.device)
            output = self.gnn(data)
            score = output["cancer_score"][0].item()
            rev = output["reversion_prob"][0].item()
        return score, rev

    def _compute_importance_scores(self, solution: PartialSolution) -> Dict[str, float]:
        if self.gnn is None:
            # Gradient-based importance when GNN is unavailable
            diff = (self.cancer_attractor - self.normal_attractor).abs().cpu().numpy()
            diff_norm = diff / (diff.max() + 1e-8)
            scores = {self.genes[i]: float(diff_norm[i]) for i in range(self.n_genes)}
            for i in solution.activate + solution.repress:
                scores[self.genes[i]] = min(1.0, scores[self.genes[i]] * 2.0)
            return scores
        with torch.no_grad():
            data = build_grn_graph_data(
                self.grn, self.genes,
                self.cancer_attractor, self.normal_attractor,
                solution.activate, solution.repress,
            ).to(self.device)
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=self.device)
            output = self.gnn(data)
            importance_raw = output["gene_importance"].cpu().numpy()
        importance_abs = np.abs(importance_raw)
        importance_norm = importance_abs / (importance_abs.max() + 1e-8)
        return {self.genes[i]: float(importance_norm[i]) for i in range(self.n_genes)}
