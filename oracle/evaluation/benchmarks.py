from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class AttractorRecoveryBenchmark:
    """Benchmark: does the attractor finding recover known cancer attractor states?"""

    def __init__(self, known_attractors: Dict[str, np.ndarray]):
        self.known_attractors = known_attractors

    def evaluate(
        self,
        predicted_attractors: List[np.ndarray],
        attractor_labels: List[str],
    ) -> Dict[str, float]:
        results = {}
        for name, known in self.known_attractors.items():
            best_dist = float("inf")
            for pred in predicted_attractors:
                d = float(np.linalg.norm(known - pred[:len(known)]))
                best_dist = min(best_dist, d)
            results[f"attractor_{name}_l2_dist"] = best_dist

        results["mean_recovery_l2"] = float(np.mean(list(results.values())))
        logger.info(f"Attractor recovery: {results}")
        return results


class PerturbationEfficacyBenchmark:
    """Benchmark: does the predicted switch set actually revert cancer score?"""

    def __init__(self, cancer_score_fn, baseline_threshold: float = 0.5):
        self.cancer_score_fn = cancer_score_fn
        self.baseline_threshold = baseline_threshold

    def evaluate(
        self,
        switch_set,
        perturbed_expressions: np.ndarray,
    ) -> Dict[str, float]:
        import torch
        x = torch.tensor(perturbed_expressions, dtype=torch.float32)
        with torch.no_grad():
            scores = self.cancer_score_fn(x).cpu().numpy()

        results = {
            "mean_cancer_score_after": float(scores.mean()),
            "fraction_below_threshold": float((scores < self.baseline_threshold).mean()),
            "predicted_reversion_probability": switch_set.predicted_reversion_probability,
            "validated_reversion_fraction": switch_set.validated_reversion_fraction,
        }
        logger.info(f"Perturbation efficacy: {results}")
        return results


class MolecularPropertyBenchmark:
    """Benchmark: do designed TCIP molecules satisfy druglikeness criteria?"""

    def evaluate(self, tcip_molecules: list) -> Dict[str, float]:
        n = len(tcip_molecules)
        if n == 0:
            return {"n_molecules": 0}

        mw_list = [m.molecular_weight for m in tcip_molecules]
        qed_list = [m.qed_score for m in tcip_molecules if hasattr(m, "qed_score")]
        passes_ro5 = [m.passes_ro5 for m in tcip_molecules]
        valid_smiles = [m for m in tcip_molecules if m.smiles and m.smiles != ""]

        results = {
            "n_molecules": n,
            "n_valid_smiles": len(valid_smiles),
            "mean_mw": float(np.mean(mw_list)),
            "mw_std": float(np.std(mw_list)),
            "fraction_passes_ro5": float(np.mean(passes_ro5)),
        }
        if qed_list:
            results["mean_qed"] = float(np.mean(qed_list))

        logger.info(f"Molecular property benchmark: {results}")
        return results


class PipelineEndToEndBenchmark:
    """Runs all benchmarks in sequence on a complete pipeline output."""

    def run_all(
        self,
        oracle_output,
        known_attractors: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Dict]:
        results = {}

        mol_bench = MolecularPropertyBenchmark()
        if hasattr(oracle_output, "tcd_output") and oracle_output.tcd_output:
            results["molecular"] = mol_bench.evaluate(oracle_output.tcd_output.tcip_molecules)

        if hasattr(oracle_output, "rsp_output") and oracle_output.rsp_output:
            rsp = oracle_output.rsp_output
            results["rsp_summary"] = {
                "n_genes_to_activate": len(rsp.switch_set.genes_to_activate),
                "n_genes_to_repress": len(rsp.switch_set.genes_to_repress),
                "validated_reversion_fraction": rsp.switch_set.validated_reversion_fraction,
            }

        if known_attractors and hasattr(oracle_output, "cam_output") and oracle_output.cam_output:
            cam = oracle_output.cam_output
            att_bench = AttractorRecoveryBenchmark(known_attractors)
            results["attractor_recovery"] = att_bench.evaluate(
                cam.all_attractors, cam.attractor_labels
            )

        return results
