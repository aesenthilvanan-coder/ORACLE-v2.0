"""TCD module evaluator: molecule quality, ternary complex validity, drug-likeness."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

import numpy as np

from oracle.evaluation.metrics import molecule_validity, molecule_novelty, molecule_diversity

logger = logging.getLogger(__name__)


class TCDEvaluator:
    """Evaluates TCD module outputs: generated TCIP molecules and ternary complexes.

    Metrics:
    - Molecular validity (valid SMILES fraction)
    - Novelty (fraction not in training set)
    - Diversity (mean pairwise Tanimoto dissimilarity)
    - Drug-likeness: QED, SA score, Lipinski compliance
    - Ternary complex validity: clash score, interface energy, writer positioning
    """

    def __init__(self, training_smiles: Optional[Set[str]] = None) -> None:
        self.training_smiles = training_smiles or set()
        self.results: Dict[str, Any] = {}

    def evaluate_molecules(
        self,
        smiles_list: List[str],
        compute_properties: bool = True,
    ) -> Dict[str, Any]:
        """Evaluate a batch of generated SMILES strings.

        Returns validity, novelty, diversity and optional property distributions.
        """
        validity = molecule_validity(smiles_list)
        novelty = molecule_novelty(smiles_list, self.training_smiles)
        diversity = molecule_diversity(smiles_list)

        result: Dict[str, Any] = {
            "validity": validity,
            "novelty": novelty,
            "diversity": diversity,
            "n_generated": len(smiles_list),
        }

        if compute_properties:
            props = self._compute_property_distributions(smiles_list)
            result.update(props)

        self.results["molecules"] = result
        logger.info(
            "Molecule eval: validity=%.3f, novelty=%.3f, diversity=%.3f",
            validity, novelty, diversity
        )
        return result

    def evaluate_ternary_complexes(
        self,
        validation_results: List[Any],
    ) -> Dict[str, float]:
        """Evaluate a batch of ValidationResult objects from TernaryComplexValidator."""
        if not validation_results:
            return {}

        passed = [v for v in validation_results if getattr(v, "passed", False)]
        pass_rate = len(passed) / max(len(validation_results), 1)

        clash_scores = [getattr(v, "clash_score", float("nan")) for v in validation_results]
        interface_energies = [getattr(v, "interface_energy", float("nan")) for v in validation_results]
        sa_scores = [getattr(v, "sa_score", float("nan")) for v in validation_results]

        result = {
            "pass_rate": pass_rate,
            "mean_clash_score": float(np.nanmean(clash_scores)),
            "mean_interface_energy": float(np.nanmean(interface_energies)),
            "mean_sa_score": float(np.nanmean(sa_scores)),
            "n_total": len(validation_results),
            "n_passed": len(passed),
        }
        self.results["ternary"] = result
        logger.info(
            "Ternary eval: pass_rate=%.3f (%d/%d)",
            pass_rate, len(passed), len(validation_results)
        )
        return result

    def evaluate_binding_affinity(
        self,
        predicted_ki: np.ndarray,
        experimental_ki: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Evaluate predicted binding affinity (pKi) predictions."""
        result: Dict[str, float] = {
            "mean_predicted_ki": float(np.mean(predicted_ki)),
            "fraction_below_100nM": float((predicted_ki < 100e-9).mean()) if predicted_ki.mean() < 1 else float((predicted_ki < 0.1).mean()),
        }

        if experimental_ki is not None and len(experimental_ki) == len(predicted_ki):
            # Pearson correlation
            corr = float(np.corrcoef(np.log(predicted_ki + 1e-15), np.log(experimental_ki + 1e-15))[0, 1])
            result["pearson_r"] = corr
            # RMSE in log space
            rmse = float(np.sqrt(np.mean((np.log10(predicted_ki + 1e-15) - np.log10(experimental_ki + 1e-15)) ** 2)))
            result["log_rmse"] = rmse

        self.results["binding"] = result
        return result

    def evaluate_drug_likeness(self, smiles_list: List[str]) -> Dict[str, float]:
        """Compute mean QED, SA score, and Lipinski compliance."""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, QED

            qed_scores, sa_scores, lipinski = [], [], []
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                qed_scores.append(QED.qed(mol))
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                lipinski.append(
                    int(mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)
                )
                # Synthetic accessibility approximation via ring complexity
                n_rings = mol.GetRingInfo().NumRings()
                n_rotatable = Descriptors.NumRotatableBonds(mol)
                sa_approx = min(6.0, 1.0 + 0.3 * n_rings + 0.1 * n_rotatable)
                sa_scores.append(sa_approx)

            result = {
                "mean_qed": float(np.mean(qed_scores)) if qed_scores else float("nan"),
                "mean_sa_score": float(np.mean(sa_scores)) if sa_scores else float("nan"),
                "lipinski_compliance": float(np.mean(lipinski)) if lipinski else float("nan"),
                "n_evaluated": len(qed_scores),
            }
        except ImportError:
            result = {"error": "rdkit not available"}

        self.results["drug_likeness"] = result
        return result

    def _compute_property_distributions(self, smiles_list: List[str]) -> Dict[str, Any]:
        """Compute MW, LogP, HBD, HBA distributions for valid molecules."""
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors

            mw_list, logp_list = [], []
            for s in smiles_list:
                mol = Chem.MolFromSmiles(s)
                if mol:
                    mw_list.append(Descriptors.MolWt(mol))
                    logp_list.append(Descriptors.MolLogP(mol))

            return {
                "mean_mw": float(np.mean(mw_list)) if mw_list else float("nan"),
                "mean_logp": float(np.mean(logp_list)) if logp_list else float("nan"),
            }
        except ImportError:
            return {}

    def summary(self) -> Dict[str, Any]:
        return self.results.copy()
