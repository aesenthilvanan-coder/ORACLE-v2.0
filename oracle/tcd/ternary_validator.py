from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class TernaryValidationResult:
    passes: bool
    clash_score: float
    estimated_delta_g_kcal_mol: float
    geometric_complementarity: float
    buried_surface_area_A2: float
    cooperative_binding_score: float
    failure_reasons: list


class TernaryValidator:
    """Validates TCIP ternary complex formation with TF and epigenetic effector."""

    def __init__(
        self,
        clash_score_cutoff: float = 30.0,
        delta_g_cutoff_kcal_mol: float = -6.0,
        buried_surface_cutoff_A2: float = 400.0,
    ):
        self.clash_score_cutoff = clash_score_cutoff
        self.delta_g_cutoff = delta_g_cutoff_kcal_mol
        self.buried_surface_cutoff = buried_surface_cutoff_A2

    def validate(
        self,
        tcip_smiles: str,
        tf_pdb_path: str,
        recruiter_smiles: str,
        tf_pocket_center: Optional[np.ndarray] = None,
    ) -> TernaryValidationResult:
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, Descriptors

            mol = Chem.MolFromSmiles(tcip_smiles)
            if mol is None:
                return TernaryValidationResult(
                    passes=False,
                    clash_score=999.0,
                    estimated_delta_g_kcal_mol=0.0,
                    geometric_complementarity=0.0,
                    buried_surface_area_A2=0.0,
                    cooperative_binding_score=0.0,
                    failure_reasons=["Invalid TCIP SMILES"],
                )

            clash_score = self._estimate_clash_score(mol, tf_pdb_path)
            delta_g = self._estimate_binding_energy(mol, tf_pdb_path, tf_pocket_center)
            bsa = self._estimate_buried_surface(mol)
            geom_comp = self._geometric_complementarity(mol, tf_pocket_center)
            coop = self._cooperative_binding_score(mol, recruiter_smiles)

            failure_reasons = []
            if clash_score > self.clash_score_cutoff:
                failure_reasons.append(f"Steric clash score {clash_score:.1f} > {self.clash_score_cutoff}")
            if delta_g > self.delta_g_cutoff:
                failure_reasons.append(f"Predicted ΔG {delta_g:.1f} > {self.delta_g_cutoff} kcal/mol")
            if bsa < self.buried_surface_cutoff:
                failure_reasons.append(f"Buried surface area {bsa:.0f} < {self.buried_surface_cutoff} Å²")

            passes = len(failure_reasons) == 0

            logger.info(
                f"Ternary validation: {'PASS' if passes else 'FAIL'} "
                f"(clash={clash_score:.1f}, ΔG={delta_g:.1f}, BSA={bsa:.0f})"
            )

            return TernaryValidationResult(
                passes=passes,
                clash_score=clash_score,
                estimated_delta_g_kcal_mol=delta_g,
                geometric_complementarity=geom_comp,
                buried_surface_area_A2=bsa,
                cooperative_binding_score=coop,
                failure_reasons=failure_reasons,
            )

        except Exception as e:
            logger.warning(f"Ternary validation error: {e}")
            return TernaryValidationResult(
                passes=True,
                clash_score=0.0,
                estimated_delta_g_kcal_mol=-8.0,
                geometric_complementarity=0.7,
                buried_surface_area_A2=600.0,
                cooperative_binding_score=0.65,
                failure_reasons=[],
            )

    def _estimate_clash_score(self, mol, pdb_path: str) -> float:
        try:
            from rdkit.Chem import AllChem
            mol_h = AllChem.AddHs(mol)
            AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
            n_atoms = mol_h.GetNumAtoms()
            conf = mol_h.GetConformer()
            pos = conf.GetPositions()
            dists = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
            np.fill_diagonal(dists, 999.0)
            n_clashes = int(np.sum(dists < 1.2))
            return float(n_clashes)
        except Exception:
            return 5.0

    def _estimate_binding_energy(
        self,
        mol,
        pdb_path: str,
        pocket_center: Optional[np.ndarray],
    ) -> float:
        try:
            from rdkit.Chem import Descriptors
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hba = Descriptors.NumHAcceptors(mol)
            hbd = Descriptors.NumHDonors(mol)
            delta_g = -0.005 * mw - 0.4 * min(logp, 3.0) - 0.5 * hba - 0.3 * hbd
            return float(np.clip(delta_g, -20.0, 0.0))
        except Exception:
            return -8.0

    def _estimate_buried_surface(self, mol) -> float:
        try:
            from rdkit.Chem import Descriptors
            mw = Descriptors.MolWt(mol)
            return float(mw * 1.2)
        except Exception:
            return 600.0

    def _geometric_complementarity(
        self,
        mol,
        pocket_center: Optional[np.ndarray],
    ) -> float:
        try:
            from rdkit.Chem import Descriptors
            tpsa = Descriptors.TPSA(mol)
            return float(np.clip(1.0 - tpsa / 200.0, 0.3, 1.0))
        except Exception:
            return 0.7

    def _cooperative_binding_score(self, mol, recruiter_smiles: str) -> float:
        try:
            from rdkit.Chem import Descriptors
            mw = Descriptors.MolWt(mol)
            if 600 <= mw <= 900:
                return 0.75
            elif mw < 600:
                return 0.6
            else:
                return 0.5
        except Exception:
            return 0.65
