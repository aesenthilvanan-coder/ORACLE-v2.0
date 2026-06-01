"""
tcip_scorer.py
--------------
Scores and ranks the pool of generated TCIP molecule candidates.

TCIPScorer combines four signal sources into a single composite score:
  * QED (Quantitative Estimate of Drug-likeness) — captures overall
    drug-like character using a smooth desirability function.
  * SA score (synthetic accessibility) — penalises molecules that are
    difficult to synthesize in the laboratory.
  * Predicted TF binding affinity — the warhead's estimated Ki against
    the target TF pocket.
  * Ternary complex validation score — captures geometry, clash, and
    writer-positioning quality from TernaryComplexValidator.

The module also defines the TCIPMolecule and TCDOutput dataclasses that
carry the full pipeline output downstream to the patient report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TCIPMolecule:
    """
    A fully characterized TCIP molecule candidate.

    Attributes
    ----------
    target_tf : str
        HGNC symbol of the targeted transcription factor.
    perturbation_type : str
        "activation" or "repression".
    writer_eraser : str
        Name of the recruited epigenetic regulator.
    full_smiles : str
        SMILES of the complete TCIP (warhead + linker + recruiter).
    tf_warhead_smiles : str
        SMILES of the TF-binding warhead fragment.
    linker_smiles : str
        SMILES of the linker fragment.
    recruiter_smiles : str
        SMILES of the recruiter warhead fragment.
    molecular_weight : float
        Molecular weight in Da.
    logP : float
        Calculated LogP (lipophilicity).
    tpsa : float
        Topological polar surface area in Å².
    sa_score : float
        Synthetic accessibility score (1–10).
    qed : float
        QED drug-likeness score (0–1).
    predicted_tf_binding_affinity : float
        Predicted Ki in nM for TF binding.
    predicted_writer_binding_affinity : float
        Predicted Ki in nM for writer/eraser binding.
    ternary_complex_score : float
        Composite ternary complex validation score (0–1).
    validation_result : Any
        Full ValidationResult object.
    ternary_complex_structure : Any
        Assembled ternary complex dict.
    mol_image : Any
        2-D depiction image as np.ndarray (H, W, 3) or None.
    """

    target_tf: str
    perturbation_type: str
    writer_eraser: str
    full_smiles: str
    tf_warhead_smiles: str
    linker_smiles: str
    recruiter_smiles: str
    molecular_weight: float
    logP: float
    tpsa: float
    sa_score: float
    qed: float
    predicted_tf_binding_affinity: float
    predicted_writer_binding_affinity: float
    ternary_complex_score: float
    validation_result: Any
    ternary_complex_structure: Any
    mol_image: Any  # np.ndarray


@dataclass
class TCIPDetail:
    """
    Per-TCIP detailed scoring breakdown, used in TCDOutput.per_tcip.

    Attributes
    ----------
    tcip : TCIPMolecule
        The scored TCIP molecule.
    composite_score : float
        Weighted composite score.
    qed_contribution : float
        QED component (weight 0.2).
    sa_contribution : float
        Inverted SA score component (weight 0.2).
    affinity_contribution : float
        TF binding affinity component (weight 0.3).
    ternary_contribution : float
        Ternary validation component (weight 0.3).
    rank : int
        Rank among all candidates (1 = best).
    """

    tcip: TCIPMolecule
    composite_score: float
    qed_contribution: float
    sa_contribution: float
    affinity_contribution: float
    ternary_contribution: float
    rank: int = 0


@dataclass
class TCDOutput:
    """
    Full output of the Transcriptional CIP Designer pipeline.

    Attributes
    ----------
    tcip_molecules : List[TCIPMolecule]
        Ranked list of TCIP candidates.
    per_tcip : List[TCIPDetail]
        Detailed scoring breakdown per molecule.
    rsp_output : Any
        Output from the Reversion Switch Predictor (Module 1).
    cam_output : Any
        Output from the Cancer Attractor Mapper (Module 2).
    n_molecules : int
        Total number of candidates generated.
    cancer_type : str
        Cancer type being targeted.
    patient_id : str
        Patient / sample identifier.
    predicted_reversion_probability : float
        Estimated probability of phenotypic reversion given the best TCIP.
    """

    tcip_molecules: List[TCIPMolecule]
    per_tcip: List[TCIPDetail]
    rsp_output: Any
    cam_output: Any
    n_molecules: int
    cancer_type: str
    patient_id: str
    predicted_reversion_probability: float


# ---------------------------------------------------------------------------
# Scorer class
# ---------------------------------------------------------------------------


class TCIPScorer:
    """
    Scores and ranks TCIP molecule candidates.

    Scoring formula
    ---------------
    score = 0.20 * QED
          + 0.20 * (1 - SA_norm)       # inverted SA (higher = easier)
          + 0.30 * affinity_norm        # normalized from predicted Ki
          + 0.30 * ternary_score        # from ValidationResult

    where:
      SA_norm      = (sa_score - 1) / 9  in [0, 1]
      affinity_norm = exp(-Ki_nM / 1000)  (saturates at high affinity)

    Parameters
    ----------
    config : TCDConfig
        Pipeline configuration.
    """

    # Weights for the four scoring components
    W_QED: float = 0.20
    W_SA: float = 0.20
    W_AFFINITY: float = 0.30
    W_TERNARY: float = 0.30

    def __init__(self, config: Any) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, tcip_molecules: List[Any]) -> List[float]:
        """
        Score a list of TCIPMolecule objects.

        Parameters
        ----------
        tcip_molecules : List[TCIPMolecule]
            TCIP candidates to score.

        Returns
        -------
        List[float]
            Composite score for each molecule (same order as input).
        """
        return [self._score_single(tcip) for tcip in tcip_molecules]

    def rank(self, tcip_molecules: List[Any]) -> List[Any]:
        """
        Rank TCIP molecules by composite score (descending).

        Parameters
        ----------
        tcip_molecules : List[TCIPMolecule]
            Unranked candidates.

        Returns
        -------
        List[TCIPMolecule]
            Sorted list (best first).
        """
        scores = self.score(tcip_molecules)
        indexed = sorted(
            enumerate(tcip_molecules),
            key=lambda t: scores[t[0]],
            reverse=True,
        )
        ranked = [mol for _, mol in indexed]
        logger.info(
            "TCIPScorer: ranked %d molecules. Top score=%.4f.",
            len(ranked),
            scores[indexed[0][0]] if scores else 0.0,
        )
        return ranked

    def build_details(
        self, tcip_molecules: List[TCIPMolecule]
    ) -> List[TCIPDetail]:
        """
        Build TCIPDetail objects for all molecules, including rank.

        Parameters
        ----------
        tcip_molecules : List[TCIPMolecule]
            Already-ranked list (best first).

        Returns
        -------
        List[TCIPDetail]
        """
        details: List[TCIPDetail] = []
        for rank_idx, tcip in enumerate(tcip_molecules, start=1):
            qed_c, sa_c, aff_c, tern_c, composite = self._breakdown(tcip)
            details.append(
                TCIPDetail(
                    tcip=tcip,
                    composite_score=composite,
                    qed_contribution=qed_c * self.W_QED,
                    sa_contribution=sa_c * self.W_SA,
                    affinity_contribution=aff_c * self.W_AFFINITY,
                    ternary_contribution=tern_c * self.W_TERNARY,
                    rank=rank_idx,
                )
            )
        return details

    # ------------------------------------------------------------------
    # Scoring internals
    # ------------------------------------------------------------------

    def _score_single(self, tcip: Any) -> float:
        """
        Compute composite score for a single TCIPMolecule.

        Parameters
        ----------
        tcip : TCIPMolecule
            The molecule to score.

        Returns
        -------
        float in [0, 1]
        """
        _, _, _, _, composite = self._breakdown(tcip)
        return composite

    def _breakdown(
        self, tcip: Any
    ) -> tuple:
        """
        Compute individual score components and composite.

        Returns
        -------
        (qed_c, sa_c, aff_c, ternary_c, composite)
        All values in [0, 1].
        """
        import math

        # QED component
        qed = self._get_qed(tcip)
        qed_c = float(np.clip(qed, 0.0, 1.0))

        # SA score component (inverted: 1.0 = easiest, 0.0 = hardest)
        sa = self._get_sa_score(tcip)
        sa_c = float(np.clip((10.0 - sa) / 9.0, 0.0, 1.0))

        # Affinity component: exp(-Ki_nM / 1000)
        ki = self._get_tf_affinity(tcip)
        aff_c = float(math.exp(-max(0.0, ki) / 1000.0))

        # Ternary complex score
        ternary_c = float(np.clip(self._get_ternary_score(tcip), 0.0, 1.0))

        composite = (
            self.W_QED * qed_c
            + self.W_SA * sa_c
            + self.W_AFFINITY * aff_c
            + self.W_TERNARY * ternary_c
        )
        composite = float(np.clip(composite, 0.0, 1.0))
        return qed_c, sa_c, aff_c, ternary_c, composite

    # ------------------------------------------------------------------
    # Property accessors (handle both TCIPMolecule and plain dicts)
    # ------------------------------------------------------------------

    def _get_qed(self, tcip: Any) -> float:
        """Retrieve or compute QED for the molecule."""
        if hasattr(tcip, "qed") and tcip.qed > 0.0:
            return tcip.qed
        smiles = self._get_smiles(tcip)
        return self._compute_qed(smiles)

    def _get_sa_score(self, tcip: Any) -> float:
        """Retrieve or compute SA score."""
        if hasattr(tcip, "sa_score") and tcip.sa_score > 0.0:
            return tcip.sa_score
        return 5.0  # moderate default

    def _get_tf_affinity(self, tcip: Any) -> float:
        """Retrieve predicted TF binding affinity (Ki in nM)."""
        if hasattr(tcip, "predicted_tf_binding_affinity"):
            return float(tcip.predicted_tf_binding_affinity)
        if isinstance(tcip, dict):
            return float(tcip.get("predicted_tf_binding_affinity", 1000.0))
        return 1000.0

    def _get_ternary_score(self, tcip: Any) -> float:
        """Retrieve ternary complex validation score."""
        if hasattr(tcip, "ternary_complex_score"):
            return float(tcip.ternary_complex_score)
        if hasattr(tcip, "validation_result"):
            vr = tcip.validation_result
            if vr is not None and hasattr(vr, "validation_score"):
                return float(vr.validation_score)
        if isinstance(tcip, dict):
            return float(tcip.get("ternary_complex_score", 0.5))
        return 0.5

    def _get_smiles(self, tcip: Any) -> str:
        """Extract SMILES from a TCIPMolecule or dict."""
        if hasattr(tcip, "full_smiles") and tcip.full_smiles:
            return tcip.full_smiles
        if hasattr(tcip, "smiles") and tcip.smiles:
            return tcip.smiles
        if isinstance(tcip, dict):
            return tcip.get("full_smiles") or tcip.get("smiles") or "C"
        return "C"

    # ------------------------------------------------------------------
    # QED computation
    # ------------------------------------------------------------------

    def _compute_qed(self, smiles: str) -> float:
        """
        Compute QED score using RDKit.

        Falls back to a heuristic Lipinski-based estimate if RDKit is
        unavailable.

        Parameters
        ----------
        smiles : str
            SMILES string.

        Returns
        -------
        float in [0, 1]
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import QED

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.3
            return float(QED.qed(mol))

        except ImportError:
            pass
        except Exception:
            return 0.3

        # Heuristic fallback using property estimates
        return self._heuristic_qed(smiles)

    def _heuristic_qed(self, smiles: str) -> float:
        """
        Heuristic QED estimate when RDKit is unavailable.

        Uses SMILES-level proxies:
          * Length as proxy for MW
          * Count of O/N atoms as proxy for H-bond count
          * Presence of rings

        Returns a value in [0.1, 0.9].
        """
        smi = smiles or "C"
        mw_proxy = len(smi) * 3.0  # rough MW estimate
        n_heteroatoms = smi.count("O") + smi.count("N") + smi.count("S")
        n_rings = smi.count("1") + smi.count("2") + smi.count("3")

        # Penalise very large / very small molecules
        mw_score = max(0.0, 1.0 - abs(mw_proxy - 450.0) / 450.0)

        # Penalise excessive polarity
        het_score = max(0.0, 1.0 - max(0, n_heteroatoms - 5) / 10.0)

        # Reward rings (drug-like)
        ring_score = min(1.0, n_rings / 4.0)

        qed_est = (mw_score + het_score + ring_score) / 3.0
        return float(np.clip(qed_est, 0.1, 0.9))

    # ------------------------------------------------------------------
    # Utility: compute all properties for a raw SMILES TCIP
    # ------------------------------------------------------------------

    def compute_properties(self, smiles: str) -> dict:
        """
        Compute all relevant physicochemical properties for a SMILES.

        Returns a dict with keys: mw, logP, tpsa, qed, sa_score, hbd,
        hba, n_rotatable_bonds.
        """
        props = {
            "mw": 0.0,
            "logP": 0.0,
            "tpsa": 0.0,
            "qed": 0.0,
            "sa_score": 5.0,
            "hbd": 0,
            "hba": 0,
            "n_rotatable_bonds": 0,
        }
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors, QED
            from rdkit.Chem import RDConfig
            import sys, os

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return props

            props["mw"] = float(Descriptors.MolWt(mol))
            props["logP"] = float(Descriptors.MolLogP(mol))
            props["tpsa"] = float(Descriptors.TPSA(mol))
            props["qed"] = float(QED.qed(mol))
            props["hbd"] = int(rdMolDescriptors.CalcNumHBD(mol))
            props["hba"] = int(rdMolDescriptors.CalcNumHBA(mol))
            props["n_rotatable_bonds"] = int(
                rdMolDescriptors.CalcNumRotatableBonds(mol)
            )

            # SA score via sascorer contrib
            try:
                sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
                if sa_path not in sys.path:
                    sys.path.append(sa_path)
                import sascorer
                props["sa_score"] = float(sascorer.calculateScore(mol))
            except Exception:
                n_atoms = mol.GetNumHeavyAtoms()
                n_rings = mol.GetRingInfo().NumRings()
                props["sa_score"] = float(
                    min(10.0, max(1.0, 2.0 + 0.05 * n_atoms + 0.3 * n_rings))
                )

        except ImportError:
            pass
        except Exception as exc:
            logger.debug("compute_properties failed for %s: %s", smiles[:30], exc)

        return props

    def render_molecule_image(
        self, smiles: str, width: int = 300, height: int = 300
    ) -> Optional[np.ndarray]:
        """
        Render a 2-D depiction of *smiles* as an RGB numpy array.

        Parameters
        ----------
        smiles : str
            Molecule SMILES.
        width, height : int
            Image dimensions in pixels.

        Returns
        -------
        np.ndarray of shape (height, width, 3) or None if RDKit unavailable.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
            from rdkit.Chem.Draw import rdMolDraw2D
            from PIL import Image
            import io

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()

            try:
                import cairosvg
                png_bytes = cairosvg.svg2png(bytestring=svg.encode(), output_width=width, output_height=height)
                img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
                return np.array(img)
            except ImportError:
                pass

            # Fallback: use RDKit PNG renderer
            img = Draw.MolToImage(mol, size=(width, height))
            return np.array(img.convert("RGB"))

        except (ImportError, Exception):
            return None
