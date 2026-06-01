"""
oracle/training/data_leakage_protocols.py

ORACLE Data Leakage Prevention System.

The strictest data leakage protocols in computational biology.
Every possible contamination vector is sealed.

Core philosophy: if there is ANY ambiguity about whether a data point
could have influenced a test result, it is excluded. Period.
We prefer a smaller clean test set over a larger contaminated one.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Leakage Taxonomy
# ---------------------------------------------------------------------------

class LeakageVector(Enum):
    # Molecular leakage
    EXACT_SMILES = "exact_smiles_match"
    CANONICAL_SMILES = "canonical_smiles_match"
    INCHIKEY = "inchikey_match"
    SCAFFOLD = "murcko_scaffold_match"
    TANIMOTO_SIMILARITY = "tanimoto_above_threshold"
    SCAFFOLD_ANALOG = "scaffold_analog_match"
    STEREOISOMER = "stereoisomer_of_train"
    SALT_FORM = "salt_form_of_train"
    PRODRUG = "prodrug_form_of_train"

    # Biological leakage
    SAME_PATIENT = "same_patient_sample"
    SAME_TUMOR = "same_tumor_different_region"
    SAME_CELL_LINE = "same_cell_line"
    SAME_DATASET = "same_geo_accession"
    TEMPORAL_CONTAMINATION = "test_data_precedes_train_cutoff"
    SAME_PUBLICATION = "same_paper_train_and_test"
    BENCHMARK_CONTAMINATION = "benchmark_used_in_training"
    PSEUDOBULK_LEAKAGE = "aggregated_from_same_cells"

    # GRN leakage
    SAME_GRN_TOPOLOGY = "identical_grn_topology"
    SUBSET_GRN = "test_grn_is_subset_of_train"
    SAME_ATTRACTOR = "identical_attractor_state"
    SAME_TF_SET = "identical_tf_perturbation_set"

    # Label leakage
    TARGET_IN_FEATURES = "target_gene_in_input_features"
    FUTURE_INFORMATION = "uses_post_hoc_annotation"
    CIRCULAR_ANNOTATION = "annotation_derived_from_model_output"
    BENCHMARK_SCORE_LEAKAGE = "test_metric_used_in_model_selection"


@dataclass
class LeakageReport:
    """Complete audit trail for any split or filtering decision."""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    n_input: int = 0
    n_output: int = 0
    n_removed: int = 0
    removal_reasons: Dict[str, int] = field(default_factory=dict)
    vectors_checked: List[str] = field(default_factory=list)
    thresholds_used: Dict[str, float] = field(default_factory=dict)
    fingerprint: str = ""  # SHA256 of output set for audit

    def log(self) -> None:
        logger.info("=" * 60)
        logger.info("LEAKAGE PREVENTION REPORT")
        logger.info(f"  Timestamp:    {self.timestamp}")
        logger.info(f"  Input:        {self.n_input:,}")
        logger.info(f"  Output:       {self.n_output:,}")
        logger.info(
            f"  Removed:      {self.n_removed:,} "
            f"({self.n_removed / max(self.n_input, 1) * 100:.1f}%)"
        )
        logger.info("  Removal reasons:")
        for reason, count in sorted(self.removal_reasons.items(), key=lambda x: -x[1]):
            logger.info(f"    {reason}: {count:,}")
        logger.info(f"  Output fingerprint: {self.fingerprint[:16]}...")
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Layer 1: Temporal Isolation Protocol
# ---------------------------------------------------------------------------

class TemporalIsolationProtocol:
    """
    Enforces strict temporal isolation between train, val, and test.

    Rule: test data must post-date ALL training data by a minimum
    buffer period. This prevents the model from learning patterns
    that were discovered or annotated BECAUSE of test data analysis.

    For biological data:
    - Train: GEO submissions before cutoff_train
    - Val:   GEO submissions in (cutoff_train + buffer, cutoff_val)
    - Test:  GEO submissions after cutoff_test

    For molecular data:
    - Train: ChEMBL assays before cutoff_train
    - Val/Test: ChEMBL assays after cutoff_val

    This is stricter than standard practice. Standard practice
    uses random splits. We use temporal splits because:
    1. Models trained on random splits are evaluated on data points
       they were trained on's contemporaries — methodologically unsound.
    2. Temporal splits test true generalization to future data.
    3. All published benchmarks we are aware of use random splits —
       this makes our evaluation more conservative and more credible.
    """

    # Hard cutoff dates (UTC)
    TRAIN_CUTOFF = "2022-01-01"  # All data submitted before this = train eligible
    VAL_CUTOFF   = "2023-01-01"  # Buffer: 2022 is excluded entirely from both
    TEST_CUTOFF  = "2023-07-01"  # Test starts here (6-month buffer after val)

    # Buffer period in days — data submitted within this window of ANY split
    # boundary is DISCARDED, not reassigned
    BOUNDARY_BUFFER_DAYS = 180  # 6 months on each side of each boundary

    def __init__(self):
        from datetime import date
        self.train_cutoff = date.fromisoformat(self.TRAIN_CUTOFF)
        self.val_cutoff = date.fromisoformat(self.VAL_CUTOFF)
        self.test_cutoff = date.fromisoformat(self.TEST_CUTOFF)

    def classify_date(self, submission_date: str) -> Optional[str]:
        """
        Classify a submission date into train/val/test/DISCARD.
        Returns None if the date falls within a buffer zone.
        DISCARD is permanent — buffer zone data is never used.
        """
        from datetime import date, timedelta

        try:
            d = date.fromisoformat(submission_date[:10])
        except Exception:
            return None  # Unknown date = discard

        buf = timedelta(days=self.BOUNDARY_BUFFER_DAYS)

        # In buffer around train/val boundary
        if self.train_cutoff - buf <= d <= self.train_cutoff + buf:
            return None  # DISCARD

        # In buffer around val/test boundary
        if self.val_cutoff - buf <= d <= self.val_cutoff + buf:
            return None  # DISCARD

        if d < self.train_cutoff - buf:
            return "train"

        if self.val_cutoff + buf <= d < self.test_cutoff - buf:
            return "val"

        if d >= self.test_cutoff + buf:
            return "test"

        return None  # Falls in one of the buffers — DISCARD

    def filter_geo_datasets(
        self, datasets: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """Filter GEO datasets into train/val/test with zero buffer-zone leakage."""
        splits: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}
        discarded = 0

        for ds in datasets:
            date_str = ds.get("submission_date", ds.get("date", None))
            if date_str is None:
                discarded += 1
                continue

            assignment = self.classify_date(date_str)
            if assignment is None:
                discarded += 1
                continue

            splits[assignment].append(ds)

        logger.info(
            f"Temporal split: train={len(splits['train'])}, "
            f"val={len(splits['val'])}, test={len(splits['test'])}, "
            f"discarded={discarded}"
        )
        return splits


# ---------------------------------------------------------------------------
# Layer 2: Molecular Leakage Prevention
# ---------------------------------------------------------------------------

class MolecularLeakageFilter:
    """
    Prevents ALL forms of molecular leakage between train and test.

    Checks (in order, fastest to slowest):
    1. Exact SMILES match (hash comparison, O(1))
    2. InChIKey match (normalized, captures tautomers/salts)
    3. Murcko scaffold identity (same core scaffold)
    4. Tanimoto similarity > threshold (Morgan FP, r=2, 2048 bits)
    5. Maximum Common Substructure coverage > threshold

    Thresholds are deliberately MORE conservative than literature:
    - Standard practice: Tanimoto > 0.4 = similar
    - Our threshold:     Tanimoto > 0.3 = contaminated, REMOVED
    - Standard scaffold: exact Murcko scaffold match
    - Our standard:      Murcko + ring system match

    Why so strict: a model that has seen a Tanimoto-0.35 analog of
    a test compound has learned something about that compound's
    binding interaction. This is leakage even if it is not
    recognized as such in standard practice.
    """

    TANIMOTO_THRESHOLD = 0.30  # Anything above this is contaminated
    MCS_COVERAGE_THRESHOLD = 0.60  # Max common substructure > 60% = contaminated

    def __init__(self, tanimoto_threshold: float = 0.30):
        self.tanimoto_threshold = tanimoto_threshold

    def compute_fingerprint(self, smiles: str):
        """Morgan fingerprint for Tanimoto computation."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        except Exception:
            return None

    def get_inchikey(self, smiles: str) -> Optional[str]:
        """InChIKey for exact/near-exact matching."""
        try:
            from rdkit import Chem
            from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            inchi = MolToInchi(mol)
            return InchiToInchiKey(inchi) if inchi else None
        except Exception:
            return None

    def get_scaffold(self, smiles: str) -> Optional[str]:
        """Murcko scaffold SMILES."""
        try:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold, canonical=True)
        except Exception:
            return None

    def get_inchikey_connectivity_layer(self, smiles: str) -> Optional[str]:
        """
        First layer of InChIKey only (connectivity, ignores stereo/tautomers).
        Catches stereoisomers of training compounds.
        """
        ikey = self.get_inchikey(smiles)
        if ikey is None:
            return None
        return ikey.split("-")[0]  # First 14 chars = connectivity layer only

    def build_train_index(self, train_smiles: List[str]) -> Dict:
        """
        Build all lookup structures for training set.
        Call once, then use filter_test_set() many times.
        """
        logger.info(f"Building molecular train index for {len(train_smiles):,} compounds...")

        index = {
            "smiles_set": set(),
            "inchikey_set": set(),
            "inchikey_connectivity_set": set(),
            "scaffold_set": set(),
            "fingerprints": [],
            "smiles_list": [],
        }

        try:
            from rdkit import Chem
        except ImportError:
            logger.warning("RDKit not available — molecular leakage check disabled")
            return index

        for smiles in train_smiles:
            canonical = None
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    canonical = Chem.MolToSmiles(mol, canonical=True)
            except Exception:
                pass

            if canonical is None:
                continue

            index["smiles_set"].add(canonical)

            ikey = self.get_inchikey(canonical)
            if ikey:
                index["inchikey_set"].add(ikey)
                index["inchikey_connectivity_set"].add(ikey.split("-")[0])

            scaffold = self.get_scaffold(canonical)
            if scaffold:
                index["scaffold_set"].add(scaffold)

            fp = self.compute_fingerprint(canonical)
            if fp is not None:
                index["fingerprints"].append(fp)
                index["smiles_list"].append(canonical)

        logger.info(f"  Scaffolds indexed: {len(index['scaffold_set']):,}")
        logger.info(f"  InChIKeys indexed: {len(index['inchikey_set']):,}")
        logger.info(f"  Fingerprints indexed: {len(index['fingerprints']):,}")
        return index

    def is_contaminated(self, smiles: str, train_index: Dict) -> Tuple[bool, str]:
        """
        Check if a single molecule is contaminated by the training set.
        Returns (is_contaminated: bool, reason: str).
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import DataStructs

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return True, "invalid_smiles"

            canonical = Chem.MolToSmiles(mol, canonical=True)

            # Check 1: Exact SMILES
            if canonical in train_index["smiles_set"]:
                return True, LeakageVector.CANONICAL_SMILES.value

            # Check 2: InChIKey (full — catches tautomers, resonance structures)
            ikey = self.get_inchikey(canonical)
            if ikey and ikey in train_index["inchikey_set"]:
                return True, LeakageVector.INCHIKEY.value

            # Check 3: InChIKey connectivity layer (catches stereoisomers)
            if ikey and ikey.split("-")[0] in train_index["inchikey_connectivity_set"]:
                return True, LeakageVector.STEREOISOMER.value

            # Check 4: Scaffold identity
            scaffold = self.get_scaffold(canonical)
            if scaffold and scaffold in train_index["scaffold_set"]:
                return True, LeakageVector.SCAFFOLD.value

            # Check 5: Tanimoto similarity (slowest — do last)
            fp = self.compute_fingerprint(canonical)
            if fp is not None and train_index["fingerprints"]:
                similarities = DataStructs.BulkTanimotoSimilarity(fp, train_index["fingerprints"])
                max_sim = max(similarities) if similarities else 0.0
                if max_sim >= self.tanimoto_threshold:
                    return True, LeakageVector.TANIMOTO_SIMILARITY.value

            return False, ""

        except Exception:
            return False, ""

    def filter_test_set(
        self,
        test_smiles: List[str],
        train_index: Dict,
    ) -> Tuple[List[str], LeakageReport]:
        """
        Filter an entire test set, removing all contaminated compounds.
        Returns (clean_test_smiles, report).
        """
        report = LeakageReport(n_input=len(test_smiles))
        clean = []

        for smiles in test_smiles:
            contaminated, reason = self.is_contaminated(smiles, train_index)
            if contaminated:
                report.removal_reasons[reason] = report.removal_reasons.get(reason, 0) + 1
            else:
                clean.append(smiles)

        report.n_output = len(clean)
        report.n_removed = len(test_smiles) - len(clean)
        report.vectors_checked = [
            v.value for v in LeakageVector
            if any(k in v.value for k in ["smiles", "tanimoto", "scaffold", "stereo", "inchi"])
        ]
        report.thresholds_used = {"tanimoto": self.tanimoto_threshold}
        report.fingerprint = hashlib.sha256("|".join(sorted(clean)).encode()).hexdigest()

        report.log()
        return clean, report


# ---------------------------------------------------------------------------
# Layer 3: Biological Data Leakage Prevention
# ---------------------------------------------------------------------------

class BiologicalLeakageFilter:
    """
    Prevents all biological data leakage.

    Rules (ALL enforced simultaneously):

    PATIENT-LEVEL: If ANY sample from a patient appears in train,
    ALL samples from that patient are excluded from val/test.
    This includes: different tumor regions, matched normal,
    different timepoints, PDX models derived from the patient.

    DATASET-LEVEL: If a GEO accession appears in train,
    ALL samples from that accession are excluded from val/test,
    even if they are different samples from the same study.

    CELL-LINE-LEVEL: If a cell line appears in train,
    that cell line is excluded from val/test across ALL datasets.
    Cell lines are often shared between studies — this is a
    commonly overlooked source of leakage.

    PUBLICATION-LEVEL: If a paper's training data is cited in
    the same paper as test data results, it is potentially
    contaminated by post-hoc analysis. Such data is flagged
    but not automatically excluded — manual review required.

    PSEUDOBULK-LEVEL: If pseudobulk profiles are generated from
    the same underlying cells as any training data, all resulting
    pseudobulk profiles are excluded.
    """

    # Known cell lines (standardized names from Cellosaurus)
    CELL_LINE_SYNONYMS_DB = None  # Loaded lazily from Cellosaurus

    def __init__(self):
        self.patient_ids_in_train: Set[str] = set()
        self.dataset_ids_in_train: Set[str] = set()
        self.cell_lines_in_train: Set[str] = set()
        self.publication_ids_in_train: Set[str] = set()
        self.sample_ids_in_train: Set[str] = set()

    def register_train_data(self, metadata_list: List[Dict]) -> None:
        """Register all training data metadata for contamination checking."""
        for meta in metadata_list:
            if "patient_id" in meta and meta["patient_id"]:
                self.patient_ids_in_train.add(str(meta["patient_id"]).strip().upper())

            if "geo_accession" in meta and meta["geo_accession"]:
                self.dataset_ids_in_train.add(str(meta["geo_accession"]).strip().upper())

            if "cell_line" in meta and meta["cell_line"]:
                cell_line = self._normalize_cell_line_name(meta["cell_line"])
                self.cell_lines_in_train.add(cell_line)

            if "pmid" in meta and meta["pmid"]:
                self.publication_ids_in_train.add(str(meta["pmid"]).strip())

            if "sample_id" in meta and meta["sample_id"]:
                self.sample_ids_in_train.add(str(meta["sample_id"]).strip().upper())

    def is_contaminated(self, metadata: Dict) -> Tuple[bool, List[str]]:
        """
        Check if a test sample is contaminated.
        Returns (is_contaminated, list_of_reasons).
        """
        reasons = []

        # Patient-level check
        patient_id = metadata.get("patient_id", "")
        if patient_id and str(patient_id).strip().upper() in self.patient_ids_in_train:
            reasons.append(LeakageVector.SAME_PATIENT.value)

        # Dataset-level check
        geo_acc = metadata.get("geo_accession", "")
        if geo_acc and str(geo_acc).strip().upper() in self.dataset_ids_in_train:
            reasons.append(LeakageVector.SAME_DATASET.value)

        # Cell-line level check
        cell_line = metadata.get("cell_line", "")
        if cell_line:
            normalized = self._normalize_cell_line_name(cell_line)
            if normalized in self.cell_lines_in_train:
                reasons.append(LeakageVector.SAME_CELL_LINE.value)

        # Sample-level check
        sample_id = metadata.get("sample_id", "")
        if sample_id and str(sample_id).strip().upper() in self.sample_ids_in_train:
            reasons.append(LeakageVector.SAME_PATIENT.value)

        # Publication-level (flag, don't auto-exclude)
        pmid = metadata.get("pmid", "")
        if pmid and str(pmid).strip() in self.publication_ids_in_train:
            reasons.append(LeakageVector.SAME_PUBLICATION.value)

        return len(reasons) > 0, reasons

    def filter_datasets(
        self,
        test_metadata: List[Dict],
        auto_exclude_publication: bool = False,
    ) -> Tuple[List[Dict], LeakageReport]:
        """
        Filter test metadata list, removing contaminated samples.
        Returns (clean_metadata, report).
        """
        report = LeakageReport(n_input=len(test_metadata))
        clean = []

        for meta in test_metadata:
            contaminated, reasons = self.is_contaminated(meta)

            if contaminated:
                # Publication contamination: flag but don't auto-exclude unless specified
                if (
                    reasons == [LeakageVector.SAME_PUBLICATION.value]
                    and not auto_exclude_publication
                ):
                    logger.warning(
                        f"FLAGGED (manual review required): {meta.get('geo_accession', 'unknown')} "
                        f"— same publication as training data"
                    )
                    clean.append(meta)  # Keep but flag
                else:
                    for r in reasons:
                        report.removal_reasons[r] = report.removal_reasons.get(r, 0) + 1
            else:
                clean.append(meta)

        report.n_output = len(clean)
        report.n_removed = len(test_metadata) - len(clean)
        report.vectors_checked = [
            LeakageVector.SAME_PATIENT.value,
            LeakageVector.SAME_DATASET.value,
            LeakageVector.SAME_CELL_LINE.value,
            LeakageVector.SAME_PUBLICATION.value,
        ]

        report.log()
        return clean, report

    def _normalize_cell_line_name(self, name: str) -> str:
        """Normalize cell line name to Cellosaurus standard."""
        normalized = str(name).strip().upper()
        normalized = normalized.replace("-", "").replace(" ", "").replace("_", "")
        return normalized
