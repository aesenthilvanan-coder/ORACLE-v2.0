#!/usr/bin/env python3
"""
Stage 0: Foundation Molecular Pretraining Dataset Builder.

Builds 10.2 billion augmented training examples from 14 molecular databases.
Output: sharded .pkl.gz files, 100k examples each (~100k shards total).

Augmentation pipeline per molecule:
  - 10x  SMILES randomization
  -  5x  ETKDG conformer generation
  -  8x  SE(3) random rotation per conformer
  -  3x  Fragment masking (BERT-style)
  Total: ~26-53x per molecule, capped at TARGET_EXAMPLES globally.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import multiprocessing
import os
import pickle
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("oracle.build_pretrain")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHARD_SIZE = 100_000
TARGET_EXAMPLES = 10_200_000_000  # 10.2 billion

DATASET_SOURCES: Dict[str, dict] = {
    "zinc20_leads": {
        "url": "https://zinc20.docking.org/tranches/all.smi.gz",
        "format": "smi_gz",
        "n_compounds_approx": 1_400_000_000,
        "priority": 1,
        "license": "free_academic",
    },
    "zinc20_instock": {
        "url": "https://zinc20.docking.org/tranches/in-stock.smi.gz",
        "format": "smi_gz",
        "n_compounds_approx": 230_000_000,
        "priority": 1,
    },
    "pubchem_bioactive": {
        "url": "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz",
        "format": "smi_gz",
        "n_compounds_approx": 100_000_000,
        "priority": 1,
    },
    "chembl_33": {
        "url": "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_33/chembl_33_chemreps.txt.gz",
        "format": "chembl_tsv_gz",
        "n_compounds_approx": 2_400_000,
        "priority": 1,
    },
    "enamine_real_sample": {
        "url": "https://enamine.net/compound-collections/real-compounds/real-space",
        "format": "smi_chunks",
        "n_compounds_approx": 500_000_000,
        "priority": 3,
        "license": "registration_required",
    },
    "gdb17_sample": {
        "url": "http://gdb.unibe.ch/downloads/",
        "format": "smi_gz",
        "n_compounds_approx": 2_000_000_000,
        "priority": 3,
    },
    "gdb11": {
        "url": "http://gdb.unibe.ch/downloads/gdb11.tar.gz",
        "format": "smi_gz",
        "n_compounds_approx": 26_000_000,
        "priority": 2,
    },
    "excape_db": {
        "url": "https://zenodo.org/records/173258/files/pubchem.chembl.dataset4publication_inchi_smiles.tsv.xz",
        "format": "tsv_xz",
        "n_compounds_approx": 70_000_000,
        "priority": 1,
        "has_bioactivity": True,
    },
    "bindingdb": {
        "url": "https://www.bindingdb.org/bind/downloads/BindingDB_All.tsv.zip",
        "format": "tsv_zip",
        "n_compounds_approx": 2_900_000,
        "priority": 1,
        "has_bioactivity": True,
    },
    "guacamol": {
        "url": "https://ndownloader.figshare.com/files/13612745",
        "format": "smi_gz",
        "n_compounds_approx": 1_600_000,
        "priority": 2,
    },
    "moses": {
        "url": "https://media.githubusercontent.com/media/molecularsets/moses/master/data/dataset_v1.csv",
        "format": "csv",
        "n_compounds_approx": 1_900_000,
        "priority": 2,
    },
    "coconut": {
        "url": "https://coconut.naturalproducts.net/download/smiles",
        "format": "smi",
        "n_compounds_approx": 400_000,
        "priority": 2,
    },
    "drugbank": {
        "url": "https://go.drugbank.com/releases/latest#structures",
        "format": "sdf",
        "n_compounds_approx": 14_000,
        "priority": 1,
        "license": "academic_free",
    },
    "protac_db": {
        "url": "http://cadd.zju.edu.cn/protacdb/",
        "format": "csv",
        "n_compounds_approx": 3_000,
        "priority": 1,
        "has_bifunctional": True,
    },
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MoleculeRecord:
    smiles: str
    source: str
    aug_type: str  # "canonical", "random_smiles", "conformer_se3", "masked"
    coords: Optional[np.ndarray] = None  # (N, 3) float32 if 3D available
    bioactivity: Optional[dict] = None   # {"target_id", "activity_nM", "activity_type"}
    masked_atom_indices: List[int] = field(default_factory=list)
    canonical_smiles: str = ""
    inchikey: str = ""


# ---------------------------------------------------------------------------
# Standardizer
# ---------------------------------------------------------------------------

class MoleculeStandardizer:
    """Standardize molecules: desalt, normalize, uncharge, canonical tautomer."""

    def __init__(
        self,
        min_mw: float = 50.0,
        max_mw: float = 1500.0,
        min_atoms: int = 5,
        max_atoms: int = 100,
    ):
        self.min_mw = min_mw
        self.max_mw = max_mw
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms

    def standardize(self, smiles: str) -> Optional[str]:
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            from rdkit.Chem.MolStandardize import rdMolStandardize

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Remove fragments (keep largest)
            mol = rdMolStandardize.FragmentParent(mol)

            # Normalize
            normalizer = rdMolStandardize.Normalizer()
            mol = normalizer.normalize(mol)

            # Uncharge
            uncharger = rdMolStandardize.Uncharger()
            mol = uncharger.uncharge(mol)

            # Canonical tautomer
            te = rdMolStandardize.TautomerEnumerator()
            mol = te.Canonicalize(mol)

            if mol is None:
                return None

            # Validate
            mw = Descriptors.MolWt(mol)
            n_atoms = mol.GetNumHeavyAtoms()

            if not (self.min_mw <= mw <= self.max_mw):
                return None
            if not (self.min_atoms <= n_atoms <= self.max_atoms):
                return None

            return Chem.MolToSmiles(mol, canonical=True)

        except Exception:
            return None

    def get_inchikey(self, smiles: str) -> str:
        try:
            from rdkit import Chem
            from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return ""
            inchi = MolToInchi(mol)
            return InchiToInchiKey(inchi) if inchi else ""
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# Augmenter
# ---------------------------------------------------------------------------

class SMILESAugmenter:
    """Generate augmented training examples from a canonical SMILES."""

    def __init__(
        self,
        n_smiles_aug: int = 10,
        n_conformers: int = 5,
        n_se3_aug: int = 8,
        random_seed: int = 42,
    ):
        self.n_smiles_aug = n_smiles_aug
        self.n_conformers = n_conformers
        self.n_se3_aug = n_se3_aug
        self.rng = np.random.default_rng(random_seed)

    def randomize_smiles(self, smiles: str) -> List[str]:
        """Generate n random SMILES variants via atom reordering."""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return []
            results = set()
            n_atoms = mol.GetNumAtoms()
            for _ in range(self.n_smiles_aug * 3):
                if len(results) >= self.n_smiles_aug:
                    break
                new_order = np.random.permutation(n_atoms).tolist()
                renumbered = Chem.RenumberAtoms(mol, new_order)
                s = Chem.MolToSmiles(renumbered, canonical=False)
                if s and s != smiles:
                    results.add(s)
            return list(results)[: self.n_smiles_aug]
        except Exception:
            return []

    def generate_conformers(self, smiles: str) -> List[np.ndarray]:
        """Generate ETKDG 3D conformers, return list of (N, 3) coordinate arrays."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return []
            mol = Chem.AddHs(mol)

            params = AllChem.ETKDGv3()
            params.randomSeed = int(self.rng.integers(0, 2**31))
            params.numThreads = 1

            cids = AllChem.EmbedMultipleConfs(mol, numConfs=self.n_conformers, params=params)
            if len(cids) == 0:
                return []

            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=1)
            mol = Chem.RemoveHs(mol)

            coords_list = []
            for cid in cids:
                conf = mol.GetConformer(cid)
                pos = np.array(conf.GetPositions(), dtype=np.float32)
                coords_list.append(pos)
            return coords_list
        except Exception:
            return []

    def se3_rotate(self, coords: np.ndarray) -> List[np.ndarray]:
        """Apply n_se3_aug random SE(3) rotations to a coordinate array."""
        results = []
        for _ in range(self.n_se3_aug):
            # Random rotation via QR decomposition
            q, _ = np.linalg.qr(self.rng.standard_normal((3, 3)))
            if np.linalg.det(q) < 0:
                q[:, 0] *= -1
            # Random translation (center + small jitter)
            center = coords.mean(axis=0)
            rotated = (coords - center) @ q.T
            rotated += self.rng.standard_normal(3) * 0.5
            results.append(rotated.astype(np.float32))
        return results

    def mask_fragments(
        self, smiles: str, n_masks: int = 3
    ) -> List[Tuple[str, List[int]]]:
        """Generate BERT-style fragment masks. Returns (smiles, masked_atom_indices) pairs."""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return []
            n_atoms = mol.GetNumHeavyAtoms()
            if n_atoms < 4:
                return []
            results = []
            for _ in range(n_masks):
                mask_frac = float(self.rng.uniform(0.1, 0.25))
                n_masked = max(1, int(n_atoms * mask_frac))
                indices = self.rng.choice(n_atoms, size=n_masked, replace=False).tolist()
                results.append((smiles, indices))
            return results
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

class PretrainingDatasetBuilder:
    """Build the 10.2B example Stage 0 pretraining dataset."""

    def __init__(
        self,
        output_dir: Path,
        n_workers: int = 8,
        max_examples: Optional[int] = None,
        generate_3d: bool = True,
        n_smiles_aug: int = 10,
        n_conformers: int = 5,
        n_se3_aug: int = 8,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_workers = n_workers
        self.max_examples = max_examples or TARGET_EXAMPLES
        self.generate_3d = generate_3d

        self.standardizer = MoleculeStandardizer()
        self.augmenter = SMILESAugmenter(
            n_smiles_aug=n_smiles_aug,
            n_conformers=n_conformers,
            n_se3_aug=n_se3_aug,
        )

        self._shard_buffer: List[dict] = []
        self._shard_idx: int = 0
        self._n_total: int = 0
        self._seen_inchikeys: set = set()

    def build(self) -> int:
        """Build full dataset. Returns total examples written."""
        logger.info("=" * 70)
        logger.info("ORACLE Stage 0: Building 10.2B Pretraining Dataset")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Target: {self.max_examples:,} examples")
        logger.info("=" * 70)

        # Priority-ordered sources
        ordered = sorted(
            DATASET_SOURCES.items(),
            key=lambda x: x[1].get("priority", 99),
        )

        for source_name, source_info in ordered:
            if self._n_total >= self.max_examples:
                break

            source_path = self.output_dir.parent / "raw" / "molecules" / source_name
            if not source_path.exists():
                # Try common extensions
                for ext in [".smi", ".smi.gz", ".tsv.gz", ".tsv.xz", ".csv"]:
                    candidate = source_path.parent / (source_name + ext)
                    if candidate.exists():
                        source_path = candidate
                        break
                else:
                    logger.warning(f"Source {source_name} not found at {source_path} — skipping")
                    continue

            remaining = self.max_examples - self._n_total
            aug_factor = self.augmenter.n_smiles_aug + self.augmenter.n_se3_aug + 3
            max_from_source = max(1, remaining // max(1, aug_factor))

            n = self._process_source(source_name, source_path, source_info, max_from_source)
            logger.info(f"  {source_name}: +{n:,} examples (total={self._n_total:,})")

        # Flush remaining partial shard
        if self._shard_buffer:
            self._flush_shard()

        # Write manifest
        manifest = {
            "n_examples": self._n_total,
            "n_shards": self._shard_idx,
            "shard_size": SHARD_SIZE,
            "target_examples": TARGET_EXAMPLES,
            "sources": list(DATASET_SOURCES.keys()),
        }
        with open(self.output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"\nDataset complete: {self._n_total:,} examples across {self._shard_idx} shards")
        return self._n_total

    def _process_source(
        self,
        source_name: str,
        source_path: Path,
        source_info: dict,
        max_from_source: int,
    ) -> int:
        n_added = 0
        for smiles, bioactivity in self._iter_source(source_path, source_info):
            if n_added >= max_from_source or self._n_total >= self.max_examples:
                break

            canonical = self.standardizer.standardize(smiles)
            if canonical is None:
                continue

            # Deduplication via InChIKey
            ikey = self.standardizer.get_inchikey(canonical)
            if ikey and ikey in self._seen_inchikeys:
                continue
            if ikey:
                self._seen_inchikeys.add(ikey)

            base_record = {
                "smiles": canonical,
                "source": source_name,
                "aug_type": "canonical",
                "bioactivity": bioactivity,
                "canonical_smiles": canonical,
                "inchikey": ikey,
            }
            self._shard_buffer.append(base_record)
            self._n_total += 1
            n_added += 1

            if len(self._shard_buffer) >= SHARD_SIZE:
                self._flush_shard()
                if self._n_total >= self.max_examples:
                    break

            # Generate augmentations
            augs = self._generate_augmentations(canonical, base_record)
            for aug in augs:
                if self._n_total >= self.max_examples:
                    break
                self._shard_buffer.append(aug)
                self._n_total += 1
                n_added += 1
                if len(self._shard_buffer) >= SHARD_SIZE:
                    self._flush_shard()

            # Fragment masks
            masks = self._generate_fragment_masks(canonical, base_record)
            for masked in masks:
                if self._n_total >= self.max_examples:
                    break
                self._shard_buffer.append(masked)
                self._n_total += 1
                n_added += 1
                if len(self._shard_buffer) >= SHARD_SIZE:
                    self._flush_shard()

        return n_added

    def _generate_augmentations(self, canonical_smiles: str, base_record: dict) -> List[dict]:
        records = []

        # SMILES randomization
        for rand_smi in self.augmenter.randomize_smiles(canonical_smiles):
            records.append({
                **base_record,
                "smiles": rand_smi,
                "aug_type": "random_smiles",
                "coords": None,
            })

        # 3D conformers + SE(3) rotations
        if self.generate_3d:
            for conf_coords in self.augmenter.generate_conformers(canonical_smiles):
                for rotated_coords in self.augmenter.se3_rotate(conf_coords):
                    records.append({
                        **base_record,
                        "aug_type": "conformer_se3",
                        "coords": rotated_coords,
                    })

        return records

    def _generate_fragment_masks(
        self, smiles: str, base_record: dict, n_masks: int = 3
    ) -> List[dict]:
        records = []
        for masked_smiles, indices in self.augmenter.mask_fragments(smiles, n_masks):
            records.append({
                **base_record,
                "smiles": masked_smiles,
                "aug_type": "masked",
                "masked_atom_indices": indices,
            })
        return records

    def _flush_shard(self) -> None:
        if not self._shard_buffer:
            return
        path = self.output_dir / f"shard_{self._shard_idx:06d}.pkl.gz"
        with gzip.open(path, "wb") as f:
            pickle.dump(self._shard_buffer, f, protocol=4)
        logger.debug(f"Flushed shard {self._shard_idx} ({len(self._shard_buffer)} records)")
        self._shard_buffer = []
        self._shard_idx += 1

    # ------------------------------------------------------------------
    # Source iterators
    # ------------------------------------------------------------------

    def _iter_source(
        self, path: Path, source_info: dict
    ) -> Iterator[Tuple[str, Optional[dict]]]:
        fmt = source_info.get("format", "smi")
        if fmt == "smi_gz":
            yield from self._iter_smi_gz(path)
        elif fmt == "smi":
            yield from self._iter_smi(path)
        elif fmt in ("chembl_tsv_gz", "chembl_tsv"):
            yield from self._iter_chembl_tsv(path)
        elif fmt == "tsv_xz":
            yield from self._iter_excape(path)
        elif fmt == "tsv_zip":
            yield from self._iter_bindingdb(path)
        elif fmt == "csv":
            yield from self._iter_csv(path)
        elif fmt in ("sdf_directory", "sdf"):
            yield from self._iter_sdf_directory(path)
        else:
            logger.warning(f"Unknown format {fmt} — skipping")

    def _iter_smi_gz(self, path: Path) -> Iterator[Tuple[str, None]]:
        try:
            with gzip.open(str(path), "rt", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    # PubChem CID-SMILES format: CID<tab>SMILES (first col is numeric)
                    if parts[0].isdigit() and len(parts) >= 2:
                        yield parts[1], None
                    else:
                        yield parts[0], None
        except Exception as e:
            logger.debug(f"_iter_smi_gz failed for {path}: {e}")

    def _iter_smi(self, path: Path) -> Iterator[Tuple[str, None]]:
        try:
            with open(path, "r", errors="ignore") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        yield parts[0], None
        except Exception as e:
            logger.debug(f"_iter_smi failed for {path}: {e}")

    def _iter_chembl_tsv(self, path: Path) -> Iterator[Tuple[str, None]]:
        """Iterate ChEMBL TSV: chembl_id\tsmiles\tinchikey columns."""
        try:
            import pandas as pd
            df = pd.read_csv(path, sep="\t", nrows=None, usecols=lambda c: "smiles" in c.lower() or c == "canonical_smiles")
            smiles_col = next(
                (c for c in df.columns if "smiles" in c.lower()), df.columns[0]
            )
            for smiles in df[smiles_col].dropna():
                yield str(smiles), None
        except Exception as e:
            logger.debug(f"_iter_chembl_tsv failed for {path}: {e}")

    def _iter_excape(self, path: Path) -> Iterator[Tuple[str, Optional[dict]]]:
        """Iterate ExCAPE-DB TSV.xz: SMILES + activity data."""
        try:
            import lzma
            import pandas as pd
            with lzma.open(str(path), "rt") as f:
                for chunk in pd.read_csv(f, sep="\t", chunksize=50_000):
                    smiles_col = next(
                        (c for c in chunk.columns if "smiles" in c.lower()), None
                    )
                    if smiles_col is None:
                        break
                    act_col = next(
                        (c for c in chunk.columns if "pchembl" in c.lower() or "activity" in c.lower()), None
                    )
                    for _, row in chunk.iterrows():
                        smiles = str(row[smiles_col])
                        bioactivity = None
                        if act_col and not pd.isna(row[act_col]):
                            try:
                                pchembl = float(row[act_col])
                                activity_nM = 10 ** (9 - pchembl)
                                bioactivity = {"activity_nM": activity_nM, "activity_type": "pChEMBL"}
                            except ValueError:
                                pass
                        yield smiles, bioactivity
        except Exception as e:
            logger.debug(f"_iter_excape failed for {path}: {e}")

    def _iter_bindingdb(self, path: Path) -> Iterator[Tuple[str, Optional[dict]]]:
        """Iterate BindingDB TSV.zip with Ki/Kd/IC50 activity data."""
        try:
            import zipfile
            import io
            import csv

            with zipfile.ZipFile(str(path)) as zf:
                tsv_name = next(
                    (n for n in zf.namelist() if n.endswith(".tsv")), zf.namelist()[0]
                )
                with zf.open(tsv_name) as raw:
                    reader = csv.DictReader(
                        io.TextIOWrapper(raw, encoding="utf-8", errors="ignore"),
                        delimiter="\t",
                    )
                    for row in reader:
                        smiles = None
                        for key in row:
                            if "smiles" in key.lower() or "ligand" in key.lower():
                                smiles = row[key].strip()
                                if smiles:
                                    break
                        if not smiles:
                            continue

                        activity_nM = None
                        activity_type = None

                        for ic50_col in row:
                            if "ic50" in ic50_col.lower():
                                val = row[ic50_col].strip()
                                if val:
                                    try:
                                        activity_nM = float(val)
                                        activity_type = "IC50"
                                    except ValueError:
                                        pass
                                break

                        if activity_nM is None:
                            for ki_col in row:
                                if "ki" in ki_col.lower() and "(" in ki_col:
                                    val = row[ki_col].strip()
                                    if val:
                                        try:
                                            activity_nM = float(val)
                                            activity_type = "Ki"
                                        except ValueError:
                                            pass
                                    break

                        if activity_nM is None:
                            for kd_col in row:
                                if "kd" in kd_col.lower() and "(" in kd_col:
                                    val = row[kd_col].strip()
                                    if val:
                                        try:
                                            activity_nM = float(val)
                                            activity_type = "Kd"
                                        except ValueError:
                                            pass
                                    break

                        target_col = next(
                            (k for k in row if "target" in k.lower() or "protein" in k.lower()), None
                        )
                        bioactivity = None
                        if activity_nM is not None:
                            bioactivity = {
                                "target_id": row[target_col] if target_col else None,
                                "activity_nM": activity_nM,
                                "activity_type": activity_type,
                            }
                        yield smiles, bioactivity

        except Exception as e:
            logger.debug(f"_iter_bindingdb failed for {path}: {e}")

    def _iter_csv(self, path: Path) -> Iterator[Tuple[str, None]]:
        """Iterate CSV file with SMILES column."""
        try:
            import pandas as pd
            df = pd.read_csv(path, nrows=None)
            smiles_col = next(
                (c for c in df.columns if "smiles" in c.lower()), df.columns[0]
            )
            for smiles in df[smiles_col].dropna():
                yield str(smiles), None
        except Exception as e:
            logger.debug(f"_iter_csv failed for {path}: {e}")

    def _iter_sdf_directory(self, path: Path) -> Iterator[Tuple[str, None]]:
        """Iterate over directory of SDF files."""
        try:
            from rdkit import Chem

            sdf_files = list(Path(path).glob("*.sdf.gz")) + list(Path(path).glob("*.sdf"))
            for sdf_path in sdf_files:
                if sdf_path.suffix == ".gz":
                    supplier = Chem.ForwardSDMolSupplier(
                        gzip.open(str(sdf_path)), removeHs=True
                    )
                else:
                    supplier = Chem.ForwardSDMolSupplier(str(sdf_path), removeHs=True)

                for mol in supplier:
                    if mol is not None:
                        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                        if smiles:
                            yield smiles, None
        except Exception as e:
            logger.debug(f"_iter_sdf_directory failed for {path}: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Stage 0 molecular pretraining dataset")
    p.add_argument("--output_dir", default="./data/processed/pretrain_shards",
                   help="Output directory for shards")
    p.add_argument("--n_workers", type=int, default=8)
    p.add_argument("--max_examples", type=int, default=None,
                   help=f"Cap total examples (default: {TARGET_EXAMPLES:,})")
    p.add_argument("--no_3d", action="store_true", help="Skip 3D conformer generation")
    p.add_argument("--n_smiles_aug", type=int, default=10)
    p.add_argument("--n_conformers", type=int, default=5)
    p.add_argument("--n_se3_aug", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    builder = PretrainingDatasetBuilder(
        output_dir=Path(args.output_dir),
        n_workers=args.n_workers,
        max_examples=args.max_examples,
        generate_3d=not args.no_3d,
        n_smiles_aug=args.n_smiles_aug,
        n_conformers=args.n_conformers,
        n_se3_aug=args.n_se3_aug,
    )
    n = builder.build()
    logger.info(f"Done: {n:,} total training examples")


if __name__ == "__main__":
    main()
