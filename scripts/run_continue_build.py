#!/usr/bin/env python3
"""Continue building Stage 0 pretraining shards from where the last run left off.

Starts at shard 55 (after the 5.5M PubChem examples already written) and processes
the remaining downloaded sources: zinc20_instock, chembl_33, excape_db, guacamol, moses.
"""
import sys, os, json, logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("logs/dataset_continue.log"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("oracle.dataset_continue")

from scripts.build_pretrain_dataset import (
    PretrainingDatasetBuilder, SHARD_SIZE, TARGET_EXAMPLES,
)

SHARD_DIR = Path("data/pretrain_shards")
MOLECULES_DIR = Path("data/raw/molecules")

# Sources already processed (skip them)
ALREADY_PROCESSED = {"pubchem_bioactive"}

# Count existing shards so we start writing after them
existing_shards = sorted(SHARD_DIR.glob("shard_*.pkl.gz"))
start_shard = len(existing_shards)
existing_examples = start_shard * SHARD_SIZE

logger.info(f"Resuming from shard {start_shard} ({existing_examples:,} examples already written)")
logger.info(f"Target: {TARGET_EXAMPLES:,} total examples")

builder = PretrainingDatasetBuilder(
    output_dir=SHARD_DIR,
    n_workers=0,          # macOS spawn env safety
    generate_3d=False,
    n_smiles_aug=10,
    n_conformers=0,
    n_se3_aug=0,
)

# Patch builder state to continue from the right shard index
builder._shard_idx = start_shard
builder._n_total = existing_examples

# Process remaining sources in priority order
remaining_sources = {
    "zinc20_instock": {
        "format": "smi_gz",
        "n_compounds_approx": 230_000_000,
        "priority": 1,
    },
    "chembl_33": {
        "format": "chembl_tsv_gz",
        "n_compounds_approx": 2_400_000,
        "priority": 1,
    },
    "excape_db": {
        "format": "tsv_xz",
        "n_compounds_approx": 70_000_000,
        "priority": 1,
        "has_bioactivity": True,
    },
    "guacamol": {
        "format": "smi_gz",
        "n_compounds_approx": 1_600_000,
        "priority": 2,
    },
    "moses": {
        "format": "csv",
        "n_compounds_approx": 1_900_000,
        "priority": 2,
    },
}

for source_name, source_info in sorted(remaining_sources.items(), key=lambda x: x[1]["priority"]):
    if builder._n_total >= TARGET_EXAMPLES:
        break

    # Try to find the source file
    source_path = MOLECULES_DIR / source_name
    if not source_path.exists():
        for ext in [".smi", ".smi.gz", ".tsv.gz", ".tsv.xz", ".csv"]:
            candidate = MOLECULES_DIR / (source_name + ext)
            if candidate.exists():
                source_path = candidate
                break
        else:
            logger.warning(f"Source {source_name} not found in {MOLECULES_DIR} — skipping")
            continue

    remaining = TARGET_EXAMPLES - builder._n_total
    aug_factor = builder.augmenter.n_smiles_aug + 3
    max_from_source = max(1, remaining // max(1, aug_factor))

    logger.info(f"Processing {source_name} (up to {max_from_source:,} raw molecules)…")
    n = builder._process_source(source_name, source_path, source_info, max_from_source)
    logger.info(f"  {source_name}: +{n:,} examples (total={builder._n_total:,})")

# Flush any partial shard
if builder._shard_buffer:
    builder._flush_shard()

# Update manifest
manifest = {
    "n_examples": builder._n_total,
    "n_shards": builder._shard_idx,
    "shard_size": SHARD_SIZE,
    "target_examples": TARGET_EXAMPLES,
    "sources": list(ALREADY_PROCESSED) + list(remaining_sources.keys()),
    "status": "partial" if builder._n_total < TARGET_EXAMPLES else "complete",
}
with open(SHARD_DIR / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

logger.info(f"Dataset build phase complete: {builder._n_total:,} examples across {builder._shard_idx} shards")
