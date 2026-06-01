#!/usr/bin/env python3
"""Build MMseqs2-clustered train/val/test splits for TF protein sequences.

Ensures zero data leakage: no two TFs with ≥30% sequence identity appear
in different splits.  Sequences are fetched from UniProt REST API.

Usage
-----
    python scripts/build_tf_splits.py \
        --identity 0.30 \
        --ratios 0.70 0.15 0.15 \
        --output data/splits/tf_sequences
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("oracle.build_tf_splits")


# ---------------------------------------------------------------------------
# Known TF set (mirrors oracle/tcd/tf_structurer.py _TF_UNIPROT_MAP)
# ---------------------------------------------------------------------------

TF_UNIPROT_MAP: Dict[str, str] = {
    "MYC": "P01106",
    "TP53": "P04637",
    "RUNX1": "Q01196",
    "CEBPA": "P49715",
    "ESR1": "P03372",
    "STAT3": "P40763",
    "HIF1A": "Q16665",
    "SOX2": "P48431",
    "FOXA1": "P55317",
    "MITF": "O75030",
    "SNAI1": "O95863",
    "CDX2": "Q99626",
    "HOXA9": "P31269",
    "MEIS1": "O00470",
    "ZEB1": "P37275",
    "TWIST1": "Q15672",
    "OLIG2": "Q13516",
    "PAX3": "P23760",
    "SPI1": "P17947",
    "IRF8": "Q02556",
    # Extended cancer-relevant TFs
    "SNAI2": "O43623",
    "BRCA1": "P38398",
    "EZH2": "Q15910",
    "KLF4": "O43474",
    "OCT4": "Q01860",
    "NANOG": "Q9H9S0",
    "GATA1": "P15976",
    "GATA3": "P23771",
    "AR": "P10275",
    "CTNNB1": "P35222",
    "VHL": "P40337",
    "SMAD4": "Q13485",
    "NRF2": "Q16236",
    "FOXO3": "O43524",
    "E2F1": "Q01094",
    "RB1": "P06400",
    "MYB": "P10242",
    "NFKB1": "P19838",
    "YAP1": "P46937",
    "TAZ": "Q9GZV5",
}


# ---------------------------------------------------------------------------
# UniProt sequence fetcher
# ---------------------------------------------------------------------------


def fetch_uniprot_sequence(uniprot_id: str, retries: int = 3) -> Optional[str]:
    """Fetch canonical amino-acid sequence from UniProt REST API."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                lines = resp.text.strip().splitlines()
                seq = "".join(l for l in lines if not l.startswith(">"))
                return seq
            logger.warning(
                "UniProt %s returned HTTP %d (attempt %d/%d)",
                uniprot_id, resp.status_code, attempt + 1, retries,
            )
        except requests.RequestException as exc:
            logger.warning("UniProt fetch error for %s: %s", uniprot_id, exc)
        time.sleep(1.0)
    return None


def fetch_all_sequences(
    tf_map: Dict[str, str],
    cache_path: Optional[Path] = None,
) -> Dict[str, str]:
    """Fetch sequences for all TFs; load/save from cache if provided."""
    if cache_path and cache_path.exists():
        with open(cache_path) as fh:
            cached: Dict[str, str] = json.load(fh)
        missing = {tf: uid for tf, uid in tf_map.items() if tf not in cached}
        if not missing:
            logger.info("All %d TF sequences loaded from cache.", len(cached))
            return cached
        logger.info(
            "Loaded %d cached sequences; fetching %d missing.",
            len(cached), len(missing),
        )
        sequences = dict(cached)
    else:
        sequences: Dict[str, str] = {}
        missing = tf_map

    for tf_name, uniprot_id in missing.items():
        seq = fetch_uniprot_sequence(uniprot_id)
        if seq:
            sequences[tf_name] = seq
            logger.info("Fetched %s (%s): %d aa", tf_name, uniprot_id, len(seq))
        else:
            logger.warning("Could not fetch sequence for %s (%s)", tf_name, uniprot_id)
        time.sleep(0.2)  # polite rate limiting

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as fh:
            json.dump(sequences, fh, indent=2)
        logger.info("Cached %d sequences to %s", len(sequences), cache_path)

    return sequences


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build MMseqs2-clustered TF splits")
    p.add_argument("--identity", type=float, default=0.30,
                   help="MMseqs2 sequence identity threshold (default 0.30 = 30%%)")
    p.add_argument("--coverage", type=float, default=0.80,
                   help="MMseqs2 coverage threshold (default 0.80)")
    p.add_argument("--ratios", type=float, nargs=3, default=[0.70, 0.15, 0.15],
                   metavar=("TRAIN", "VAL", "TEST"),
                   help="Train/val/test ratios (must sum to 1.0)")
    p.add_argument("--output", default="data/splits/tf_sequences",
                   help="Output directory for split JSON files")
    p.add_argument("--cache", default="data/splits/tf_sequences/sequences.json",
                   help="Path to cache fetched UniProt sequences")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mmseqs_bin", default="mmseqs",
                   help="MMseqs2 binary (default: mmseqs from PATH)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ratios = tuple(args.ratios)
    assert abs(sum(ratios) - 1.0) < 1e-4, f"Ratios must sum to 1.0, got {sum(ratios)}"

    from oracle.utils.sequence_split import mmseqs2_split, save_splits

    output_dir = Path(ROOT / args.output)
    cache_path = Path(ROOT / args.cache)

    # 1. Fetch sequences
    sequences = fetch_all_sequences(TF_UNIPROT_MAP, cache_path=cache_path)
    if not sequences:
        logger.error("No sequences fetched — check network / UniProt availability.")
        sys.exit(1)
    logger.info("Total sequences available: %d", len(sequences))

    # 2. Cluster and split
    splits = mmseqs2_split(
        sequences=sequences,
        ratios=ratios,
        identity=args.identity,
        coverage=args.coverage,
        seed=args.seed,
        mmseqs_bin=args.mmseqs_bin,
    )

    # 3. Save
    save_splits(splits, str(output_dir))

    # 4. Summary
    print("\n=== MMseqs2 Split Summary (identity ≥ {:.0f}%) ===".format(
        args.identity * 100
    ))
    for name in ("train", "val", "test"):
        ids = splits[name]
        print(f"  {name:5s}: {len(ids):3d} TFs  {ids}")

    total = sum(len(v) for v in splits.values())
    print(f"\n  Total assigned: {total} / {len(sequences)}")
    print(f"  Splits written to: {output_dir}")


if __name__ == "__main__":
    main()
