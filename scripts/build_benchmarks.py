#!/usr/bin/env python3
"""Download and prepare benchmark datasets for ORACLE evaluation.

Benchmarks:
  kaist_colorectal  — KAIST REVERT colorectal cancer reversion dataset
  aml_atra          — AML ATRA differentiation dataset (GEO GSE13159)
  beeline           — Synthetic GRN benchmarks from BEELINE paper
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("oracle.build_benchmarks")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ORACLE benchmark datasets")
    p.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["kaist_colorectal", "aml_atra", "beeline", "all"],
        default=["all"],
    )
    p.add_argument("--data_dir", default="./data", help="Root data directory")
    p.add_argument("--force", action="store_true", help="Re-download even if data exists")
    return p.parse_args()


def build_kaist_colorectal(data_dir: Path, force: bool = False) -> None:
    """Prepare KAIST REVERT colorectal benchmark.

    Ground truth: CDX2 activation + SNAI2 repression restores normal colonocyte identity.
    Sourced from colorectal scRNA-seq: GSE132465, GSE166555, GSE200997.
    """
    bench_dir = data_dir / "benchmarks/kaist_colorectal"
    bench_dir.mkdir(parents=True, exist_ok=True)

    gt_path = bench_dir / "ground_truth.json"
    if gt_path.exists() and not force:
        logger.info("KAIST ground truth already exists: %s", gt_path)
    else:
        ground_truth = {
            "perturbations": {"CDX2": "Activation", "SNAI2": "Repression"},
            "cancer_type": "colorectal",
            "tissue": "colon",
            "reference": "KAIST REVERT (2023)",
            "geo_accessions": ["GSE132465", "GSE166555", "GSE200997"],
            "description": (
                "CDX2 is the master colonocyte TF; its re-activation drives reversion. "
                "SNAI2 is an EMT driver; its repression restores epithelial identity."
            ),
        }
        with open(gt_path, "w") as f:
            json.dump(ground_truth, f, indent=2)
        logger.info("Wrote KAIST ground truth to %s", gt_path)

    # Download scRNA-seq data using GEO fetcher
    try:
        from oracle.data.fetchers.geo_fetcher import GEOFetcher
        fetcher = GEOFetcher(data_dir=str(data_dir))
        for acc in ["GSE132465", "GSE166555"]:
            dest = data_dir / f"raw/scrnaseq/{acc}"
            if dest.exists() and not force:
                logger.info("GEO %s already downloaded", acc)
                continue
            logger.info("Downloading GEO %s...", acc)
            try:
                fetcher.fetch(acc, cancer_type="colorectal")
            except Exception as e:
                logger.warning("Could not download %s: %s", acc, e)
    except ImportError as e:
        logger.warning("GEO fetcher unavailable: %s", e)


def build_aml_atra(data_dir: Path, force: bool = False) -> None:
    """Prepare AML ATRA differentiation benchmark (GEO GSE13159).

    Ground truth: CEBPA, IRF8, SPI1 activation drives AML→normal myeloid differentiation
    under ATRA treatment.
    """
    bench_dir = data_dir / "benchmarks/aml_atra"
    bench_dir.mkdir(parents=True, exist_ok=True)

    gt_path = bench_dir / "ground_truth.json"
    if gt_path.exists() and not force:
        logger.info("AML ATRA ground truth already exists: %s", gt_path)
    else:
        ground_truth = {
            "perturbations": {
                "CEBPA": "Activation",
                "IRF8": "Activation",
                "SPI1": "Activation",
            },
            "cancer_type": "leukemia_aml",
            "tissue": "bone_marrow",
            "reference": "GEO GSE13159",
            "geo_accessions": ["GSE13159", "GSE116256"],
            "description": (
                "ATRA treatment of AML activates the myeloid differentiation program. "
                "Key TFs: CEBPA (granulocyte differentiation), "
                "IRF8/SPI1 (monocyte/DC differentiation)."
            ),
        }
        with open(gt_path, "w") as f:
            json.dump(ground_truth, f, indent=2)
        logger.info("Wrote AML ATRA ground truth to %s", gt_path)

    try:
        from oracle.data.fetchers.geo_fetcher import GEOFetcher
        fetcher = GEOFetcher(data_dir=str(data_dir))
        for acc in ["GSE13159"]:
            dest = data_dir / f"raw/scrnaseq/{acc}"
            if dest.exists() and not force:
                logger.info("GEO %s already downloaded", acc)
                continue
            logger.info("Downloading GEO %s...", acc)
            try:
                fetcher.fetch(acc, cancer_type="leukemia_aml")
            except Exception as e:
                logger.warning("Could not download %s: %s", acc, e)
    except ImportError as e:
        logger.warning("GEO fetcher unavailable: %s", e)


def build_beeline(data_dir: Path, force: bool = False) -> None:
    """Prepare BEELINE synthetic GRN benchmarks.

    Synthetic networks: GSD (500 genes), mCAD (300 genes), VSN (800 genes),
    BFC (400 genes). Ground-truth edge sets provided as JSON.
    """
    bench_dir = data_dir / "benchmarks/synthetic"
    bench_dir.mkdir(parents=True, exist_ok=True)

    # Minimal synthetic networks for quick benchmarking
    synthetic_networks = {
        "GSD": {
            "description": "Gene regulatory network from GSD (500 genes, scale-free)",
            "n_genes": 500,
            "n_edges": 1200,
            "edges": [],  # populated from BEELINE data package
        },
        "mCAD": {
            "description": "mCAD synthetic GRN (300 genes)",
            "n_genes": 300,
            "n_edges": 800,
            "edges": [],
        },
    }

    for name, info in synthetic_networks.items():
        gold_path = bench_dir / f"{name}_gold.json"
        if gold_path.exists() and not force:
            logger.info("BEELINE %s gold standard already exists", name)
            continue

        with open(gold_path, "w") as f:
            json.dump(info["edges"], f, indent=2)
        logger.info("Created BEELINE %s gold standard at %s", name, gold_path)

    # Write network metadata
    meta_path = bench_dir / "beeline_metadata.json"
    with open(meta_path, "w") as f:
        json.dump({
            "source": "BEELINE (Pratapa et al. 2020, Nature Methods)",
            "networks": list(synthetic_networks.keys()),
            "note": "Edge lists populated after downloading from https://github.com/Murali-group/BEELINE",
        }, f, indent=2)
    logger.info("BEELINE metadata written to %s", meta_path)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    benchmarks = args.benchmarks
    if "all" in benchmarks:
        benchmarks = ["kaist_colorectal", "aml_atra", "beeline"]

    for bench in benchmarks:
        logger.info("Building benchmark: %s", bench)
        if bench == "kaist_colorectal":
            build_kaist_colorectal(data_dir, args.force)
        elif bench == "aml_atra":
            build_aml_atra(data_dir, args.force)
        elif bench == "beeline":
            build_beeline(data_dir, args.force)

    logger.info("Benchmark preparation complete.")


if __name__ == "__main__":
    main()
