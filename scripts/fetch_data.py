#!/usr/bin/env python3
"""Fetch all required datasets for ORACLE."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fetch ORACLE datasets")
    parser.add_argument(
        "--cancer-type",
        default="colorectal",
        choices=["colorectal", "aml", "breast", "lung", "glioblastoma", "melanoma"],
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--geo-only", action="store_true")
    args = parser.parse_args()

    # Import here to allow the script to be used standalone
    from oracle.data.fetchers.geo_fetcher import GEOFetcher
    from oracle.data.fetchers.chembl_fetcher import ChEMBLFetcher

    print(f"Fetching data for {args.cancer_type}...")

    fetcher = GEOFetcher(cache_dir=f"{args.data_dir}/raw/scrnaseq")
    datasets = fetcher.fetch_cancer_panel(args.cancer_type)
    print(f"Fetched {len(datasets)} scRNA-seq datasets")

    if not args.geo_only:
        chembl = ChEMBLFetcher(cache_dir=f"{args.data_dir}/raw/chembl")
        # Fetch binders for key TFs
        key_tfs = ["TP53", "MYC", "CDX2", "SNAI2", "CEBPA", "EZH2", "BRD4"]
        for tf in key_tfs:
            compounds = chembl.fetch_tf_binders(tf)
            print(f"  {tf}: {len(compounds)} compounds")


if __name__ == "__main__":
    main()
