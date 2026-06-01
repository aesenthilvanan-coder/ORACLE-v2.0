#!/usr/bin/env python3
"""Preprocess all raw scRNA-seq datasets."""

import argparse
import scanpy as sc
from pathlib import Path
from oracle.cam.preprocessing import CancerAttractionPreprocessor
from oracle.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Run CAM preprocessing on all fetched scRNA-seq datasets."
    )
    parser.add_argument("--config", default="configs/base_config.yaml")
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args()

    config = load_config(args.config)
    preprocessor = CancerAttractionPreprocessor(config.cam)

    raw_dir = Path(args.data_dir) / "raw" / "scrnaseq"
    out_dir = Path(args.data_dir) / "processed" / "anndata"
    out_dir.mkdir(parents=True, exist_ok=True)

    h5ad_files = list(raw_dir.glob("*.h5ad"))
    if not h5ad_files:
        print(f"No .h5ad files found in {raw_dir}. Run scripts/fetch_data.py first.")
        return

    for h5ad_file in h5ad_files:
        print(f"Processing {h5ad_file.name}...")
        adata = sc.read_h5ad(h5ad_file)
        print(f"  Input shape: {adata.shape}")
        adata_processed = preprocessor.run(adata)
        out_path = out_dir / h5ad_file.name
        adata_processed.write_h5ad(out_path)
        print(f"  Saved to {out_path}  (shape: {adata_processed.shape})")

    print(f"\nPreprocessing complete. {len(h5ad_files)} datasets processed.")


if __name__ == "__main__":
    main()
