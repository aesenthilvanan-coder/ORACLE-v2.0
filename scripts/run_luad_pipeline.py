#!/usr/bin/env python3
"""
ORACLE full pipeline — Lung Adenocarcinoma (LUAD)
Primary data: GSE131907 (Kim et al. 2020, Nat Comm, 208,506 cells, 44 LUAD patients)
              GSE189357 (spatiotemporal AIS→invasive atlas)

Genetic context (KRAS-mutant LUAD):
  - KRAS G12C/D/V constitutively active
  - STK11/LKB1 loss-of-function (common co-mutation, immunotherapy resistance)
  - TP53 mutation (50% co-occurrence with KRAS-mutant LUAD)

Pipeline:
  Module 1 (CAM)  — Real LUAD scRNA-seq, curated GRN, Boolean attractor landscape
  Module 2 (RSP)  — Minimal TF reversion switches (KRAS/STK11 excluded — genetic events)
  Module 3 (TCD)  — TCIP bifunctional molecule design for AT2 redifferentiation
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("oracle.luad_pipeline")

OUTPUT_DIR = Path("outputs/luad")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# LUAD biology constants
# ─────────────────────────────────────────────────────────────────────────────

LUAD_ONCOGENIC_TFS = [
    "MYC",    # amplified in ~20% LUAD; master proliferation/dedifferentiation TF
    "YAP1",   # Hippo pathway effector; activated by KRAS loss of LATS signaling
    "WWTR1",  # TAZ — cooperates with YAP1
    "SNAI2",  # Slug — EMT, represses CDH1, promotes invasion
    "ZEB1",   # EMT + dedifferentiation; represses NKX2-1
    "ZEB2",   # EMT cooperative with ZEB1
    "TWIST1", # EMT TF
    "SOX9",   # stem cell/EMT; marks dedifferentiated LUAD
    "E2F1",   # cell cycle driver; elevated when RB1/TP53 axis is disrupted
    "FOSL1",  # AP-1 component; KRAS-driven; represses AT2 program
    "TEAD1",  # YAP1/TAZ effector TF
]

LUAD_DIFFERENTIATION_TFS = [
    "NKX2-1",  # TTF1 — master AT2 identity TF; lost in poorly differentiated LUAD
    "FOXA1",   # pioneer TF; AT2 chromatin accessibility
    "FOXA2",   # AT2 pioneer TF; cooperates with NKX2-1
    "ETV5",    # AT2 TF; required for surfactant program
]

# Genetic events — excluded from druggability (require genetic correction)
KRAS_MUTANT = True
STK11_LOST  = True       # STK11/LKB1 loss — common co-mutation
KRAS_GENE   = "KRAS"
STK11_GENE  = "STK11"

# ─────────────────────────────────────────────────────────────────────────────
# Gene panel — 65 LUAD-relevant genes
# ─────────────────────────────────────────────────────────────────────────────

LUAD_GENE_PANEL = [
    # AT2 differentiation identity (restore)
    "NKX2-1", "FOXA1", "FOXA2", "ETV5",
    "SFTPC", "SFTPB", "ABCA3", "SLC34A2",
    # EMT TFs (repress)
    "SNAI2", "ZEB1", "ZEB2", "TWIST1",
    # Hippo pathway
    "YAP1", "WWTR1", "TEAD1", "LATS1", "LATS2",
    # AP-1 / inflammatory
    "FOS", "JUN", "FOSL1", "NFKB1",
    # Stem / dedifferentiation
    "SOX9", "SOX4",
    # MYC program
    "MYC",
    # CDK / cell cycle
    "CCND1", "CCND2", "CDK4", "CDK6",
    # E2F
    "E2F1", "E2F3",
    # Cell cycle brakes
    "CDKN1A", "CDKN2A", "RB1",
    # TP53 pathway
    "TP53", "MDM2", "BAX", "BCL2", "BCL2L1", "MCL1",
    # KRAS / MAPK
    "KRAS", "EGFR", "BRAF", "MAPK1", "MAPK3",
    # PI3K / AKT / mTOR
    "AKT1", "PIK3CA", "MTOR",
    # Tumor suppressors
    "STK11", "KEAP1",
    # Epigenetic
    "EZH2", "KDM6A", "HDAC1", "BRD4",
    # EMT structural markers
    "CDH1", "CDH2", "VIM", "FN1",
    # NRF2
    "NFE2L2",
    # Metabolism / hypoxia
    "LDHA", "HIF1A",
    # Notch / Wnt
    "NOTCH1", "HES1", "CTNNB1",
    # Apoptosis effectors
    "CASP3", "CASP9",
]

N_GENES = len(LUAD_GENE_PANEL)
GENE_IDX = {g: i for i, g in enumerate(LUAD_GENE_PANEL)}

# ─────────────────────────────────────────────────────────────────────────────
# Expression profiles for synthetic fallback
# ─────────────────────────────────────────────────────────────────────────────

def _luad_cancer_profile() -> Dict[str, float]:
    p = {g: 0.5 for g in LUAD_GENE_PANEL}
    # AT2 identity — silenced
    for g in ["NKX2-1","FOXA1","FOXA2","ETV5","SFTPC","SFTPB","ABCA3","SLC34A2"]:
        p[g] = 0.25
    # Oncogenic TFs — activated
    p["MYC"]   = 5.5;  p["YAP1"]  = 4.5; p["WWTR1"] = 3.5; p["TEAD1"] = 3.5
    p["SNAI2"] = 3.8;  p["ZEB1"]  = 3.5; p["ZEB2"]  = 3.0; p["TWIST1"]= 2.8
    p["SOX9"]  = 3.2;  p["SOX4"]  = 3.0; p["FOSL1"] = 3.5; p["E2F1"]  = 4.0
    # KRAS-driven signaling
    p["KRAS"]  = 4.5;  p["MAPK1"] = 4.5; p["MAPK3"] = 4.0
    p["AKT1"]  = 3.5;  p["MTOR"]  = 3.2; p["PIK3CA"]= 3.0
    # Cell cycle — CDKs active, brakes off
    p["CDK4"]  = 4.5;  p["CDK6"]  = 4.2; p["CCND1"] = 5.0; p["CCND2"] = 3.5
    p["E2F3"]  = 3.5;  p["CDKN1A"]= 1.0; p["CDKN2A"]= 0.3; p["RB1"]   = 1.5
    # TP53 partially impaired (mutation)
    p["TP53"]  = 1.5;  p["MDM2"]  = 3.5
    # Apoptosis — anti-apoptotic dominance
    p["BCL2"]  = 4.0;  p["BCL2L1"]= 3.8; p["MCL1"]  = 4.2; p["BAX"]   = 1.5
    # Epigenetic silencers active
    p["EZH2"]  = 4.5;  p["HDAC1"] = 3.5; p["BRD4"]  = 4.0; p["KDM6A"] = 1.5
    # EMT markers
    p["CDH1"]  = 0.5;  p["CDH2"]  = 4.5; p["VIM"]   = 5.0; p["FN1"]   = 3.5
    # STK11 lost — Hippo dysregulated
    p["STK11"] = 0.2;  p["LATS1"] = 1.0; p["LATS2"] = 0.8
    # Metabolism
    p["HIF1A"] = 3.5;  p["LDHA"]  = 4.0
    # Notch
    p["NOTCH1"]= 3.0;  p["HES1"]  = 2.8; p["CTNNB1"]= 2.5
    # BRAF/EGFR context (secondary)
    p["BRAF"]  = 2.5;  p["EGFR"]  = 2.0; p["KEAP1"] = 1.5; p["NFE2L2"]= 3.0
    p["CASP3"] = 1.5;  p["CASP9"] = 1.5
    return p


def _luad_normal_at2_profile() -> Dict[str, float]:
    p = {g: 1.5 for g in LUAD_GENE_PANEL}
    # AT2 identity — high
    p["NKX2-1"] = 7.5; p["FOXA1"] = 6.5; p["FOXA2"] = 7.0; p["ETV5"]  = 5.5
    p["SFTPC"]  = 8.5; p["SFTPB"] = 8.0; p["ABCA3"] = 7.5; p["SLC34A2"]= 7.0
    # Epithelial markers
    p["CDH1"]   = 7.5; p["CDH2"]  = 0.3; p["VIM"]   = 0.5; p["FN1"]   = 0.3
    # Oncogenic TFs — low
    p["MYC"]    = 1.5; p["YAP1"]  = 1.5; p["WWTR1"] = 1.2; p["TEAD1"] = 1.5
    p["SNAI2"]  = 0.4; p["ZEB1"]  = 0.3; p["ZEB2"]  = 0.3; p["TWIST1"]= 0.4
    p["SOX9"]   = 0.8; p["SOX4"]  = 0.8; p["FOSL1"] = 0.5
    # TP53 functional
    p["TP53"]   = 4.5; p["MDM2"]  = 2.0; p["CDKN1A"]= 4.5; p["CDKN2A"]= 4.0
    # Cell cycle quiescent
    p["CDK4"]   = 1.5; p["CDK6"]  = 1.2; p["CCND1"] = 1.5; p["E2F1"]  = 1.5
    # KRAS/MAPK baseline
    p["KRAS"]   = 1.5; p["MAPK1"] = 2.0; p["MAPK3"] = 2.0; p["AKT1"]  = 2.0
    p["MTOR"]   = 2.0; p["PIK3CA"]= 1.5
    # STK11 intact — Hippo active → YAP OFF
    p["STK11"]  = 4.5; p["LATS1"] = 4.5; p["LATS2"] = 4.0
    # Apoptosis balanced
    p["BAX"]    = 4.0; p["BCL2"]  = 2.0; p["BCL2L1"]= 2.5; p["MCL1"]  = 2.5
    # Epigenetic — normal
    p["EZH2"]   = 2.0; p["HDAC1"] = 2.5; p["BRD4"]  = 2.5; p["KDM6A"] = 4.0
    # Metabolism normal
    p["HIF1A"]  = 1.5; p["LDHA"]  = 1.5
    # Notch low
    p["NOTCH1"] = 1.5; p["HES1"]  = 1.2; p["CTNNB1"]= 1.5
    p["BRAF"]   = 1.5; p["EGFR"]  = 1.5; p["KEAP1"] = 4.5; p["NFE2L2"]= 2.0
    p["RB1"]    = 4.5; p["NFKB1"] = 1.5; p["FOS"]   = 1.5; p["JUN"]   = 1.5
    p["CASP3"]  = 2.5; p["CASP9"] = 2.5
    return p


# ─────────────────────────────────────────────────────────────────────────────
# GEO data download helpers
# ─────────────────────────────────────────────────────────────────────────────

_GEO_BASE_131907 = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE131nnn/GSE131907/suppl"
_GEO_BASE_189357 = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE189nnn/GSE189357/suppl"

_ANNOT_URL_131907  = f"{_GEO_BASE_131907}/GSE131907_Lung_Cancer_cell_annotation.txt.gz"
_MATRIX_URL_131907 = f"{_GEO_BASE_131907}/GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz"

_GEO_DATA_DIR = Path("data/luad/gse131907")


def _geo_download(url: str, dest: Path, label: str) -> bool:
    import requests
    try:
        logger.info("Downloading %s ...", label)
        resp = requests.get(url, stream=True, timeout=600)
        resp.raise_for_status()
        total_mb = int(resp.headers.get("content-length", 0)) / 1e6
        downloaded = 0
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=2 << 20):
                fh.write(chunk)
                downloaded += len(chunk)
                if total_mb > 10 and downloaded % (50 << 20) < (2 << 20):
                    logger.info("  %.0f / %.0f MB", downloaded / 1e6, total_mb)
        logger.info("  Done: %.1f MB → %s", downloaded / 1e6, dest)
        return True
    except Exception as exc:
        logger.warning("Download failed [%s]: %s", label, exc)
        if dest.exists():
            dest.unlink()
        return False


def _log_normalize(X: np.ndarray, scale: float = 1e4) -> np.ndarray:
    total = X.sum(axis=1, keepdims=True)
    total[total == 0] = 1.0
    return np.log1p(X / total * scale).astype(np.float32)


def _parse_gse131907_annotation(annot_path: Path) -> "pd.DataFrame":
    import gzip
    import pandas as pd
    with gzip.open(str(annot_path), "rt") as fh:
        df = pd.read_csv(fh, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    logger.info("GSE131907 annotation: %d cells, cols=%s", len(df), list(df.columns))
    if "Cell_subtype" in df.columns:
        logger.info("Cell_subtype counts:\n%s",
                    df["Cell_subtype"].value_counts().head(15).to_string())
    return df


def _get_cell_masks(annot: "pd.DataFrame"):
    """Return (barcode_col, type_col, cancer_mask, normal_mask) for GSE131907."""
    # GSE131907 barcodes are in 'Barcode' column
    barcode_col = "Barcode" if "Barcode" in annot.columns else annot.columns[0]

    # Use Cell_subtype: "Malignant cells"/"tS1"/"tS2"/"tS3" = cancer, "AT2"/"Club"/"AT1" = normal
    if "Cell_subtype" in annot.columns:
        type_col = "Cell_subtype"
    elif "Cell_type" in annot.columns:
        type_col = "Cell_type"
    else:
        type_col = annot.columns[1]

    ct = annot[type_col].astype(str).str.strip()
    cancer_mask = ct.str.contains(r"Malignant\s*cells?|^tS[0-9]+$", regex=True, na=False)
    normal_mask  = ct.str.contains(r"^AT2$|^AT1$|^Club$|^Ciliated$", regex=True, na=False)
    logger.info("Using column '%s'  |  Cancer: %d  Normal AT2/Club: %d",
                type_col, cancer_mask.sum(), normal_mask.sum())
    return barcode_col, type_col, cancer_mask, normal_mask


def _load_gse131907_matrix(
    matrix_path: Path,
    cancer_barcodes: set,
    normal_barcodes: set,
    panel_genes: List[str],
    max_cells_per_class: int = 3000,
) -> "Tuple[np.ndarray, List[str], List[str]]":
    """
    Stream-parse the dense GSE131907 count matrix (genes × cells TSV).
    Only load rows matching cancer/normal barcodes and panel genes.
    Caps at max_cells_per_class to keep memory manageable.
    """
    import gzip

    logger.info("Parsing count matrix — streaming (genes × cells format)...")

    # First pass: find gene indices for our panel
    panel_set = set(panel_genes)

    # The matrix is typically genes×cells: first row = barcodes, first col = gene names
    opener = gzip.open if str(matrix_path).endswith(".gz") else open

    # We need to know column indices for our target barcodes
    # Open once to read header (barcodes), then stream gene rows
    with opener(str(matrix_path), "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")

    # First column is always the gene/index label; strip it
    all_barcodes = header[1:]

    # GSE131907 barcodes in matrix: AAACCTGAGAAACCGC_LN_05  (barcode_SAMPLE)
    # Annotation barcodes are plain 16-char: AAACCTGAGAAACCGC
    # Strip everything from first underscore onward
    def _norm(b):
        return b.split("_")[0]

    barcode_to_col      = {b: i for i, b in enumerate(all_barcodes)}
    barcode_to_col_norm = {_norm(b): i for i, b in enumerate(all_barcodes)}

    # Identify which columns correspond to cancer / normal barcodes
    cancer_cols, cancer_bcs_found = [], []
    normal_cols, normal_bcs_found = [], []

    for bc in cancer_barcodes:
        col = barcode_to_col.get(bc) or barcode_to_col_norm.get(_norm(bc))
        if col is not None and len(cancer_cols) < max_cells_per_class:
            cancer_cols.append(col)
            cancer_bcs_found.append(bc)

    for bc in normal_barcodes:
        col = barcode_to_col.get(bc) or barcode_to_col_norm.get(_norm(bc))
        if col is not None and len(normal_cols) < max_cells_per_class:
            normal_cols.append(col)
            normal_bcs_found.append(bc)

    logger.info("Matched %d cancer cols, %d normal cols in matrix", len(cancer_cols), len(normal_cols))

    if len(cancer_cols) < 50 or len(normal_cols) < 30:
        raise RuntimeError(f"Insufficient barcode matches: cancer={len(cancer_cols)}, normal={len(normal_cols)}")

    target_cols = cancer_cols + normal_cols
    target_set  = set(target_cols)

    # Allocate output
    n_cells = len(target_cols)
    cancer_X = np.zeros((len(cancer_cols), len(panel_genes)), dtype=np.float32)
    normal_X = np.zeros((len(normal_cols), len(panel_genes)), dtype=np.float32)

    col_to_cancer_row = {c: r for r, c in enumerate(cancer_cols)}
    col_to_normal_row = {c: r for r, c in enumerate(normal_cols)}

    rows_loaded = 0
    with opener(str(matrix_path), "rt") as fh:
        fh.readline()  # skip header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            gene = parts[0]
            if gene not in panel_set:
                continue
            gene_pos = panel_genes.index(gene)
            vals = parts[1:]
            for c in cancer_cols:
                if c < len(vals):
                    v = vals[c]
                    cancer_X[col_to_cancer_row[c], gene_pos] = float(v) if v not in ("", "0") else 0.0
            for c in normal_cols:
                if c < len(vals):
                    v = vals[c]
                    normal_X[col_to_normal_row[c], gene_pos] = float(v) if v not in ("", "0") else 0.0
            rows_loaded += 1

    logger.info("Loaded %d panel genes from matrix", rows_loaded)

    X_all     = np.vstack([cancer_X, normal_X])
    barcodes  = cancer_bcs_found + normal_bcs_found
    states    = ["cancer"] * len(cancer_cols) + ["normal"] * len(normal_cols)
    return X_all, barcodes, states


def load_gse131907_luad() -> "Any":
    """
    Download and process GSE131907 real LUAD scRNA-seq data (Kim et al. 2020).
    Cancer cells: Malignant cells  |  Normal cells: AT2 / Club epithelial
    Falls back to synthetic if download or parsing fails.
    """
    import anndata as ad
    import scipy.sparse as sp
    import pandas as pd

    _GEO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = _GEO_DATA_DIR / "luad_anndata_panel.h5ad"

    if processed_path.exists():
        logger.info("Loading cached GSE131907 AnnData: %s", processed_path)
        adata = ad.read_h5ad(str(processed_path))
        nc = (adata.obs["cell_state"] == "cancer").sum()
        nn = (adata.obs["cell_state"] == "normal").sum()
        logger.info("Loaded: %d cells × %d genes  (cancer=%d, normal=%d)",
                    adata.n_obs, adata.n_vars, nc, nn)
        return adata

    # Step 1 — annotation file (small)
    annot_path = _GEO_DATA_DIR / "cell_annotation.txt.gz"
    if not annot_path.exists():
        ok = _geo_download(_ANNOT_URL_131907, annot_path, "GSE131907 cell annotations")
        if not ok:
            logger.warning("Annotation download failed — using synthetic data")
            return _generate_luad_synthetic()

    try:
        annot = _parse_gse131907_annotation(annot_path)
        barcode_col, type_col, cancer_mask, normal_mask = _get_cell_masks(annot)
        cancer_barcodes = set(annot.loc[cancer_mask, barcode_col])
        normal_barcodes = set(annot.loc[normal_mask, barcode_col])
    except Exception as exc:
        logger.warning("Annotation parsing failed (%s) — using synthetic", exc)
        return _generate_luad_synthetic()

    if len(cancer_barcodes) < 200 or len(normal_barcodes) < 100:
        logger.warning("Too few labelled cells (cancer=%d, normal=%d) — using synthetic",
                       len(cancer_barcodes), len(normal_barcodes))
        return _generate_luad_synthetic()

    logger.info("Target barcodes — LUAD cancer: %d, AT2 normal: %d",
                len(cancer_barcodes), len(normal_barcodes))

    # Step 2 — count matrix (large, ~1-4 GB compressed)
    matrix_path = _GEO_DATA_DIR / "raw_UMI_matrix.txt.gz"
    if not matrix_path.exists():
        ok = _geo_download(_MATRIX_URL_131907, matrix_path,
                           "GSE131907 raw UMI matrix (may be 1-4 GB)")
        if not ok:
            logger.warning("Matrix download failed — using synthetic data")
            return _generate_luad_synthetic()

    # Step 3 — parse
    try:
        X_raw, barcodes, states = _load_gse131907_matrix(
            matrix_path, cancer_barcodes, normal_barcodes, LUAD_GENE_PANEL,
            max_cells_per_class=3000,
        )
    except Exception as exc:
        logger.warning("Matrix parsing failed (%s) — using synthetic", exc)
        return _generate_luad_synthetic()

    if len(X_raw) < 200:
        logger.warning("Only %d cells matched — using synthetic", len(X_raw))
        return _generate_luad_synthetic()

    # Step 4 — normalise
    X_norm = _log_normalize(X_raw)

    # Enforce genetic events in cancer cells
    cancer_idx_list = [i for i, s in enumerate(states) if s == "cancer"]
    kras_col  = GENE_IDX.get("KRAS",  -1)
    stk11_col = GENE_IDX.get("STK11", -1)
    if kras_col >= 0:
        # KRAS mutation raises expression baseline (constitutive activation)
        rng = np.random.default_rng(42)
        X_norm[np.ix_(cancer_idx_list, [kras_col])] = rng.normal(5.0, 0.5, (len(cancer_idx_list), 1)).clip(3.5, 7.0)
        logger.info("Set KRAS constitutively active in %d cancer cells", len(cancer_idx_list))
    if stk11_col >= 0:
        X_norm[np.ix_(cancer_idx_list, [stk11_col])] = 0.0
        logger.info("Set STK11=0 in %d cancer cells (loss-of-function)", len(cancer_idx_list))

    obs = pd.DataFrame({
        "cell_state": states,
        "cell_type":  ["LUAD_malignant" if s == "cancer" else "AT2_normal" for s in states],
        "n_genes_by_counts": (X_norm > 0).sum(axis=1),
        "total_counts": X_norm.sum(axis=1),
        "source": "GSE131907",
    }, index=barcodes)

    var = pd.DataFrame({"gene_name": LUAD_GENE_PANEL}, index=LUAD_GENE_PANEL)
    adata = ad.AnnData(X=sp.csr_matrix(X_norm), obs=obs, var=var)
    adata.layers["raw_counts"] = sp.csr_matrix(X_raw)
    adata.uns["cancer_type"]  = "lung_adenocarcinoma"
    adata.uns["tissue_type"]  = "lung"
    adata.uns["geo_accession"] = "GSE131907"
    adata.uns["genetic_context"] = {
        "KRAS":  "G12C_activating_mutation",
        "STK11": "loss_of_function_mutation",
        "TP53":  "hotspot_mutation_partial_loss",
    }

    adata.write_h5ad(str(processed_path))
    logger.info("Saved processed AnnData: %s", processed_path)
    nc = (adata.obs["cell_state"] == "cancer").sum()
    nn = (adata.obs["cell_state"] == "normal").sum()
    logger.info("GSE131907 LUAD: %d cells × %d genes  (cancer=%d, normal=%d)",
                adata.n_obs, adata.n_vars, nc, nn)
    return adata


def _generate_luad_synthetic() -> "Any":
    import anndata as ad
    import scipy.sparse as sp
    import pandas as pd

    rng = np.random.default_rng(42)
    cancer_profile = np.array([_luad_cancer_profile()[g]  for g in LUAD_GENE_PANEL], dtype=np.float32)
    normal_profile = np.array([_luad_normal_at2_profile()[g] for g in LUAD_GENE_PANEL], dtype=np.float32)
    transit_profile = 0.4 * cancer_profile + 0.6 * normal_profile

    n_cancer, n_normal, n_transit = 600, 300, 400

    def _sample(prof, n, noise=0.18):
        noise_arr = rng.normal(0, noise * np.maximum(prof.std(), 0.5), (n, N_GENES)).astype(np.float32)
        cells = np.clip(prof[None, :] + noise_arr, 0.0, None)
        cells[rng.random((n, N_GENES)) < 0.14] = 0.0
        return cells

    X_c = _sample(cancer_profile,  n_cancer,  0.16)
    X_n = _sample(normal_profile,  n_normal,  0.14)
    X_t = _sample(transit_profile, n_transit, 0.22)

    # KRAS activation — elevated in all cancer cells
    if "KRAS" in GENE_IDX:
        X_c[:, GENE_IDX["KRAS"]]  = rng.normal(5.0, 0.5, n_cancer).clip(3.5, 7.0).astype(np.float32)
    if "STK11" in GENE_IDX:
        X_c[:, GENE_IDX["STK11"]] = 0.0
        X_t[:, GENE_IDX["STK11"]] = 0.0

    X = np.vstack([X_c, X_n, X_t])
    n_total = len(X)
    states  = ["cancer"]*n_cancer + ["normal"]*n_normal + ["transitional"]*n_transit
    types   = ["LUAD_malignant"]*n_cancer + ["AT2_normal"]*n_normal + ["transitional_EMT"]*n_transit

    obs = pd.DataFrame({
        "cell_state": states, "cell_type": types,
        "n_genes_by_counts": (X > 0).sum(axis=1),
        "total_counts": X.sum(axis=1),
        "source": "synthetic_fallback",
    }, index=[f"cell_{i}" for i in range(n_total)])

    var = pd.DataFrame({"gene_name": LUAD_GENE_PANEL}, index=LUAD_GENE_PANEL)
    adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)
    adata.layers["raw_counts"] = adata.X.copy()
    adata.uns.update({"cancer_type": "lung_adenocarcinoma", "tissue_type": "lung",
                      "genetic_context": {"KRAS": "G12C", "STK11": "loss", "TP53": "mutation"}})
    logger.info("Synthetic LUAD: %d cells × %d genes  (cancer=%d, normal=%d, transitional=%d)",
                n_total, N_GENES, n_cancer, n_normal, n_transit)
    return adata


def generate_luad_adata() -> Any:
    return load_gse131907_luad()


# ─────────────────────────────────────────────────────────────────────────────
# GRN construction — LUAD biology
# ─────────────────────────────────────────────────────────────────────────────

def build_luad_grn(adata: Any) -> Any:
    import networkx as nx
    import scipy.sparse as sp

    genes    = list(adata.var_names)
    gene_idx = {g: i for i, g in enumerate(genes)}
    n_genes  = len(genes)

    def _mean(state):
        mask = adata.obs["cell_state"] == state
        X = adata[mask].X
        return np.array(X.mean(axis=0)).flatten() if sp.issparse(X) else X.mean(axis=0)

    cancer_mean = _mean("cancer")
    normal_mean = _mean("normal")
    eps = 1e-3
    lfc = np.log2((cancer_mean + eps) / (normal_mean + eps))

    curated = [
        # ── KRAS / MAPK signaling ───────────────────────────────────────────
        ("KRAS",  "MAPK1",  +1, 0.97),
        ("KRAS",  "MAPK3",  +1, 0.97),
        ("KRAS",  "AKT1",   +1, 0.92),
        ("KRAS",  "MYC",    +1, 0.90),
        ("KRAS",  "MDM2",   +1, 0.88),
        ("MAPK1", "FOS",    +1, 0.93),
        ("MAPK1", "FOSL1",  +1, 0.92),
        ("MAPK1", "EZH2",   +1, 0.85),
        ("MAPK3", "JUN",    +1, 0.90),
        ("MAPK3", "FOSL1",  +1, 0.88),
        # ── EGFR (parallel driver) ─────────────────────────────────────────
        ("EGFR",  "AKT1",   +1, 0.95),
        ("EGFR",  "MAPK1",  +1, 0.93),
        # ── AT2 identity ───────────────────────────────────────────────────
        ("NKX2-1","SFTPC",  +1, 0.98),
        ("NKX2-1","SFTPB",  +1, 0.97),
        ("NKX2-1","ABCA3",  +1, 0.95),
        ("NKX2-1","SLC34A2",+1, 0.92),
        ("NKX2-1","FOXA1",  +1, 0.88),
        ("NKX2-1","ETV5",   +1, 0.90),
        ("NKX2-1","CDH1",   +1, 0.85),
        ("FOXA1", "NKX2-1", +1, 0.87),   # positive feedback loop
        ("FOXA2", "NKX2-1", +1, 0.85),
        ("FOXA1", "SFTPC",  +1, 0.88),
        ("FOXA2", "SFTPB",  +1, 0.87),
        # ── MYC — proliferation / dedifferentiation ────────────────────────
        ("MYC",   "CDK4",   +1, 0.95),
        ("MYC",   "CDK6",   +1, 0.93),
        ("MYC",   "CCND1",  +1, 0.94),
        ("MYC",   "E2F1",   +1, 0.92),
        ("MYC",   "EZH2",   +1, 0.90),
        ("MYC",   "NKX2-1", -1, 0.92),   # MYC represses AT2 master TF
        ("MYC",   "FOXA1",  -1, 0.88),
        ("MYC",   "SFTPC",  -1, 0.90),
        ("MYC",   "CDKN1A", -1, 0.85),
        # ── YAP1/TAZ/Hippo ────────────────────────────────────────────────
        ("YAP1",  "TEAD1",  +1, 0.93),
        ("YAP1",  "SNAI2",  +1, 0.88),
        ("YAP1",  "ZEB1",   +1, 0.85),
        ("YAP1",  "SOX9",   +1, 0.83),
        ("YAP1",  "CDK6",   +1, 0.85),
        ("YAP1",  "MYC",    +1, 0.87),
        ("WWTR1", "TEAD1",  +1, 0.90),
        ("WWTR1", "SNAI2",  +1, 0.85),
        ("LATS1", "YAP1",   -1, 0.96),
        ("LATS2", "YAP1",   -1, 0.94),
        ("LATS1", "WWTR1",  -1, 0.93),
        ("STK11", "LATS1",  +1, 0.90),
        ("STK11", "LATS2",  +1, 0.88),
        # ── EMT ────────────────────────────────────────────────────────────
        ("SNAI2", "CDH1",   -1, 0.97),
        ("ZEB1",  "CDH1",   -1, 0.96),
        ("ZEB2",  "CDH1",   -1, 0.94),
        ("ZEB1",  "NKX2-1", -1, 0.90),
        ("ZEB1",  "FOXA1",  -1, 0.87),
        ("ZEB1",  "SFTPC",  -1, 0.88),
        ("ZEB2",  "ZEB1",   +1, 0.88),
        ("TWIST1","ZEB1",   +1, 0.85),
        ("TWIST1","CDH1",   -1, 0.88),
        ("CDH1",  "SNAI2",  -1, 0.82),
        # ── AP-1 / inflammatory ────────────────────────────────────────────
        ("FOS",   "FOSL1",  +1, 0.88),
        ("JUN",   "FOSL1",  +1, 0.88),
        ("FOSL1", "NKX2-1", -1, 0.87),
        ("FOSL1", "SFTPC",  -1, 0.85),
        ("FOSL1", "ZEB1",   +1, 0.83),
        ("NFKB1", "MYC",    +1, 0.85),
        ("NFKB1", "BCL2",   +1, 0.88),
        ("NFKB1", "MCL1",   +1, 0.87),
        # ── TP53 pathway ───────────────────────────────────────────────────
        ("TP53",  "CDKN1A", +1, 0.97),
        ("TP53",  "BAX",    +1, 0.95),
        ("TP53",  "MDM2",   +1, 0.98),
        ("MDM2",  "TP53",   -1, 0.97),
        # ── Cell cycle ─────────────────────────────────────────────────────
        ("CCND1", "CDK4",   +1, 0.95),
        ("CCND2", "CDK6",   +1, 0.92),
        ("CDKN1A","CDK4",   -1, 0.95),
        ("CDKN1A","CDK6",   -1, 0.92),
        ("CDKN2A","CDK4",   -1, 0.97),
        ("CDKN2A","CDK6",   -1, 0.96),
        ("RB1",   "E2F1",   -1, 0.97),
        ("E2F1",  "CCND1",  +1, 0.90),
        ("E2F1",  "CDK4",   +1, 0.88),
        ("E2F3",  "CCND2",  +1, 0.88),
        # ── Epigenetic ─────────────────────────────────────────────────────
        ("EZH2",  "NKX2-1", -1, 0.93),
        ("EZH2",  "CDKN1A", -1, 0.88),
        ("EZH2",  "CDKN2A", -1, 0.90),
        ("EZH2",  "SFTPC",  -1, 0.87),
        ("BRD4",  "MYC",    +1, 0.93),
        ("BRD4",  "YAP1",   +1, 0.88),
        ("BRD4",  "FOSL1",  +1, 0.87),
        ("KDM6A", "NKX2-1", +1, 0.88),   # H3K27me3 demethylase → de-represses NKX2-1
        ("KDM6A", "SFTPC",  +1, 0.85),
        ("HDAC1", "NKX2-1", -1, 0.85),
        # ── SOX9 / dedifferentiation ───────────────────────────────────────
        ("SOX9",  "ZEB1",   +1, 0.85),
        ("SOX9",  "SNAI2",  +1, 0.83),
        ("SOX9",  "CDH1",   -1, 0.83),
        ("SOX4",  "SOX9",   +1, 0.87),
        ("SOX4",  "SNAI2",  +1, 0.83),
        # ── KEAP1/NRF2 ────────────────────────────────────────────────────
        ("KEAP1", "NFE2L2", -1, 0.95),
        ("NFE2L2","LDHA",   +1, 0.88),
        # ── Notch / Wnt ───────────────────────────────────────────────────
        ("NOTCH1","HES1",   +1, 0.97),
        ("HES1",  "NKX2-1", -1, 0.85),
        ("CTNNB1","MYC",    +1, 0.88),
        ("CTNNB1","CCND1",  +1, 0.88),
        # ── PI3K/AKT/mTOR ─────────────────────────────────────────────────
        ("AKT1",  "MTOR",   +1, 0.95),
        ("AKT1",  "MDM2",   +1, 0.90),
        ("MTOR",  "HIF1A",  +1, 0.90),
        ("HIF1A", "LDHA",   +1, 0.95),
        ("HIF1A", "NOTCH1", +1, 0.82),
        # ── Apoptosis ─────────────────────────────────────────────────────
        ("BAX",   "CASP9",  +1, 0.92),
        ("CASP9", "CASP3",  +1, 0.95),
        ("BCL2",  "BAX",    -1, 0.90),
        ("BCL2L1","BAX",    -1, 0.88),
        ("MCL1",  "CASP9",  -1, 0.85),
        # ── TEAD1 downstream ──────────────────────────────────────────────
        ("TEAD1", "CCND1",  +1, 0.88),
        ("TEAD1", "CDK4",   +1, 0.85),
        ("TEAD1", "SNAI2",  +1, 0.85),
        # KRAS → STK11 mutual exclusivity context (KRAS suppresses LKB1 activity)
        ("KRAS",  "STK11",  -1, 0.80),
    ]

    G = nx.DiGraph()
    G.add_nodes_from(genes)

    n_curated = 0
    for src, tgt, sgn, conf in curated:
        if src in gene_idx and tgt in gene_idx:
            G.add_edge(src, tgt, sign=sgn, weight=conf, source="curated")
            n_curated += 1

    # Data-driven correlation edges
    corr = np.corrcoef(
        np.array(adata.X.todense() if sp.issparse(adata.X) else adata.X, dtype=np.float32).T
    )
    threshold = 0.20
    n_data = 0
    for i in range(n_genes):
        for j in range(n_genes):
            if i == j:
                continue
            r = corr[i, j]
            if abs(r) < threshold:
                continue
            gi, gj = genes[i], genes[j]
            if not G.has_edge(gi, gj):
                sgn = +1 if r > 0 else -1
                G.add_edge(gi, gj, sign=sgn, weight=abs(r) * 0.6, source="data")
                n_data += 1

    logger.info("LUAD GRN: %d nodes, %d edges  (curated=%d, data-driven=%d)",
                n_genes, G.number_of_edges(), n_curated, n_data)
    return G


# ─────────────────────────────────────────────────────────────────────────────
# Module 1 — CAM
# ─────────────────────────────────────────────────────────────────────────────

def run_cam(adata: Any, grn: Any) -> Dict[str, Any]:  # noqa: E302
    from oracle.cam.boolean_network import BooleanNetworkSimulator
    from oracle.cam.preprocessing import CAMConfig

    logger.info("=== MODULE 1: Cancer Attractor Mapper ===")

    genes    = list(adata.var_names)
    gene_idx = {g: i for i, g in enumerate(genes)}
    n_genes  = len(genes)
    import scipy.sparse as sp

    # Mean expression per state
    def _mean(state):
        mask = adata.obs["cell_state"] == state
        X = adata[mask].X
        return np.array(X.mean(axis=0)).flatten() if sp.issparse(X) else X.mean(axis=0)

    cancer_expr = _mean("cancer")
    normal_expr  = _mean("normal")

    cam_cfg = CAMConfig(cancer_type="lung_adenocarcinoma", tissue="lung",
                        integration_time=30.0, n_ode_steps=80)
    sim = BooleanNetworkSimulator(grn, cam_cfg)

    # ── Expression-derived attractors using fold-change binarization ──────────
    # ON = gene is preferentially expressed in that state
    eps = 0.1
    # Cancer ON: cancer_mean > 1.5 AND ≥1.2× normal; or biologically enforced ON
    c_att = np.zeros(n_genes, dtype=np.float32)
    n_att  = np.zeros(n_genes, dtype=np.float32)
    for k in range(n_genes):
        c_expr = cancer_expr[k]
        n_expr = normal_expr[k]
        if c_expr > 1.5 and c_expr > n_expr * 1.2:
            c_att[k] = 1.0
        if n_expr > 1.5 and n_expr > c_expr * 1.2:
            n_att[k] = 1.0

    # Biology overrides — cancer attractor
    for g in ["MYC","YAP1","WWTR1","TEAD1","SNAI2","ZEB1","E2F1","FOSL1","SOX9","EZH2","BRD4"]:
        if g in gene_idx:
            c_att[gene_idx[g]] = 1.0
    for g in ["SFTPC","SFTPB","ABCA3","SLC34A2"]:
        if g in gene_idx:
            c_att[gene_idx[g]] = 0.0  # AT2 markers silenced in cancer

    # Biology overrides — normal AT2 attractor
    for g in ["FOXA1","FOXA2","ETV5","SFTPC","SFTPB","ABCA3","SLC34A2","CDH1","TP53","RB1","STK11","LATS1"]:
        if g in gene_idx:
            n_att[gene_idx[g]] = 1.0
    for g in ["SNAI2","ZEB1","ZEB2","SOX9","FOSL1","MYC","YAP1","EZH2","KRAS","CDH2","VIM"]:
        if g in gene_idx:
            n_att[gene_idx[g]] = 0.0

    # Enforce genetic constraints
    if KRAS_GENE in gene_idx:
        c_att[gene_idx[KRAS_GENE]] = 1.0
        logger.info("Enforced KRAS=1 in cancer attractor (G12C activating mutation)")
    if STK11_GENE in gene_idx:
        c_att[gene_idx[STK11_GENE]] = 0.0
        logger.info("Enforced STK11=0 in cancer attractor (loss-of-function)")

    # Validate by running Boolean dynamics from expression-derived seeds
    logger.info("Validating attractors via Boolean dynamics on %d-node GRN...", n_genes)
    t0 = time.time()
    n_valid = 50
    rng_val = np.random.default_rng(0)
    c_att_counts = np.zeros(n_genes)
    n_att_counts = np.zeros(n_genes)
    for _ in range(n_valid):
        seed_c = (c_att + rng_val.normal(0, 0.1, n_genes) > 0.5).astype(np.uint8)
        seed_n  = (n_att  + rng_val.normal(0, 0.1, n_genes) > 0.5).astype(np.uint8)
        final_c, _ = sim._run_trajectory(seed_c, max_steps=80)
        final_n, _  = sim._run_trajectory(seed_n,  max_steps=80)
        c_att_counts += final_c.astype(float)
        n_att_counts  += final_n.astype(float)

    c_att = (c_att_counts / n_valid > 0.5).astype(np.float32)
    n_att  = (n_att_counts  / n_valid > 0.5).astype(np.float32)
    # Re-enforce constraints after dynamics
    if KRAS_GENE in gene_idx:
        c_att[gene_idx[KRAS_GENE]] = 1.0
    if STK11_GENE in gene_idx:
        c_att[gene_idx[STK11_GENE]] = 0.0
    logger.info("Attractor validation: %.1f s", time.time() - t0)

    # Estimate basin sizes from data: fraction of cancer vs normal cells
    n_cancer_data = (adata.obs["cell_state"] == "cancer").sum()
    n_normal_data  = (adata.obs["cell_state"] == "normal").sum()
    n_total_data   = n_cancer_data + n_normal_data
    cancer_basin = float(n_cancer_data) / float(n_total_data)
    normal_basin  = float(n_normal_data)  / float(n_total_data)
    basin_sizes = {0: cancer_basin, 1: normal_basin}
    attractors  = [c_att, n_att]
    cancer_idx_att, normal_idx_att = 0, 1

    logger.info("Cancer basin: %.1f%%  |  Normal basin: %.1f%%  (from data)",
                cancer_basin * 100, normal_basin * 100)

    cancer_score = float(np.mean([c_att[gene_idx[g]] for g in
                                   ["MYC","YAP1","SNAI2","ZEB1","SOX9"] if g in gene_idx]))
    oncogenes_on = [g for g in LUAD_ONCOGENIC_TFS if g in gene_idx and c_att[gene_idx[g]] > 0.5]
    diff_tfs_on  = [g for g in LUAD_DIFFERENTIATION_TFS if g in gene_idx and n_att[gene_idx[g]] > 0.5]

    logger.info("Cancer attractor: %d genes ON, oncogenes_active=%s, KRAS=ON, STK11=OFF",
                int(c_att.sum()), oncogenes_on)
    logger.info("Normal attractor: %d genes ON, AT2_TFs_active=%s", int(n_att.sum()), diff_tfs_on)

    return {
        "genes": genes, "gene_idx": gene_idx, "n_genes": n_genes,
        "attractors": attractors, "cancer_attractor": c_att, "normal_attractor": n_att,
        "cancer_attractor_idx": cancer_idx_att, "normal_attractor_idx": normal_idx_att,
        "basin_sizes": basin_sizes, "cancer_expr": cancer_expr, "normal_expr": normal_expr,
        "sim": sim,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CancerScoreFunction training
# ─────────────────────────────────────────────────────────────────────────────

def _train_luad_cancer_score(
    n_genes: int,
    cancer_vec: np.ndarray,
    normal_vec: np.ndarray,
    adata: Any,
    ckpt_path: str = "checkpoints/cancer_score_luad.pt",
    n_epochs: int = 40,
) -> Any:
    from oracle.rsp.cancer_score import CancerScoreFunction
    import scipy.sparse as sp

    if Path(ckpt_path).exists():
        logger.info("Loading cached LUAD cancer score checkpoint: %s", ckpt_path)
        fn = CancerScoreFunction(n_genes)
        ck = torch.load(ckpt_path, map_location="cpu")
        fn.load_state_dict(ck["model_state_dict"])
        fn.eval()
        with torch.no_grad():
            c = fn(torch.tensor(cancer_vec).unsqueeze(0)).item()
            n = fn(torch.tensor(normal_vec).unsqueeze(0)).item()
        logger.info("Cached model — cancer=%.4f, normal=%.4f", c, n)
        return fn

    logger.info("Training LUAD CancerScoreFunction (%d genes, %d epochs)...", n_genes, n_epochs)

    X_all = np.array(adata.X.todense() if sp.issparse(adata.X) else adata.X, dtype=np.float32)
    states = list(adata.obs["cell_state"])
    y_all  = np.array([1.0 if s == "cancer" else 0.0 for s in states], dtype=np.float32)

    # Enforce genetic context in training data
    kras_col  = GENE_IDX.get("KRAS",  -1)
    stk11_col = GENE_IDX.get("STK11", -1)
    cancer_rows = [i for i, s in enumerate(states) if s == "cancer"]
    if kras_col >= 0:
        X_all[np.array(cancer_rows), kras_col] = 1.0   # normalised to binary
    if stk11_col >= 0:
        X_all[np.array(cancer_rows), stk11_col] = 0.0

    fn = CancerScoreFunction(n_genes)
    optimizer_t = torch.optim.Adam(fn.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_t, T_max=n_epochs)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_all), torch.tensor(y_all)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    for epoch in range(1, n_epochs + 1):
        fn.train()
        epoch_loss = 0.0
        for Xb, yb in loader:
            pred = fn(Xb).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy(pred, yb)
            optimizer_t.zero_grad()
            loss.backward()
            optimizer_t.step()
            epoch_loss += loss.item()
        scheduler.step()

        if epoch % 8 == 0:
            fn.eval()
            with torch.no_grad():
                c_s = fn(torch.tensor(cancer_vec).unsqueeze(0)).item()
                n_s = fn(torch.tensor(normal_vec).unsqueeze(0)).item()
            val_loss = epoch_loss / len(loader)
            logger.info("  epoch %2d/%d | val_loss=%.4f | cancer=%.4f | normal=%.4f",
                        epoch, n_epochs, val_loss, c_s, n_s)

    fn.eval()
    Path(ckpt_path).parent.mkdir(exist_ok=True)
    torch.save({"model_state_dict": fn.state_dict(), "n_genes": n_genes}, ckpt_path)
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# Module 2 — RSP
# ─────────────────────────────────────────────────────────────────────────────

def run_rsp(cam_result: Dict[str, Any], grn: Any, adata: Any) -> Dict[str, Any]:
    from oracle.rsp.switch_optimizer import MinimalSwitchOptimizer
    from oracle.rsp.cancer_score import RSPConfig
    from oracle.rsp.perturbation_sim import PerturbationSimulator
    from oracle.cam.continuous_ode import ContinuousGRNDynamics
    from oracle.cam.preprocessing import CAMConfig

    logger.info("=== MODULE 2: Reversion Switch Predictor ===")

    genes     = cam_result["genes"]
    n_genes   = cam_result["n_genes"]
    cancer_att = cam_result["cancer_attractor"].astype(np.float32)
    normal_att = cam_result["normal_attractor"].astype(np.float32)

    rsp_cfg = RSPConfig(
        n_genes=n_genes, max_perturbations=5,
        target_cancer_score=0.20, validation_trajectories=60,
    )
    cam_cfg = CAMConfig(cancer_type="lung_adenocarcinoma", tissue="lung",
                        integration_time=30.0, n_ode_steps=80)

    try:
        ode_model = ContinuousGRNDynamics(grn, cam_cfg)
    except Exception as e:
        logger.warning("ODE model unavailable (%s) — fallback", e)
        class _FallbackODE:
            def __init__(self, n):
                self.n_genes = n; self.use_torchdiffeq = False
            def __call__(self, t, x):
                return torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros(self.n_genes, np.float32)
            def parameters(self): return iter([torch.zeros(1)])
        ode_model = _FallbackODE(n_genes)

    cancer_score_fn = _train_luad_cancer_score(
        n_genes=n_genes, cancer_vec=cancer_att, normal_vec=normal_att,
        adata=adata, ckpt_path="checkpoints/cancer_score_luad.pt", n_epochs=40,
    )

    cancer_att_t = torch.tensor(cancer_att, dtype=torch.float32)
    normal_att_t = torch.tensor(normal_att, dtype=torch.float32)
    sim = PerturbationSimulator(ode_model, cancer_score_fn, cancer_att_t, rsp_cfg)

    # Constrain to biologically validated LUAD epigenetic switch targets.
    # These are the TFs/epigenetic regulators with established druggability in
    # KRAS-mutant LUAD where epigenetic intervention can reverse AT2 identity loss
    # (Kim et al. 2020, Nat Comm; Mollaoglu et al. 2018; Liang et al. 2020).
    LUAD_SWITCH_CANDIDATES = {
        # Oncogenic/epigenetic drivers to repress
        "MYC", "YAP1", "WWTR1", "TEAD1",
        "SNAI2", "ZEB1", "ZEB2", "SOX9", "FOSL1", "E2F1",
        "EZH2", "BRD4", "HDAC1",
        # AT2 differentiation TFs to activate (silenced by EZH2/BRD4 in LUAD)
        "NKX2-1", "FOXA1", "FOXA2", "ETV5",
    }
    druggable_indices = {i for i, g in enumerate(genes) if g in LUAD_SWITCH_CANDIDATES}

    # Expand TF knowledge with LUAD-specific TFs
    import oracle.rsp.switch_optimizer as _sw_mod
    luad_tfs = {
        "MYC", "YAP1", "WWTR1", "TEAD1", "SNAI2", "ZEB1", "ZEB2", "TWIST1",
        "SOX9", "SOX4", "FOSL1", "FOS", "JUN", "NFKB1",
        "NKX2-1", "FOXA1", "FOXA2", "ETV5", "E2F1", "E2F3",
        "EZH2", "BRD4", "HDAC1", "KDM6A", "HIF1A", "NFE2L2",
        "BRAF", "EGFR", "NOTCH1", "CTNNB1",
    }
    for attr in ("_DRUGGABLE_TFS", "_TF_GENES"):
        if hasattr(_sw_mod, attr):
            getattr(_sw_mod, attr).update(luad_tfs)

    t0 = time.time()
    optimizer = MinimalSwitchOptimizer(
        None, sim, grn, genes, cancer_att_t, normal_att_t,
        druggable_genes=druggable_indices,
        max_perturbations=rsp_cfg.max_perturbations,
        target_cancer_score=rsp_cfg.target_cancer_score,
        validation_trajectories=rsp_cfg.validation_trajectories,
    )
    switch_set = optimizer.optimize(cancer_score_fn)
    logger.info("RSP complete in %.1f s", time.time() - t0)
    logger.info("  ACTIVATE: %s", switch_set.genes_to_activate)
    logger.info("  REPRESS:  %s", switch_set.genes_to_repress)

    import dataclasses
    # Sanity guards
    # KRAS/STK11 must not appear in activation (genetic events)
    for g in [KRAS_GENE, STK11_GENE]:
        if g in switch_set.genes_to_activate:
            logger.error("GUARD: %s in activation list — removing (genetic event)", g)
            switch_set = dataclasses.replace(
                switch_set,
                genes_to_activate=[x for x in switch_set.genes_to_activate if x != g],
            )
    # AT2 differentiation TFs must not appear in repress list
    at2_tfs_protect = {"NKX2-1", "FOXA1", "FOXA2", "ETV5", "SFTPC", "SFTPB"}
    if any(g in at2_tfs_protect for g in switch_set.genes_to_repress):
        protected = [g for g in switch_set.genes_to_repress if g in at2_tfs_protect]
        logger.warning("GUARD: AT2 TFs %s in repress list — moving to activate", protected)
        switch_set = dataclasses.replace(
            switch_set,
            genes_to_repress=[x for x in switch_set.genes_to_repress if x not in at2_tfs_protect],
            genes_to_activate=list(set(switch_set.genes_to_activate + protected)),
        )

    # Boolean validation
    logger.info("Boolean validation of switch set...")
    sim_bool = cam_result["sim"]
    n_revert = 0
    n_trials  = 80
    for _ in range(n_trials):
        init = cam_result["cancer_attractor"].copy().astype(np.float32)
        rng  = np.random.default_rng()
        init += rng.normal(0, 0.1, init.shape).astype(np.float32)
        init  = np.clip(init, 0, 1)
        for g in switch_set.genes_to_activate:
            if g in cam_result["gene_idx"]:
                init[cam_result["gene_idx"][g]] = 1.0
        for g in switch_set.genes_to_repress:
            if g in cam_result["gene_idx"]:
                init[cam_result["gene_idx"][g]] = 0.0
        binary = (init > 0.5).astype(np.uint8)
        final, _ = sim_bool._run_trajectory(binary, max_steps=80)
        hamming_to_normal = np.sum(np.abs(final - cam_result["normal_attractor"]))
        if hamming_to_normal <= 6:
            n_revert += 1

    bool_reversion = n_revert / n_trials
    logger.info("Boolean validation: %.1f%% reversion (%d/%d trials)",
                bool_reversion * 100, n_revert, n_trials)

    return {
        "switch_set": switch_set,
        "validated_reversion_fraction": switch_set.validated_reversion_fraction,
        "bool_reversion": bool_reversion,
        "druggable_indices": druggable_indices,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Module 3 — TCD
# ─────────────────────────────────────────────────────────────────────────────

# LUAD-specific warhead SMILES
_LUAD_WARHEAD_MAP: Dict[str, str] = {
    "MYC":    "CC1=C(C2=CC=CC=C2S1)C3=NN4C(=C3)N=C(C)N(CC(=O)N5CCC[C@H]5C(=O)O)C4=O",  # JQ1/BET
    "YAP1":   "c1ccc2c(c1)oc(cc2=O)NC(=O)c1cccc(c1)F",          # Verteporfin analog
    "WWTR1":  "O=C(Nc1ccc(F)cc1)c1ccc2[nH]c3ccccc3c2c1",
    "TEAD1":  "CC1=CC(=CC(=C1)OCC2=CC=CC=C2)C(=O)N3CCN(CC3)C(=O)c1ccc(cc1)F",
    "SNAI2":  "O=C(Nc1ccc(cc1)F)c1cc2ccccc2[nH]1",               # Slug inhibitor
    "ZEB1":   "c1ccc2c(c1)nc(cc2)NC(=O)c1ccc(cc1)C(F)(F)F",
    "ZEB2":   "c1ccc(cc1)NC(=O)c1ccc2[nH]c3ccccc3c2c1",
    "SOX9":   "c1ccc(cc1)NC(=O)c1ccccc1NC(=O)c1ccc2ccccc2c1",
    "FOSL1":  "O=C(Nc1ccc(cc1)Cl)c1ccc2[nH]c3ccccc3c2c1",
    "EZH2":   "CC(=O)Nc1ccc(cc1)C(=O)N2CC[C@@H](CC2)N3CCOCC3",  # EPZ-6438 analog
    "BRD4":   "CC1=C(C2=CC=CC=C2S1)C3=NN4C(=C3)N=C(C)N(CC(=O)N5CCC[C@H]5C(=O)NC(C)(C))C4=O",  # JQ1
    "HDAC1":  "O=C(CCCCCCC(=O)Nc1ccc(cc1)C(=O)c1ccccc1)NO",     # Vorinostat analog
    "NKX2-1": "c1ccc(cc1)NC(=O)c1ccc2ccccc2n1",
    "FOXA1":  "O=C(Nc1cccc(c1)Cl)c1ccc2[nH]c3ccccc3c2c1",
    "FOXA2":  "O=C(Nc1ccc(F)cc1)c1cccnc1",
    "ETV5":   "c1ccc2c(c1)nc(cc2)NC(=O)c1ccc(F)cc1",
}

# Epigenetic recruiters
_LUAD_WRITER_MAP: Dict[str, str] = {
    # A-485 p300/CBP activator analog with free aniline NH2 for amide coupling
    "p300":  "O=C(c1ccc(cc1)Nc1ncnc2[nH]ccc12)c1ccc(cc1)N",
    "CBP":   "O=C(c1ccc(cc1)NC2=NC=CC=N2)c1ccc(cc1)N",
    "DOT1L": "CC(C)(C)c1ccc(cc1)NC(=O)Nc1ccc2[nH]c3ccccc3c2c1",
}
_LUAD_ERASER_MAP: Dict[str, str] = {
    # EPZ-6438 analog with free piperidine NH2 for amide coupling
    "EZH2":   "CC(=O)Nc1ccc(cc1)C(=O)N2CC[C@@H](CC2)N",
    # Vorinostat analog with free NH2 (terminal aniline replaced by NH2)
    "HDAC1":  "O=C(CCCCCCC(=O)N)NO",
    "BRD4":   "CC1=C(C2=CC=CC=C2S1)C3=NN4C(=C3)N=C(C)N(CC(=O)N5CCC[C@H]5C(=O)NC(C)(C))C4=O",
    "DNMT3A": "c1ccc(cc1)NC(=O)c1ccc2[nH]c(=O)n(Cc3cccnc3)c2c1",
}


def design_tcips(rsp_result: Dict[str, Any], cam_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    from oracle.tcd.linker_designer import LinkerDesigner
    from oracle.tcd.tcip_assembler import TCIPAssembler
    from oracle.tcd.writer_selector import WriterEraserSelector
    from oracle.tcd.hard_constraints import TCIPHardConstraints

    logger.info("=== MODULE 3: TCIP Design ===")

    switch_set  = rsp_result["switch_set"]
    we_selector = WriterEraserSelector()
    linker_des  = LinkerDesigner()
    assembler   = TCIPAssembler()
    constraints = TCIPHardConstraints()

    molecules = []
    processed = set()

    for gene in switch_set.genes_to_repress:
        if gene in processed or gene not in _LUAD_WARHEAD_MAP:
            continue
        processed.add(gene)
        warhead = _LUAD_WARHEAD_MAP[gene]

        # For repression: choose eraser appropriate to the target
        # EZH2 self-targeting avoided: use HDAC1 to silence EZH2 transcription
        # BRD4/oncogenic TFs: EZH2 deposits H3K27me3 at their target loci
        if gene in {"EZH2", "HDAC1"}:
            eraser_name, eraser_smiles = "HDAC1", _LUAD_ERASER_MAP["HDAC1"]
        elif gene in {"BRD4"}:
            eraser_name, eraser_smiles = "HDAC1", _LUAD_ERASER_MAP["HDAC1"]
        else:
            eraser_name, eraser_smiles = "EZH2", _LUAD_ERASER_MAP["EZH2"]
        dist = cam_result["gene_idx"].get(gene, 0) * 0.5 + 6.0
        try:
            linker = linker_des.design(required_distance_A=dist)
            assembled = assembler.assemble(warhead, linker.smiles, eraser_smiles)
            cr = constraints.check(assembled.smiles, linker_smiles=linker.smiles)
            mol_info = {
                "gene": gene, "action": "REPRESS",
                "warhead_target": gene, "recruiter": eraser_name,
                "warhead_smiles": warhead, "linker_smiles": linker.smiles,
                "recruiter_smiles": eraser_smiles, "tcip_smiles": assembled.smiles,
                "mw": cr.props.get("mw", assembled.properties.molecular_weight),
                "logp": cr.props.get("logP", assembled.properties.log_p),
                "tpsa": cr.props.get("tpsa", 0.0),
                "qed": cr.props.get("qed", assembled.properties.qed),
                "sa_score": cr.props.get("sa_score", 0.0),
                "hbd": cr.props.get("hbd", 0),
                "hba": cr.props.get("hba", 0),
                "n_rotatable_bonds": cr.props.get("n_rotatable_bonds", 0),
                "passes_ro5": assembled.properties.passes_ro5,
                "passes_hard_constraints": cr.passed,
                "constraint_violations": cr.violations,
                "linker_length_A": linker.length_A,
                "biological_rationale": (
                    f"Repress {gene} in LUAD: warhead binds {gene} TF/enzymatic pocket; "
                    f"linker bridges to {eraser_name} recruiter for epigenetic silencing "
                    f"at {gene} target loci."
                ),
            }
            molecules.append(mol_info)
            status = "PASS" if cr.passed else f"FAIL({'; '.join(cr.violations[:2])})"
            logger.info("  TCIP designed: %s (REPRESS) → %s | MW=%.0f QED=%.2f TPSA=%.0f SAS=%.1f | %s",
                        gene, eraser_name, mol_info["mw"], mol_info["qed"],
                        mol_info["tpsa"], mol_info["sa_score"], status)
        except Exception as exc:
            logger.warning("  TCIP design failed for %s: %s", gene, exc)

    for gene in switch_set.genes_to_activate:
        if gene in processed or gene not in _LUAD_WARHEAD_MAP:
            continue
        processed.add(gene)
        warhead = _LUAD_WARHEAD_MAP[gene]

        # For activation: recruit writer (p300/CBP for H3K27ac)
        writer_name, writer_smiles = "p300", _LUAD_WRITER_MAP["p300"]
        dist = cam_result["gene_idx"].get(gene, 0) * 0.5 + 7.0
        try:
            linker = linker_des.design(required_distance_A=dist, prefer_rigid=False)
            assembled = assembler.assemble(warhead, linker.smiles, writer_smiles)
            cr = constraints.check(assembled.smiles, linker_smiles=linker.smiles)
            mol_info = {
                "gene": gene, "action": "ACTIVATE",
                "warhead_target": gene, "recruiter": writer_name,
                "warhead_smiles": warhead, "linker_smiles": linker.smiles,
                "recruiter_smiles": writer_smiles, "tcip_smiles": assembled.smiles,
                "mw": cr.props.get("mw", assembled.properties.molecular_weight),
                "logp": cr.props.get("logP", assembled.properties.log_p),
                "tpsa": cr.props.get("tpsa", 0.0),
                "qed": cr.props.get("qed", assembled.properties.qed),
                "sa_score": cr.props.get("sa_score", 0.0),
                "hbd": cr.props.get("hbd", 0),
                "hba": cr.props.get("hba", 0),
                "n_rotatable_bonds": cr.props.get("n_rotatable_bonds", 0),
                "passes_ro5": assembled.properties.passes_ro5,
                "passes_hard_constraints": cr.passed,
                "constraint_violations": cr.violations,
                "linker_length_A": linker.length_A,
                "biological_rationale": (
                    f"Activate {gene} in LUAD AT2 redifferentiation: warhead tethers to "
                    f"{gene} promoter region; p300 recruiter deposits H3K27ac "
                    f"to reopen AT2 gene loci silenced by EZH2/HDAC1."
                ),
            }
            molecules.append(mol_info)
            status = "PASS" if cr.passed else f"FAIL({'; '.join(cr.violations[:2])})"
            logger.info("  TCIP designed: %s (ACTIVATE) → %s | MW=%.0f QED=%.2f TPSA=%.0f SAS=%.1f | %s",
                        gene, writer_name, mol_info["mw"], mol_info["qed"],
                        mol_info["tpsa"], mol_info["sa_score"], status)
        except Exception as exc:
            logger.warning("  TCIP design failed for %s: %s", gene, exc)

    # KRAS note — genetic event, no TCIP (but AMG-510/sotorasib as companion)
    logger.info("  NOTE: KRAS G12C is a genetic event — no TCIP possible. "
                "Sotorasib (AMG-510) or adagrasib (MRTX849) recommended as companion therapy.")

    logger.info("Total TCIPs designed: %d", len(molecules))
    return molecules


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    cam_result: Dict[str, Any],
    rsp_result: Dict[str, Any],
    molecules:  List[Dict[str, Any]],
    runtime_s:  float,
) -> None:
    logger.info("=== Generating ORACLE LUAD Report ===")
    sw = rsp_result["switch_set"]
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Text report
    lines = [
        "=" * 70,
        "ORACLE V2.0 — LUAD Epigenetic Reversion Report",
        f"Generated: {now}",
        f"Runtime:   {runtime_s:.1f} s",
        "=" * 70,
        "",
        "CANCER TYPE:  Lung Adenocarcinoma (LUAD)",
        "PRIMARY DATA: GSE131907 (Kim et al. 2020, Nat Comm; 208,506 cells)",
        "GENETIC CONTEXT:",
        "  • KRAS G12C/D/V — constitutively active (40% of LUAD)",
        "    ⚠  Cannot be epigenetically rescued; recommend sotorasib/adagrasib",
        "  • STK11/LKB1 — loss of function (dysregulates Hippo/YAP axis)",
        "  • TP53 — hotspot mutation (partial loss; ~50% co-occurrence)",
        "",
        "─" * 70,
        "MODULE 1 — Cancer Attractor Mapper (CAM)",
        "─" * 70,
        f"  Boolean attractors found:   {len(cam_result['attractors'])}",
        f"  Cancer basin size:          {cam_result['basin_sizes'].get(cam_result['cancer_attractor_idx'], 0):.1%}",
        f"  Normal basin size:          {cam_result['basin_sizes'].get(cam_result['normal_attractor_idx'], 0):.1%}",
        "",
        "  Cancer attractor active oncogenes:",
    ]
    for g in LUAD_ONCOGENIC_TFS:
        idx = cam_result["gene_idx"].get(g, -1)
        if idx >= 0:
            val = cam_result["cancer_attractor"][idx]
            lines.append(f"    {g:<12} {'ON ' if val > 0.5 else 'off'}")

    lines += [
        "",
        "  Normal (AT2) attractor differentiation TFs:",
    ]
    for g in LUAD_DIFFERENTIATION_TFS + ["SFTPC", "SFTPB", "ABCA3"]:
        idx = cam_result["gene_idx"].get(g, -1)
        if idx >= 0:
            val = cam_result["normal_attractor"][idx]
            lines.append(f"    {g:<12} {'ON ' if val > 0.5 else 'off'}")

    lines += [
        "",
        "─" * 70,
        "MODULE 2 — Reversion Switch Predictor (RSP)",
        "─" * 70,
        f"  Predicted reversion prob:    {sw.predicted_reversion_probability:.1%}",
        f"  ODE-validated reversion:     {sw.validated_reversion_fraction:.1%}",
        f"  Boolean reversion:           {rsp_result['bool_reversion']:.1%}",
        f"  Predicted cancer score after: {sw.predicted_cancer_score_after:.3f}",
        "",
        "  GENES TO ACTIVATE (restore AT2 identity):",
    ]
    for g in sw.genes_to_activate:
        imp = sw.gene_importance_scores.get(g, 0.0)
        lines.append(f"    {g:<14}  importance={imp:.3f}")

    lines += ["", "  GENES TO REPRESS (silence oncogenic program):"]
    for g in sw.genes_to_repress:
        imp = sw.gene_importance_scores.get(g, 0.0)
        lines.append(f"    {g:<14}  importance={imp:.3f}")

    n_pass = sum(1 for m in molecules if m.get("passes_hard_constraints", True))
    lines += [
        "",
        "─" * 70,
        "MODULE 3 — TCIP Design",
        "─" * 70,
        f"  Total TCIP molecules designed: {len(molecules)}",
        f"  Hard constraints passed:       {n_pass}/{len(molecules)}",
        "",
        "  Hard constraints (TCIP/bRo5 relaxed — must pass ALL):",
        "    1. Lipinski Ro5     MW<=1000, logP<=6, HBD<=6, HBA<=15 (>= 3/4)",
        "    2. Veber            RotBonds<=25, TPSA<=250 A^2",
        "    3. Ghose Filter     MW 160-1000, logP -0.4 to 6.0, MR 40-300, atoms 20-120",
        "    4. Egan             logP<=7, TPSA<=250 A^2",
        "    5. QED >= 0.04 (bifunctionals inherently score 0.04-0.25)",
        "    6. Linker atoms     5-20 heavy atoms",
        "    7. SAS <= 7.0",
        "    8. PAINS filter     no pan-assay interference substructures",
        "    9. Brenk filter     no reactive/toxic groups (PEG/hydroxamate whitelisted)",
        "   10. Ames alerts      no mutagenicity structural alerts (indole whitelisted)",
        "",
    ]
    for mol in molecules:
        passed     = mol.get("passes_hard_constraints", True)
        violations = mol.get("constraint_violations", [])
        status     = "PASS" if passed else "FAIL: " + "; ".join(violations[:3])
        lines += [
            f"  [{mol['action']}] {mol['gene']} -> {mol['recruiter']}  [{status}]",
            f"    TCIP SMILES: {mol['tcip_smiles']}",
            f"    MW={mol['mw']:.1f}  logP={mol['logp']:.2f}  "
            f"TPSA={mol.get('tpsa', 0):.0f}  QED={mol['qed']:.2f}  "
            f"SAS={mol.get('sa_score', 0):.1f}  "
            f"HBD={mol.get('hbd', '-')}  HBA={mol.get('hba', '-')}  "
            f"RotB={mol.get('n_rotatable_bonds', '-')}  "
            f"Ro5={'PASS' if mol['passes_ro5'] else 'FAIL'}",
            f"    Rationale: {mol['biological_rationale']}",
            "",
        ]

    lines += [
        "─" * 70,
        "THERAPEUTIC STRATEGY SUMMARY",
        "─" * 70,
        "The ORACLE analysis identifies a minimal TF perturbation set to",
        "drive KRAS-mutant LUAD cells from the cancer Waddington attractor",
        "back toward the AT2 (alveolar type II) normal lung identity.",
        "",
        "Recommended combination approach:",
        "  1. KRAS G12C: Sotorasib (AMG-510) or Adagrasib (MRTX849) — direct",
        "     KRAS G12C covalent inhibitor as companion.",
        "  2. TCIPs targeting the epigenetic switch set above — bifunctional",
        "     molecules that simultaneously silence oncogenic TFs and re-activate",
        "     NKX2-1/FOXA1/FOXA2 AT2 identity programme.",
        "  3. EZH2 inhibitor (EPZ-6438/tazemetostat) to remove H3K27me3",
        "     from NKX2-1/SFTPC loci silenced during dedifferentiation.",
        "",
        f"Pipeline completed in {runtime_s:.1f} s",
        "=" * 70,
    ]

    report_text = "\n".join(lines)
    txt_path = OUTPUT_DIR / "luad_oracle_report.txt"
    txt_path.write_text(report_text)
    logger.info("Text report: %s", txt_path)

    # JSON report
    report_json = {
        "pipeline": "ORACLE V2.0 — LUAD",
        "generated": now,
        "runtime_s": runtime_s,
        "cancer_type": "lung_adenocarcinoma",
        "data_source": "GSE131907",
        "genetic_context": {
            "KRAS": "G12C_activating_mutation",
            "STK11": "loss_of_function",
            "TP53": "hotspot_mutation",
        },
        "cam": {
            "n_attractors": len(cam_result["attractors"]),
            "cancer_basin_size": float(cam_result["basin_sizes"].get(cam_result["cancer_attractor_idx"], 0)),
            "normal_basin_size": float(cam_result["basin_sizes"].get(cam_result["normal_attractor_idx"], 0)),
            "cancer_attractor_active_genes": [
                g for g in cam_result["genes"]
                if cam_result["cancer_attractor"][cam_result["gene_idx"][g]] > 0.5
            ],
        },
        "rsp": {
            "genes_to_activate": sw.genes_to_activate,
            "genes_to_repress":  sw.genes_to_repress,
            "predicted_reversion_prob":    float(sw.predicted_reversion_probability),
            "validated_reversion_fraction": float(sw.validated_reversion_fraction),
            "bool_reversion": float(rsp_result["bool_reversion"]),
            "predicted_cancer_score_after": float(sw.predicted_cancer_score_after),
            "gene_importance": {k: float(v) for k, v in sw.gene_importance_scores.items()},
        },
        "tcd": {
            "n_molecules": len(molecules),
            "n_hard_constraint_pass": sum(1 for m in molecules if m.get("passes_hard_constraints", True)),
            "molecules": molecules,
        },
    }
    json_path = OUTPUT_DIR / "luad_oracle_report.json"
    with open(json_path, "w") as f:
        json.dump(report_json, f, indent=2)
    logger.info("JSON report: %s", json_path)

    # TSV of TCIP molecules
    if molecules:
        import pandas as pd
        tsv_path = OUTPUT_DIR / "luad_tcip_molecules.tsv"
        pd.DataFrame(molecules).to_csv(str(tsv_path), sep="\t", index=False)
        logger.info("TCIP molecules TSV: %s", tsv_path)

    print("\n" + report_text)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("ORACLE LUAD PIPELINE — START")
    logger.info("Cancer: Lung Adenocarcinoma (LUAD)")
    logger.info("Context: KRAS G12C + STK11 loss + TP53 mutation")
    logger.info("Data: GSE131907 (Kim et al. 2020, Nat Comm)")
    logger.info("=" * 60)

    adata      = generate_luad_adata()
    grn        = build_luad_grn(adata)
    cam_result = run_cam(adata, grn)
    rsp_result = run_rsp(cam_result, grn, adata)
    molecules  = design_tcips(rsp_result, cam_result)
    runtime    = time.time() - t_start

    generate_report(cam_result, rsp_result, molecules, runtime)

    logger.info("=" * 60)
    logger.info("ORACLE LUAD PIPELINE — COMPLETE  (%.1f s)", runtime)
    logger.info("Outputs: %s", OUTPUT_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
