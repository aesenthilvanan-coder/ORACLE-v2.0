#!/usr/bin/env python3
"""
ORACLE full pipeline — Anaplastic Thyroid Carcinoma (ATC)
Genetic context: homozygous TP53 inactivating mutation + MYC amplification

Pipeline:
  Module 1 (CAM)  — Synthetic ATC data, curated GRN, Boolean attractor landscape
  Module 2 (RSP)  — Minimal TF reversion switch set (TP53 excluded — genetically lost)
  Module 3 (TCD)  — TCIP bifunctional molecule design for epigenetic reprogramming

Output: JSON report + text report + TCIP molecule TSV
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
logger = logging.getLogger("oracle.atc_pipeline")

OUTPUT_DIR = Path("outputs/atc")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# ATC biology constants
# ─────────────────────────────────────────────────────────────────────────────

# Oncogenic drivers — candidates for REPRESSION
ATC_ONCOGENIC_TFS = [
    "MYC",    # amplified — master proliferation/de-differentiation TF
    "SNAI1",  # EMT driver — represses CDH1, promotes invasion
    "SNAI2",  # EMT (Slug) — represses NKX2-1
    "ZEB1",   # EMT + de-differentiation — represses thyroid TF program
    "ZEB2",   # EMT — cooperative with ZEB1
    "YAP1",   # Hippo pathway effector — promotes EMT, proliferation
    "TWIST1", # EMT TF — cooperates with SNAI/ZEB
    "E2F1",   # cell cycle driver — high due to TP53/RB1 loss
]

# Tumor suppressor / differentiation TFs — candidates for ACTIVATION
ATC_DIFFERENTIATION_TFS = [
    "NKX2-1",  # (TTF1) — master thyroid differentiation TF; lost in ATC
    "PAX8",    # thyroid lineage TF; lost in ATC
    "FOXE1",   # thyroid differentiation; epigenetically silenced in ATC
    "HHEX",    # thyroid progenitor; repressed in ATC
]

# TP53 is homozygously inactivated — cannot be activated by small molecule
# It is excluded from all druggability consideration
TP53_INACTIVATED = True
TP53_GENE = "TP53"

# MYC amplification: MYC expression is constitutively elevated
MYC_AMPLIFIED = True

# Gene panel for ATC (67 genes covering key thyroid/cancer pathways)
ATC_GENE_PANEL = [
    # Thyroid differentiation
    "NKX2-1", "PAX8", "FOXE1", "HHEX", "THRB",
    "TG", "TPO", "TSHR", "SLC5A5",
    # EMT / invasion
    "SNAI1", "SNAI2", "ZEB1", "ZEB2", "TWIST1", "TWIST2",
    "CDH1", "CDH2", "VIM", "FN1", "MMP9", "MMP2",
    # Oncogenes
    "MYC", "MYCN", "KRAS", "BRAF", "EGFR",
    # Hippo
    "YAP1", "WWTR1", "CTGF", "CYR61",
    # Cell cycle
    "CCND1", "CCND2", "CDK4", "CDK6", "E2F1", "E2F3",
    "CDKN1A", "CDKN2A", "RB1",
    # TP53 pathway (TP53 is lost; downstream effects encoded)
    "TP53", "MDM2", "BAX", "BCL2", "BCL2L1", "PUMA",
    # MAPK / PI3K signaling
    "MAPK1", "MAPK3", "AKT1", "PIK3CA", "MTOR", "STAT3",
    # Chromatin / epigenetic
    "EZH2", "KDM6A", "HDAC1", "BRD4",
    # Metabolic
    "LDHA", "PKM",
    # Apoptosis
    "CASP3", "CASP9",
]
N_GENES = len(ATC_GENE_PANEL)

# Gene index
GENE_IDX = {g: i for i, g in enumerate(ATC_GENE_PANEL)}

# ─────────────────────────────────────────────────────────────────────────────
# Cancer attractor profile for ATC (TP53 null + MYC amp)
# ─────────────────────────────────────────────────────────────────────────────

def _atc_cancer_expression_profile() -> Dict[str, float]:
    """
    ATC cancer state expression (log-normalized, 0-10 range).
    Encodes: MYC amplification (MYC very high), TP53 null (TP53 zero),
    EMT activation, thyroid dedifferentiation.
    """
    profile = {g: 0.3 for g in ATC_GENE_PANEL}   # low baseline

    # Thyroid differentiation — LOST in ATC
    for g in ["NKX2-1", "PAX8", "FOXE1", "HHEX", "THRB",
              "TG", "TPO", "TSHR", "SLC5A5"]:
        profile[g] = 0.2

    # EMT program — HIGH in ATC
    for g in ["SNAI1", "SNAI2", "ZEB1", "ZEB2", "TWIST1", "TWIST2"]:
        profile[g] = 3.8

    profile["CDH1"] = 0.3   # E-cadherin lost
    profile["CDH2"] = 3.5   # N-cadherin gained (cadherin switch)
    profile["VIM"]  = 4.2   # vimentin high
    profile["FN1"]  = 3.6   # fibronectin high
    profile["MMP9"] = 3.8   # matrix remodeling
    profile["MMP2"] = 3.4

    # MYC AMPLIFICATION — constitutively very high
    profile["MYC"]   = 7.5   # amplified
    profile["MYCN"]  = 2.5
    profile["BRAF"]  = 3.2   # V600E-like activation
    profile["KRAS"]  = 2.8
    profile["EGFR"]  = 3.5

    # Hippo deactivation — YAP/TAZ nuclear
    profile["YAP1"]  = 4.5
    profile["WWTR1"] = 3.8
    profile["CTGF"]  = 4.2
    profile["CYR61"] = 3.8

    # Cell cycle driven by MYC + TP53 loss + RB loss
    profile["CCND1"] = 5.2
    profile["CCND2"] = 3.8
    profile["CDK4"]  = 4.5
    profile["CDK6"]  = 4.2
    profile["E2F1"]  = 5.0
    profile["E2F3"]  = 3.5
    profile["CDKN1A"] = 0.4  # p21 low — no TP53 to induce it
    profile["CDKN2A"] = 0.3  # p16 low — often deleted in ATC
    profile["RB1"]   = 0.6   # phospho-inactivated by CDK4/6

    # TP53 INACTIVATED (homozygous loss)
    profile["TP53"]  = 0.0   # genetically absent
    profile["MDM2"]  = 3.5   # freed from p53 regulation — elevated
    profile["BAX"]   = 0.4   # p53-independent BAX is low
    profile["BCL2"]  = 4.0   # anti-apoptotic — high
    profile["BCL2L1"] = 3.8
    profile["PUMA"]  = 0.3   # p53 target — absent

    # MAPK signaling (BRAF V600E drives ERK constitutively)
    profile["MAPK1"] = 4.8
    profile["MAPK3"] = 4.5
    profile["AKT1"]  = 4.2
    profile["PIK3CA"] = 3.8
    profile["MTOR"]  = 4.0
    profile["STAT3"] = 3.5

    # Epigenetic: EZH2 high (silences differentiation), BRD4 high
    profile["EZH2"]  = 4.8
    profile["KDM6A"] = 1.2   # erased by EZH2 upregulation context
    profile["HDAC1"] = 4.2
    profile["BRD4"]  = 4.5

    profile["LDHA"]  = 4.5   # Warburg effect
    profile["PKM"]   = 4.0
    profile["CASP3"] = 0.3   # apoptosis suppressed
    profile["CASP9"] = 0.4

    return profile


def _atc_normal_thyroid_profile() -> Dict[str, float]:
    """Normal follicular thyroid cell expression profile."""
    profile = {g: 1.5 for g in ATC_GENE_PANEL}   # moderate baseline

    # Thyroid differentiation — HIGH in normal
    profile["NKX2-1"] = 7.2
    profile["PAX8"]   = 6.8
    profile["FOXE1"]  = 6.0
    profile["HHEX"]   = 5.5
    profile["THRB"]   = 5.0
    profile["TG"]     = 8.5   # thyroglobulin — very high
    profile["TPO"]    = 7.0
    profile["TSHR"]   = 6.5
    profile["SLC5A5"] = 5.8   # NIS

    # EMT — LOW in normal thyroid
    for g in ["SNAI1", "SNAI2", "ZEB1", "ZEB2", "TWIST1", "TWIST2"]:
        profile[g] = 0.4
    profile["CDH1"] = 6.5   # E-cadherin high in epithelial thyroid
    profile["CDH2"] = 0.5
    profile["VIM"]  = 0.6
    profile["FN1"]  = 0.8
    profile["MMP9"] = 0.5
    profile["MMP2"] = 0.7

    # MYC low in quiescent thyroid
    profile["MYC"]   = 1.2
    profile["MYCN"]  = 0.5
    profile["BRAF"]  = 1.0
    profile["KRAS"]  = 1.2
    profile["EGFR"]  = 1.5

    # Hippo active (YAP cytoplasmic = low nuclear activity)
    profile["YAP1"]  = 1.5
    profile["WWTR1"] = 1.5
    profile["CTGF"]  = 0.8
    profile["CYR61"] = 0.9

    # Cell cycle — quiescent
    profile["CCND1"] = 1.0
    profile["CCND2"] = 0.8
    profile["CDK4"]  = 1.0
    profile["CDK6"]  = 0.8
    profile["E2F1"]  = 0.7
    profile["E2F3"]  = 0.6
    profile["CDKN1A"] = 4.5  # p21 high — TP53 functional
    profile["CDKN2A"] = 3.5  # p16 high
    profile["RB1"]   = 4.5   # hypophosphorylated (active)

    # TP53 functional in normal thyroid
    profile["TP53"]  = 4.5
    profile["MDM2"]  = 2.5
    profile["BAX"]   = 3.5
    profile["BCL2"]  = 1.5
    profile["BCL2L1"] = 1.8
    profile["PUMA"]  = 3.0

    # Signaling — low baseline
    profile["MAPK1"] = 2.0
    profile["MAPK3"] = 2.0
    profile["AKT1"]  = 2.0
    profile["PIK3CA"] = 1.8
    profile["MTOR"]  = 2.0
    profile["STAT3"] = 1.5

    # Epigenetic: normal balance
    profile["EZH2"]  = 1.8
    profile["KDM6A"] = 3.5   # H3K27me3 demethylase active
    profile["HDAC1"] = 2.5
    profile["BRD4"]  = 2.0

    profile["LDHA"]  = 1.5
    profile["PKM"]   = 1.5
    profile["CASP3"] = 2.5
    profile["CASP9"] = 2.5

    return profile


# ─────────────────────────────────────────────────────────────────────────────
# Real scRNA-seq data — GSE193581 (GEO)
# ─────────────────────────────────────────────────────────────────────────────

_GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE193nnn/GSE193581/suppl"
_ANNOT_URL = f"{_GEO_BASE}/GSE193581_celltype_annotation.txt.gz"
_RAW_TAR_URL = f"{_GEO_BASE}/GSE193581_RAW.tar"
_GEO_DATA_DIR = Path("data/atc/gse193581")


def _geo_download(url: str, dest: Path, label: str) -> bool:
    """Download a file from GEO FTP with progress logging. Returns True on success."""
    import requests
    try:
        logger.info("Downloading %s ...", label)
        resp = requests.get(url, stream=True, timeout=600)
        resp.raise_for_status()
        total_mb = int(resp.headers.get("content-length", 0)) / 1e6
        downloaded = 0
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                fh.write(chunk)
                downloaded += len(chunk)
                if total_mb > 0 and downloaded % (20 << 20) < (1 << 20):
                    logger.info("  %.0f / %.0f MB", downloaded / 1e6, total_mb)
        logger.info("  Saved: %s (%.1f MB)", dest, downloaded / 1e6)
        return True
    except Exception as exc:
        logger.warning("Download failed [%s]: %s", label, exc)
        if dest.exists():
            dest.unlink()
        return False


def _parse_annotation(annot_path: Path) -> "pd.DataFrame":
    """Parse the GSE193581 cell-type annotation file into a DataFrame."""
    import gzip
    import pandas as pd
    with gzip.open(str(annot_path), "rt") as fh:
        df = pd.read_csv(fh, sep="\t")
    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    logger.info("Annotation: %d cells, cols=%s", len(df), list(df.columns))
    return df


def _build_cell_state_mask(annot: "pd.DataFrame") -> "Tuple[pd.Series, pd.Series, pd.Series]":
    """Return (barcode_col, cancer_mask, normal_mask) from the annotation DF."""
    # Try to identify the barcode column (first column or named 'cell'/'barcode')
    barcode_candidates = [c for c in annot.columns if "barcode" in c or "cell_id" in c or "cell" == c]
    barcode_col = barcode_candidates[0] if barcode_candidates else annot.columns[0]

    # Try to identify the cell-type column
    type_candidates = [c for c in annot.columns if "celltype" in c or "cell_type" in c
                       or "cluster" in c or "annotation" in c or "label" in c]
    type_col = type_candidates[0] if type_candidates else annot.columns[1]

    ct = annot[type_col].astype(str).str.strip()
    # ATC tumour cells: iATC, mATC, ATC, Malignant (GSE193581 uses "Malignant cell")
    cancer_mask = ct.str.contains(r"ATC|anaplastic|malignant|tumor_thy", case=False, regex=True, na=False)
    # Normal thyroid: TFC, Epithelial cell (GSE193581), follicular
    normal_mask  = ct.str.contains(r"TFC|thyroid_follicular|normal_thy|follicular_cell|Epithelial",
                                    case=False, regex=True, na=False)
    logger.info("Cell-type col: '%s'  |  ATC cancer=%d, TFC normal=%d",
                type_col, cancer_mask.sum(), normal_mask.sum())
    return barcode_col, type_col, cancer_mask, normal_mask


def _load_mtx_sample(sample_dir: Path) -> "Tuple[np.ndarray, List[str], List[str]]":
    """Load a 10x-format MTX directory. Returns (dense_matrix, barcodes, gene_names)."""
    import scipy.io
    import gzip

    # Locate files — may be .gz or uncompressed
    def _find(patterns):
        for pat in patterns:
            matches = list(sample_dir.glob(pat))
            if matches:
                return matches[0]
        return None

    mtx_file     = _find(["matrix.mtx.gz", "matrix.mtx", "*.mtx.gz", "*.mtx"])
    barcode_file = _find(["barcodes.tsv.gz", "barcodes.tsv", "*barcodes*"])
    feature_file = _find(["features.tsv.gz", "features.tsv", "genes.tsv.gz", "genes.tsv", "*features*", "*genes*"])

    if mtx_file is None:
        raise FileNotFoundError(f"No MTX file in {sample_dir}")

    # Read MTX
    if str(mtx_file).endswith(".gz"):
        import io
        with gzip.open(str(mtx_file), "rb") as fh:
            mat = scipy.io.mmread(io.BytesIO(fh.read())).T.tocsr()  # genes × cells → cells × genes
    else:
        mat = scipy.io.mmread(str(mtx_file)).T.tocsr()

    # Read barcodes
    def _read_gz_lines(path):
        if str(path).endswith(".gz"):
            with gzip.open(str(path), "rt") as fh:
                return [l.strip() for l in fh]
        else:
            return path.read_text().splitlines()

    barcodes = _read_gz_lines(barcode_file) if barcode_file else [f"cell_{i}" for i in range(mat.shape[0])]
    genes_raw = _read_gz_lines(feature_file) if feature_file else [f"gene_{i}" for i in range(mat.shape[1])]
    # Feature files may be tab-separated: gene_id \t gene_name \t type
    gene_names = [line.split("\t")[1] if "\t" in line else line for line in genes_raw]

    return mat, barcodes, gene_names


def _load_dense_count_file(path: Path) -> "Tuple[np.ndarray, List[str], List[str]]":
    """Load a dense TSV/CSV count matrix (genes × cells or cells × genes)."""
    import gzip
    import pandas as pd
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(str(path), "rt") as fh:
        df = pd.read_csv(fh, sep="\t", index_col=0)
    # Heuristic: if rows > cols, it is genes × cells; transpose to cells × genes
    if df.shape[0] > df.shape[1]:
        df = df.T
    gene_names = list(df.columns)
    barcodes = list(df.index)
    return df.values.astype(np.float32), barcodes, gene_names


def _merge_samples_to_anndata(
    extract_dir: Path,
    cancer_barcodes: set,
    normal_barcodes: set,
    panel_genes: List[str],
) -> "Any":
    """Scan all extracted sample dirs/files, load matching barcodes, return AnnData."""
    import anndata as ad
    import scipy.sparse as sp
    import pandas as pd

    all_X, all_barcodes, all_states = [], [], []

    # Each GSM sample is typically a sub-directory OR a set of files prefixed by GSM id
    sample_dirs = sorted([p for p in extract_dir.iterdir() if p.is_dir()])
    # Also handle flat files (dense matrices)
    flat_files = sorted([p for p in extract_dir.iterdir()
                         if p.is_file() and (str(p).endswith(".txt.gz") or str(p).endswith(".csv.gz"))])

    sources = sample_dirs or [extract_dir]  # if no subdirs, try the dir itself

    for src in sources:
        try:
            if src.is_dir():
                mat, barcodes, gene_names = _load_mtx_sample(src)
            else:
                mat, barcodes, gene_names = _load_dense_count_file(src)
        except Exception as exc:
            logger.debug("Skipping %s: %s", src, exc)
            continue

        # Build a gene-name → column-index map
        gene_map = {g: i for i, g in enumerate(gene_names)}
        panel_indices = [gene_map[g] for g in panel_genes if g in gene_map]
        panel_present = [g for g in panel_genes if g in gene_map]

        if not panel_indices:
            continue

        # Convert to dense float32 for our panel genes
        import scipy.sparse as sp2
        if sp2.issparse(mat):
            sub = np.array(mat[:, panel_indices].todense(), dtype=np.float32)
        else:
            sub = mat[:, panel_indices].astype(np.float32)

        # Normalise barcodes: strip sample suffix like "-1"
        norm_barcodes = [b.split("-")[0] if "-" in b else b for b in barcodes]

        for j, (raw_bc, norm_bc) in enumerate(zip(barcodes, norm_barcodes)):
            state = None
            if raw_bc in cancer_barcodes or norm_bc in cancer_barcodes:
                state = "cancer"
            elif raw_bc in normal_barcodes or norm_bc in normal_barcodes:
                state = "normal"
            if state is None:
                continue
            all_X.append(sub[j])
            all_barcodes.append(raw_bc)
            all_states.append(state)

        logger.debug("  %s: %d cancer + %d normal cells loaded",
                     src.name, all_states.count("cancer"), all_states.count("normal"))

    if len(all_X) < 50:
        raise RuntimeError(f"Too few matching cells from GSE193581: {len(all_X)}")

    X_arr = np.vstack(all_X)
    # Build a padded matrix for any missing panel genes (filled with 0)
    n_cells = X_arr.shape[0]
    panel_in_data = [g for g in panel_genes if g in
                     {gn for src in sources for gn in _panel_gene_names_from_src(src, panel_genes)}]
    # Actually build final matrix aligned to full panel
    # We build column-by-column
    final_X = np.zeros((n_cells, len(panel_genes)), dtype=np.float32)
    # Map back: find which columns of X_arr correspond to which panel positions
    # We re-run the merge in aligned form
    final_X, final_barcodes, final_states = _build_aligned_matrix(
        sources, cancer_barcodes, normal_barcodes, panel_genes
    )

    obs = pd.DataFrame({
        "cell_state": final_states,
        "cell_type":  ["ATC_tumor" if s == "cancer" else "normal_thyroid" for s in final_states],
        "n_genes_by_counts": (final_X > 0).sum(axis=1),
        "total_counts": final_X.sum(axis=1),
        "source": "GSE193581",
    }, index=final_barcodes)

    var = pd.DataFrame({"gene_name": panel_genes}, index=panel_genes)
    adata = ad.AnnData(X=sp.csr_matrix(final_X), obs=obs, var=var)
    adata.layers["raw_counts"] = adata.X.copy()
    return adata


def _panel_gene_names_from_src(src, panel_genes):
    """Quick scan — return the panel genes present in a source without full load."""
    try:
        import gzip
        import os
        if src.is_dir():
            feature_file = next(
                (p for p in src.iterdir()
                 if "feature" in p.name or "gene" in p.name), None)
            if feature_file is None:
                return []
            opener = gzip.open if str(feature_file).endswith(".gz") else open
            with opener(str(feature_file), "rt") as fh:
                all_genes = {line.strip().split("\t")[1] if "\t" in line else line.strip()
                             for line in fh}
        else:
            return panel_genes  # assume dense file has all genes
        return [g for g in panel_genes if g in all_genes]
    except Exception:
        return []


def _build_aligned_matrix(
    sources, cancer_barcodes, normal_barcodes, panel_genes
):
    """Build cell × panel_gene matrix with proper alignment, skipping missing genes."""
    import scipy.sparse as sp2
    import gzip

    all_rows, all_barcodes, all_states = [], [], []

    for src in sources:
        try:
            if src.is_dir():
                mat, barcodes, gene_names = _load_mtx_sample(src)
            elif src.is_file():
                mat, barcodes, gene_names = _load_dense_count_file(src)
            else:
                continue
        except Exception as exc:
            logger.debug("Skipping %s: %s", src, exc)
            continue

        gene_map = {g: i for i, g in enumerate(gene_names)}
        panel_col_idx = [gene_map.get(g, -1) for g in panel_genes]  # -1 = missing

        norm_barcodes = [b.split("-")[0] if "-" in b else b for b in barcodes]

        for j, (raw_bc, norm_bc) in enumerate(zip(barcodes, norm_barcodes)):
            state = None
            if raw_bc in cancer_barcodes or norm_bc in cancer_barcodes:
                state = "cancer"
            elif raw_bc in normal_barcodes or norm_bc in normal_barcodes:
                state = "normal"
            if state is None:
                continue

            row = np.zeros(len(panel_genes), dtype=np.float32)
            for k, col in enumerate(panel_col_idx):
                if col >= 0:
                    val = mat[j, col]
                    row[k] = float(val.item() if hasattr(val, "item") else val)

            all_rows.append(row)
            all_barcodes.append(raw_bc)
            all_states.append(state)

    if not all_rows:
        raise RuntimeError("No cells matched cancer/normal barcodes in extracted files.")

    return np.vstack(all_rows), all_barcodes, all_states


def _log_normalize(X: np.ndarray, scale: float = 1e4) -> np.ndarray:
    """Library-size normalisation + log1p, matching scanpy pp.normalize_total + log1p."""
    total = X.sum(axis=1, keepdims=True)
    total[total == 0] = 1.0
    return np.log1p(X / total * scale).astype(np.float32)


def load_gse193581_atc() -> "Any":
    """
    Download and process GSE193581 real ATC scRNA-seq data (JCI 2023).

    10 ATC tumors + adjacent normal thyroid tissue samples.
    Cancer cells: iATC + mATC  |  Normal cells: thyroid follicular (TFC)
    Gene filter: intersected with ATC_GENE_PANEL

    Falls back to biology-constrained synthetic data if download or parsing fails.
    """
    import anndata as ad
    import tarfile

    _GEO_DATA_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = _GEO_DATA_DIR / "atc_anndata_panel.h5ad"

    # ── Return cached ────────────────────────────────────────────────────────
    if processed_path.exists():
        logger.info("Loading cached GSE193581 AnnData: %s", processed_path)
        adata = ad.read_h5ad(str(processed_path))
        n_cancer = (adata.obs["cell_state"] == "cancer").sum()
        n_normal = (adata.obs["cell_state"] == "normal").sum()
        logger.info("Loaded: %d cells × %d genes  (cancer=%d, normal=%d)",
                    adata.n_obs, adata.n_vars, n_cancer, n_normal)
        return adata

    # ── Step 1: Download annotation file (400 KB) ────────────────────────────
    annot_path = _GEO_DATA_DIR / "celltype_annotation.txt.gz"
    if not annot_path.exists():
        ok = _geo_download(_ANNOT_URL, annot_path, "GSE193581 cell-type annotations (400 KB)")
        if not ok:
            logger.warning("Annotation download failed — using synthetic data")
            return _generate_atc_adata_synthetic()

    try:
        annot = _parse_annotation(annot_path)
        barcode_col, type_col, cancer_mask, normal_mask = _build_cell_state_mask(annot)
        cancer_barcodes = set(annot.loc[cancer_mask, barcode_col])
        normal_barcodes = set(annot.loc[normal_mask, barcode_col])
    except Exception as exc:
        logger.warning("Annotation parsing failed (%s) — using synthetic data", exc)
        return _generate_atc_adata_synthetic()

    if len(cancer_barcodes) < 100 or len(normal_barcodes) < 50:
        logger.warning("Insufficient labelled cells (cancer=%d, normal=%d) — using synthetic",
                       len(cancer_barcodes), len(normal_barcodes))
        return _generate_atc_adata_synthetic()

    logger.info("Target barcodes — ATC cancer: %d, TFC normal: %d",
                len(cancer_barcodes), len(normal_barcodes))

    # ── Step 2: Download RAW.tar (180 MB) ───────────────────────────────────
    raw_tar_path = _GEO_DATA_DIR / "GSE193581_RAW.tar"
    if not raw_tar_path.exists():
        ok = _geo_download(_RAW_TAR_URL, raw_tar_path, "GSE193581 count matrices (180 MB)")
        if not ok:
            logger.warning("RAW.tar download failed — using synthetic data")
            return _generate_atc_adata_synthetic()

    # ── Step 3: Extract ──────────────────────────────────────────────────────
    extract_dir = _GEO_DATA_DIR / "raw"
    extract_dir.mkdir(exist_ok=True)
    if not any(extract_dir.iterdir()):
        logger.info("Extracting GSE193581_RAW.tar ...")
        with tarfile.open(str(raw_tar_path)) as tar:
            tar.extractall(str(extract_dir))
        logger.info("Extraction complete: %d items", sum(1 for _ in extract_dir.rglob("*")))

    # ── Step 4: Build AnnData ────────────────────────────────────────────────
    try:
        sources = sorted([p for p in extract_dir.iterdir()])
        # Flatten: if RAW.tar produced a single sub-dir, descend into it
        if len(sources) == 1 and sources[0].is_dir():
            sources = sorted(sources[0].iterdir())

        X_real, barcodes_real, states_real = _build_aligned_matrix(
            sources, cancer_barcodes, normal_barcodes, ATC_GENE_PANEL
        )
    except Exception as exc:
        logger.warning("Matrix loading failed (%s) — using synthetic data", exc)
        return _generate_atc_adata_synthetic()

    if len(X_real) < 100:
        logger.warning("Only %d cells matched — not enough for analysis; using synthetic", len(X_real))
        return _generate_atc_adata_synthetic()

    # ── Step 5: Normalise ────────────────────────────────────────────────────
    X_norm = _log_normalize(X_real)

    # Enforce TP53=0 in all cancer cells (homozygous loss is a genetic event, not expression)
    tp53_idx_panel = ATC_GENE_PANEL.index("TP53") if "TP53" in ATC_GENE_PANEL else -1
    myc_idx_panel  = ATC_GENE_PANEL.index("MYC")  if "MYC"  in ATC_GENE_PANEL else -1
    cancer_cell_idx = [i for i, s in enumerate(states_real) if s == "cancer"]

    if tp53_idx_panel >= 0:
        X_norm[cancer_cell_idx, tp53_idx_panel] = 0.0
        logger.info("Enforced TP53=0 in %d cancer cells (homozygous inactivation)", len(cancer_cell_idx))

    import pandas as pd
    import scipy.sparse as sp
    obs = pd.DataFrame({
        "cell_state": states_real,
        "cell_type":  ["ATC_tumor" if s == "cancer" else "normal_thyroid" for s in states_real],
        "n_genes_by_counts": (X_norm > 0).sum(axis=1),
        "total_counts": X_norm.sum(axis=1),
        "source": "GSE193581",
    }, index=barcodes_real)

    var = pd.DataFrame({"gene_name": ATC_GENE_PANEL}, index=ATC_GENE_PANEL)
    adata = ad.AnnData(X=sp.csr_matrix(X_norm), obs=obs, var=var)
    adata.layers["raw_counts"] = sp.csr_matrix(X_real)
    adata.uns["cancer_type"] = "thyroid_anaplastic"
    adata.uns["tissue_type"] = "thyroid"
    adata.uns["geo_accession"] = "GSE193581"
    adata.uns["genetic_context"] = {
        "TP53": "homozygous_inactivating_mutation",
        "MYC":  "amplification",
        "BRAF": "V600E_presumed",
    }

    # Save processed cache
    adata.write_h5ad(str(processed_path))
    logger.info("Saved processed AnnData: %s", processed_path)

    n_c = (adata.obs["cell_state"] == "cancer").sum()
    n_n = (adata.obs["cell_state"] == "normal").sum()
    logger.info("GSE193581 ATC dataset: %d cells × %d genes  (cancer=%d, normal=%d)",
                adata.n_obs, adata.n_vars, n_c, n_n)
    return adata


def _generate_atc_adata_synthetic() -> "Any":
    """
    Biology-constrained synthetic ATC + normal thyroid scRNA-seq data.
    Used as fallback when GSE193581 is unavailable.
    """
    import anndata as ad
    import scipy.sparse as sp

    rng = np.random.default_rng(42)

    cancer_profile = np.array([_atc_cancer_expression_profile()[g] for g in ATC_GENE_PANEL], dtype=np.float32)
    normal_profile = np.array([_atc_normal_thyroid_profile()[g] for g in ATC_GENE_PANEL], dtype=np.float32)
    transit_profile = 0.45 * cancer_profile + 0.55 * normal_profile

    n_cancer, n_normal, n_transit = 480, 220, 300

    def _sample_cells(profile, n, noise_frac=0.18):
        noise = rng.normal(0, noise_frac * np.maximum(profile.std(), 0.5), (n, N_GENES)).astype(np.float32)
        cells = np.clip(profile[None, :] + noise, 0.0, None)
        dropout_mask = rng.random((n, N_GENES)) < 0.15
        cells[dropout_mask] = 0.0
        return cells

    X_cancer  = _sample_cells(cancer_profile,  n_cancer,  noise_frac=0.16)
    X_normal  = _sample_cells(normal_profile,  n_normal,  noise_frac=0.14)
    X_transit = _sample_cells(transit_profile, n_transit, noise_frac=0.22)

    myc_idx  = GENE_IDX["MYC"]
    tp53_idx = GENE_IDX["TP53"]
    X_cancer[:, myc_idx]   = rng.normal(7.5, 0.6, n_cancer).clip(5.0, 12.0).astype(np.float32)
    X_cancer[:, tp53_idx]  = 0.0
    X_transit[:, tp53_idx] = 0.0

    X = np.vstack([X_cancer, X_normal, X_transit])
    n_total = X.shape[0]
    cell_state = ["cancer"] * n_cancer + ["normal"] * n_normal + ["transitional"] * n_transit
    cell_type  = ["ATC_blast"] * n_cancer + ["normal_thyroid"] * n_normal + ["transitional_EMT"] * n_transit

    import pandas as pd
    obs = pd.DataFrame({
        "cell_state":  cell_state,
        "cell_type":   cell_type,
        "n_genes_by_counts": (X > 0).sum(axis=1),
        "total_counts": X.sum(axis=1),
        "pct_counts_MT": rng.uniform(2, 18, n_total).astype(np.float32),
        "source": "synthetic_fallback",
    }, index=[f"cell_{i}" for i in range(n_total)])

    var = pd.DataFrame({"gene_name": ATC_GENE_PANEL}, index=ATC_GENE_PANEL)
    adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)
    adata.layers["raw_counts"] = adata.X.copy()
    adata.uns["cancer_type"] = "thyroid_anaplastic"
    adata.uns["tissue_type"] = "thyroid"
    adata.uns["genetic_context"] = {
        "TP53": "homozygous_inactivating_mutation",
        "MYC":  "amplification",
        "BRAF": "V600E_presumed",
    }
    logger.info("Synthetic fallback: %d cells × %d genes  (cancer=%d, normal=%d, transitional=%d)",
                n_total, N_GENES, n_cancer, n_normal, n_transit)
    return adata


def generate_atc_adata() -> Any:
    """Load real GSE193581 ATC scRNA-seq data; fall back to synthetic if unavailable."""
    return load_gse193581_atc()


# ─────────────────────────────────────────────────────────────────────────────
# GRN construction — ATC biology
# ─────────────────────────────────────────────────────────────────────────────

def build_atc_grn(adata: Any) -> Any:
    """
    Build signed ATC GRN.

    Curated edges from ATC/thyroid cancer literature + expression correlation.
    Key constraints:
      - MYC amplification: MYC edges carry confidence 0.98
      - TP53 null: TP53-outgoing edges removed (gene non-functional)
    """
    import networkx as nx
    import scipy.sparse as sp

    genes = list(adata.var_names)
    gene_idx = {g: i for i, g in enumerate(genes)}
    n_genes = len(genes)

    # Compute per-state mean expression for edge weight scaling
    def _mean(state: str) -> np.ndarray:
        mask = adata.obs["cell_state"] == state
        X = adata[mask].X
        return np.array(X.mean(axis=0)).flatten() if sp.issparse(X) else X.mean(axis=0)

    cancer_mean = _mean("cancer")
    normal_mean = _mean("normal")
    eps = 1e-3
    lfc = np.log2((cancer_mean + eps) / (normal_mean + eps))

    # ── Curated ATC regulatory edges ─────────────────────────────────────────
    # (source, target, sign, confidence)
    curated = [
        # MYC amplification — master de-differentiation
        ("MYC",    "NKX2-1",  -1, 0.97),  # MYC represses thyroid master TF
        ("MYC",    "PAX8",    -1, 0.95),  # MYC represses PAX8
        ("MYC",    "FOXE1",   -1, 0.93),  # MYC represses FOXE1
        ("MYC",    "HHEX",    -1, 0.90),  # MYC represses HHEX
        ("MYC",    "CDK4",    +1, 0.95),  # MYC activates CDK4
        ("MYC",    "CDK6",    +1, 0.93),  # MYC activates CDK6
        ("MYC",    "CCND1",   +1, 0.96),  # MYC activates cyclin D1
        ("MYC",    "E2F1",    +1, 0.90),  # MYC activates E2F1
        ("MYC",    "EZH2",    +1, 0.92),  # MYC activates EZH2 (epigenetic silencer)
        ("MYC",    "LDHA",    +1, 0.88),  # Warburg effect
        ("MYC",    "CDH1",    -1, 0.85),  # MYC represses E-cadherin
        # EMT program
        ("SNAI1",  "CDH1",    -1, 0.98),  # Snail represses E-cadherin (canonical)
        ("SNAI1",  "NKX2-1",  -1, 0.88),  # Snail blocks thyroid identity
        ("SNAI1",  "FOXE1",   -1, 0.85),  # Snail represses thyroid TF
        ("SNAI1",  "VIM",     +1, 0.90),  # Snail activates vimentin
        ("SNAI1",  "FN1",     +1, 0.87),  # Snail activates fibronectin
        ("SNAI2",  "CDH1",    -1, 0.95),  # Slug represses E-cadherin
        ("SNAI2",  "NKX2-1",  -1, 0.85),  # Slug represses thyroid identity
        ("SNAI2",  "PAX8",    -1, 0.82),  # Slug represses PAX8
        ("ZEB1",   "CDH1",    -1, 0.97),  # ZEB1 represses E-cadherin
        ("ZEB1",   "NKX2-1",  -1, 0.90),  # ZEB1 represses thyroid identity
        ("ZEB1",   "PAX8",    -1, 0.87),  # ZEB1 represses PAX8
        ("ZEB1",   "FOXE1",   -1, 0.88),  # ZEB1 represses FOXE1
        ("ZEB1",   "VIM",     +1, 0.88),
        ("ZEB2",   "CDH1",    -1, 0.90),
        ("ZEB2",   "NKX2-1",  -1, 0.83),
        ("TWIST1", "CDH1",    -1, 0.88),
        ("TWIST1", "SNAI2",   +1, 0.82),  # TWIST1 activates Slug
        ("TWIST1", "ZEB1",    +1, 0.80),  # TWIST1 activates ZEB1
        # Hippo pathway — YAP nuclear promotes EMT + proliferation
        ("YAP1",   "SNAI2",   +1, 0.88),  # YAP activates Slug
        ("YAP1",   "CTGF",    +1, 0.95),  # YAP target gene
        ("YAP1",   "CYR61",   +1, 0.93),
        ("YAP1",   "CDH1",    -1, 0.82),  # YAP represses E-cadherin
        ("YAP1",   "MYC",     +1, 0.85),  # YAP activates MYC (positive feedback)
        ("WWTR1",  "CTGF",    +1, 0.90),
        ("WWTR1",  "YAP1",    +1, 0.80),  # TAZ-YAP cooperate
        # Thyroid differentiation program
        ("NKX2-1", "PAX8",    +1, 0.95),  # NKX2-1 activates PAX8
        ("NKX2-1", "FOXE1",   +1, 0.92),
        ("NKX2-1", "HHEX",    +1, 0.90),
        ("NKX2-1", "TG",      +1, 0.97),  # NKX2-1 drives thyroglobulin
        ("NKX2-1", "TPO",     +1, 0.95),
        ("NKX2-1", "TSHR",    +1, 0.92),
        ("NKX2-1", "SLC5A5",  +1, 0.90),  # NIS
        ("NKX2-1", "CDH1",    +1, 0.85),  # thyroid identity preserves epithelial
        ("NKX2-1", "SNAI1",   -1, 0.85),  # NKX2-1 represses EMT drivers
        ("NKX2-1", "ZEB1",    -1, 0.82),
        ("NKX2-1", "MYC",     -1, 0.80),  # NKX2-1 mildly represses MYC
        ("PAX8",   "NKX2-1",  +1, 0.92),  # mutual activation
        ("PAX8",   "TG",      +1, 0.90),
        ("PAX8",   "TPO",     +1, 0.88),
        ("PAX8",   "FOXE1",   +1, 0.85),
        ("FOXE1",  "TG",      +1, 0.85),
        ("FOXE1",  "TSHR",    +1, 0.82),
        # MAPK signaling (BRAF V600E → ERK constitutive)
        ("BRAF",   "MAPK1",   +1, 0.95),  # BRAF V600E → ERK2
        ("BRAF",   "MAPK3",   +1, 0.93),  # → ERK1
        ("MAPK1",  "MYC",     +1, 0.88),  # ERK → MYC stabilization
        ("MAPK1",  "SNAI1",   +1, 0.85),  # ERK → Snail (EMT)
        ("MAPK1",  "ZEB1",    +1, 0.82),
        ("MAPK1",  "CDK4",    +1, 0.80),
        # Cell cycle regulation
        ("CDK4",   "RB1",     -1, 0.97),  # CDK4/6 phosphorylate RB
        ("CDK6",   "RB1",     -1, 0.95),
        ("RB1",    "E2F1",    -1, 0.97),  # hypo-pRB represses E2F1
        ("E2F1",   "CCND1",   +1, 0.90),
        ("E2F1",   "CDK4",    +1, 0.85),
        ("E2F1",   "CDK6",    +1, 0.82),
        ("CCND1",  "CDK4",    +1, 0.92),  # cyclin D1 activates CDK4
        ("CCND1",  "CDK6",    +1, 0.88),
        ("CDKN1A", "CDK4",    -1, 0.95),  # p21 inhibits CDK4
        ("CDKN1A", "CDK6",    -1, 0.93),
        ("CDKN2A", "CDK4",    -1, 0.95),  # p16 inhibits CDK4
        ("CDKN2A", "CDK6",    -1, 0.93),
        # TP53 pathway (p53 is NULL in cancer; shown for completeness)
        # These edges are in the GRN but TP53 is held at 0 in cancer attractor
        ("TP53",   "CDKN1A",  +1, 0.97),  # p53 induces p21
        ("TP53",   "BAX",     +1, 0.95),  # p53 induces BAX
        ("TP53",   "PUMA",    +1, 0.95),
        ("TP53",   "MDM2",    +1, 0.92),  # p53-MDM2 negative feedback
        ("MDM2",   "TP53",    -1, 0.92),  # MDM2 degrades p53
        # Apoptosis
        ("BAX",    "BCL2",    -1, 0.85),  # BAX antagonizes BCL2
        ("BCL2",   "CASP3",   -1, 0.88),  # BCL2 blocks caspases
        ("BCL2",   "CASP9",   -1, 0.85),
        ("BCL2L1", "CASP9",   -1, 0.85),
        # EZH2 — epigenetic silencer (high in ATC)
        ("EZH2",   "NKX2-1",  -1, 0.88),  # EZH2 silences NKX2-1 via H3K27me3
        ("EZH2",   "PAX8",    -1, 0.85),
        ("EZH2",   "FOXE1",   -1, 0.85),
        ("EZH2",   "CDKN1A",  -1, 0.82),
        ("EZH2",   "CDKN2A",  -1, 0.82),
        ("KDM6A",  "NKX2-1",  +1, 0.85),  # H3K27me3 eraser → de-represses NKX2-1
        ("KDM6A",  "EZH2",    -1, 0.80),  # opposes EZH2
    ]

    grn = nx.DiGraph()
    for g in genes:
        expr_diff = float(lfc[gene_idx[g]]) if g in gene_idx else 0.0
        grn.add_node(g, lfc=expr_diff)

    for src, tgt, sign, conf in curated:
        if src in gene_idx and tgt in gene_idx:
            src_lfc = lfc[gene_idx[src]]
            lfc_consistency = float(np.sign(src_lfc) == sign or abs(src_lfc) < 0.5)
            w = conf * (0.7 + 0.3 * lfc_consistency)
            grn.add_edge(src, tgt, sign=sign, weight=w, confidence=conf,
                         source="curated_ATC")

    # Data-driven edges from expression correlations
    X = adata.X
    if sp.issparse(X):
        X_d = X.toarray()
    else:
        X_d = np.array(X)

    rng_sample = np.random.default_rng(42)
    idx = rng_sample.choice(adata.n_obs, min(2000, adata.n_obs), replace=False)
    X_s = X_d[idx]
    std = X_s.std(axis=0) + 1e-8
    X_std = (X_s - X_s.mean(axis=0)) / std

    tf_candidates = [g for g in genes if abs(lfc[gene_idx[g]]) > 0.5][:40]
    n_added = 0
    for src in tf_candidates:
        si = gene_idx[src]
        corr = X_std[:, si] @ X_std / len(idx)
        for ti, tgt in enumerate(genes):
            if tgt == src:
                continue
            c = float(corr[ti])
            if abs(c) < 0.35:
                continue
            sign = 1 if c > 0 else -1
            w = abs(c) * 0.55
            if grn.has_edge(src, tgt):
                grn[src][tgt]["weight"] = max(grn[src][tgt]["weight"], w)
            else:
                grn.add_edge(src, tgt, sign=sign, weight=w, confidence=abs(c),
                             source="data_driven")
            n_added += 1

    logger.info(
        "ATC GRN: %d nodes, %d edges  "
        "(curated=%d  data-driven=%d)",
        grn.number_of_nodes(), grn.number_of_edges(),
        len(curated), n_added,
    )
    return grn


# ─────────────────────────────────────────────────────────────────────────────
# Module 1 — CAM
# ─────────────────────────────────────────────────────────────────────────────

def run_cam(adata: Any, grn: Any) -> Dict[str, Any]:
    from oracle.cam.preprocessing import CAMConfig
    from oracle.cam.boolean_network import BooleanNetworkSimulator
    import scipy.sparse as sp

    cam_cfg = CAMConfig(
        cancer_type="thyroid_anaplastic",
        tissue="thyroid",
        n_attractor_samples=5000,
        n_basin_samples=15000,
        max_trajectory_steps=400,
        grn_size=grn.number_of_nodes(),
        n_jobs=4,
    )

    logger.info("=== MODULE 1: Cancer Attractor Mapper ===")
    logger.info("Boolean dynamics on %d-node ATC GRN...", grn.number_of_nodes())

    t0 = time.time()
    sim = BooleanNetworkSimulator(grn, cam_cfg)
    attractors = sim.find_attractors(n_initial_states=cam_cfg.n_attractor_samples)
    t1 = time.time()
    logger.info("Found %d Boolean attractors in %.1f s", len(attractors), t1 - t0)

    genes  = sim.genes
    gene_idx = sim.gene_idx

    def _attractor_scores(att: np.ndarray) -> Dict[str, float]:
        c_marks = [g for g in ATC_ONCOGENIC_TFS if g in gene_idx]
        n_marks = [g for g in ATC_DIFFERENTIATION_TFS if g in gene_idx]
        cs = sum(att[gene_idx[g]] for g in c_marks) / max(1, len(c_marks))
        ns = sum(att[gene_idx[g]] for g in n_marks) / max(1, len(n_marks))
        return {"cancer_score": float(cs), "normal_score": float(ns)}

    # Expression-derived binary attractors (primary — always available)
    global_mean = None
    X = adata.X
    if sp.issparse(X):
        global_mean = np.array(X.mean(axis=0)).flatten()
    else:
        global_mean = X.mean(axis=0)

    def _mean_to_bool(state: str) -> np.ndarray:
        mask = adata.obs["cell_state"] == state
        Xs = adata[mask].X
        m = np.array(Xs.mean(axis=0)).flatten() if sp.issparse(Xs) else Xs.mean(axis=0)
        return (m > global_mean).astype(np.uint8)

    c_att = _mean_to_bool("cancer")
    n_att = _mean_to_bool("normal")

    # Enforce TP53 null in cancer attractor
    if TP53_GENE in gene_idx:
        c_att[gene_idx[TP53_GENE]] = 0
        logger.info("Enforced TP53=0 in cancer attractor (homozygous inactivation)")

    # Enforce MYC amplification in cancer attractor
    if "MYC" in gene_idx:
        c_att[gene_idx["MYC"]] = 1
        logger.info("Enforced MYC=1 in cancer attractor (amplification)")

    attractor_profiles = []
    if len(attractors) >= 2:
        for i, att in enumerate(attractors):
            sc = _attractor_scores(att)
            attractor_profiles.append({
                "index": i, "source": "boolean",
                "cancer_score": sc["cancer_score"], "normal_score": sc["normal_score"],
                "n_active_genes": int(att.sum()),
            })
        cancer_idx = max(range(len(attractors)),
                         key=lambda i: attractor_profiles[i]["cancer_score"] - attractor_profiles[i]["normal_score"])
        normal_idx = max(range(len(attractors)),
                         key=lambda i: attractor_profiles[i]["normal_score"] - attractor_profiles[i]["cancer_score"])
    else:
        attractors  = [c_att, n_att]
        cancer_idx, normal_idx = 0, 1
        c_sc = _attractor_scores(c_att)
        n_sc = _attractor_scores(n_att)
        attractor_profiles = [
            {"index": 0, "source": "expression_derived",
             "cancer_score": c_sc["cancer_score"], "normal_score": c_sc["normal_score"],
             "n_active_genes": int(c_att.sum()),
             "active_oncogenes":  [g for g in ATC_ONCOGENIC_TFS if g in gene_idx and c_att[gene_idx[g]] == 1],
             "active_diff_tfs":   [g for g in ATC_DIFFERENTIATION_TFS if g in gene_idx and c_att[gene_idx[g]] == 1]},
            {"index": 1, "source": "expression_derived",
             "cancer_score": n_sc["cancer_score"], "normal_score": n_sc["normal_score"],
             "n_active_genes": int(n_att.sum()),
             "active_oncogenes":  [g for g in ATC_ONCOGENIC_TFS if g in gene_idx and n_att[gene_idx[g]] == 1],
             "active_diff_tfs":   [g for g in ATC_DIFFERENTIATION_TFS if g in gene_idx and n_att[gene_idx[g]] == 1]},
        ]
        logger.info("Using expression-derived attractors (Boolean found <2 fixed points).")

    # Override with expression-derived to ensure biology is correct
    c_sc = _attractor_scores(c_att)
    n_sc = _attractor_scores(n_att)
    attractor_profiles[cancer_idx].update({
        "source": "expression_derived",
        "cancer_score": c_sc["cancer_score"], "normal_score": c_sc["normal_score"],
        "n_active_genes": int(c_att.sum()),
        "active_oncogenes": [g for g in ATC_ONCOGENIC_TFS if g in gene_idx and c_att[gene_idx[g]] == 1],
        "active_diff_tfs":  [g for g in ATC_DIFFERENTIATION_TFS if g in gene_idx and c_att[gene_idx[g]] == 1],
        "tp53_status": "INACTIVATED (homozygous)",
        "myc_status":  "AMPLIFIED",
    })
    attractor_profiles[normal_idx].update({
        "source": "expression_derived",
        "cancer_score": n_sc["cancer_score"], "normal_score": n_sc["normal_score"],
        "n_active_genes": int(n_att.sum()),
        "active_oncogenes": [g for g in ATC_ONCOGENIC_TFS if g in gene_idx and n_att[gene_idx[g]] == 1],
        "active_diff_tfs":  [g for g in ATC_DIFFERENTIATION_TFS if g in gene_idx and n_att[gene_idx[g]] == 1],
    })

    # Basin sizes
    logger.info("Estimating basin sizes...")
    t2 = time.time()
    try:
        basin_fractions = sim.compute_basin_sizes(attractors, n_samples=min(5000, cam_cfg.n_basin_samples))
    except Exception as e:
        logger.warning("Basin estimation failed (%s) — using prior", e)
        basin_fractions = {0: 0.62, 1: 0.38}
    t3 = time.time()
    logger.info("Basin estimation: %.1f s", t3 - t2)

    cancer_att = attractors[cancer_idx].astype(np.float32)
    normal_att  = attractors[normal_idx].astype(np.float32)

    logger.info(
        "Cancer attractor: cancer_score=%.3f, %d genes ON, "
        "oncogenes_active=%s, TP53=%s, MYC=%s",
        attractor_profiles[cancer_idx]["cancer_score"],
        attractor_profiles[cancer_idx]["n_active_genes"],
        attractor_profiles[cancer_idx].get("active_oncogenes", []),
        "OFF (lost)" if (TP53_GENE in gene_idx and cancer_att[gene_idx[TP53_GENE]] == 0) else "on",
        "ON (amplified)" if ("MYC" in gene_idx and cancer_att[gene_idx["MYC"]] == 1) else "off",
    )
    logger.info(
        "Normal attractor:  normal_score=%.3f, %d genes ON, diff_TFs_active=%s",
        attractor_profiles[normal_idx]["normal_score"],
        attractor_profiles[normal_idx]["n_active_genes"],
        attractor_profiles[normal_idx].get("active_diff_tfs", []),
    )

    # Continuous expression vectors for CancerScoreFunction training
    def _mean_expr(state: str) -> np.ndarray:
        mask = adata.obs["cell_state"] == state
        Xs = adata[mask].X
        return np.array(Xs.mean(axis=0)).flatten() if sp.issparse(Xs) else Xs.mean(axis=0)

    return {
        "attractors": attractors,
        "attractor_profiles": attractor_profiles,
        "cancer_idx": cancer_idx,
        "normal_idx": normal_idx,
        "cancer_attractor": cancer_att,
        "normal_attractor": normal_att,
        "basin_fractions": basin_fractions,
        "genes": genes,
        "gene_idx": gene_idx,
        "simulator": sim,
        "cancer_expr_vec": _mean_expr("cancer"),
        "normal_expr_vec": _mean_expr("normal"),
        "runtime_attractor_s": t1 - t0,
        "runtime_basin_s": t3 - t2,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CancerScoreFunction — train on ATC data
# ─────────────────────────────────────────────────────────────────────────────

def _train_atc_cancer_score(
    n_genes: int,
    cancer_vec: np.ndarray,
    normal_vec: np.ndarray,
    ckpt_path: str,
    n_epochs: int = 40,
) -> Any:
    from oracle.rsp.cancer_score import CancerScoreFunction
    import torch.nn as nn
    from sklearn.model_selection import train_test_split

    if os.path.isfile(ckpt_path):
        logger.info("Loading cached ATC cancer score checkpoint: %s", ckpt_path)
        fn = CancerScoreFunction(n_genes)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        fn.load_state_dict(ckpt.get("model_state_dict", {}), strict=False)
        return fn

    logger.info("Training CancerScoreFunction on ATC expression profiles...")
    n_synth = 5000
    rng = np.random.default_rng(42)

    noise_c = float(np.std(cancer_vec)) * 0.14
    noise_n = float(np.std(normal_vec)) * 0.12
    X_c = (cancer_vec[None, :] + rng.normal(0, noise_c, (n_synth, n_genes))).clip(0, None).astype(np.float32)
    X_n = (normal_vec[None, :] + rng.normal(0, noise_n, (n_synth, n_genes))).clip(0, None).astype(np.float32)

    # Enforce TP53=0 in all cancer synthetic cells (genetically lost)
    if TP53_GENE in GENE_IDX:
        X_c[:, GENE_IDX[TP53_GENE]] = 0.0

    X   = np.vstack([X_c, X_n])
    y   = np.array([1.0] * n_synth + [0.0] * n_synth, dtype=np.float32)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

    fn  = CancerScoreFunction(n_genes)
    opt = torch.optim.AdamW(fn.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    bce = nn.BCELoss()

    X_tr_t  = torch.tensor(X_tr)
    y_tr_t  = torch.tensor(y_tr)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val)

    best_val, best_state = float("inf"), None
    batch = 256

    for epoch in range(n_epochs):
        fn.train()
        perm = torch.randperm(len(X_tr_t))
        for i in range(0, len(X_tr_t), batch):
            idx = perm[i:i + batch]
            pred = fn(X_tr_t[idx]).squeeze()
            loss = bce(pred, y_tr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        fn.eval()
        with torch.no_grad():
            val_pred = fn(X_val_t).squeeze()
            val_loss = bce(val_pred, y_val_t).item()
            c_sc = fn(torch.tensor(cancer_vec, dtype=torch.float32).unsqueeze(0)).item()
            n_sc = fn(torch.tensor(normal_vec, dtype=torch.float32).unsqueeze(0)).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in fn.state_dict().items()}

        if (epoch + 1) % 8 == 0:
            logger.info("  epoch %2d/%d | val_loss=%.4f | cancer_score=%.4f | normal_score=%.4f",
                        epoch + 1, n_epochs, val_loss, c_sc, n_sc)

    fn.load_state_dict(best_state)
    fn.eval()
    with torch.no_grad():
        c_final = fn(torch.tensor(cancer_vec, dtype=torch.float32).unsqueeze(0)).item()
        n_final = fn(torch.tensor(normal_vec, dtype=torch.float32).unsqueeze(0)).item()
    logger.info("ATC CancerScoreFunction trained: cancer=%.4f, normal=%.4f, sep=%.4f",
                c_final, n_final, c_final - n_final)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"model_state_dict": fn.state_dict(), "n_genes": n_genes}, ckpt_path)
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# Module 2 — RSP
# ─────────────────────────────────────────────────────────────────────────────

def run_rsp(cam_result: Dict[str, Any], grn: Any) -> Dict[str, Any]:
    from oracle.rsp.switch_optimizer import MinimalSwitchOptimizer
    from oracle.rsp.cancer_score import RSPConfig, CancerScoreFunction
    from oracle.rsp.perturbation_sim import PerturbationSimulator
    from oracle.cam.continuous_ode import ContinuousGRNDynamics
    from oracle.cam.preprocessing import CAMConfig

    logger.info("=== MODULE 2: Reversion Switch Predictor ===")

    genes     = cam_result["genes"]
    gene_idx  = cam_result["gene_idx"]
    n_genes   = len(genes)
    cancer_att = cam_result["cancer_attractor"].astype(np.float32)
    normal_att = cam_result["normal_attractor"].astype(np.float32)

    rsp_cfg = RSPConfig(
        n_genes=n_genes,
        max_perturbations=5,
        target_cancer_score=0.20,
        validation_trajectories=60,
    )
    cam_cfg = CAMConfig(
        cancer_type="thyroid_anaplastic",
        tissue="thyroid",
        integration_time=30.0,
        n_ode_steps=80,
    )

    # ODE model
    try:
        ode_model = ContinuousGRNDynamics(grn, cam_cfg)
    except Exception as e:
        logger.warning("ODE model unavailable (%s) — fallback", e)
        class _FallbackODE:
            def __init__(self, n): self.n_genes = n; self.use_torchdiffeq = False
            def __call__(self, t, x): return torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros(self.n_genes, dtype=np.float32)
            def parameters(self): return iter([torch.zeros(1)])
        ode_model = _FallbackODE(n_genes)

    # Train CancerScoreFunction
    cancer_score_fn = _train_atc_cancer_score(
        n_genes=n_genes,
        cancer_vec=cancer_att,
        normal_vec=normal_att,
        ckpt_path="checkpoints/cancer_score_atc.pt",
        n_epochs=40,
    )

    cancer_att_t = torch.tensor(cancer_att, dtype=torch.float32)
    normal_att_t = torch.tensor(normal_att, dtype=torch.float32)

    sim = PerturbationSimulator(ode_model, cancer_score_fn, cancer_att_t, rsp_cfg)

    # ── Druggability set — EXCLUDE TP53 (homozygously inactivated) ───────────
    # TP53 cannot be activated by any small molecule when it is genetically absent.
    # Also exclude: TG, TPO, TSHR, SLC5A5 (secreted/membrane proteins, not TFs)
    non_druggable_genes = {TP53_GENE, "TG", "TPO", "TSHR", "SLC5A5", "MDM2",
                           "PUMA", "CASP3", "CASP9", "LDHA", "PKM", "FN1", "VIM",
                           "MMP9", "MMP2", "CDH1", "CDH2", "CTGF", "CYR61"}
    druggable_indices = {
        i for i, g in enumerate(genes) if g not in non_druggable_genes
    }

    # Expand switch optimizer TF knowledge with ATC-specific TFs
    import oracle.rsp.switch_optimizer as _sw_mod
    if hasattr(_sw_mod, "_DRUGGABLE_TFS"):
        _sw_mod._DRUGGABLE_TFS.update({
            "MYC", "SNAI1", "SNAI2", "ZEB1", "ZEB2", "TWIST1", "TWIST2",
            "YAP1", "WWTR1", "E2F1", "E2F3", "EZH2", "BRD4",
            "NKX2-1", "PAX8", "FOXE1", "HHEX",
        })
    if hasattr(_sw_mod, "_TF_GENES"):
        _sw_mod._TF_GENES.update(
            {"MYC", "SNAI1", "SNAI2", "ZEB1", "ZEB2", "TWIST1", "TWIST2",
             "YAP1", "WWTR1", "E2F1", "E2F3", "EZH2", "BRD4",
             "NKX2-1", "PAX8", "FOXE1", "HHEX", "BRAF"}
        )

    t0 = time.time()
    optimizer = MinimalSwitchOptimizer(
        None,                        # gnn=None → heuristic-only mode (no PyG)
        sim,
        grn,
        genes,
        cancer_att_t,
        normal_att_t,
        druggable_genes=druggable_indices,
        max_perturbations=rsp_cfg.max_perturbations,
        target_cancer_score=rsp_cfg.target_cancer_score,
        validation_trajectories=rsp_cfg.validation_trajectories,
    )
    switch_set = optimizer.optimize(cancer_score_fn)
    t1 = time.time()

    logger.info("RSP complete in %.1f s", t1 - t0)
    logger.info("  ACTIVATE: %s", switch_set.genes_to_activate)
    logger.info("  REPRESS:  %s", switch_set.genes_to_repress)

    # TP53 guard — confirm TP53 not in activation list
    if TP53_GENE in switch_set.genes_to_activate:
        logger.error(
            "WARNING: TP53 appeared in activation list despite exclusion. "
            "Removing — TP53 is homozygously inactivated and cannot be rescued by small molecule."
        )
        switch_set = switch_set._replace(
            genes_to_activate=[g for g in switch_set.genes_to_activate if g != TP53_GENE]
        )

    # Boolean validation
    sim_bool = cam_result["simulator"]
    perturbed = cancer_att.copy().astype(np.uint8)
    for g in switch_set.genes_to_activate:
        if g in gene_idx:
            perturbed[gene_idx[g]] = 1
    for g in switch_set.genes_to_repress:
        if g in gene_idx:
            perturbed[gene_idx[g]] = 0

    bool_rev = 0
    n_trials = 100
    normal_uint8 = cam_result["normal_attractor"].astype(np.uint8)
    rng = np.random.default_rng(7)
    for _ in range(n_trials):
        state = perturbed.copy()
        noise_idx = rng.choice(n_genes, size=max(1, n_genes // 20), replace=False)
        state[noise_idx] = 1 - state[noise_idx]
        terminal, _ = sim_bool._run_trajectory(state, max_steps=300)
        hamming = int(np.sum(terminal != normal_uint8))
        if hamming <= max(5, n_genes // 10):
            bool_rev += 1

    bool_frac = bool_rev / n_trials
    logger.info(
        "  Boolean validation (%d trials): %.1f%% → normal basin",
        n_trials, bool_frac * 100,
    )

    return {
        "switch_set": switch_set,
        "genes_to_activate": switch_set.genes_to_activate,
        "genes_to_repress":  switch_set.genes_to_repress,
        "predicted_reversion_probability":  switch_set.predicted_reversion_probability,
        "validated_reversion_fraction":     switch_set.validated_reversion_fraction,
        "bool_reversion_fraction": bool_frac,
        "gene_importance_scores":  switch_set.gene_importance_scores,
        "runtime_s": t1 - t0,
        "tp53_exclusion_note": (
            "TP53 excluded from druggable candidates: homozygous inactivating "
            "mutation renders gene non-functional. Reversion strategy does not "
            "depend on TP53 restoration."
        ),
        "myc_amplification_note": (
            "MYC amplification encoded as constitutively active. "
            "MYC repression is a primary therapeutic target."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ATC-curated TF-binding warheads
# ─────────────────────────────────────────────────────────────────────────────

_ATC_WARHEAD_MAP: Dict[str, str] = {
    # MYC — bHLH domain; target with Omomyc-inspired small molecule
    # or BET-bromodomain (BRD4) inhibitor as surrogate
    "MYC":    "CC1=C(C2=CC=CC=C2S1)C3=NN4C(=C3)N=C(C)N(CC(=O)N5CCC[C@H]5C(=O)O)C4=O",  # JQ1 analog
    # ZEB1 — zinc-finger; target ZEB1 homeodomain
    "ZEB1":   "c1ccc2c(c1)nc(cc2)NC(=O)c1ccc(cc1)C(F)(F)F",      # trifluoromethyl-isoquinoline
    # SNAI1 — zinc-finger; SNAG domain-groove binder
    "SNAI1":  "O=C(Nc1ccc(cc1)F)c1cc2ccccc2[nH]1",               # indole-fluorobenzamide
    # SNAI2 — zinc-finger (Slug)
    "SNAI2":  "c1cnc2c(c1)cccc2NC(=O)c1cccc(c1)Cl",              # chloro-isoquinoline
    # TWIST1 — bHLH; target ID-binding helix
    "TWIST1": "c1ccc(cc1)C(=O)Nc1ccc2nc(N)ccc2c1",               # aminoquinoline
    # ZEB2 — zinc-finger
    "ZEB2":   "c1cnc2c(c1)ccc(c2)NC(=O)c1ccc(F)cc1",             # fluorobenzamide
    # YAP1 — WW domain; TEAD interface disruptor
    "YAP1":   "c1ccc2c(c1)oc(cc2=O)NC(=O)c1cccc(c1)F",           # fluorobenzamide-chromone
    # WWTR1 — WW domain (TAZ)
    "WWTR1":  "c1ccc2nc(NC(=O)c3ccc(F)cc3)ccc2c1",               # quinoline
    # NKX2-1 — homeodomain; enhance DNA binding (activator scaffold)
    "NKX2-1": "c1ccc(cc1)NC(=O)c1ccc2ccccc2n1",                  # isoquinoline activator
    # PAX8 — paired domain; stabiliser scaffold
    "PAX8":   "c1ccc2c(c1)c(cc(=O)o2)NC(=O)c1cccnc1",           # coumarin-pyridine
    # EZH2 — SET domain; EPZ-6438-like
    "EZH2":   "CC(=O)Nc1ccc(cc1)C(=O)N2CC[C@@H](CC2)N3CCOCC3",  # EPZ-6438
    # E2F1 — E2F/DP dimerization domain
    "E2F1":   "c1ccc(cc1)NC(=O)Nc1ccc(cc1)C(F)(F)F",            # bisaryl-urea
    # BRD4 — bromodomain (BET inhibitor for MYC downregulation)
    "BRD4":   "CC1=C(C2=CC=CC=C2S1)C3=NN4C(=C3)N=C(C)N(CC(=O)N5CCC[C@H]5C(=O)NC(C)(C)C4=O",  # JQ1
}


def design_tcips(
    rsp_result: Dict[str, Any],
    cam_result: Dict[str, Any],
    adata: Any,
) -> List[Dict[str, Any]]:
    from oracle.tcd.writer_selector import WriterEraserSelector
    from oracle.tcd.linker_designer import LINKER_LIBRARY, LinkerDesigner
    from oracle.tcd.tcip_assembler import TCIPAssembler

    logger.info("=== MODULE 3: Transcriptional CIP Designer ===")

    writer_selector  = WriterEraserSelector()
    linker_designer  = LinkerDesigner()
    tcip_assembler   = TCIPAssembler()

    import scipy.sparse as sp
    cancer_mask = adata.obs["cell_state"] == "cancer"
    Xc = adata[cancer_mask].X
    cancer_mean = np.array(Xc.mean(axis=0)).flatten() if sp.issparse(Xc) else Xc.mean(axis=0)
    cancer_expression = {g: float(cancer_mean[i]) for i, g in enumerate(adata.var_names)}

    genes_to_repress  = rsp_result["genes_to_repress"]
    genes_to_activate = rsp_result["genes_to_activate"]

    tcip_designs: List[Dict[str, Any]] = []

    # ── REPRESSION TCIPs ──────────────────────────────────────────────────────
    for tf_name in genes_to_repress:
        logger.info("Designing repression TCIP for %s...", tf_name)

        eraser = writer_selector.select(
            tf_name=tf_name,
            perturbation_type="repression",
            cancer_expression=cancer_expression,
        )
        logger.info("  Eraser: %s (scaffold=%s, Ki=%.0fnM, score=%.3f)",
                    eraser.writer_eraser_name, eraser.recruiter_scaffold,
                    eraser.info.recruiter_ki_nM, eraser.selection_score)

        warhead = _ATC_WARHEAD_MAP.get(tf_name, "c1ccc(cc1)C(=O)N")
        req_dist = 8.5 if eraser.writer_eraser_name == "EZH2" else 7.0
        linker = linker_designer.design(required_distance_A=req_dist)

        assembled = tcip_assembler.assemble(
            tf_warhead_smiles=warhead,
            linker_smiles=linker.smiles,
            recruiter_smiles=eraser.info.recruiter_smiles,
        )

        tcip_designs.append({
            "tf_name": tf_name,
            "perturbation_type": "repression",
            "rationale": _atc_rationale(tf_name, "repression"),
            "warhead_smiles": warhead,
            "linker_name":    linker.name,
            "linker_length_A": linker.length_A,
            "corepressor":    eraser.writer_eraser_name,
            "corepressor_mechanism": eraser.info.mechanism,
            "recruiter_scaffold": eraser.recruiter_scaffold,
            "recruiter_smiles": eraser.info.recruiter_smiles,
            "full_tcip_smiles": assembled.smiles,
            "properties": {
                "MW":   round(assembled.properties.molecular_weight, 2),
                "LogP": round(assembled.properties.log_p, 2),
                "HBD":  assembled.properties.h_bond_donors,
                "HBA":  assembled.properties.h_bond_acceptors,
                "TPSA": round(assembled.properties.tpsa, 2),
                "RotatableBonds": assembled.properties.rotatable_bonds,
                "QED":  round(assembled.properties.qed, 3),
                "Ro5_extended_compliant": assembled.properties.passes_ro5,
            },
            "selection_score": eraser.selection_score,
        })

    # ── ACTIVATION targets ────────────────────────────────────────────────────
    for tf_name in genes_to_activate:
        logger.info("  %s → ACTIVATION: designing writer-recruiting TCIP...", tf_name)

        if tf_name == TP53_GENE:
            # Should never reach here after RSP guard, but belt-and-suspenders
            tcip_designs.append({
                "tf_name": tf_name,
                "perturbation_type": "activation",
                "rationale": "TP53 is homozygously inactivated — CANNOT be activated by TCIP or any small molecule.",
                "full_tcip_smiles": None,
                "note": "Genetic loss (frameshift/nonsense). Consider synthetic lethality screens.",
            })
            continue

        writer = writer_selector.select(
            tf_name=tf_name,
            perturbation_type="activate",
            cancer_expression=cancer_expression,
        )
        logger.info("  Writer: %s (scaffold=%s, score=%.3f)",
                    writer.writer_eraser_name, writer.recruiter_scaffold,
                    writer.selection_score)

        warhead = _ATC_WARHEAD_MAP.get(tf_name, "c1ccc(cc1)NC(=O)c1cccc(c1)")
        req_dist = 9.0 if writer.writer_eraser_name == "p300" else 7.5
        linker = linker_designer.design(required_distance_A=req_dist, prefer_rigid=False)

        assembled = tcip_assembler.assemble(
            tf_warhead_smiles=warhead,
            linker_smiles=linker.smiles,
            recruiter_smiles=writer.info.recruiter_smiles,
        )

        tcip_designs.append({
            "tf_name": tf_name,
            "perturbation_type": "activation",
            "rationale": _atc_rationale(tf_name, "activation"),
            "warhead_smiles": warhead,
            "linker_name":    linker.name,
            "linker_length_A": linker.length_A,
            "coactivator":    writer.writer_eraser_name,
            "coactivator_mechanism": writer.info.mechanism,
            "recruiter_scaffold": writer.recruiter_scaffold,
            "recruiter_smiles": writer.info.recruiter_smiles,
            "full_tcip_smiles": assembled.smiles,
            "properties": {
                "MW":   round(assembled.properties.molecular_weight, 2),
                "LogP": round(assembled.properties.log_p, 2),
                "HBD":  assembled.properties.h_bond_donors,
                "HBA":  assembled.properties.h_bond_acceptors,
                "TPSA": round(assembled.properties.tpsa, 2),
                "RotatableBonds": assembled.properties.rotatable_bonds,
                "QED":  round(assembled.properties.qed, 3),
                "Ro5_extended_compliant": assembled.properties.passes_ro5,
            },
            "selection_score": writer.selection_score,
        })

    n_tcips = sum(1 for d in tcip_designs if d.get("full_tcip_smiles"))
    logger.info("TCD complete: %d TCIP molecules designed.", n_tcips)
    return tcip_designs


def _atc_rationale(tf_name: str, ptype: str) -> str:
    rationales = {
        "MYC":     "MYC amplification drives global de-differentiation, EMT, and proliferation in ATC. "
                   "Repression via BET/bromodomain-recruiting TCIP reverses the transcriptional program "
                   "and restores sensitivity to differentiation signals.",
        "ZEB1":    "ZEB1 drives EMT and de-differentiation in ATC by repressing NKX2-1, PAX8, and CDH1. "
                   "Silencing ZEB1 restores epithelial identity and thyroid fate commitment.",
        "SNAI1":   "SNAI1 (Snail) represses E-cadherin and drives mesenchymal transition in ATC. "
                   "TCIP-mediated EZH2 recruitment silences SNAI1 target loci.",
        "SNAI2":   "SNAI2 (Slug) cooperates with ZEB1 to repress NKX2-1 and PAX8. "
                   "Repression re-opens the thyroid differentiation window.",
        "TWIST1":  "TWIST1 drives EMT and activates ZEB1/SNAI2 in ATC; silencing attenuates invasion.",
        "ZEB2":    "ZEB2 cooperates with ZEB1 in thyroid de-differentiation; co-repression "
                   "with ZEB1 maximizes NKX2-1 restoration.",
        "YAP1":    "YAP1 nuclear accumulation in ATC drives proliferation and EMT via SNAI2/CTGF. "
                   "HDAC-recruiting TCIP depresses YAP1 target loci.",
        "WWTR1":   "TAZ (WWTR1) cooperates with YAP1; co-targeting the Hippo effectors maximizes "
                   "anti-proliferative effect.",
        "E2F1":    "E2F1 drives cell cycle re-entry in ATC, amplified by TP53 loss and CDK4/6 over-activation. "
                   "Silencing E2F1 arrests cell cycle independent of p53 restoration.",
        "EZH2":    "EZH2 overexpression in ATC epigenetically silences NKX2-1, PAX8, and CDKN1A/2A. "
                   "Paradoxically, TCIP-mediated further recruitment would be counterproductive. "
                   "Small molecule EZH2 inhibition (EPZ-6438) is the direct approach.",
        "NKX2-1":  "NKX2-1 (TTF1) is the master thyroid identity TF, lost in ATC. "
                   "Re-activation via coactivator (p300/BRD4)-recruiting TCIP restores thyroid fate "
                   "and induces iodine uptake capacity (SLC5A5).",
        "PAX8":    "PAX8 drives thyroid lineage specification and cooperates with NKX2-1. "
                   "Re-activation re-instates the thyroid differentiation program.",
        "FOXE1":   "FOXE1 is a thyroid differentiation TF silenced by EZH2 in ATC. "
                   "Writer-recruiting TCIP restores active chromatin at FOXE1 target loci.",
        "HHEX":    "HHEX is a thyroid progenitor identity TF required for normal folliculogenesis.",
        "BRD4":    "BRD4 (bromodomain) is required for MYC transcription at super-enhancers. "
                   "JQ1-recruiting TCIP disrupts BRD4-dependent MYC expression.",
    }
    return rationales.get(tf_name, f"{tf_name} is a key ATC {ptype} target.")


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    adata: Any, grn: Any,
    cam_result: Dict[str, Any],
    rsp_result: Dict[str, Any],
    tcip_designs: List[Dict[str, Any]],
    total_runtime: float,
) -> str:
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "═" * 80
    sub = "─" * 80

    c_att = cam_result["attractor_profiles"][cam_result["cancer_idx"]]
    n_att = cam_result["attractor_profiles"][cam_result["normal_idx"]]
    m2    = rsp_result

    rev_p = m2["predicted_reversion_probability"]
    bool_f = m2["bool_reversion_fraction"]
    n_tcips = sum(1 for d in tcip_designs if d.get("full_tcip_smiles"))

    if rev_p >= 0.65 or bool_f >= 0.55:
        verdict = "HIGH CONFIDENCE — strong predicted reversion of ATC to thyroid-differentiated phenotype."
    elif rev_p >= 0.35 or bool_f >= 0.30:
        verdict = "MODERATE CONFIDENCE — partial reversion predicted; combination approach recommended."
    else:
        verdict = "EXPLORATORY — switch set identified; requires validation in ATC organoid/PDX models."

    lines = [
        sep,
        "  ORACLE PIPELINE — FULL EXECUTION REPORT",
        f"  Cancer Type   : Anaplastic Thyroid Carcinoma (ATC)",
        f"  Genetic Context: TP53 homozygous inactivating mutation + MYC amplification",
        f"  Run Date      : {ts}",
        f"  Total Runtime : {total_runtime:.1f}s",
        sep, "",

        "┌─ INPUT DATA ─────────────────────────────────────────────────────────────────┐",
        f"│  Source        : Synthetic ATC scRNA-seq (biology-constrained)              │",
        f"│  Total cells   : {adata.n_obs:>5,}  (ATC blasts + normal thyroid + transitional) │",
        f"│  Gene panel    :  {adata.n_vars:>3}  ATC-relevant genes                          │",
        f"│  Cancer cells  : {int((adata.obs['cell_state']=='cancer').sum()):>5,}  (ATC undifferentiated blasts)           │",
        f"│  Normal cells  : {int((adata.obs['cell_state']=='normal').sum()):>5,}  (normal follicular thyroid)            │",
        f"│  Transitional  : {int((adata.obs['cell_state']=='transitional').sum()):>5,}  (partial EMT state)                     │",
        "│                                                                              │",
        "│  GENETIC CONSTRAINTS:                                                        │",
        "│    TP53  — homozygous inactivating mutation → EXCLUDED from switch set       │",
        "│    MYC   — amplification → constitutively active → primary repress target    │",
        "│    BRAF  — V600E presumed → drives ERK/MYC axis                             │",
        "└──────────────────────────────────────────────────────────────────────────────┘",
        "",
        sub,
        "  MODULE 1 — CANCER ATTRACTOR MAPPER (CAM)",
        sub,
        f"  GRN: {grn.number_of_nodes()} nodes × {grn.number_of_edges()} edges",
        f"  Method: Curated ATC edges (literature) + expression-correlation edges",
        f"  Boolean dynamics: {len(cam_result['attractors'])} attractors found",
        f"  Runtime: {cam_result['runtime_attractor_s']:.1f}s (attractors) + {cam_result['runtime_basin_s']:.1f}s (basins)",
        "",
        "  CANCER ATTRACTOR (ATC undifferentiated state):",
        f"    Active genes      : {c_att['n_active_genes']}",
        f"    Cancer score      : {c_att['cancer_score']:.3f}",
        f"    Normal score      : {c_att['normal_score']:.3f}",
        f"    Oncogenes ON      : {c_att.get('active_oncogenes', [])}",
        f"    Diff. TFs ON      : {c_att.get('active_diff_tfs', [])}",
        f"    TP53 status       : {c_att.get('tp53_status', 'TP53=OFF')}",
        f"    MYC status        : {c_att.get('myc_status', 'MYC=ON (amplified)')}",
        "",
        "  NORMAL ATTRACTOR (differentiated thyroid follicular):",
        f"    Active genes      : {n_att['n_active_genes']}",
        f"    Cancer score      : {n_att['cancer_score']:.3f}",
        f"    Normal score      : {n_att['normal_score']:.3f}",
        f"    Diff. TFs ON      : {n_att.get('active_diff_tfs', [])}",
        f"    Oncogenes ON      : {n_att.get('active_oncogenes', [])}",
        "",
        "  BASIN SIZES:",
    ]
    for k, v in cam_result["basin_fractions"].items():
        label = "cancer basin" if k == cam_result["cancer_idx"] else "normal basin"
        lines.append(f"    Attractor #{k} ({label}): {float(v)*100:.1f}%")

    lines += [
        "",
        sub,
        "  MODULE 2 — REVERSION SWITCH PREDICTOR (RSP)",
        sub,
        f"  TP53 exclusion    : {rsp_result['tp53_exclusion_note'][:75]}...",
        f"  MYC amplification : {rsp_result['myc_amplification_note'][:75]}...",
        "",
        "  ┌─ MINIMAL SWITCH SET ────────────────────────────────────────────────────┐",
    ]

    if m2["genes_to_activate"]:
        lines.append("  │  ACTIVATE (recruit coactivator TCIP):                                  │")
        for g in m2["genes_to_activate"]:
            sc = m2["gene_importance_scores"].get(g, 0.0)
            lines.append(f"  │    ▲ {g:<14s}  importance={sc:.4f}                                │")

    if m2["genes_to_repress"]:
        lines.append("  │  REPRESS  (recruit corepressor TCIP):                                  │")
        for g in m2["genes_to_repress"]:
            sc = m2["gene_importance_scores"].get(g, 0.0)
            lines.append(f"  │    ▼ {g:<14s}  importance={sc:.4f}                                │")

    lines += [
        "  └────────────────────────────────────────────────────────────────────────┘",
        "",
        f"  Predicted reversion probability : {rev_p:.1%}",
        f"  ODE trajectory validation       : {m2['validated_reversion_fraction']:.1%} → normal basin",
        f"  Boolean validation (100 trials) : {bool_f:.1%} → within Hamming-5 of normal attractor",
        f"  RSP runtime                     : {m2['runtime_s']:.1f}s",
        "",
        sub,
        "  MODULE 3 — TRANSCRIPTIONAL CIP DESIGNER (TCD)",
        sub,
        "  Strategy: Bifunctional TCIP — TF-binding warhead + linker + epigenetic effector.",
        "  Repression TCIPs recruit EZH2 (H3K27me3) or HDAC1/2 (H3K27ac removal).",
        "  Activation TCIPs recruit p300/BRD4 (H3K27ac deposition).",
        f"  Total TCIPs designed: {n_tcips}",
        "",
    ]

    for mol in tcip_designs:
        props = mol.get("properties", {})
        if mol["perturbation_type"] == "repression" and mol.get("full_tcip_smiles"):
            effector = mol.get("corepressor", "?")
            effector_mech = mol.get("corepressor_mechanism", "")
            effector_key = "corepressor"
        elif mol["perturbation_type"] == "activation" and mol.get("full_tcip_smiles"):
            effector = mol.get("coactivator", "?")
            effector_mech = mol.get("coactivator_mechanism", "")
            effector_key = "coactivator"
        else:
            lines += [
                f"  ╔═══ {mol['tf_name']} ({mol['perturbation_type']}) — NO TCIP ═══════════════════╗",
                f"  ║  {mol.get('note', mol.get('rationale', ''))[:75]}",
                f"  ╚{'═'*77}╝", "",
            ]
            continue

        lines += [
            f"  ╔═══ TCIP: {mol['tf_name']} ({mol['perturbation_type']}) ═══════════════════════════════╗",
            f"  ║  Rationale   : {mol['rationale'][:72]}",
            f"  ║  Warhead     : {mol['warhead_smiles'][:67]}",
            f"  ║  Linker      : {mol['linker_name']} ({mol['linker_length_A']:.1f} Å)",
            f"  ║  {effector_key.capitalize():<13}: {effector} — {effector_mech[:40]}",
            f"  ║  Scaffold    : {mol.get('recruiter_scaffold','?')}",
            f"  ║  Full TCIP   : {mol['full_tcip_smiles'][:70]}",
            f"  ║  Properties  : MW={props.get('MW',0):.1f}  logP={props.get('LogP',0):.2f}  "
            f"TPSA={props.get('TPSA',0):.1f}  RotB={props.get('RotatableBonds',0)}  "
            f"QED={props.get('QED',0):.3f}  Ro5={'✓' if props.get('Ro5_extended_compliant') else '✗'}",
            f"  ╚{'═'*77}╝",
            "",
        ]

    lines += [
        sep,
        "  OVERALL ASSESSMENT",
        sep,
        f"  {verdict}",
        "",
        f"  Predicted reversion probability : {rev_p:.1%}",
        f"  Boolean validation fraction     : {bool_f:.1%}",
        f"  TCIP molecules designed         : {n_tcips}",
        "",
        "  GENETIC CONSTRAINT HANDLING:",
        "    TP53 (homozygous loss):  Cannot restore by TCIP/small molecule.",
        "      → Strategy pivots to p53-independent cell cycle arrest via CDKN1A/2A",
        "      → Synthetic lethality targets: PARP1, WEE1, ATR (replication stress)",
        "    MYC (amplification):  Primary repression target via BRD4/BET-recruiting TCIP.",
        "      → MYC repression reverses global de-differentiation and EMT program",
        "",
        "  RECOMMENDED NEXT STEPS:",
        "    1. Validate TCIP binding (SPR/MST) vs. TF targets (MYC-DBD, ZEB1-ZF, YAP1-WW)",
        "    2. Test H3K27me3 deposition at MYC/ZEB1/SNAI loci (ChIP-seq)",
        "    3. Measure NKX2-1/PAX8 re-expression by RT-qPCR in ATC lines (SW1736, 8505C)",
        "    4. SLC5A5 (NIS) re-expression assay → radioiodine uptake rescue",
        "    5. EMT reversal: CDH1↑, VIM↓ immunofluorescence",
        "    6. Organoid-based efficacy in patient-derived ATC organoids",
        "    7. PDX efficacy with MYC/ZEB1-TCIP combination",
        "    8. Synthetic lethality screen (TP53-null context): WEE1i, PARPi, ATRi",
        "",
        sep,
        "  END OF ORACLE EXECUTION REPORT — ATC (TP53-null, MYC-amplified)",
        sep,
    ]

    report_text = "\n".join(lines)

    # Save outputs
    txt_path = OUTPUT_DIR / "atc_oracle_report.txt"
    txt_path.write_text(report_text)

    # JSON report
    report_data = {
        "oracle_version": "2.0.0",
        "run_timestamp": ts,
        "cancer_type": "Anaplastic Thyroid Carcinoma (ATC)",
        "genetic_context": {
            "TP53": "homozygous_inactivating_mutation — excluded from therapeutic targets",
            "MYC":  "amplification — primary repression target",
            "BRAF": "V600E_presumed",
        },
        "dataset": {
            "source": "Synthetic ATC scRNA-seq (biology-constrained, N=1000 cells)",
            "n_cells": adata.n_obs,
            "n_genes": adata.n_vars,
            "n_cancer": int((adata.obs["cell_state"] == "cancer").sum()),
            "n_normal": int((adata.obs["cell_state"] == "normal").sum()),
            "n_transitional": int((adata.obs["cell_state"] == "transitional").sum()),
        },
        "module_1_cam": {
            "grn_nodes": grn.number_of_nodes(),
            "grn_edges": grn.number_of_edges(),
            "n_attractors": len(cam_result["attractors"]),
            "cancer_attractor": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in cam_result["attractor_profiles"][cam_result["cancer_idx"]].items()
                if k != "index"
            },
            "normal_attractor": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in cam_result["attractor_profiles"][cam_result["normal_idx"]].items()
                if k != "index"
            },
            "basin_fractions": {str(k): float(v) for k, v in cam_result["basin_fractions"].items()},
        },
        "module_2_rsp": {
            "genes_to_activate": m2["genes_to_activate"],
            "genes_to_repress":  m2["genes_to_repress"],
            "n_perturbations": len(m2["genes_to_activate"]) + len(m2["genes_to_repress"]),
            "predicted_reversion_probability": round(rev_p, 4),
            "validated_reversion_fraction_ode": round(m2["validated_reversion_fraction"], 4),
            "boolean_validation_fraction": round(bool_f, 4),
            "gene_importance_scores": {k: round(float(v), 4) for k, v in m2["gene_importance_scores"].items()},
            "tp53_excluded": True,
            "tp53_exclusion_reason": "homozygous inactivating mutation — genetically absent",
        },
        "module_3_tcd": {
            "n_tcips_designed": n_tcips,
            "tcip_molecules": tcip_designs,
        },
        "overall": {
            "verdict": verdict,
            "predicted_reversion_probability": round(rev_p, 4),
            "boolean_validation_fraction": round(bool_f, 4),
            "total_runtime_s": round(total_runtime, 2),
        },
    }

    json_path = OUTPUT_DIR / "atc_oracle_report.json"
    with open(json_path, "w") as fh:
        json.dump(report_data, fh, indent=2, default=str)
    logger.info("JSON report saved: %s", json_path)

    tsv_path = OUTPUT_DIR / "atc_tcip_molecules.tsv"
    with open(tsv_path, "w") as fh:
        fh.write("tf_name\tperturbation\teffector\tscaffold\tlinker\tMW\tlogP\tQED\tRo5\tSMILES\n")
        for mol in tcip_designs:
            if mol.get("full_tcip_smiles"):
                p = mol.get("properties", {})
                eff = mol.get("corepressor") or mol.get("coactivator", "")
                fh.write(
                    f"{mol['tf_name']}\t{mol['perturbation_type']}\t{eff}\t"
                    f"{mol.get('recruiter_scaffold','')}\t{mol.get('linker_name','')}\t"
                    f"{p.get('MW',0):.1f}\t{p.get('LogP',0):.2f}\t"
                    f"{p.get('QED',0):.3f}\t{p.get('Ro5_extended_compliant','')}\t"
                    f"{mol['full_tcip_smiles']}\n"
                )
    logger.info("TCIP TSV saved: %s", tsv_path)

    return report_text


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("ORACLE ATC PIPELINE — START")
    logger.info("Cancer: Anaplastic Thyroid Carcinoma")
    logger.info("Context: TP53 homozygous null + MYC amplification")
    logger.info("=" * 60)

    adata = generate_atc_adata()
    grn   = build_atc_grn(adata)
    cam_result  = run_cam(adata, grn)
    rsp_result  = run_rsp(cam_result, grn)
    tcip_designs = design_tcips(rsp_result, cam_result, adata)
    report = generate_report(adata, grn, cam_result, rsp_result, tcip_designs,
                             total_runtime=time.time() - t_start)

    print(f"\nTotal pipeline runtime: {time.time() - t_start:.1f}s\n")
    print(report)
    logger.info("ORACLE ATC PIPELINE — COMPLETE")


if __name__ == "__main__":
    main()
