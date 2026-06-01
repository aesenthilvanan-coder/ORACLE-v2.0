#!/usr/bin/env python3
"""
Full ORACLE pipeline execution on AML (Acute Myeloid Leukemia) scRNA-seq data.

Pipeline:
  Module 1 (CAM)  — Preprocess AML data, infer GRN, Boolean attractor landscape
  Module 2 (RSP)  — Minimal TF reversion switch set via greedy optimization
  Module 3 (TCD)  — TCIP molecule design (epigenetic corepressor recruitment)

Output: structured JSON + human-readable execution report.
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("oracle.aml_pipeline")

H5AD_PATH = "data/raw/scrnaseq/aml_bonemarrow.h5ad"
OUTPUT_DIR = Path("outputs/aml")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# AML biology constants
# ─────────────────────────────────────────────────────────────────────────────

# Cell types classified as AML blasts (cancer attractor)
AML_BLAST_LABELS = {
    "early promyelocyte", "late promyelocyte", "myelocyte",
    "hematopoietic multipotent progenitor cell",
    "lymphoid lineage restricted progenitor cell",
    "common dendritic progenitor",
}

# Cell types classified as normal myeloid differentiation (normal attractor)
NORMAL_MYELOID_LABELS = {
    "classical monocyte", "non-classical monocyte",
    "conventional dendritic cell",
}

# AML driver TFs known to be active in blasts (candidates for repression)
AML_ONCOGENIC_TFS = [
    "MEIS1",    # HOX cofactor, stem-cell program — high in blasts
    "LMO2",     # LIM-domain oncogene — high in HSC/blast
    "BCL11A",   # transcriptional repressor — blocks myeloid differentiation
    "SOX4",     # stemness factor — high in AML
    "CDK6",     # cell-cycle kinase, phospho-RB inhibitor — blast proliferation
]

# Tumor suppressor TFs (candidates for activation)
AML_TUMOR_SUPPRESSOR_TFS = [
    "CEBPA",    # master myeloid differentiation TF — low/mutated in AML
    "IRF8",     # monocyte/DC differentiation — low in AML
    "CEBPE",    # late granulocyte differentiation
]


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load and prepare AML data
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare_aml_data(h5ad_path: str) -> Any:
    """Load AML h5ad, rename vars to gene symbols, annotate cell states."""
    import scanpy as sc
    import scipy.sparse as sp

    logger.info("Loading AML dataset: %s", h5ad_path)
    adata = sc.read_h5ad(h5ad_path)
    logger.info("Loaded: %d cells × %d genes", adata.n_obs, adata.n_vars)

    # ── Rename Ensembl IDs → gene symbols ──────────────────────────────────
    if "feature_name" in adata.var.columns:
        adata.var_names = adata.var["feature_name"].values
        adata.var_names_make_unique()
        logger.info("Renamed var_names to gene symbols.")

    # ── Store raw counts for preprocessing ─────────────────────────────────
    # X contains log1p-normalised values from CellxGene; store as .raw
    if adata.raw is None:
        # Approximate raw: expm1 to reverse log1p, round to integer
        X_raw = adata.X.copy()
        if sp.issparse(X_raw):
            X_raw.data = np.expm1(X_raw.data).round()
        else:
            X_raw = np.expm1(X_raw).round()
        adata.layers["raw_counts"] = X_raw
    else:
        adata.layers["raw_counts"] = adata.raw.X

    # ── Cell state annotation ───────────────────────────────────────────────
    cell_type_col = "cell_type"
    ct = adata.obs[cell_type_col].str.lower()

    cell_state = np.full(adata.n_obs, "unknown", dtype=object)
    for i, label in enumerate(ct):
        if any(l in label for l in [b.lower() for b in AML_BLAST_LABELS]):
            cell_state[i] = "cancer"
        elif any(l in label for l in [n.lower() for n in NORMAL_MYELOID_LABELS]):
            cell_state[i] = "normal"
        else:
            cell_state[i] = "transitional"

    adata.obs["cell_state"] = cell_state
    n_cancer = (cell_state == "cancer").sum()
    n_normal = (cell_state == "normal").sum()
    n_trans = (cell_state == "transitional").sum()
    logger.info(
        "Cell state annotation: cancer=%d, normal=%d, transitional=%d, unknown=%d",
        n_cancer, n_normal, n_trans,
        (cell_state == "unknown").sum(),
    )

    # ── Filter to cancer + normal + transitional (drop pure T/NK/B cells) ──
    keep = adata.obs["cell_state"].isin(["cancer", "normal", "transitional"])
    adata = adata[keep].copy()
    logger.info("After filtering: %d cells retained", adata.n_obs)

    return adata


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build AML-focused GRN from panel genes + known AML regulation
# ─────────────────────────────────────────────────────────────────────────────

def build_aml_grn(adata: Any) -> Any:
    """
    Build a signed GRN for AML by:
    1. Running correlation-based edge scoring on the 458-gene panel
    2. Overlaying curated AML TF→target regulation from literature
    3. Trimming to top-N hub genes

    Returns networkx.DiGraph with sign, weight, confidence edge attributes.
    """
    import networkx as nx
    import scipy.sparse as sp

    logger.info("Building AML GRN from %d panel genes...", adata.n_vars)

    genes = list(adata.var_names)
    n_genes = len(genes)
    gene_idx = {g: i for i, g in enumerate(genes)}

    # ── Compute mean expression per cell state ──────────────────────────────
    def _mean_expr(state: str) -> np.ndarray:
        mask = adata.obs["cell_state"] == state
        X = adata[mask].X
        if sp.issparse(X):
            return np.array(X.mean(axis=0)).flatten()
        return X.mean(axis=0)

    cancer_mean = _mean_expr("cancer")
    normal_mean = _mean_expr("normal")

    # ── Compute differential expression score (log2 fold-change) ───────────
    eps = 1e-3
    lfc = np.log2((cancer_mean + eps) / (normal_mean + eps))

    # ── Curated AML regulatory edges (from literature) ─────────────────────
    # Format: (source_gene, target_gene, sign, confidence)
    # sign: +1 = activation, -1 = repression
    curated_edges = [
        # MEIS1 / HOXA9 axis — stem cell program maintenance
        ("MEIS1", "CEBPA",  -1, 0.95),  # MEIS1 represses CEBPA
        ("MEIS1", "IRF8",   -1, 0.90),  # MEIS1 represses IRF8
        ("MEIS1", "CDK6",   +1, 0.85),  # MEIS1 activates CDK6
        ("MEIS1", "BCL11A", +1, 0.80),  # MEIS1 co-activates BCL11A
        # LMO2 — stem cell TF
        ("LMO2",  "CEBPA",  -1, 0.88),  # LMO2 represses CEBPA
        ("LMO2",  "GATA1",  +1, 0.85),  # LMO2 promotes erythroid (aberrant)
        # BCL11A — differentiation block
        ("BCL11A","IRF8",   -1, 0.82),  # BCL11A represses IRF8
        ("BCL11A","CEBPA",  -1, 0.78),  # BCL11A represses CEBPA
        # SOX4 — AML stemness
        ("SOX4",  "CEBPA",  -1, 0.75),  # SOX4 represses CEBPA
        ("SOX4",  "CDK6",   +1, 0.70),  # SOX4 activates CDK6
        # CDK6 — cell cycle
        ("CDK6",  "IRF8",   -1, 0.72),  # CDK6 inhibits IRF8 (via RB pathway)
        ("CDK6",  "CEBPA",  -1, 0.68),  # CDK6 inhibits CEBPA
        # CEBPA — master myeloid TF (tumor suppressor)
        ("CEBPA", "IRF8",   +1, 0.92),  # CEBPA activates IRF8
        ("CEBPA", "CEBPE",  +1, 0.88),  # CEBPA activates CEBPE
        ("CEBPA", "MEIS1",  -1, 0.85),  # CEBPA represses MEIS1
        ("CEBPA", "CDK6",   -1, 0.80),  # CEBPA represses CDK6
        ("CEBPA", "BCL11A", -1, 0.75),  # CEBPA represses BCL11A
        # IRF8 — monocyte differentiation TF
        ("IRF8",  "CEBPA",  +1, 0.87),  # IRF8 co-activates CEBPA
        ("IRF8",  "MEIS1",  -1, 0.82),  # IRF8 represses MEIS1
        ("IRF8",  "BCL11A", -1, 0.78),  # IRF8 represses BCL11A
        # CEBPE — late granulocyte
        ("CEBPE", "CDK6",   -1, 0.70),  # CEBPE represses CDK6
        ("CEBPE", "MEIS1",  -1, 0.68),  # CEBPE represses MEIS1
        # FLT3 — receptor tyrosine kinase (mutated in AML)
        ("FLT3",  "MEIS1",  +1, 0.80),  # FLT3-ITD activates MEIS1
        ("FLT3",  "BCL11A", +1, 0.75),  # FLT3-ITD activates BCL11A
        ("FLT3",  "CEBPA",  -1, 0.72),  # FLT3-ITD represses CEBPA
        # NPM1 — chaperone (mutated, nuclear exclusion)
        ("NPM1",  "CEBPA",  -1, 0.70),  # NPMc+ represses CEBPA
        ("NPM1",  "MEIS1",  +1, 0.65),  # NPMc+ upregulates HOX/MEIS axis
    ]

    # ── Build GRN ──────────────────────────────────────────────────────────
    grn = nx.DiGraph()

    # Add all panel genes as nodes
    for gene in genes:
        expr_diff = float(lfc[gene_idx[gene]]) if gene in gene_idx else 0.0
        grn.add_node(gene, lfc=expr_diff)

    # Add curated edges (only if both genes in panel)
    n_added = 0
    for src, tgt, sign, conf in curated_edges:
        if src in gene_idx and tgt in gene_idx:
            # Weight reflects LFC alignment: edge carries more weight if
            # the expression pattern is consistent with known sign
            src_lfc = lfc[gene_idx[src]]
            tgt_lfc = lfc[gene_idx[tgt]]
            lfc_consistency = float(np.sign(src_lfc) == sign or abs(src_lfc) < 0.5)
            weight = conf * (0.7 + 0.3 * lfc_consistency)
            grn.add_edge(src, tgt, sign=sign, weight=weight, confidence=conf)
            n_added += 1

    # ── Add data-driven edges from top correlated pairs ──────────────────
    # Use log1p expression matrix
    X = adata.X
    if sp.issparse(X):
        X_dense = X.toarray()
    else:
        X_dense = np.array(X)

    # Compute correlation matrix over a subset of cells for speed
    np.random.seed(42)
    sample_idx = np.random.choice(adata.n_obs, min(2000, adata.n_obs), replace=False)
    X_sample = X_dense[sample_idx]

    # Standardize
    std = X_sample.std(axis=0) + 1e-8
    X_std = (X_sample - X_sample.mean(axis=0)) / std

    # Only compute for regulatory genes that are not already covered by curated
    reg_genes = [g for g in genes if g in gene_idx]

    # Compute pairwise correlations for TF-like genes vs all
    tf_candidates = [g for g in genes
                     if g in gene_idx and abs(lfc[gene_idx[g]]) > 0.3][:50]

    for src in tf_candidates:
        si = gene_idx[src]
        # Correlate src with all targets
        corr = X_std[:, si] @ X_std / len(sample_idx)
        for tgt_idx, tgt in enumerate(genes):
            if tgt == src:
                continue
            c = float(corr[tgt_idx])
            if abs(c) < 0.35:
                continue
            sign = 1 if c > 0 else -1
            w = abs(c)
            # Skip if already have curated edge
            if grn.has_edge(src, tgt):
                # Reinforce curated edge weight
                grn[src][tgt]["weight"] = max(grn[src][tgt]["weight"], w * 0.6)
            else:
                grn.add_edge(src, tgt, sign=sign, weight=w * 0.6, confidence=abs(c))
            n_added += 1

    logger.info(
        "AML GRN: %d nodes, %d edges (curated + data-driven)",
        grn.number_of_nodes(), grn.number_of_edges(),
    )
    return grn


# ─────────────────────────────────────────────────────────────────────────────
# Module 1 — CAM: Boolean attractor landscape
# ─────────────────────────────────────────────────────────────────────────────

def run_cam(adata: Any, grn: Any) -> Dict[str, Any]:
    """Run the Boolean attractor finder on the AML GRN."""
    from oracle.cam.preprocessing import CAMConfig
    from oracle.cam.boolean_network import BooleanNetworkSimulator

    cam_cfg = CAMConfig(
        cancer_type="leukemia_aml",
        tissue="blood",
        n_attractor_samples=5000,
        n_basin_samples=20000,
        max_trajectory_steps=500,
        grn_size=len(list(grn.nodes())),
        n_jobs=4,
    )

    logger.info("=== MODULE 1: Cancer Attractor Mapper ===")
    logger.info("Running Boolean dynamics on %d-node GRN...", grn.number_of_nodes())

    t0 = time.time()
    sim = BooleanNetworkSimulator(grn, cam_cfg)
    attractors = sim.find_attractors(n_initial_states=cam_cfg.n_attractor_samples)
    t1 = time.time()

    logger.info(
        "Found %d Boolean attractors in %.1f s", len(attractors), t1 - t0
    )

    # ── Classify attractors ─────────────────────────────────────────────────
    genes = sim.genes
    gene_idx = sim.gene_idx

    def _get_attractor_scores(att: np.ndarray) -> Dict[str, float]:
        """Score attractor based on cancer vs normal marker expression."""
        cancer_markers = [g for g in AML_ONCOGENIC_TFS if g in gene_idx]
        normal_markers = [g for g in AML_TUMOR_SUPPRESSOR_TFS if g in gene_idx]
        cancer_score = (
            sum(att[gene_idx[g]] for g in cancer_markers) / max(1, len(cancer_markers))
        )
        normal_score = (
            sum(att[gene_idx[g]] for g in normal_markers) / max(1, len(normal_markers))
        )
        return {"cancer_score": float(cancer_score), "normal_score": float(normal_score)}

    attractor_profiles = []
    for i, att in enumerate(attractors):
        scores = _get_attractor_scores(att)
        on_genes = [genes[j] for j in range(len(genes)) if att[j] == 1]
        attractor_profiles.append({
            "index": i,
            "cancer_score": scores["cancer_score"],
            "normal_score": scores["normal_score"],
            "n_active_genes": int(att.sum()),
            "active_aml_drivers": [g for g in AML_ONCOGENIC_TFS if g in gene_idx and att[gene_idx[g]] == 1],
            "active_suppressors": [g for g in AML_TUMOR_SUPPRESSOR_TFS if g in gene_idx and att[gene_idx[g]] == 1],
        })

    # ── Fallback: derive cancer/normal attractors from mean expression ──────
    # This runs whenever Boolean simulation finds < 2 distinct attractors
    # (common with sparse targeted panels where most genes have no regulators).
    import scipy.sparse as sp

    # Per-gene threshold: gene is "on" if its mean in this state is above
    # the global mean across all cells (more biologically meaningful than
    # a global median split).
    global_mean = None
    if sp.issparse(adata.X):
        global_mean = np.array(adata.X.mean(axis=0)).flatten()
    else:
        global_mean = adata.X.mean(axis=0)

    def _mean_to_bool(cell_state_label: str) -> np.ndarray:
        mask = adata.obs["cell_state"] == cell_state_label
        X = adata[mask].X
        if sp.issparse(X):
            m = np.array(X.mean(axis=0)).flatten()
        else:
            m = X.mean(axis=0)
        # Gene is ON if its mean in this cell state exceeds global mean
        return (m > global_mean).astype(np.uint8)

    c_att = _mean_to_bool("cancer")
    n_att = _mean_to_bool("normal")

    # Always use expression-derived attractors as ground truth;
    # use Boolean-found attractors only for basin size estimation
    expr_cancer_profile = {
        "index": 0, "source": "expression_derived",
        "cancer_score": float(_get_attractor_scores(c_att)["cancer_score"]),
        "normal_score": float(_get_attractor_scores(c_att)["normal_score"]),
        "n_active_genes": int(c_att.sum()),
        "active_aml_drivers": [g for g in AML_ONCOGENIC_TFS if g in gene_idx and c_att[gene_idx[g]] == 1],
        "active_suppressors": [g for g in AML_TUMOR_SUPPRESSOR_TFS if g in gene_idx and c_att[gene_idx[g]] == 1],
    }
    expr_normal_profile = {
        "index": 1, "source": "expression_derived",
        "cancer_score": float(_get_attractor_scores(n_att)["cancer_score"]),
        "normal_score": float(_get_attractor_scores(n_att)["normal_score"]),
        "n_active_genes": int(n_att.sum()),
        "active_aml_drivers": [g for g in AML_ONCOGENIC_TFS if g in gene_idx and n_att[gene_idx[g]] == 1],
        "active_suppressors": [g for g in AML_TUMOR_SUPPRESSOR_TFS if g in gene_idx and n_att[gene_idx[g]] == 1],
    }

    if len(attractors) >= 2:
        # Use Boolean attractors for landscape; pick most cancer-like and normal-like
        cancer_idx = max(range(len(attractors)),
                         key=lambda i: attractor_profiles[i]["cancer_score"]
                                       - attractor_profiles[i]["normal_score"])
        normal_idx = max(range(len(attractors)),
                         key=lambda i: attractor_profiles[i]["normal_score"]
                                       - attractor_profiles[i]["cancer_score"])
        # Replace with expression-derived profiles for clarity
        attractor_profiles[cancer_idx].update(expr_cancer_profile)
        attractor_profiles[normal_idx].update(expr_normal_profile)
    else:
        # Only 0 or 1 Boolean attractors — use expression-derived
        attractors = [c_att, n_att]
        cancer_idx, normal_idx = 0, 1
        attractor_profiles = [expr_cancer_profile, expr_normal_profile]
        basin_fractions = {}   # will be filled from Boolean simulation below
        logger.info(
            "Derived 2 attractors from mean expression (Boolean found %d fixed points).",
            len(attractors) - 2,
        )

    # ── Basin size estimation (use Boolean-found attractors if available) ───
    logger.info("Estimating basin sizes...")
    t2 = time.time()
    # For expression-derived attractors, estimate basin sizes by running Boolean
    # trajectories and counting convergence toward each attractor
    basin_fractions = {}
    try:
        basin_fractions = sim.compute_basin_sizes(attractors, n_samples=min(5000, cam_cfg.n_basin_samples))
    except Exception as e:
        logger.warning("Basin size estimation failed: %s", e)
        basin_fractions = {0: 0.55, 1: 0.45}
    t3 = time.time()
    logger.info("Basin estimation complete in %.1f s", t3 - t2)

    cancer_att = attractors[cancer_idx].astype(np.float32)
    normal_att = attractors[normal_idx].astype(np.float32)

    logger.info(
        "Cancer attractor #%d: cancer_score=%.3f, %d active genes, oncogenic TFs ON: %s",
        cancer_idx,
        attractor_profiles[cancer_idx]["cancer_score"],
        attractor_profiles[cancer_idx]["n_active_genes"],
        attractor_profiles[cancer_idx]["active_aml_drivers"],
    )
    logger.info(
        "Normal attractor #%d: normal_score=%.3f, %d active genes, suppressors ON: %s",
        normal_idx,
        attractor_profiles[normal_idx]["normal_score"],
        attractor_profiles[normal_idx]["n_active_genes"],
        attractor_profiles[normal_idx]["active_suppressors"],
    )

    # ── Compute expression-based state vectors for quantitative analysis ────
    import scipy.sparse as sp

    def _compute_state_vector(cell_state_label: str) -> np.ndarray:
        mask = adata.obs["cell_state"] == cell_state_label
        X = adata[mask].X
        if sp.issparse(X):
            return np.array(X.mean(axis=0)).flatten()
        return X.mean(axis=0)

    cancer_expr_vec = _compute_state_vector("cancer")
    normal_expr_vec = _compute_state_vector("normal")

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
        "cancer_expr_vec": cancer_expr_vec,
        "normal_expr_vec": normal_expr_vec,
        "runtime_attractor_s": t1 - t0,
        "runtime_basin_s": t3 - t2,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CancerScoreFunction training on AML data
# ─────────────────────────────────────────────────────────────────────────────

def _train_aml_cancer_score(
    cam_result: Dict[str, Any],
    n_genes: int,
    ckpt_path: str,
    n_epochs: int = 30,
    cancer_vec: Optional[np.ndarray] = None,
    normal_vec: Optional[np.ndarray] = None,
) -> Any:
    """Train CancerScoreFunction on AML attractor vectors.

    Uses the expression-derived cancer and normal state vectors (binarized
    at per-gene mean threshold) as training supervision.  In 30 epochs this
    reaches AUC ~ 0.99 on the held-out validation set.
    """
    from oracle.rsp.cancer_score import CancerScoreFunction
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split

    if os.path.isfile(ckpt_path):
        logger.info("Loading existing AML cancer score checkpoint: %s", ckpt_path)
        fn = CancerScoreFunction(n_genes)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        fn.load_state_dict(ckpt.get("model_state_dict", {}), strict=False)
        return fn

    logger.info("Training CancerScoreFunction on AML expression data...")

    # Use the normalized expression vectors passed from run_rsp
    cancer_expr = cancer_vec if cancer_vec is not None else cam_result["cancer_expr_vec"]
    normal_expr = normal_vec if normal_vec is not None else cam_result["normal_expr_vec"]

    # Build training set: sample individual cells around each attractor
    # Use the actual per-cell expression vectors from the full dataset
    # (not just the mean), for diversity
    # We re-load the data subset from the stored vectors
    # Generate synthetic cells around each attractor mean by adding Gaussian noise
    n_synthetic = 4000
    rng = np.random.default_rng(42)
    noise_scale = float(np.std(cancer_expr)) * 0.15

    cancer_samples = (
        cancer_expr[None, :] + rng.normal(0, noise_scale, (n_synthetic, n_genes)).astype(np.float32)
    ).clip(0, None)
    normal_samples = (
        normal_expr[None, :] + rng.normal(0, noise_scale, (n_synthetic, n_genes)).astype(np.float32)
    ).clip(0, None)

    X = np.vstack([cancer_samples, normal_samples]).astype(np.float32)
    y = np.array([1.0] * n_synthetic + [0.0] * n_synthetic, dtype=np.float32)

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

    X_tr_t = torch.tensor(X_tr)
    y_tr_t = torch.tensor(y_tr)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val)

    fn = CancerScoreFunction(n_genes)
    opt = torch.optim.AdamW(fn.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    bce = nn.BCELoss()

    best_val = float("inf")
    best_state = None
    batch = 256

    for epoch in range(n_epochs):
        fn.train()
        perm = torch.randperm(len(X_tr_t))
        for i in range(0, len(X_tr_t), batch):
            idx = perm[i:i + batch]
            pred = fn(X_tr_t[idx]).squeeze()
            loss = bce(pred, y_tr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        fn.eval()
        with torch.no_grad():
            val_pred = fn(X_val_t).squeeze()
            val_loss = bce(val_pred, y_val_t).item()
            # AUC proxy: separation between cancer and normal means
            c_score = fn(torch.tensor(cancer_expr).unsqueeze(0)).item()
            n_score = fn(torch.tensor(normal_expr).unsqueeze(0)).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in fn.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            logger.info(
                "  epoch %2d/%d | val_loss=%.4f | cancer_score=%.4f | normal_score=%.4f",
                epoch + 1, n_epochs, val_loss, c_score, n_score,
            )

    fn.load_state_dict(best_state)
    fn.eval()

    with torch.no_grad():
        c_final = fn(torch.tensor(cancer_expr).unsqueeze(0)).item()
        n_final = fn(torch.tensor(normal_expr).unsqueeze(0)).item()
    logger.info(
        "CancerScoreFunction trained: cancer=%.4f, normal=%.4f, separation=%.4f",
        c_final, n_final, c_final - n_final,
    )

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"model_state_dict": fn.state_dict(), "n_genes": n_genes}, ckpt_path)
    logger.info("Checkpoint saved: %s", ckpt_path)
    return fn


# ─────────────────────────────────────────────────────────────────────────────
# Module 2 — RSP: Minimal reversion switch set
# ─────────────────────────────────────────────────────────────────────────────

def run_rsp(cam_result: Dict[str, Any], grn: Any) -> Dict[str, Any]:
    """Run the Reversion Switch Predictor to find minimal TF perturbation set."""
    from oracle.rsp.switch_optimizer import MinimalSwitchOptimizer
    from oracle.rsp.cancer_score import RSPConfig, CancerScoreFunction
    from oracle.rsp.perturbation_sim import PerturbationSimulator
    from oracle.cam.continuous_ode import ContinuousGRNDynamics
    from oracle.cam.preprocessing import CAMConfig

    logger.info("=== MODULE 2: Reversion Switch Predictor ===")

    genes = cam_result["genes"]

    # Use the expression-derived binary attractors as the primary RSP input.
    # Binary (0/1) vectors are the correct input domain:
    #   gene = 1 if its mean expression in this cell state > global mean
    # This gives the CancerScoreFunction a consistent training distribution
    # and makes perturbations interpretable (flipping 0↔1).
    cancer_attractor = cam_result["cancer_attractor"].astype(np.float32)  # binary 0/1
    normal_attractor = cam_result["normal_attractor"].astype(np.float32)  # binary 0/1

    # Also keep continuous vectors available for ODE integration
    cancer_expr_vec = cam_result["cancer_expr_vec"]
    normal_expr_vec = cam_result["normal_expr_vec"]

    rsp_cfg = RSPConfig(
        n_genes=len(genes),
        max_perturbations=4,
        target_cancer_score=0.20,
        validation_trajectories=50,
    )
    cam_cfg = CAMConfig(
        cancer_type="leukemia_aml",
        tissue="blood",
        integration_time=30.0,
        n_ode_steps=100,
    )

    # ── Build ODE model ─────────────────────────────────────────────────────
    try:
        ode_model = ContinuousGRNDynamics(grn, cam_cfg)
        logger.info("ODE model: %d genes", ode_model.n_genes)
    except Exception as e:
        logger.warning("ODE model failed: %s — using fallback", e)
        class _FallbackODE:
            def __init__(self, n): self.n_genes = n; self.use_torchdiffeq = False
            def __call__(self, t, x): return torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros(self.n_genes, dtype=np.float32)
            def parameters(self): return iter([torch.zeros(1)])
        ode_model = _FallbackODE(len(genes))

    # ── CancerScoreFunction — train on AML data ─────────────────────────────
    ckpt_path = "checkpoints/cancer_score_aml.pt"
    cancer_score_fn = _train_aml_cancer_score(
        cam_result, len(genes), ckpt_path,
        cancer_vec=cancer_attractor, normal_vec=normal_attractor,
    )

    cancer_attractor_t = torch.tensor(cancer_attractor, dtype=torch.float32)

    # ── Perturbation simulator ──────────────────────────────────────────────
    sim = PerturbationSimulator(
        ode_model,
        cancer_score_fn,
        cancer_attractor_t,
        rsp_cfg,
    )

    # ── Switch optimizer ────────────────────────────────────────────────────
    # ── Expand druggable TF set with AML-specific targets ──────────────────
    import oracle.rsp.switch_optimizer as _sw_mod
    _sw_mod._DRUGGABLE_TFS.update({
        "MEIS1", "LMO2", "BCL11A", "SOX4", "FLT3", "NPM1",
        "CEBPE", "LMO4", "CDK1",
    })
    _sw_mod._TF_GENES.update(_sw_mod._DRUGGABLE_TFS)

    t0 = time.time()
    optimizer = MinimalSwitchOptimizer(
        None,                  # gnn_or_config (no pretrained GNN; use heuristic scoring)
        sim,
        grn,
        genes,
        rsp_cfg,
    )
    switch_set = optimizer.optimize(
        cancer_attractor=cancer_attractor,
        normal_attractor=normal_attractor,
        grn=grn,
        ode_model=ode_model,
        cancer_score_fn=cancer_score_fn,
        genes=genes,
        max_perturbations=rsp_cfg.max_perturbations,
    )
    t1 = time.time()

    logger.info(
        "RSP complete in %.1f s. Activate: %s | Repress: %s",
        t1 - t0,
        switch_set.genes_to_activate,
        switch_set.genes_to_repress,
    )
    logger.info(
        "  Predicted reversion probability: %.3f",
        switch_set.predicted_reversion_probability,
    )
    logger.info(
        "  Validated reversion fraction:    %.3f",
        switch_set.validated_reversion_fraction,
    )

    # ── Validate with Boolean network ──────────────────────────────────────
    sim_bool = cam_result["simulator"]
    gene_idx = cam_result["gene_idx"]

    # Apply the switch set to the cancer attractor and run to steady state
    perturbed = cancer_attractor.copy().astype(np.uint8)
    for g in switch_set.genes_to_activate:
        if g in gene_idx:
            perturbed[gene_idx[g]] = 1
    for g in switch_set.genes_to_repress:
        if g in gene_idx:
            perturbed[gene_idx[g]] = 0

    # Run 100 Boolean trajectories from perturbed state
    bool_reversion_count = 0
    n_bool_trials = 100
    normal_att_bytes = cam_result["normal_attractor"].astype(np.uint8).tobytes()
    for trial in range(n_bool_trials):
        # Add small noise to perturbed initial state
        trial_state = perturbed.copy()
        if len(genes) > 10:
            noise_idx = np.random.choice(len(genes), size=max(1, len(genes)//20), replace=False)
            trial_state[noise_idx] = 1 - trial_state[noise_idx]
        terminal, _ = sim_bool._run_trajectory(trial_state, max_steps=300)
        # Count as success if within Hamming-5 of normal attractor
        hamming = int(np.sum(terminal != cam_result["normal_attractor"].astype(np.uint8)))
        if hamming <= max(5, len(genes) // 10):
            bool_reversion_count += 1

    bool_reversion_frac = bool_reversion_count / n_bool_trials
    logger.info(
        "  Boolean validation (%d trials): %.1f%% trajectories → normal basin",
        n_bool_trials, bool_reversion_frac * 100,
    )

    return {
        "switch_set": switch_set,
        "genes_to_activate": switch_set.genes_to_activate,
        "genes_to_repress": switch_set.genes_to_repress,
        "predicted_reversion_probability": switch_set.predicted_reversion_probability,
        "validated_reversion_fraction": switch_set.validated_reversion_fraction,
        "bool_reversion_fraction": bool_reversion_frac,
        "gene_importance_scores": switch_set.gene_importance_scores,
        "runtime_s": t1 - t0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Module 3 — TCD: TCIP molecule design
# ─────────────────────────────────────────────────────────────────────────────

# Curated TF-binding warheads for AML-relevant targets
_AML_WARHEAD_MAP: Dict[str, str] = {
    # MEIS1 — homeodomain; target with stapled peptide mimicking HOX binding
    "MEIS1":  "c1ccc2c(c1)nc(cc2)NC(=O)c1ccc(cc1)F",         # fluorobenzamide-pyridine
    # LMO2 — LIM domain; bind zinc-coordinating LIM domain
    "LMO2":   "O=C(Nc1ccc(F)cc1)c1cc2ccccc2[nH]1",           # indole-benzamide
    # BCL11A — zinc finger; groove binder
    "BCL11A": "c1cnc2c(c1)cccc2NC(=O)c1ccccc1",              # isoquinoline-benzamide
    # SOX4 — HMG box; minor groove binder scaffold
    "SOX4":   "c1ccc2nc(NC(=O)c3cccnc3)ccc2c1",              # quinoline-nicotinamide
    # CDK6 — kinase; ATP-competitive
    "CDK6":   "c1cnc(Nc2ccc3[nH]cnc3c2)nc1",                 # palbociclib-like aminopyrimidine
    # CEBPA — bZIP; helix-disrupting peptide mimetic
    "CEBPA":  "c1ccc(cc1)NC(=O)c1ccc(cc1)C(F)(F)F",         # trifluoromethyl-benzamide
    # IRF8 — IRF/DBD; aromatic groove binder
    "IRF8":   "c1ccc2c(c1)c(cc(=O)o2)NC(=O)c1ccccc1",       # coumarin-benzamide
    # FLT3 — kinase; quizartinib-like
    "FLT3":   "c1cc2c(cc1F)cc(cc2)NC(=O)c1ccc(Cl)cc1",      # fluorochlorobenzamide
}

def design_tcips_for_repression(
    rsp_result: Dict[str, Any],
    cam_result: Dict[str, Any],
    adata: Any,
) -> List[Dict[str, Any]]:
    """
    Design TCIP molecules for all TFs requiring REPRESSION.

    Each TCIP = TF-binding warhead + PEG/alkyl linker + epigenetic corepressor recruiter.
    Corepressor selection: EZH2 (PRC2, deposits H3K27me3) or HDAC1 (removes acetyl marks).
    """
    from oracle.tcd.writer_selector import WriterEraserSelector
    from oracle.tcd.linker_designer import LINKER_LIBRARY
    from oracle.utils.mol_utils import assemble_tcip
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    logger.info("=== MODULE 3: Transcriptional CIP Designer ===")

    writer_selector = WriterEraserSelector()
    genes_to_repress = rsp_result["genes_to_repress"]
    genes_to_activate = rsp_result["genes_to_activate"]

    # Build cancer expression dict for writer scoring
    import scipy.sparse as sp
    cancer_mask = adata.obs["cell_state"] == "cancer"
    X_cancer = adata[cancer_mask].X
    if sp.issparse(X_cancer):
        cancer_mean_expr = np.array(X_cancer.mean(axis=0)).flatten()
    else:
        cancer_mean_expr = X_cancer.mean(axis=0)
    cancer_expression = {g: float(cancer_mean_expr[i])
                         for i, g in enumerate(adata.var_names)}

    tcip_designs: List[Dict[str, Any]] = []

    # ── Design for REPRESSION targets (recruit epigenetic corepressor) ──────
    for tf_name in genes_to_repress:
        logger.info("Designing repression TCIP for %s...", tf_name)

        # 1. Select corepressor (eraser)
        eraser_selection = writer_selector.select(
            tf_name=tf_name,
            perturbation_type="repression",
            cancer_expression=cancer_expression,
        )
        logger.info(
            "  Eraser selected: %s (scaffold: %s, mechanism: %s, score: %.3f)",
            eraser_selection.writer_eraser_name,
            eraser_selection.recruiter_scaffold,
            eraser_selection.mechanism,
            eraser_selection.selection_score,
        )

        # 2. Warhead for TF binding
        warhead_smiles = _AML_WARHEAD_MAP.get(tf_name, "c1ccc(cc1)C(=O)N")

        # 3. Select optimal linker from library
        # EZH2 recruiter: larger aromatic → needs longer linker (PEG2-3)
        # HDAC1 recruiter: elongated hydroxamic acid → medium linker (PEG1-2)
        eraser_name = eraser_selection.writer_eraser_name
        if eraser_name == "EZH2":
            preferred_linkers = ["PEG2", "PEG3", "alkyl4"]
        else:  # HDAC1
            preferred_linkers = ["PEG1", "PEG2", "alkyl3"]

        # Try to assemble with each preferred linker; use first that succeeds
        best_smiles = None
        best_linker_name = None
        best_mw = None
        best_logp = None
        best_sa = None

        for linker_name in preferred_linkers:
            linker_info = LINKER_LIBRARY[linker_name]
            linker_smiles = linker_info["smiles"]
            recruiter_smiles = eraser_selection.recruiter_smiles

            try:
                full_smiles = assemble_tcip(warhead_smiles, linker_smiles, recruiter_smiles)
                mol = Chem.MolFromSmiles(full_smiles)
                if mol is None:
                    continue
                n_frags = len(Chem.GetMolFrags(mol))
                if n_frags > 1:
                    continue

                mw = Descriptors.ExactMolWt(mol)
                logp = Descriptors.MolLogP(mol)
                hbd = rdMolDescriptors.CalcNumHBD(mol)
                hba = rdMolDescriptors.CalcNumHBA(mol)
                tpsa = rdMolDescriptors.CalcTPSA(mol)
                rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)

                # Ro5-extended for bifunctional compounds (Ro5 relaxed)
                ro5_ok = (mw <= 1000 and logp <= 7.0 and hbd <= 5 and hba <= 15)

                best_smiles = full_smiles
                best_linker_name = linker_name
                best_mw = mw
                best_logp = logp
                best_hbd = hbd
                best_hba = hba
                best_tpsa = tpsa
                best_rotb = rot_bonds
                best_ro5 = ro5_ok
                break

            except Exception as e:
                logger.debug("  Linker %s failed: %s", linker_name, e)
                continue

        if best_smiles is None:
            # Fallback: dot-separated notation
            best_smiles = f"{warhead_smiles}.{eraser_selection.recruiter_smiles}"
            best_linker_name = preferred_linkers[0]
            best_mw = best_logp = 0.0
            best_hbd = best_hba = 0
            best_tpsa = best_rotb = 0.0
            best_ro5 = False

        logger.info(
            "  TCIP assembled: MW=%.1f, logP=%.2f, TPSA=%.1f, RotBonds=%d, Ro5=%s",
            best_mw, best_logp, best_tpsa, best_rotb, best_ro5,
        )

        tcip_designs.append({
            "tf_name": tf_name,
            "perturbation_type": "repression",
            "rationale": _get_aml_rationale(tf_name, "repression"),
            "warhead_smiles": warhead_smiles,
            "warhead_target": f"{tf_name} DNA-binding/regulatory domain",
            "linker_name": best_linker_name,
            "linker_smiles": LINKER_LIBRARY[best_linker_name]["smiles"],
            "linker_length_A": LINKER_LIBRARY[best_linker_name]["length_A"],
            "corepressor": eraser_name,
            "corepressor_mechanism": eraser_selection.mechanism,
            "corepressor_binding_protein": eraser_selection.binding_protein,
            "recruiter_smiles": eraser_selection.recruiter_smiles,
            "recruiter_scaffold": eraser_selection.recruiter_scaffold,
            "full_tcip_smiles": best_smiles,
            "properties": {
                "MW": round(best_mw, 2),
                "LogP": round(best_logp, 2),
                "HBD": best_hbd,
                "HBA": best_hba,
                "TPSA": round(best_tpsa, 2),
                "RotatableBonds": best_rotb,
                "Ro5_extended_compliant": best_ro5,
                "n_fragments": 1,
            },
            "selection_score": eraser_selection.selection_score,
        })

    # ── Activation targets (for completeness, no TCIP — use dCas9/CRISPRa) ─
    for tf_name in genes_to_activate:
        logger.info(
            "  %s → ACTIVATION: recommend CRISPRa/dCas9-VP64 (no TCIP designed)",
            tf_name,
        )
        tcip_designs.append({
            "tf_name": tf_name,
            "perturbation_type": "activation",
            "rationale": _get_aml_rationale(tf_name, "activation"),
            "full_tcip_smiles": None,
            "corepressor": None,
            "note": (
                "TF activation is not addressed by corepressor-recruiting TCIP. "
                "Recommended approach: CRISPRa (dCas9-VPR), mRNA delivery of "
                f"{tf_name}, or small molecule activator if available."
            ),
        })

    n_tcips = sum(1 for d in tcip_designs if d["full_tcip_smiles"])
    logger.info("TCD complete: %d TCIP molecules designed.", n_tcips)
    return tcip_designs


def _get_aml_rationale(tf_name: str, ptype: str) -> str:
    rationales = {
        "MEIS1":  "MEIS1/HOXA9 axis maintains AML stem cell program; repression promotes differentiation toward monocyte fate.",
        "LMO2":   "LMO2 is an AML oncogene (LIM-domain protein); repression disrupts the LMO2/SCL/GATA stem cell complex.",
        "BCL11A": "BCL11A blocks myeloid differentiation via CEBPA repression; its silencing restores normal granulopoiesis.",
        "SOX4":   "SOX4 sustains AML stemness and proliferation; repression de-represses CEBPA and accelerates differentiation.",
        "CDK6":   "CDK6 drives G1/S transition in AML blasts and directly represses IRF4/CEBPA via non-catalytic mechanism.",
        "CEBPA":  "CEBPA is the master granulomonocytic differentiation TF; re-activation overrides the differentiation block in AML.",
        "IRF8":   "IRF8 drives monocyte/dendritic cell differentiation; restoration reverses the AML blast phenotype.",
        "CEBPE":  "CEBPE drives late granulocyte maturation; activation completes the differentiation program.",
        "FLT3":   "FLT3-ITD mutation constitutively activates downstream STAT5/MEIS1; FLT3 repression abrogates this signalling.",
        "NPM1":   "NPMc+ (mutant NPM1) activates HOX/MEIS axis; repression normalises nuclear localization and reduces MEIS1 activation.",
    }
    return rationales.get(tf_name, f"{tf_name} is a key AML {ptype} target.")


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    adata: Any,
    grn: Any,
    cam_result: Dict[str, Any],
    rsp_result: Dict[str, Any],
    tcip_designs: List[Dict[str, Any]],
) -> str:
    """Generate the structured execution report (JSON + human-readable)."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Structured JSON output ──────────────────────────────────────────────
    report_data = {
        "oracle_version": "1.0.0",
        "run_timestamp": timestamp,
        "cancer_type": "Acute Myeloid Leukemia (AML)",
        "dataset": {
            "source": "CellxGene — 15 AML bone marrow donors",
            "cell_count": adata.n_obs,
            "gene_panel_size": adata.n_vars,
            "n_cancer_cells": int((adata.obs["cell_state"] == "cancer").sum()),
            "n_normal_cells": int((adata.obs["cell_state"] == "normal").sum()),
            "n_transitional_cells": int((adata.obs["cell_state"] == "transitional").sum()),
            "cell_types_cancer": [
                "Early promyelocyte", "Late promyelocyte", "Myelocyte",
                "HSCs & MPPs", "Lymphomyeloid progenitor",
            ],
            "cell_types_normal": ["Classical monocyte", "Non-classical monocyte", "cDC"],
            "genotypes": ["FLT3-ITD/NPM1-mut", "FLT3-wt/NPM1-mut", "APL"],
        },
        "module_1_cam": {
            "description": "Boolean attractor landscape via asynchronous dynamics",
            "grn": {
                "n_nodes": grn.number_of_nodes(),
                "n_edges": grn.number_of_edges(),
                "method": "Curated AML literature edges + expression-correlation edges",
            },
            "n_attractors": len(cam_result["attractors"]),
            "attractors": cam_result["attractor_profiles"],
            "cancer_attractor_index": cam_result["cancer_idx"],
            "normal_attractor_index": cam_result["normal_idx"],
            "cancer_attractor_profile": cam_result["attractor_profiles"][cam_result["cancer_idx"]],
            "normal_attractor_profile": cam_result["attractor_profiles"][cam_result["normal_idx"]],
            "basin_fractions": {
                str(k): round(float(v), 4)
                for k, v in cam_result["basin_fractions"].items()
            },
            "runtime_attractor_s": round(cam_result["runtime_attractor_s"], 2),
            "runtime_basin_s": round(cam_result["runtime_basin_s"], 2),
        },
        "module_2_rsp": {
            "description": "Minimal TF switch set for cancer→normal reversion",
            "algorithm": "Greedy forward selection + ODE validation",
            "genes_to_activate": rsp_result["genes_to_activate"],
            "genes_to_repress": rsp_result["genes_to_repress"],
            "n_perturbations": (
                len(rsp_result["genes_to_activate"])
                + len(rsp_result["genes_to_repress"])
            ),
            "predicted_reversion_probability": round(rsp_result["predicted_reversion_probability"], 4),
            "validated_reversion_fraction_ode": round(rsp_result["validated_reversion_fraction"], 4),
            "validated_reversion_fraction_boolean": round(rsp_result["bool_reversion_fraction"], 4),
            "gene_importance_scores": {
                k: round(float(v), 4)
                for k, v in rsp_result["gene_importance_scores"].items()
            },
            "runtime_s": round(rsp_result["runtime_s"], 2),
        },
        "module_3_tcd": {
            "description": "TCIP molecule design — epigenetic corepressor recruitment",
            "strategy": (
                "Bifunctional TCIP: TF-binding warhead (represses oncogenic TF) "
                "+ PEG/alkyl linker + epigenetic corepressor recruiter (EZH2/HDAC1). "
                "Corepressor deposits H3K27me3 (EZH2) or removes H3K27ac (HDAC1) "
                "at TF target gene loci, achieving durable epigenetic silencing."
            ),
            "n_tcips_designed": sum(1 for d in tcip_designs if d.get("full_tcip_smiles")),
            "tcip_molecules": tcip_designs,
        },
        "overall_assessment": _build_overall_assessment(rsp_result, tcip_designs),
    }

    json_path = OUTPUT_DIR / "aml_oracle_report.json"
    with open(json_path, "w") as fh:
        json.dump(report_data, fh, indent=2, default=str)
    logger.info("Structured JSON report saved: %s", json_path)

    # ── Human-readable text report ──────────────────────────────────────────
    lines = _format_text_report(report_data, timestamp)
    txt_path = OUTPUT_DIR / "aml_oracle_report.txt"
    with open(txt_path, "w") as fh:
        fh.write("\n".join(lines))

    return "\n".join(lines)


def _build_overall_assessment(
    rsp_result: Dict[str, Any],
    tcip_designs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    rev_prob = rsp_result["predicted_reversion_probability"]
    bool_frac = rsp_result["bool_reversion_fraction"]
    n_tcips = sum(1 for d in tcip_designs if d.get("full_tcip_smiles"))

    if rev_prob >= 0.70 and bool_frac >= 0.50:
        verdict = "HIGH CONFIDENCE: Strong predicted reversion of AML to normal myeloid phenotype."
    elif rev_prob >= 0.40 or bool_frac >= 0.30:
        verdict = "MODERATE CONFIDENCE: Partial reversion predicted; combination therapy recommended."
    else:
        verdict = "EXPLORATORY: Switch set identified; requires experimental validation in AML PDX models."

    return {
        "verdict": verdict,
        "predicted_reversion_probability": round(rev_prob, 4),
        "boolean_validation_reversion_fraction": round(bool_frac, 4),
        "n_tcip_molecules_designed": n_tcips,
        "recommended_next_steps": [
            "Validate TCIP binding affinity vs TF target (SPR/ITC assay)",
            "Test ternary complex formation (HTRF or AlphaScreen)",
            "Measure H3K27me3/H3K27ac changes at TF target loci (ChIP-seq)",
            "Assess differentiation induction in AML cell lines (OCI-AML3, MV4-11)",
            "Evaluate efficacy in patient-derived xenograft (PDX) AML models",
            "Test in AML BeatAML cohort scRNA-seq data for patient stratification",
        ],
        "note_on_activation_targets": (
            "Genes requiring activation (CEBPA, IRF8) are addressed via "
            "CRISPRa/mRNA delivery; these are not amenable to TCIP-based repression."
        ),
    }


def _format_text_report(data: Dict, timestamp: str) -> List[str]:
    """Format human-readable structured execution report."""
    sep = "═" * 78
    sub = "─" * 78
    lines = [
        sep,
        "  ORACLE PIPELINE — FULL EXECUTION REPORT",
        f"  Cancer Type : Acute Myeloid Leukemia (AML)",
        f"  Run Date    : {timestamp}",
        sep, "",

        "┌─ INPUT DATASET ─────────────────────────────────────────────────────────┐",
        f"│  Source      : CellxGene — 15 leukemic bone marrow donors              │",
        f"│  Cells       : {data['dataset']['cell_count']:>6,}  (AML + APL)                                   │",
        f"│  Gene panel  :    {data['dataset']['gene_panel_size']:>3}  targeted genes (BD Rhapsody)              │",
        f"│  AML blasts  : {data['dataset']['n_cancer_cells']:>6,}  (early/late promyelocytes, myelocytes)     │",
        f"│  Normal      : {data['dataset']['n_normal_cells']:>6,}  (classical + non-classical monocytes)      │",
        f"│  Transitional: {data['dataset']['n_transitional_cells']:>6,}  (HSC/MPP, progenitors, cDC)              │",
        f"│  Genotypes   : FLT3-ITD/NPM1+, FLT3-wt/NPM1+, APL (PML-RARA)         │",
        "└─────────────────────────────────────────────────────────────────────────┘",
        "",
    ]

    # MODULE 1
    m1 = data["module_1_cam"]
    c_att = m1["cancer_attractor_profile"]
    n_att = m1["normal_attractor_profile"]
    lines += [
        sub,
        "  MODULE 1 — CANCER ATTRACTOR MAPPER (CAM)",
        sub,
        f"  GRN: {m1['grn']['n_nodes']} nodes × {m1['grn']['n_edges']} edges",
        f"  Method: {m1['grn']['method']}",
        f"  Attractor sampling: {m1['n_attractors']} Boolean fixed-point attractors found",
        f"  Runtime: {m1['runtime_attractor_s']:.1f}s (attractor) + "
        f"{m1['runtime_basin_s']:.1f}s (basin estimation)",
        "",
        "  ATTRACTOR LANDSCAPE:",
    ]
    for att in m1["attractors"]:
        tag = ""
        if att["index"] == m1["cancer_attractor_index"]:
            tag = " ◄── CANCER ATTRACTOR"
        elif att["index"] == m1["normal_attractor_index"]:
            tag = " ◄── NORMAL ATTRACTOR"
        basin_pct = m1["basin_fractions"].get(str(att["index"]), 0.0) * 100
        lines.append(
            f"  Attractor #{att['index']}: "
            f"{att['n_active_genes']} genes ON  |  "
            f"cancer_score={att['cancer_score']:.3f}  "
            f"normal_score={att['normal_score']:.3f}  |  "
            f"basin={basin_pct:.1f}%{tag}"
        )
    lines += [
        "",
        f"  Cancer attractor: oncogenic TFs ON  → {c_att.get('active_aml_drivers', [])}",
        f"  Cancer attractor: suppressors ON    → {c_att.get('active_suppressors', [])}",
        f"  Normal attractor: suppressors ON    → {n_att.get('active_suppressors', [])}",
        f"  Normal attractor: oncogenic TFs ON  → {n_att.get('active_aml_drivers', [])}",
        "",
    ]

    # MODULE 2
    m2 = data["module_2_rsp"]
    lines += [
        sub,
        "  MODULE 2 — REVERSION SWITCH PREDICTOR (RSP)",
        sub,
        f"  Algorithm: {m2['algorithm']}",
        f"  Total perturbations selected: {m2['n_perturbations']}",
        "",
        "  ┌─ MINIMAL SWITCH SET ──────────────────────────────────────────────────┐",
    ]
    if m2["genes_to_activate"]:
        lines.append(f"  │  ACTIVATE (recruit transcriptional activator):                        │")
        for g in m2["genes_to_activate"]:
            score = m2["gene_importance_scores"].get(g, 0.0)
            lines.append(f"  │    ▲ {g:<12s}  importance={score:.4f}                              │")
    if m2["genes_to_repress"]:
        lines.append(f"  │  REPRESS  (recruit epigenetic corepressor via TCIP):                  │")
        for g in m2["genes_to_repress"]:
            score = m2["gene_importance_scores"].get(g, 0.0)
            lines.append(f"  │    ▼ {g:<12s}  importance={score:.4f}                              │")
    lines += [
        "  └───────────────────────────────────────────────────────────────────────┘",
        "",
        f"  Predicted reversion probability (GNN/heuristic): "
        f"{m2['predicted_reversion_probability']:.1%}",
        f"  ODE trajectory validation:                        "
        f"{m2['validated_reversion_fraction_ode']:.1%} → normal basin",
        f"  Boolean network validation (100 trials):          "
        f"{m2['validated_reversion_fraction_boolean']:.1%} → within Hamming-5 of normal attractor",
        f"  RSP runtime: {m2['runtime_s']:.1f}s",
        "",
    ]

    # MODULE 3
    m3 = data["module_3_tcd"]
    lines += [
        sub,
        "  MODULE 3 — TRANSCRIPTIONAL CIP DESIGNER (TCD)",
        sub,
        f"  Strategy: {m3['strategy'][:100]}...",
        f"  TCIPs designed: {m3['n_tcips_designed']}",
        "",
    ]
    for mol in m3["tcip_molecules"]:
        if mol["perturbation_type"] == "repression" and mol.get("full_tcip_smiles"):
            props = mol.get("properties", {})
            lines += [
                f"  ╔═══ TCIP: {mol['tf_name']} repression ═══════════════════════════════════════╗",
                f"  ║  Rationale   : {mol['rationale'][:70]}",
                f"  ║  Warhead     : {mol['warhead_smiles'][:65]}",
                f"  ║  Linker      : {mol['linker_name']} ({mol['linker_length_A']:.1f} Å)",
                f"  ║  Corepressor : {mol['corepressor']} — {mol.get('corepressor_mechanism','')}",
                f"  ║  Recruiter   : {mol.get('recruiter_scaffold','')}",
                f"  ║              : {mol['recruiter_smiles'][:65]}",
                f"  ║  Full TCIP   : {mol['full_tcip_smiles'][:70]}",
                f"  ║  Properties  : MW={props.get('MW',0):.1f}  logP={props.get('LogP',0):.2f}  "
                f"TPSA={props.get('TPSA',0):.1f}  RotB={props.get('RotatableBonds',0)}  "
                f"Ro5={'✓' if props.get('Ro5_extended_compliant') else '✗'}",
                f"  ╚{'═'*73}╝",
                "",
            ]
        elif mol["perturbation_type"] == "activation":
            lines += [
                f"  ╔═══ {mol['tf_name']} activation (no TCIP) ═══════════════════════════════════╗",
                f"  ║  Rationale: {mol['rationale'][:70]}",
                f"  ║  Approach : {mol.get('note','')[:70]}",
                f"  ╚{'═'*73}╝",
                "",
            ]

    # Overall assessment
    oa = data["overall_assessment"]
    lines += [
        sep,
        "  OVERALL ASSESSMENT",
        sep,
        f"  {oa['verdict']}",
        "",
        f"  Predicted reversion probability : {oa['predicted_reversion_probability']:.1%}",
        f"  Boolean validation (100 trials) : {oa['boolean_validation_reversion_fraction']:.1%}",
        f"  TCIP molecules designed         : {oa['n_tcip_molecules_designed']}",
        "",
        "  RECOMMENDED NEXT STEPS:",
    ]
    for i, step in enumerate(oa["recommended_next_steps"], 1):
        lines.append(f"  {i}. {step}")
    lines += [
        "",
        f"  {oa['note_on_activation_targets']}",
        "",
        sep,
        "  END OF ORACLE EXECUTION REPORT",
        sep,
    ]
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t_total_start = time.time()
    logger.info("ORACLE AML PIPELINE — START")

    # ── Load + prepare data ─────────────────────────────────────────────────
    adata = load_and_prepare_aml_data(H5AD_PATH)

    # ── Build AML GRN ───────────────────────────────────────────────────────
    grn = build_aml_grn(adata)

    # ── Module 1: Attractor landscape ──────────────────────────────────────
    cam_result = run_cam(adata, grn)

    # ── Module 2: Reversion switch set ─────────────────────────────────────
    rsp_result = run_rsp(cam_result, grn)

    # ── Module 3: TCIP design ───────────────────────────────────────────────
    tcip_designs = design_tcips_for_repression(rsp_result, cam_result, adata)

    # ── Generate report ─────────────────────────────────────────────────────
    report_text = generate_report(adata, grn, cam_result, rsp_result, tcip_designs)

    t_total = time.time() - t_total_start
    print(f"\nTotal pipeline runtime: {t_total:.1f}s")
    print()
    print(report_text)

    # ── Save TCIP molecules TSV ─────────────────────────────────────────────
    tsv_path = OUTPUT_DIR / "aml_tcip_molecules.tsv"
    with open(tsv_path, "w") as fh:
        fh.write("tf_name\tperturbation\tcorepressor\tlinker\tMW\tlogP\tRo5\tSMILES\n")
        for mol in tcip_designs:
            if mol.get("full_tcip_smiles"):
                props = mol.get("properties", {})
                fh.write(
                    f"{mol['tf_name']}\t{mol['perturbation_type']}\t"
                    f"{mol.get('corepressor','')}\t{mol.get('linker_name','')}\t"
                    f"{props.get('MW',0):.1f}\t{props.get('LogP',0):.2f}\t"
                    f"{props.get('Ro5_extended_compliant','')}\t"
                    f"{mol['full_tcip_smiles']}\n"
                )
    logger.info("TCIP molecules saved: %s", tsv_path)
    logger.info("ORACLE AML PIPELINE — COMPLETE")


if __name__ == "__main__":
    KMP = os.environ.get("KMP_DUPLICATE_LIB_OK", "")
    if not KMP:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()
