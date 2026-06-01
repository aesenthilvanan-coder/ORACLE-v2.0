#!/usr/bin/env python3
"""
Train the CancerScoreFunction on real tumor scRNA-seq data.

Usage
-----
python scripts/train_cancer_score.py \
    --h5ad data/raw/scrnaseq/crc_untreated_epithelial.h5ad \
    --cancer-type colorectal \
    --output checkpoints/cancer_score_crc.pt

The script works for any cancer type supported by CellStateAnnotator.
Cell type annotations from the h5ad 'cell_type' column are used directly
when available (e.g., "malignant cell" = cancer).  Marker-gene-based
scoring provides a fallback for unannotated datasets.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label assignment from CellxGene cell_type annotations
# ---------------------------------------------------------------------------

# Cell types that unambiguously indicate a malignant / cancer state
CANCER_CELL_TYPES = {
    "malignant cell",
    "tumor cell",
    "cancer cell",
    "neoplastic cell",
}

# Cell type substrings that indicate normal differentiated state
NORMAL_CELL_TYPE_KEYWORDS = [
    "colonocyte",
    "goblet cell",
    "enterocyte",
    "crypt stem cell",
    "enteroendocrine",
    "tuft cell",
    "paneth cell",
    "hepatocyte",
    "pneumocyte",
    "alveolar",
    "keratinocyte",
    "fibroblast",
    "endothelial",
    "erythrocyte",
    "platelet",
    "basophil",
    "eosinophil",
    "neutrophil",
]


def assign_labels_from_annotations(cell_types: np.ndarray) -> np.ndarray:
    """Assign binary cancer (1) / normal (0) labels from cell_type strings.

    Returns -1 for cells that cannot be unambiguously assigned.
    """
    labels = np.full(len(cell_types), -1, dtype=np.int8)
    for i, ct in enumerate(cell_types):
        ct_lower = str(ct).lower()
        if ct_lower in CANCER_CELL_TYPES:
            labels[i] = 1
        elif any(kw in ct_lower for kw in NORMAL_CELL_TYPE_KEYWORDS):
            labels[i] = 0
    return labels


# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------


def load_and_preprocess(h5ad_path: str, n_hvgs: int = 2000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load h5ad, preprocess, and return (expression_matrix, labels, gene_names).

    Preprocessing:
    1. Filter cells/genes by minimal QC
    2. Normalize to 10k counts, log1p
    3. Select top n_hvgs highly variable genes
    4. Assign binary cancer/normal labels from cell_type column

    Returns
    -------
    X : np.ndarray of shape (n_labeled_cells, n_hvgs)
        Log-normalized HVG expression matrix, float32.
    y : np.ndarray of shape (n_labeled_cells,)
        Binary labels: 1 = cancer, 0 = normal.
    genes : list of str
        Selected gene names (length n_hvgs).
    """
    try:
        import scanpy as sc
        from scipy.sparse import issparse
    except ImportError:
        logger.error("scanpy and scipy are required. Install: pip install scanpy scipy")
        sys.exit(1)

    logger.info("Loading h5ad: %s", h5ad_path)
    adata = sc.read_h5ad(h5ad_path)
    logger.info("Loaded: %d cells × %d genes", *adata.shape)

    # Step 1: Assign labels from cell_type before QC filtering
    if "cell_type" in adata.obs.columns:
        cell_types = adata.obs["cell_type"].values
        raw_labels = assign_labels_from_annotations(cell_types)
        n_cancer = (raw_labels == 1).sum()
        n_normal = (raw_labels == 0).sum()
        n_ambig = (raw_labels == -1).sum()
        logger.info(
            "Cell type annotation: %d cancer, %d normal, %d ambiguous",
            n_cancer, n_normal, n_ambig,
        )
        adata.obs["cancer_label"] = raw_labels.astype(float)
        # Keep only unambiguously labeled cells
        adata = adata[raw_labels >= 0].copy()
        logger.info("After ambiguous filter: %d cells", adata.n_obs)
    else:
        logger.warning("No 'cell_type' column found; all cells will use marker-gene scoring")
        adata.obs["cancer_label"] = -1.0

    # Step 2: Minimal QC
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # Step 3: Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Step 4: HVG selection
    actual_hvgs = min(n_hvgs, adata.n_vars)
    logger.info("Selecting %d HVGs (Seurat v3)...", actual_hvgs)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=actual_hvgs)
    hvg_adata = adata[:, adata.var["highly_variable"]].copy()
    logger.info("HVG shape: %s", hvg_adata.shape)

    # Extract expression matrix
    X = hvg_adata.X
    if issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    # Labels (only labeled cells survived filtering above)
    y = adata.obs["cancer_label"].values[adata.obs["cancer_label"].values >= 0]
    y = hvg_adata.obs["cancer_label"].values.astype(np.float32)

    genes = list(hvg_adata.var_names)

    # Log class balance
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    logger.info("Training labels: %d cancer (1) | %d normal (0) | ratio=%.2f", n_pos, n_neg, n_pos / max(n_neg, 1))

    return X, y, genes


# ---------------------------------------------------------------------------
# Build DataLoader
# ---------------------------------------------------------------------------


def build_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 256,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Split into train/val and return DataLoader pair."""
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    dataset = TensorDataset(X_tensor, y_tensor)
    n_val = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    logger.info("Train: %d cells | Val: %d cells", n_train, n_val)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, list]:
    """Train model with BCE + monotonicity + smoothness losses (per spec §9.1)."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)
    bce = nn.BCELoss()

    # Class weights to handle imbalance
    all_labels = torch.cat([y for _, y in train_loader])
    pos_weight = (all_labels == 0).sum().float() / (all_labels == 1).sum().float()
    bce_weighted = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_val_loss = float("inf")
    patience_count = 0

    for epoch in range(1, n_epochs + 1):
        # ----- train -----
        model.train()
        total_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            scores = model(X_batch).squeeze(-1)

            # Classification loss (sigmoid output, use BCE)
            cls_loss = bce(scores, y_batch)

            # Monotonicity loss: cancer scores should be higher than normal scores
            cancer_mask = y_batch > 0.5
            normal_mask = ~cancer_mask
            if cancer_mask.any() and normal_mask.any():
                cancer_mean = scores[cancer_mask].mean()
                normal_mean = scores[normal_mask].mean()
                mono_loss = torch.clamp(normal_mean - cancer_mean + 0.1, min=0.0)
            else:
                mono_loss = torch.tensor(0.0, device=device)

            # Smoothness loss: penalize large variance in scores within same class
            smooth_loss = torch.tensor(0.0, device=device)
            if cancer_mask.sum() > 1:
                smooth_loss = smooth_loss + scores[cancer_mask].var()
            if normal_mask.sum() > 1:
                smooth_loss = smooth_loss + scores[normal_mask].var()

            loss = cls_loss + 0.1 * mono_loss + 0.01 * smooth_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = total_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train_loss)

        # ----- validation -----
        model.eval()
        val_loss = 0.0
        all_scores, all_labels_v = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                scores = model(X_batch).squeeze(-1)
                val_loss += bce(scores, y_batch).item()
                all_scores.append(scores.cpu())
                all_labels_v.append(y_batch.cpu())
        avg_val_loss = val_loss / max(len(val_loader), 1)
        history["val_loss"].append(avg_val_loss)

        # AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(
                torch.cat(all_labels_v).numpy(),
                torch.cat(all_scores).numpy(),
            )
        except Exception:
            auc = float("nan")
        history["val_auc"].append(auc)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | val_auc=%.4f",
                epoch, n_epochs, avg_train_loss, avg_val_loss, auc,
            )

        # Early stopping
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= 15:
            logger.info("Early stopping at epoch %d (patience=15)", epoch)
            break

    return history


# ---------------------------------------------------------------------------
# Save gene list alongside checkpoint
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: nn.Module,
    genes: List[str],
    output_path: str,
    history: Optional[Dict] = None,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_genes": model.n_genes,
            "genes": genes,
            "history": history or {},
        },
        output_path,
    )
    gene_path = Path(output_path).with_suffix(".genes.txt")
    gene_path.write_text("\n".join(genes))
    logger.info("Saved checkpoint: %s", output_path)
    logger.info("Saved gene list: %s (%d genes)", gene_path, len(genes))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CancerScoreFunction on real scRNA-seq data.")
    parser.add_argument("--h5ad", required=True, help="Path to input .h5ad file")
    parser.add_argument("--cancer-type", default="colorectal", help="Cancer type label")
    parser.add_argument("--n-hvgs", type=int, default=2000, help="Number of HVGs to select")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--n-epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output", default="checkpoints/cancer_score.pt", help="Output checkpoint path")
    args = parser.parse_args()

    # Select device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Device: %s", device)

    # Load data
    X, y, genes = load_and_preprocess(args.h5ad, n_hvgs=args.n_hvgs)
    logger.info("Expression matrix: %s | Labels: %s", X.shape, y.shape)

    # Normalize expression to [0,1] per gene (min-max)
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)
    X_range = np.where(X_max - X_min > 0, X_max - X_min, 1.0)
    X = (X - X_min) / X_range

    # DataLoaders
    train_loader, val_loader = build_loaders(X, y, batch_size=args.batch_size)

    # Build model
    from oracle.rsp.cancer_score import CancerScoreFunction
    model = CancerScoreFunction(n_genes_or_config=len(genes), hidden_dim=args.hidden_dim)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("CancerScoreFunction: %d parameters | input_dim=%d", n_params, len(genes))

    # Train
    history = train(
        model, train_loader, val_loader,
        n_epochs=args.n_epochs, lr=args.lr, device=device,
    )

    final_auc = history["val_auc"][-1] if history["val_auc"] else float("nan")
    logger.info("Training complete. Final val_auc=%.4f", final_auc)

    # Save
    save_checkpoint(model, genes, args.output, history)


if __name__ == "__main__":
    main()
