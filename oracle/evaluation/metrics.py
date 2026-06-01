"""Shared evaluation metrics for all ORACLE modules."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np


def attractor_accuracy(
    predicted_labels: List[str],
    true_labels: List[str],
) -> float:
    """Fraction of attractors with correct cancer/normal/transitional classification."""
    if not true_labels:
        return 0.0
    correct = sum(p == t for p, t in zip(predicted_labels, true_labels))
    return correct / len(true_labels)


def basin_overlap(
    predicted_basin: np.ndarray,
    true_basin: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Jaccard overlap between predicted and true attractor basin membership.

    Args:
        predicted_basin: (N,) float array of basin membership scores.
        true_basin: (N,) binary array of true basin membership.
        threshold: binarization threshold for predicted_basin.
    """
    pred_binary = predicted_basin >= threshold
    true_binary = true_basin.astype(bool)
    intersection = int((pred_binary & true_binary).sum())
    union = int((pred_binary | true_binary).sum())
    return intersection / max(union, 1)


def switch_f1(
    predicted_perturbations: Dict[str, str],
    ground_truth_perturbations: Dict[str, str],
    match_type: bool = True,
) -> Dict[str, float]:
    """F1 score for TF perturbation prediction.

    Args:
        predicted_perturbations: gene -> perturbation_type dict from RSP.
        ground_truth_perturbations: gene -> perturbation_type ground truth.
        match_type: if True, require perturbation type to match (Activation/Repression).

    Returns:
        dict with precision, recall, f1 keys.
    """
    pred_set: Set[Tuple] = set()
    for gene, ptype in predicted_perturbations.items():
        key = (gene, str(ptype)) if match_type else gene
        pred_set.add(key)

    gt_set: Set[Tuple] = set()
    for gene, ptype in ground_truth_perturbations.items():
        key = (gene, str(ptype)) if match_type else gene
        gt_set.add(key)

    tp = len(pred_set & gt_set)
    precision = tp / max(len(pred_set), 1)
    recall = tp / max(len(gt_set), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp}


def reversion_auc(
    reversion_probs: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """AUROC for reversion probability prediction.

    Uses trapezoidal rule without requiring sklearn.
    """
    sorted_indices = np.argsort(-reversion_probs)
    true_sorted = true_labels[sorted_indices]

    n_pos = int(true_sorted.sum())
    n_neg = len(true_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp, fp = 0, 0

    for label in true_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)
    return float(np.trapezoid(tpr_arr, fpr_arr))


def molecule_validity(smiles_list: List[str]) -> float:
    """Fraction of SMILES strings that parse to valid RDKit molecules."""
    try:
        from rdkit import Chem
        valid = sum(1 for s in smiles_list if Chem.MolFromSmiles(s) is not None)
        return valid / max(len(smiles_list), 1)
    except ImportError:
        return float("nan")


def molecule_novelty(
    generated_smiles: List[str],
    training_smiles: Set[str],
) -> float:
    """Fraction of valid generated molecules not in the training set."""
    try:
        from rdkit import Chem
        novel = 0
        valid = 0
        for s in generated_smiles:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                valid += 1
                canonical = Chem.MolToSmiles(mol)
                if canonical not in training_smiles:
                    novel += 1
        return novel / max(valid, 1)
    except ImportError:
        return float("nan")


def molecule_diversity(smiles_list: List[str], n_samples: int = 100) -> float:
    """Mean pairwise Tanimoto dissimilarity among a random sample of valid molecules."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs

        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        mols = [m for m in mols if m is not None]
        if len(mols) < 2:
            return 0.0

        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in mols]
        rng = np.random.default_rng(42)
        idx = rng.choice(len(fps), min(n_samples, len(fps)), replace=False)
        sampled = [fps[i] for i in idx]

        total, count = 0.0, 0
        for i in range(len(sampled)):
            for j in range(i + 1, len(sampled)):
                sim = DataStructs.TanimotoSimilarity(sampled[i], sampled[j])
                total += 1.0 - sim
                count += 1
        return total / max(count, 1)
    except ImportError:
        return float("nan")


def grn_auroc(
    predicted_weights: Dict[Tuple[str, str], float],
    true_edges: Set[Tuple[str, str]],
    all_pairs: Optional[List[Tuple[str, str]]] = None,
) -> float:
    """AUROC for GRN edge prediction against ground-truth edges.

    Args:
        predicted_weights: (TF, target) -> weight dict.
        true_edges: set of (TF, target) ground-truth edges.
        all_pairs: if provided, restrict evaluation to these pairs.
    """
    if all_pairs is None:
        all_pairs = list(predicted_weights.keys())

    scores = np.array([predicted_weights.get(p, 0.0) for p in all_pairs])
    labels = np.array([1.0 if p in true_edges else 0.0 for p in all_pairs])

    return reversion_auc(scores, labels)


def grn_early_precision(
    predicted_weights: Dict[Tuple[str, str], float],
    true_edges: Set[Tuple[str, str]],
    top_k: Optional[int] = None,
) -> float:
    """Precision among top-k highest-weighted predicted edges."""
    sorted_edges = sorted(predicted_weights.items(), key=lambda x: x[1], reverse=True)
    if top_k is None:
        top_k = max(len(true_edges), 1)

    top_edges = [e for e, _ in sorted_edges[:top_k]]
    hits = sum(1 for e in top_edges if e in true_edges)
    return hits / max(len(top_edges), 1)
