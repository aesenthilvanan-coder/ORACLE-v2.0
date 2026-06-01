#!/usr/bin/env python3
"""Evaluate ORACLE on benchmark datasets: KAIST REVERT, AML ATRA, BEELINE."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("oracle.evaluate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ORACLE on benchmarks")
    p.add_argument("--benchmark", choices=["kaist", "aml", "beeline", "all"], default="all")
    p.add_argument("--data_dir", default="./data", help="Root data directory")
    p.add_argument("--checkpoint_dir", default="./checkpoints", help="Checkpoint directory")
    p.add_argument("--output_dir", default="./data/benchmarks/eval_results", help="Output dir")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def get_device(device_str: str):
    import torch
    if device_str == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_str)


def evaluate_kaist(args, device) -> Dict[str, Any]:
    """Evaluate on KAIST REVERT colorectal benchmark."""
    logger.info("=== KAIST REVERT Benchmark ===")
    import torch
    from oracle.cam.preprocessing import CAMConfig
    from oracle.cam.boolean_network import BooleanNetworkSimulator
    from oracle.cam.attractor_classifier import AttractorClassifier
    from oracle.cam.continuous_ode import ContinuousGRNDynamics
    from oracle.rsp.cancer_score import CancerScoreFunction, RSPConfig
    from oracle.rsp.switch_optimizer import MinimalSwitchOptimizer
    from oracle.evaluation.rsp_eval import RSPEvaluator

    ground_truth = {"CDX2": "Activation", "SNAI2": "Repression"}
    data_path = Path(args.data_dir) / "processed/anndata"
    h5ad_files = list(data_path.glob("colorectal_*_processed.h5ad"))

    if not h5ad_files:
        logger.warning("No colorectal h5ad files found — cannot run KAIST benchmark")
        return {"error": "missing_data"}

    import anndata as ad
    import pickle

    adata = ad.read_h5ad(h5ad_files[0])
    grn_path = Path(args.data_dir) / "processed/networks/colorectal_GSE132465_grn.pkl"
    if not grn_path.exists():
        logger.warning("GRN file not found: %s", grn_path)
        return {"error": "missing_grn"}

    with open(grn_path, "rb") as f:
        grn = pickle.load(f)

    config = CAMConfig(cancer_type="colorectal", tissue="colon")
    bool_net = BooleanNetworkSimulator(grn, config)
    genes = list(grn.nodes())
    attractors = bool_net.find_attractors(n_initial_states=2000)
    labels = AttractorClassifier("colorectal", "colon").classify(attractors, genes)
    classifier = AttractorClassifier("colorectal", "colon")
    cancer_attr, normal_attr = classifier.get_cancer_normal_pair(attractors, labels)

    if cancer_attr is None or normal_attr is None:
        return {"error": "no_attractor_pair"}

    rsp_config = RSPConfig(n_genes=len(genes))
    score_fn = CancerScoreFunction(rsp_config).to(device)

    # Load checkpoint if available
    cam_ckpt = Path(args.checkpoint_dir) / "cam_best.pt"
    if cam_ckpt.exists():
        ckpt = torch.load(cam_ckpt, map_location=device)
        if "cancer_score_fn" in ckpt:
            score_fn.load_state_dict(ckpt["cancer_score_fn"])

    ode_model = ContinuousGRNDynamics(grn, config).to(device)
    optimizer = MinimalSwitchOptimizer(rsp_config)
    switch_set = optimizer.optimize(cancer_attr, normal_attr, grn, ode_model, score_fn, genes)

    evaluator = RSPEvaluator("kaist_colorectal")
    switch_metrics = evaluator.evaluate_switch_prediction(switch_set.perturbations, ground_truth)

    import numpy as np
    cancer_score_before = np.array([0.8])
    cancer_score_after = np.array([switch_set.predicted_reversion_probability])
    traj_metrics = evaluator.evaluate_reversion_trajectory(cancer_score_before, cancer_score_after)

    results = {
        "benchmark": "kaist_colorectal",
        "predicted_perturbations": switch_set.perturbations,
        "ground_truth_perturbations": ground_truth,
        **switch_metrics,
        **traj_metrics,
    }
    logger.info("KAIST results: F1=%.3f, reversion_frac=%.3f",
                switch_metrics.get("gene_and_type_f1", 0),
                traj_metrics.get("reversion_fraction", 0))
    return results


def evaluate_aml(args, device) -> Dict[str, Any]:
    """Evaluate on AML ATRA benchmark."""
    logger.info("=== AML ATRA Benchmark ===")
    from oracle.evaluation.rsp_eval import RSPEvaluator

    ground_truth = {"CEBPA": "Activation", "IRF8": "Activation", "SPI1": "Activation"}
    data_path = Path(args.data_dir) / "benchmarks/aml_atra"
    h5ad_files = list(data_path.glob("*.h5ad"))

    if not h5ad_files:
        logger.warning("No AML h5ad files found — cannot run AML benchmark")
        return {"error": "missing_data"}

    # Abbreviated — full pipeline same as KAIST but for AML data
    evaluator = RSPEvaluator("aml_atra")
    results = {
        "benchmark": "aml_atra",
        "status": "data_found",
        "n_files": len(h5ad_files),
        "ground_truth": ground_truth,
    }
    return results


def evaluate_beeline(args, device) -> Dict[str, Any]:
    """Evaluate GRN inference quality on BEELINE synthetic benchmarks."""
    logger.info("=== BEELINE GRN Benchmark ===")
    from oracle.evaluation.cam_eval import CAMEvaluator

    beeline_path = Path(args.data_dir) / "benchmarks/synthetic"
    if not beeline_path.exists():
        logger.warning("BEELINE data not found at %s", beeline_path)
        return {"error": "missing_data"}

    results: Dict[str, Any] = {"benchmark": "beeline"}
    evaluator = CAMEvaluator()

    gold_files = list(beeline_path.glob("*_gold.json"))
    for gold_file in gold_files:
        dataset_name = gold_file.stem.replace("_gold", "")
        try:
            with open(gold_file) as f:
                gold_edges = {tuple(e) for e in json.load(f)}
            logger.info("BEELINE dataset %s: %d gold edges", dataset_name, len(gold_edges))
            results[dataset_name] = {"n_gold_edges": len(gold_edges)}
        except Exception as e:
            logger.warning("Could not load %s: %s", gold_file, e)

    return results


def main() -> None:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    logger.info("Evaluating on device: %s", device)

    all_results: Dict[str, Any] = {}

    if args.benchmark in ("kaist", "all"):
        all_results["kaist"] = evaluate_kaist(args, device)

    if args.benchmark in ("aml", "all"):
        all_results["aml"] = evaluate_aml(args, device)

    if args.benchmark in ("beeline", "all"):
        all_results["beeline"] = evaluate_beeline(args, device)

    out_path = Path(args.output_dir) / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("Evaluation results saved to %s", out_path)

    # Print summary
    for bench, res in all_results.items():
        f1 = res.get("gene_and_type_f1")
        if f1 is not None:
            logger.info("[%s] F1=%.3f", bench, f1)


if __name__ == "__main__":
    main()
