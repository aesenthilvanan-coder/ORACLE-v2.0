#!/usr/bin/env python3
"""Train CAM module models (CancerScoreFunction + GRNTransformer)."""

import argparse
import torch
from oracle.utils.config import load_config
from oracle.training.cam_trainer import CAMTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train the Cancer Attraction Mapper (CAM) models."
    )
    parser.add_argument(
        "--config",
        default="configs/training/cam_train.yaml",
        help="Path to the CAM training configuration YAML.",
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Root data directory containing processed AnnData files.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="./checkpoints",
        help="Directory to save model checkpoints.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training CAM on device: {device}")
    print(f"Config: {args.config}")
    print(f"Data dir: {args.data_dir}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")

    config = load_config(args.config)
    trainer = CAMTrainer(config, args.data_dir, args.checkpoint_dir)
    trainer.train()
    print("CAM training complete.")


if __name__ == "__main__":
    main()
