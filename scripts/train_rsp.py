#!/usr/bin/env python3
"""Train RSP module GNN (GNNSwitchPredictor)."""

import argparse
from oracle.utils.config import load_config
from oracle.training.rsp_trainer import RSPTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train the Reversion Switch Predictor (RSP) GNN."
    )
    parser.add_argument(
        "--config",
        default="configs/training/rsp_train.yaml",
        help="Path to the RSP training configuration YAML.",
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Root data directory.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="./checkpoints",
        help="Directory to save model checkpoints.",
    )
    args = parser.parse_args()

    print(f"Training RSP GNN")
    print(f"Config: {args.config}")
    print(f"Data dir: {args.data_dir}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")

    config = load_config(args.config)
    trainer = RSPTrainer(config, args.data_dir, args.checkpoint_dir)
    trainer.train()
    print("RSP training complete.")


if __name__ == "__main__":
    main()
