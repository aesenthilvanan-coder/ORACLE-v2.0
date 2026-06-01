#!/usr/bin/env python3
"""Train TCD equivariant diffusion model."""

import argparse
from oracle.utils.config import load_config
from oracle.training.tcd_trainer import TCDTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train the Transcriptional CIP Designer (TCD) diffusion model."
    )
    parser.add_argument(
        "--config",
        default="configs/training/tcd_train.yaml",
        help="Path to the TCD training configuration YAML.",
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

    print(f"Training TCD equivariant diffusion model")
    print(f"Config: {args.config}")
    print(f"Data dir: {args.data_dir}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")

    config = load_config(args.config)
    trainer = TCDTrainer(config, args.data_dir, args.checkpoint_dir)
    trainer.train()
    print("TCD training complete.")


if __name__ == "__main__":
    main()
