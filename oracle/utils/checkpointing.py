"""
Checkpointing utilities for the ORACLE training pipelines.

Provides the ``Checkpointer`` class for saving and loading model checkpoints
with metadata (epoch, metrics, timestamps).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class Checkpointer:
    """Save and load model checkpoints with metadata.

    Checkpoints are stored as ``{model_name}_epoch{epoch:04d}.pt`` files
    inside *checkpoint_dir*.  Each checkpoint contains:

    - ``model_state_dict`` – model weights
    - ``optimizer_state_dict`` – optimiser state (if provided)
    - ``epoch`` – training epoch at save time
    - ``metrics`` – dict of metric values
    - ``timestamp`` – ISO 8601 timestamp string
    - ``model_name`` – the model name passed at construction

    Parameters
    ----------
    checkpoint_dir:
        Directory where checkpoint files are stored.  Created if needed.
    model_name:
        Human-readable model identifier used in file names.
    """

    def __init__(self, checkpoint_dir: str, model_name: str) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        model: nn.Module,
        optimizer=None,
        epoch: int = 0,
        metrics: Optional[Dict] = None,
    ) -> str:
        """Save a model checkpoint to disk.

        Parameters
        ----------
        model:
            PyTorch model whose ``state_dict`` will be saved.
        optimizer:
            Optional optimiser; its ``state_dict`` is included if provided.
        epoch:
            Current training epoch.
        metrics:
            Optional dict of scalar metric values to store alongside the
            checkpoint (e.g. ``{"val_loss": 0.42, "val_auroc": 0.91}``).

        Returns
        -------
        str
            Absolute path to the saved checkpoint file.
        """
        filename = f"{self.model_name}_epoch{epoch:04d}.pt"
        path = self.checkpoint_dir / filename

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics or {},
            "timestamp": datetime.utcnow().isoformat(),
            "model_name": self.model_name,
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, str(path))
        logger.info(
            "Checkpoint saved: %s (epoch=%d, metrics=%s)",
            path,
            epoch,
            metrics,
        )
        return str(path)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(
        self,
        model: nn.Module,
        optimizer=None,
        path: Optional[str] = None,
    ) -> Dict:
        """Load a checkpoint into *model* (and optionally *optimizer*).

        If *path* is ``None``, the most recent checkpoint in
        ``checkpoint_dir`` is loaded automatically.

        Parameters
        ----------
        model:
            PyTorch model to load weights into.
        optimizer:
            Optional optimiser to restore state for.
        path:
            Explicit path to a ``*.pt`` checkpoint file.  If ``None``, the
            latest checkpoint is used.

        Returns
        -------
        dict
            The full checkpoint dict (including ``epoch`` and ``metrics``).

        Raises
        ------
        FileNotFoundError
            If no checkpoint exists and *path* is ``None``.
        """
        if path is None:
            path = self.get_latest()
            if path is None:
                raise FileNotFoundError(
                    f"No checkpoints found in {self.checkpoint_dir}"
                )

        path = str(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # Load to CPU first to avoid CUDA/MPS device mismatch
        checkpoint = torch.load(path, map_location="cpu")

        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        epoch = checkpoint.get("epoch", 0)
        metrics = checkpoint.get("metrics", {})
        logger.info(
            "Checkpoint loaded: %s (epoch=%d, metrics=%s)", path, epoch, metrics
        )
        return checkpoint

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_latest(self) -> Optional[str]:
        """Return the path to the most recent checkpoint file.

        Files are sorted by modification time.

        Returns
        -------
        str or None
            Path to the latest ``.pt`` file, or ``None`` if none exist.
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        # Sort by modification time (most recent last → return last element)
        checkpoints.sort(key=lambda p: os.path.getmtime(p))
        return checkpoints[-1]

    def list_checkpoints(self) -> List[str]:
        """Return a sorted list of all checkpoint file paths in
        ``checkpoint_dir`` for this model.

        Returns
        -------
        list of str
        """
        pattern = f"{self.model_name}_epoch*.pt"
        matches = sorted(
            str(p) for p in self.checkpoint_dir.glob(pattern)
        )
        return matches

    def best(self, metric: str = "val_loss", lower_is_better: bool = True) -> Optional[str]:
        """Return the checkpoint with the best value for a given metric.

        Parameters
        ----------
        metric:
            Metric key to compare across checkpoints.
        lower_is_better:
            If ``True`` (default), lower metric values are considered better
            (e.g. loss).  Set to ``False`` for metrics like AUROC.

        Returns
        -------
        str or None
            Path to the best checkpoint, or ``None`` if no checkpoints exist.
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None

        best_path = None
        best_val = float("inf") if lower_is_better else float("-inf")

        for ckpt_path in checkpoints:
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                val = ckpt.get("metrics", {}).get(metric)
                if val is None:
                    continue
                val = float(val)
                if lower_is_better and val < best_val:
                    best_val = val
                    best_path = ckpt_path
                elif not lower_is_better and val > best_val:
                    best_val = val
                    best_path = ckpt_path
            except Exception as exc:
                logger.warning("Could not read checkpoint %s: %s", ckpt_path, exc)

        return best_path
