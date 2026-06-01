"""
Training callback classes for the ORACLE training pipelines.

EarlyStoppingCallback – halts training when a monitored metric stops improving
CheckpointCallback    – saves model checkpoints on a fixed epoch cadence
LoggingCallback       – logs per-epoch metrics to console and/or file
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseCallback:
    """Minimal callback interface."""

    def on_epoch_end(
        self,
        epoch: int,
        metrics: Dict[str, float],
        model: Optional[nn.Module] = None,
        optimizer=None,
    ) -> bool:
        """Called at the end of every epoch.

        Returns ``True`` to signal that training should stop early.
        """
        return False


# ---------------------------------------------------------------------------
# EarlyStoppingCallback
# ---------------------------------------------------------------------------

class EarlyStoppingCallback(BaseCallback):
    """Stop training when a monitored metric does not improve for *patience* epochs.

    Parameters
    ----------
    monitor:
        Metric key to watch, e.g. ``"val_loss"`` or ``"val_auroc"``.
    patience:
        Number of epochs without improvement before stopping.
    min_delta:
        Minimum change to qualify as an improvement.
    mode:
        ``"min"`` to look for decreasing metric (e.g. loss) or
        ``"max"`` for increasing metric (e.g. AUROC).
    verbose:
        Log messages when improvement is detected or training stops.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
        verbose: bool = True,
    ) -> None:
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode.lower()
        self.verbose = verbose

        self._best: float = float("inf") if self.mode == "min" else float("-inf")
        self._counter: int = 0
        self._stopped: bool = False

    def on_epoch_end(
        self,
        epoch: int,
        metrics: Dict[str, float],
        model: Optional[nn.Module] = None,
        optimizer=None,
    ) -> bool:
        """Returns ``True`` if training should stop."""
        value = metrics.get(self.monitor)
        if value is None:
            return False

        improved = (
            (self.mode == "min" and value < self._best - self.min_delta)
            or (self.mode == "max" and value > self._best + self.min_delta)
        )

        if improved:
            if self.verbose:
                logger.info(
                    "EarlyStopping: %s improved from %.6f to %.6f",
                    self.monitor,
                    self._best,
                    value,
                )
            self._best = value
            self._counter = 0
        else:
            self._counter += 1
            if self.verbose:
                logger.info(
                    "EarlyStopping: %s did not improve (%.6f). "
                    "Counter %d/%d",
                    self.monitor,
                    value,
                    self._counter,
                    self.patience,
                )
            if self._counter >= self.patience:
                logger.info(
                    "EarlyStopping triggered at epoch %d. "
                    "Best %s = %.6f",
                    epoch,
                    self.monitor,
                    self._best,
                )
                self._stopped = True
                return True

        return False

    @property
    def stopped(self) -> bool:
        """``True`` if early stopping was triggered."""
        return self._stopped

    def reset(self) -> None:
        """Reset internal state for reuse across experiments."""
        self._best = float("inf") if self.mode == "min" else float("-inf")
        self._counter = 0
        self._stopped = False


# ---------------------------------------------------------------------------
# CheckpointCallback
# ---------------------------------------------------------------------------

class CheckpointCallback(BaseCallback):
    """Save model checkpoints every *save_every* epochs.

    Parameters
    ----------
    checkpoint_dir:
        Directory to save checkpoint files.
    model_name:
        Prefix for checkpoint file names.
    save_every:
        Save a checkpoint every this many epochs.  Defaults to 1.
    save_best:
        Also save the best checkpoint according to *monitor*.
    monitor:
        Metric to track for best-checkpoint saving.
    mode:
        ``"min"`` or ``"max"`` for the *monitor* metric.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        model_name: str,
        save_every: int = 1,
        save_best: bool = True,
        monitor: str = "val_loss",
        mode: str = "min",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.save_every = save_every
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode.lower()

        self._best: float = float("inf") if mode == "min" else float("-inf")

    def on_epoch_end(
        self,
        epoch: int,
        metrics: Dict[str, float],
        model: Optional[nn.Module] = None,
        optimizer=None,
    ) -> bool:
        """Save checkpoint if appropriate. Never triggers early stopping."""
        if model is None:
            return False

        # Periodic save
        if epoch % self.save_every == 0:
            self._save(model, optimizer, epoch, metrics, suffix="")

        # Best-model save
        if self.save_best:
            value = metrics.get(self.monitor)
            if value is not None:
                improved = (
                    (self.mode == "min" and value < self._best)
                    or (self.mode == "max" and value > self._best)
                )
                if improved:
                    self._best = value
                    self._save(model, optimizer, epoch, metrics, suffix="_best")
                    logger.info(
                        "CheckpointCallback: new best %s=%.6f saved at epoch %d",
                        self.monitor,
                        value,
                        epoch,
                    )

        return False  # Never stops training

    def _save(
        self,
        model: nn.Module,
        optimizer,
        epoch: int,
        metrics: Dict[str, float],
        suffix: str = "",
    ) -> str:
        filename = f"{self.model_name}_epoch{epoch:04d}{suffix}.pt"
        path = self.checkpoint_dir / filename

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint, str(path))
        logger.debug("Checkpoint saved: %s", path)
        return str(path)


# ---------------------------------------------------------------------------
# LoggingCallback
# ---------------------------------------------------------------------------

class LoggingCallback(BaseCallback):
    """Log per-epoch metrics to the console and optionally to a CSV file.

    Parameters
    ----------
    log_file:
        Optional path to a CSV file where metrics are appended each epoch.
        If ``None``, only console logging is performed.
    log_every:
        Log every this many epochs.  Defaults to 1.
    """

    def __init__(
        self,
        log_file: Optional[str] = None,
        log_every: int = 1,
    ) -> None:
        self.log_file = log_file
        self.log_every = log_every
        self._header_written = False

        if log_file is not None:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(
        self,
        epoch: int,
        metrics: Dict[str, float],
        model: Optional[nn.Module] = None,
        optimizer=None,
    ) -> bool:
        """Log metrics. Never triggers early stopping."""
        if epoch % self.log_every != 0:
            return False

        # Console
        metric_str = "  ".join(
            f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in sorted(metrics.items())
        )
        logger.info("Epoch %04d | %s", epoch, metric_str)

        # CSV file
        if self.log_file is not None:
            self._write_csv(epoch, metrics)

        return False

    def _write_csv(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Append a row to the CSV log file."""
        import csv

        fields = ["epoch"] + sorted(metrics.keys())
        row = {"epoch": epoch, **metrics}

        mode = "a" if self._header_written else "w"
        with open(self.log_file, mode, newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Convenience wrappers: step() / load_best() interface
# ---------------------------------------------------------------------------

class EarlyStopping(EarlyStoppingCallback):
    """Thin wrapper adding a step(value) -> bool API."""

    def __init__(self, patience: int = 10, mode: str = "min", min_delta: float = 1e-4) -> None:
        super().__init__(monitor="val_loss", patience=patience, mode=mode, min_delta=min_delta)
        self._epoch = 0

    def step(self, value: float) -> bool:
        """Return True if training should stop."""
        import math
        if math.isnan(value):
            return False
        self._epoch += 1
        return self.on_epoch_end(self._epoch, {"val_loss": value})


class ModelCheckpoint(CheckpointCallback):
    """Thin wrapper adding step(value, model) and load_best(model) APIs."""

    def __init__(
        self,
        checkpoint_dir: str,
        filename: str = "best.pt",
        monitor: str = "val_loss",
        mode: str = "min",
    ) -> None:
        # Strip extension for model_name
        model_name = filename.replace(".pt", "").replace(".pth", "")
        super().__init__(
            checkpoint_dir=checkpoint_dir,
            model_name=model_name,
            save_every=9999,  # only save best
            save_best=True,
            monitor=monitor,
            mode=mode,
        )
        self._best_path: Optional[str] = None
        self._epoch = 0
        self._filename = filename

    def step(self, value: float, model) -> None:
        """Save a new best checkpoint if value improved."""
        import math, torch as _torch
        if math.isnan(value):
            return
        improved = (
            (self.mode == "min" and value < self._best)
            or (self.mode == "max" and value > self._best)
        )
        if improved:
            self._best = value
            path = str(Path(self.checkpoint_dir) / self._filename)
            _torch.save({"model_state_dict": model.state_dict(), "best_value": value}, path)
            self._best_path = path
            logger.debug("ModelCheckpoint: saved best (%.6f) → %s", value, path)

    def load_best(self, model) -> None:
        """Load the best checkpoint weights into model (in-place)."""
        import torch as _torch
        if self._best_path is None:
            path = str(Path(self.checkpoint_dir) / self._filename)
        else:
            path = self._best_path
        if not Path(path).exists():
            logger.warning("ModelCheckpoint: no checkpoint found at %s", path)
            return
        ckpt = _torch.load(path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        logger.info("ModelCheckpoint: loaded best weights from %s", path)
