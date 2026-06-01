"""CAM module trainer: trains ContinuousGRNDynamics and CancerScoreFunction."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from oracle.training.losses import CancerScoreLoss
from oracle.training.callbacks import EarlyStopping, ModelCheckpoint

logger = logging.getLogger(__name__)


class CAMTrainer:
    """Trains the CAM neural components (ContinuousGRNDynamics + CancerScoreFunction).

    Training objective:
        L_total = L_cls + 0.1 * L_mono + 0.01 * L_smooth

    where L_cls is BCE on cancer/normal attractor labels, L_mono enforces that
    cancer score increases monotonically along pseudotime, and L_smooth penalizes
    large gradients in score function outputs.
    """

    def __init__(
        self,
        cancer_score_fn: nn.Module,
        ode_model: Optional[nn.Module] = None,
        config: Optional[object] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.cancer_score_fn = cancer_score_fn
        self.ode_model = ode_model
        self.config = config

        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device

        lr = getattr(config, "learning_rate", 1e-3)
        wd = getattr(config, "weight_decay", 1e-4)

        params = list(cancer_score_fn.parameters())
        if ode_model is not None:
            params += list(ode_model.parameters())

        self.optimizer = AdamW(params, lr=lr, weight_decay=wd)
        self.n_epochs = getattr(config, "n_epochs", 100)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.n_epochs, eta_min=1e-6)

        self.loss_fn = CancerScoreLoss(mono_weight=0.1, smooth_weight=0.01)

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "train_auc": [], "val_auc": []
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: str = "./checkpoints",
        patience: int = 15,
    ) -> Dict[str, List[float]]:
        """Full training loop with early stopping and checkpointing."""
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        early_stop = EarlyStopping(patience=patience, mode="min")
        checkpointer = ModelCheckpoint(
            checkpoint_dir, filename="cam_best.pt", monitor="val_loss", mode="min"
        )

        self.cancer_score_fn.to(self.device)
        if self.ode_model is not None:
            self.ode_model.to(self.device)

        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()
            train_metrics = self._train_epoch(train_loader)
            val_metrics = self._val_epoch(val_loader) if val_loader else {}

            self.scheduler.step()

            self.history["train_loss"].append(train_metrics.get("loss", float("nan")))
            self.history["val_loss"].append(val_metrics.get("loss", float("nan")))

            elapsed = time.time() - t0
            logger.info(
                "CAM epoch %d/%d | train_loss=%.4f | val_loss=%.4f | %.1fs",
                epoch, self.n_epochs,
                train_metrics.get("loss", 0),
                val_metrics.get("loss", 0),
                elapsed,
            )

            val_loss = val_metrics.get("loss", float("nan"))
            checkpointer.step(val_loss, self.cancer_score_fn)
            if early_stop.step(val_loss):
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

        checkpointer.load_best(self.cancer_score_fn)
        return self.history

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.cancer_score_fn.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            x = batch["expression"].to(self.device, dtype=torch.float32)
            labels = batch["cancer_label"].to(self.device, dtype=torch.float32)
            pseudotime = batch.get("pseudotime")
            if pseudotime is not None:
                pseudotime = pseudotime.to(self.device, dtype=torch.float32)

            self.optimizer.zero_grad()
            scores = self.cancer_score_fn(x).squeeze(-1)
            loss = self.loss_fn(scores, labels, x, pseudotime)
            loss.backward()
            nn.utils.clip_grad_norm_(self.cancer_score_fn.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return {"loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def _val_epoch(self, loader: Optional[DataLoader]) -> Dict[str, float]:
        if loader is None:
            return {}
        self.cancer_score_fn.eval()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            x = batch["expression"].to(self.device, dtype=torch.float32)
            labels = batch["cancer_label"].to(self.device, dtype=torch.float32)
            pseudotime = batch.get("pseudotime")
            if pseudotime is not None:
                pseudotime = pseudotime.to(self.device, dtype=torch.float32)

            scores = self.cancer_score_fn(x).squeeze(-1)
            loss = self.loss_fn(scores, labels, x, pseudotime)
            total_loss += loss.item()
            n_batches += 1

        return {"loss": total_loss / max(n_batches, 1)}

    def save(self, path: str) -> None:
        torch.save({
            "cancer_score_fn": self.cancer_score_fn.state_dict(),
            "ode_model": self.ode_model.state_dict() if self.ode_model else None,
            "optimizer": self.optimizer.state_dict(),
            "history": self.history,
        }, path)
        logger.info("CAM trainer state saved to %s", path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.cancer_score_fn.load_state_dict(ckpt["cancer_score_fn"])
        if self.ode_model and ckpt.get("ode_model"):
            self.ode_model.load_state_dict(ckpt["ode_model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.history = ckpt.get("history", self.history)
        logger.info("CAM trainer state loaded from %s", path)


# ---------------------------------------------------------------------------
# CancerScoreTrainer — simplified interface used by master_trainer Stage 2a
# ---------------------------------------------------------------------------

class CancerScoreTrainer:
    """Trains a CancerScoreFunction on numpy arrays of cancer/normal expression."""

    def __init__(
        self,
        n_genes: int,
        n_epochs: int = 200,
        batch_size: int = 256,
        lr: float = 5e-4,
        device: Optional[torch.device] = None,
        checkpoint_path: Optional[Path] = None,
    ):
        self.n_genes = n_genes
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or torch.device("cpu")
        self.checkpoint_path = checkpoint_path
        self.model: Optional[nn.Module] = None  # Set externally
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None

    def train(
        self,
        cancer_states: np.ndarray,
        normal_states: np.ndarray,
        pt_pairs: Optional[np.ndarray] = None,
    ) -> nn.Module:
        assert self.model is not None, "Set trainer.model before calling train()"
        self.model = self.model.to(self.device)

        if self.optimizer is None:
            self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        if self.scheduler is None:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.n_epochs, eta_min=1e-6)

        X = np.vstack([cancer_states, normal_states])
        y = np.concatenate([
            np.ones(len(cancer_states), dtype=np.float32),
            np.zeros(len(normal_states), dtype=np.float32),
        ])

        for epoch in range(self.n_epochs):
            idx = np.random.permutation(len(X))
            epoch_losses: List[float] = []

            for start in range(0, len(X), self.batch_size):
                batch_idx = idx[start:start + self.batch_size]
                x_b = torch.tensor(X[batch_idx], dtype=torch.float32, device=self.device)
                y_b = torch.tensor(y[batch_idx], dtype=torch.float32, device=self.device)

                self.model.train()
                scores = self.model(x_b)
                loss = F.binary_cross_entropy(scores.squeeze(), y_b)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_losses.append(loss.item())

            self.scheduler.step()

            if epoch % 50 == 0 or epoch == self.n_epochs - 1:
                logger.info(
                    "CancerScoreTrainer epoch %d/%d: loss=%.4f",
                    epoch + 1, self.n_epochs, np.mean(epoch_losses)
                )

        if self.checkpoint_path is not None:
            Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), self.checkpoint_path)
            logger.info("CancerScoreTrainer saved to %s", self.checkpoint_path)

        return self.model
