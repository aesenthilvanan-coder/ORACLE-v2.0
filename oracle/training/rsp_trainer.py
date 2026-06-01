"""RSP module trainer: trains GNNSwitchPredictor and CancerScoreFunction jointly."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from oracle.training.losses import SwitchPredictorLoss
from oracle.training.callbacks import EarlyStopping, ModelCheckpoint

logger = logging.getLogger(__name__)


class RSPTrainer:
    """Trains the RSP module: joint GNN switch predictor + cancer score function.

    Training objective:
        L_total = L_mse(score) + L_bce(reversion) + 0.001 * L_importance
    """

    def __init__(
        self,
        switch_gnn: nn.Module,
        cancer_score_fn: nn.Module,
        config: Optional[object] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.switch_gnn = switch_gnn
        self.cancer_score_fn = cancer_score_fn
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

        all_params = list(switch_gnn.parameters()) + list(cancer_score_fn.parameters())
        self.optimizer = AdamW(all_params, lr=lr, weight_decay=wd)
        self.n_epochs = getattr(config, "n_epochs", 100)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.n_epochs, eta_min=1e-6)

        self.loss_fn = SwitchPredictorLoss(importance_weight=0.001)

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_reversion_acc": [], "val_reversion_acc": [],
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: str = "./checkpoints",
        patience: int = 15,
    ) -> Dict[str, List[float]]:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        early_stop = EarlyStopping(patience=patience, mode="min")
        checkpointer = ModelCheckpoint(
            checkpoint_dir, filename="rsp_best.pt", monitor="val_loss", mode="min"
        )

        self.switch_gnn.to(self.device)
        self.cancer_score_fn.to(self.device)

        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()
            train_m = self._train_epoch(train_loader)
            val_m = self._val_epoch(val_loader) if val_loader else {}

            self.scheduler.step()

            self.history["train_loss"].append(train_m.get("loss", float("nan")))
            self.history["val_loss"].append(val_m.get("loss", float("nan")))

            logger.info(
                "RSP epoch %d/%d | train_loss=%.4f | val_loss=%.4f | %.1fs",
                epoch, self.n_epochs,
                train_m.get("loss", 0),
                val_m.get("loss", 0),
                time.time() - t0,
            )

            val_loss = val_m.get("loss", float("nan"))
            checkpointer.step(val_loss, self.switch_gnn)
            if early_stop.step(val_loss):
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

        checkpointer.load_best(self.switch_gnn)
        return self.history

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.switch_gnn.train()
        self.cancer_score_fn.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            # batch: dict with 'graph' (PyG Data), 'cancer_score', 'reversion_label'
            graph = batch["graph"].to(self.device)
            cancer_score_gt = batch["cancer_score"].to(self.device, dtype=torch.float32)
            reversion_label = batch["reversion_label"].to(self.device, dtype=torch.float32)
            importance_gt = batch.get("importance")
            if importance_gt is not None:
                importance_gt = importance_gt.to(self.device, dtype=torch.float32)

            self.optimizer.zero_grad()

            out = self.switch_gnn(graph)
            pred_cancer_score = out["cancer_score"]
            pred_reversion = out["reversion_prob"]
            pred_importance = out.get("importance")

            loss = self.loss_fn(
                pred_cancer_score, cancer_score_gt,
                pred_reversion, reversion_label,
                pred_importance, importance_gt,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.switch_gnn.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return {"loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def _val_epoch(self, loader: Optional[DataLoader]) -> Dict[str, float]:
        if loader is None:
            return {}
        self.switch_gnn.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            graph = batch["graph"].to(self.device)
            cancer_score_gt = batch["cancer_score"].to(self.device, dtype=torch.float32)
            reversion_label = batch["reversion_label"].to(self.device, dtype=torch.float32)

            out = self.switch_gnn(graph)
            loss = self.loss_fn(
                out["cancer_score"], cancer_score_gt,
                out["reversion_prob"], reversion_label,
            )
            total_loss += loss.item()
            n_batches += 1

        return {"loss": total_loss / max(n_batches, 1)}

    def save(self, path: str) -> None:
        torch.save({
            "switch_gnn": self.switch_gnn.state_dict(),
            "cancer_score_fn": self.cancer_score_fn.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "history": self.history,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.switch_gnn.load_state_dict(ckpt["switch_gnn"])
        self.cancer_score_fn.load_state_dict(ckpt["cancer_score_fn"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.history = ckpt.get("history", self.history)


# ---------------------------------------------------------------------------
# SwitchPredictorDataset and SwitchPredictorTrainer (used by master_trainer)
# ---------------------------------------------------------------------------

class SwitchPredictorDataset(Dataset):
    """Wraps a list of {graph, target_score, target_reversion} samples."""

    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


class SwitchPredictorTrainer:
    """Trains SwitchPredictorGNN on a SwitchPredictorDataset."""

    def __init__(
        self,
        hidden_dim: int = 256,
        n_gnn_layers: int = 8,
        n_attention_heads: int = 4,
        n_epochs: int = 500,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: Optional[torch.device] = None,
        checkpoint_path: Optional[Path] = None,
    ):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or torch.device("cpu")
        self.checkpoint_path = checkpoint_path
        self.model: Optional[nn.Module] = None  # Set externally before calling train()

    def train(self, dataset: SwitchPredictorDataset) -> nn.Module:
        assert self.model is not None, "Set trainer.model before calling train()"

        self.model = self.model.to(self.device)

        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.n_epochs, eta_min=1e-6)

        # Use list-collate so each sample can be processed individually (variable graph sizes)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
        )

        for epoch in range(self.n_epochs):
            epoch_losses: List[float] = []

            for batch_samples in loader:
                optimizer.zero_grad()
                batch_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
                valid = 0

                for sample in batch_samples:
                    try:
                        graph = sample["graph"].to(self.device)
                        t_score = torch.tensor(
                            float(sample["target_score"]), dtype=torch.float32, device=self.device
                        )
                        t_rev = torch.tensor(
                            float(sample["target_reversion"]), dtype=torch.float32, device=self.device
                        )

                        if not hasattr(graph, "batch") or graph.batch is None:
                            graph.batch = torch.zeros(
                                graph.x.shape[0], dtype=torch.long, device=self.device
                            )

                        self.model.train()
                        output = self.model(graph)

                        loss = (
                            F.mse_loss(output["cancer_score"].squeeze(), t_score) +
                            F.binary_cross_entropy(output["reversion_prob"].squeeze(), t_rev)
                        )
                        batch_loss = batch_loss + loss
                        valid += 1
                    except Exception as e:
                        logger.debug(f"Sample skipped: {e}")
                        continue

                if valid > 0:
                    (batch_loss / valid).backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    epoch_losses.append((batch_loss / valid).item())

            scheduler.step()

            if epoch % 50 == 0 or epoch == self.n_epochs - 1:
                logger.info(
                    "SwitchPredictor epoch %d/%d: loss=%.4f",
                    epoch + 1, self.n_epochs,
                    np.mean(epoch_losses) if epoch_losses else float("nan"),
                )

        if self.checkpoint_path is not None:
            Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), self.checkpoint_path)
            logger.info("SwitchPredictorTrainer saved to %s", self.checkpoint_path)

        return self.model
