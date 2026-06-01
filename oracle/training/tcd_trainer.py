"""TCD module trainer: trains TCIPDiffusionModel (DDPM) for TCIP molecule generation."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from oracle.training.losses import DiffusionLoss
from oracle.training.callbacks import EarlyStopping, ModelCheckpoint

logger = logging.getLogger(__name__)


class TCDTrainer:
    """Trains the TCIPDiffusionModel (SE(3)-equivariant DDPM).

    Training objective:
        L_total = MSE(coords) + CE(atom_types)

    Uses DDPM forward process: q(x_t | x_0) = N(sqrt(alpha_bar_t)*x_0, (1-alpha_bar_t)*I)
    and trains the score network to predict the noise epsilon.
    """

    def __init__(
        self,
        diffusion_model: nn.Module,
        config: Optional[object] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.diffusion_model = diffusion_model
        self.config = config

        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device

        lr = getattr(config, "learning_rate", 1e-4)
        wd = getattr(config, "weight_decay", 1e-5)

        self.optimizer = AdamW(diffusion_model.parameters(), lr=lr, weight_decay=wd)
        self.n_epochs = getattr(config, "n_epochs", 200)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.n_epochs, eta_min=1e-6)

        self.loss_fn = DiffusionLoss()

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_coord_loss": [], "val_coord_loss": [],
            "train_atom_loss": [], "val_atom_loss": [],
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: str = "./checkpoints",
        patience: int = 25,
    ) -> Dict[str, List[float]]:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        early_stop = EarlyStopping(patience=patience, mode="min")
        checkpointer = ModelCheckpoint(
            checkpoint_dir, filename="tcd_diffusion.pt", monitor="val_loss", mode="min"
        )

        self.diffusion_model.to(self.device)

        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()
            train_m = self._train_epoch(train_loader)
            val_m = self._val_epoch(val_loader) if val_loader else {}

            self.scheduler.step()

            for k in ["loss", "coord_loss", "atom_loss"]:
                self.history[f"train_{k}"].append(train_m.get(k, float("nan")))
                self.history[f"val_{k}"].append(val_m.get(k, float("nan")))

            logger.info(
                "TCD epoch %d/%d | train_loss=%.4f (coord=%.4f, atom=%.4f) | val_loss=%.4f | %.1fs",
                epoch, self.n_epochs,
                train_m.get("loss", 0),
                train_m.get("coord_loss", 0),
                train_m.get("atom_loss", 0),
                val_m.get("loss", 0),
                time.time() - t0,
            )

            val_loss = val_m.get("loss", float("nan"))
            checkpointer.step(val_loss, self.diffusion_model)
            if early_stop.step(val_loss):
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

        checkpointer.load_best(self.diffusion_model)
        return self.history

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.diffusion_model.train()
        total, total_coord, total_atom = 0.0, 0.0, 0.0
        n = 0

        for batch in loader:
            # batch: dict with coords, atom_types, pocket_graph, recruiter_graph, geometry
            coords = batch["coords"].to(self.device, dtype=torch.float32)
            atom_types = batch["atom_types"].to(self.device, dtype=torch.long)
            pocket_graph = batch.get("pocket_graph")
            if pocket_graph is not None:
                pocket_graph = pocket_graph.to(self.device)
            recruiter_graph = batch.get("recruiter_graph")
            if recruiter_graph is not None:
                recruiter_graph = recruiter_graph.to(self.device)
            geometry = batch.get("geometry")
            if geometry is not None:
                geometry = geometry.to(self.device, dtype=torch.float32)

            # Sample random timestep
            B = coords.shape[0]
            T = getattr(self.diffusion_model, "T", 1000)
            t = torch.randint(0, T, (B,), device=self.device)

            self.optimizer.zero_grad()
            coord_loss, atom_loss = self.diffusion_model.compute_loss(
                coords, atom_types, t, pocket_graph, recruiter_graph, geometry
            )
            loss = self.loss_fn(coord_loss, atom_loss)
            loss.backward()
            nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), 1.0)
            self.optimizer.step()

            total += loss.item()
            total_coord += coord_loss.item() if isinstance(coord_loss, torch.Tensor) else coord_loss
            total_atom += atom_loss.item() if isinstance(atom_loss, torch.Tensor) else atom_loss
            n += 1

        denom = max(n, 1)
        return {"loss": total / denom, "coord_loss": total_coord / denom, "atom_loss": total_atom / denom}

    @torch.no_grad()
    def _val_epoch(self, loader: Optional[DataLoader]) -> Dict[str, float]:
        if loader is None:
            return {}
        self.diffusion_model.eval()
        total, total_coord, total_atom = 0.0, 0.0, 0.0
        n = 0

        for batch in loader:
            coords = batch["coords"].to(self.device, dtype=torch.float32)
            atom_types = batch["atom_types"].to(self.device, dtype=torch.long)
            B = coords.shape[0]
            T = getattr(self.diffusion_model, "T", 1000)
            t = torch.randint(0, T, (B,), device=self.device)

            coord_loss, atom_loss = self.diffusion_model.compute_loss(
                coords, atom_types, t
            )
            loss = self.loss_fn(coord_loss, atom_loss)
            total += loss.item()
            total_coord += coord_loss.item() if isinstance(coord_loss, torch.Tensor) else coord_loss
            total_atom += atom_loss.item() if isinstance(atom_loss, torch.Tensor) else atom_loss
            n += 1

        denom = max(n, 1)
        return {"loss": total / denom, "coord_loss": total_coord / denom, "atom_loss": total_atom / denom}

    def save(self, path: str) -> None:
        torch.save({
            "model": self.diffusion_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "history": self.history,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.diffusion_model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.history = ckpt.get("history", self.history)


# ---------------------------------------------------------------------------
# MoleculeDataset and TCIPDiffusionTrainer (used by master_trainer Stage 2d)
# ---------------------------------------------------------------------------

class MoleculeDataset(Dataset):
    """Wraps a list of molecule sample dicts (with coords, atom_types, etc.)."""

    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


class TCIPDiffusionTrainer:
    """Simplified diffusion trainer interface used by Stage 2d in master_trainer."""

    def __init__(
        self,
        hidden_dim: int = 256,
        n_egnn_layers: int = 8,
        n_timesteps: int = 1000,
        n_epochs: int = 500,
        batch_size: int = 16,
        lr: float = 5e-5,
        device: Optional[torch.device] = None,
        checkpoint_path: Optional[Path] = None,
    ):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or torch.device("cpu")
        self.checkpoint_path = checkpoint_path
        self.model: Optional[nn.Module] = None  # Set externally

    def train(self, dataset: MoleculeDataset) -> nn.Module:
        assert self.model is not None, "Set trainer.model before calling train()"
        if len(dataset) == 0:
            logger.warning("TCIPDiffusionTrainer: empty dataset, skipping training")
            return self.model

        self.model = self.model.to(self.device)
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.n_epochs, eta_min=1e-6)

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, collate_fn=lambda x: x)

        for epoch in range(self.n_epochs):
            epoch_losses: List[float] = []

            for batch_samples in loader:
                optimizer.zero_grad()
                batch_loss = torch.tensor(0.0, device=self.device)
                valid = 0

                for sample in batch_samples:
                    try:
                        coords = torch.tensor(
                            np.array(sample.get("coords", [[0, 0, 0]])),
                            dtype=torch.float32, device=self.device
                        ).unsqueeze(0)
                        atom_types = torch.tensor(
                            np.array(sample.get("atom_types", [0])),
                            dtype=torch.long, device=self.device
                        ).unsqueeze(0)

                        B = coords.shape[0]
                        t = torch.randint(0, self.model.n_timesteps, (B,), device=self.device)

                        self.model.train()
                        coord_loss, atom_loss = self.model.compute_loss(coords, atom_types, t)
                        loss = coord_loss + atom_loss
                        batch_loss = batch_loss + loss
                        valid += 1
                    except Exception as e:
                        logger.debug(f"Diffusion sample skipped: {e}")
                        continue

                if valid > 0:
                    (batch_loss / valid).backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    epoch_losses.append((batch_loss / valid).item())

            scheduler.step()

            if epoch % 100 == 0 or epoch == self.n_epochs - 1:
                logger.info(
                    "TCIPDiffusionTrainer epoch %d/%d: loss=%.4f",
                    epoch + 1, self.n_epochs,
                    np.mean(epoch_losses) if epoch_losses else float("nan"),
                )

        if self.checkpoint_path is not None:
            Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), self.checkpoint_path)
            logger.info("TCIPDiffusionTrainer saved to %s", self.checkpoint_path)

        return self.model
