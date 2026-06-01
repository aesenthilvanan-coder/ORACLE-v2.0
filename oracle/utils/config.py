"""
Configuration utilities for the ORACLE pipeline.

Provides:
- ``load_config``   – load a YAML config file via omegaconf
- ``get_device``    – intelligent device selection (MPS > CUDA > CPU)
- ``OracleConfig``  – omegaconf DictConfig wrapper
- ``CAMConfig``     – CAM module settings dataclass
- ``RSPConfig``     – RSP module settings dataclass
- ``TCDConfig``     – TCD module settings dataclass
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Sub-module config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CAMConfig:
    """Configuration for the Cancer Attractor Mapping (CAM) module.

    Attributes
    ----------
    n_genes:
        Number of genes in the regulatory network.
    n_tfs:
        Number of transcription factors to consider.
    n_hidden:
        Hidden dimension for the CancerScoreFunction MLP.
    n_layers:
        Number of MLP layers in the CancerScoreFunction.
    dropout:
        Dropout rate.
    grn_method:
        GRN inference method: ``"grnboost2"`` | ``"ppcor"`` | ``"scenic"``.
    n_attractors:
        Maximum number of attractors to identify.
    attractor_iters:
        Number of Boolean dynamics simulation iterations.
    cancer_score_threshold:
        Minimum score to classify a state as cancer.
    use_pseudotime:
        Whether to use pseudotime ordering for the monotonicity loss.
    learning_rate:
        Optimiser learning rate.
    batch_size:
        Training batch size.
    n_epochs:
        Number of training epochs.
    weight_decay:
        AdamW weight decay.
    cache_dir:
        Directory for intermediate file caches.
    output_dir:
        Directory for saving CAM outputs.
    """

    n_genes: int = 2000
    n_tfs: int = 200
    n_hidden: int = 512
    n_layers: int = 4
    dropout: float = 0.1
    grn_method: str = "grnboost2"
    n_attractors: int = 10
    attractor_iters: int = 1000
    cancer_score_threshold: float = 0.5
    use_pseudotime: bool = True
    learning_rate: float = 1e-3
    batch_size: int = 256
    n_epochs: int = 100
    weight_decay: float = 1e-4
    cache_dir: str = "./cache/cam"
    output_dir: str = "./outputs/cam"


@dataclass
class RSPConfig:
    """Configuration for the Reprogramming Switch Predictor (RSP) module.

    Attributes
    ----------
    gnn_hidden_dim:
        Hidden dimension in the SwitchPredictorGNN.
    gnn_n_layers:
        Number of message-passing layers.
    gnn_n_heads:
        Number of attention heads (for GAT-style layers).
    dropout:
        Dropout rate.
    n_perturbation_steps:
        Number of iterative perturbation steps.
    max_switches:
        Maximum number of TF switches to return.
    perturbation_magnitude:
        Delta applied to TF expression during in silico perturbation.
    top_k_candidates:
        Number of top TF candidates to score.
    learning_rate:
        Optimiser learning rate.
    batch_size:
        Training batch size.
    n_epochs:
        Number of training epochs.
    weight_decay:
        AdamW weight decay.
    importance_weight:
        Weight on the L1 sparsity regularisation term.
    cache_dir:
        Directory for intermediate file caches.
    output_dir:
        Directory for saving RSP outputs.
    """

    gnn_hidden_dim: int = 256
    gnn_n_layers: int = 4
    gnn_n_heads: int = 8
    dropout: float = 0.1
    n_perturbation_steps: int = 5
    max_switches: int = 10
    perturbation_magnitude: float = 2.0
    top_k_candidates: int = 50
    learning_rate: float = 1e-3
    batch_size: int = 64
    n_epochs: int = 150
    weight_decay: float = 1e-4
    importance_weight: float = 0.001
    cache_dir: str = "./cache/rsp"
    output_dir: str = "./outputs/rsp"


@dataclass
class TCDConfig:
    """Configuration for the TCIP Conditional Diffusion (TCD) module.

    Attributes
    ----------
    n_timesteps:
        Number of diffusion timesteps.
    beta_start:
        Starting beta value for the noise schedule.
    beta_end:
        Ending beta value for the noise schedule.
    hidden_dim:
        Hidden dimension of the diffusion model.
    n_layers:
        Number of equivariant layers.
    n_heads:
        Number of attention heads.
    dropout:
        Dropout rate.
    max_atoms:
        Maximum number of atoms in a TCIP molecule.
    atom_vocab_size:
        Size of the atom-type vocabulary.
    linker_length_range:
        Min/max linker length in number of heavy atoms.
    mw_range:
        Target molecular weight range [min, max] in Da.
    learning_rate:
        Optimiser learning rate.
    batch_size:
        Training batch size.
    n_epochs:
        Number of training epochs.
    weight_decay:
        AdamW weight decay.
    cache_dir:
        Directory for intermediate file caches.
    output_dir:
        Directory for saving TCD outputs.
    """

    n_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    hidden_dim: int = 512
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    max_atoms: int = 150
    atom_vocab_size: int = 14
    linker_length_range: List[int] = field(default_factory=lambda: [5, 20])
    mw_range: List[float] = field(default_factory=lambda: [700.0, 1200.0])
    learning_rate: float = 1e-4
    batch_size: int = 32
    n_epochs: int = 200
    weight_decay: float = 1e-5
    cache_dir: str = "./cache/tcd"
    output_dir: str = "./outputs/tcd"


# ---------------------------------------------------------------------------
# OracleConfig
# ---------------------------------------------------------------------------

class OracleConfig:
    """Thin wrapper around an omegaconf ``DictConfig`` providing typed access
    to the three sub-module configs.

    Parameters
    ----------
    cfg:
        An omegaconf ``DictConfig`` loaded from a YAML file.  Expected to
        have ``cam``, ``rsp``, and ``tcd`` top-level keys.
    """

    def __init__(self, cfg) -> None:
        self._cfg = cfg

    # ------------------------------------------------------------------
    # Sub-module config accessors
    # ------------------------------------------------------------------

    @property
    def cam(self) -> CAMConfig:
        """Return a ``CAMConfig`` populated from the config."""
        return _dict_to_dataclass(CAMConfig, self._cfg.get("cam", {}))

    @property
    def rsp(self) -> RSPConfig:
        """Return an ``RSPConfig`` populated from the config."""
        return _dict_to_dataclass(RSPConfig, self._cfg.get("rsp", {}))

    @property
    def tcd(self) -> TCDConfig:
        """Return a ``TCDConfig`` populated from the config."""
        return _dict_to_dataclass(TCDConfig, self._cfg.get("tcd", {}))

    # ------------------------------------------------------------------
    # Passthrough
    # ------------------------------------------------------------------

    def get(self, key: str, default=None):
        return self._cfg.get(key, default)

    def __getattr__(self, name: str):
        try:
            return self._cfg[name]
        except (KeyError, TypeError):
            raise AttributeError(f"OracleConfig has no attribute '{name}'")

    def __repr__(self) -> str:  # pragma: no cover
        return f"OracleConfig({self._cfg})"


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def load_config(path: str) -> OracleConfig:
    """Load an ORACLE YAML configuration file.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    OracleConfig
    """
    try:
        from omegaconf import OmegaConf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "omegaconf is required for load_config. "
            "Install with: pip install omegaconf"
        ) from exc

    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    cfg = OmegaConf.load(path)
    return OracleConfig(cfg)


def get_device(config=None):
    """Intelligently select the best available compute device.

    Priority: MPS (Apple Silicon) > CUDA > CPU.

    Parameters
    ----------
    config:
        Optional config object; if it has a ``device`` attribute that
        attribute is returned directly as a ``torch.device``.

    Returns
    -------
    torch.device
    """
    import torch

    # Respect explicit override
    if config is not None:
        device_str = None
        if hasattr(config, "device"):
            device_str = config.device
        elif hasattr(config, "get"):
            device_str = config.get("device", None)

        if device_str and device_str != "auto":
            return torch.device(device_str)

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dict_to_dataclass(cls, cfg):
    """Populate a dataclass from a dict-like config object.

    Missing keys fall back to dataclass defaults; extra keys are ignored.
    """
    import dataclasses

    if cfg is None:
        return cls()

    # Convert omegaconf DictConfig / dict
    if hasattr(cfg, "to_container"):
        cfg = cfg.to_container(resolve=True)
    elif not isinstance(cfg, dict):
        try:
            from omegaconf import OmegaConf  # type: ignore
            cfg = OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            cfg = dict(cfg) if hasattr(cfg, "__iter__") else {}

    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in cfg.items() if k in field_names}
    return cls(**filtered)
