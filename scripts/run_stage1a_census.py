#!/usr/bin/env python3
"""Stream CellxGENE Census (~53M cells × 30k genes ≈ 1.6T pairs) for Stage 1a bio pretraining."""
import sys, os, logging, importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("logs/stage1a_census.log"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("oracle.stage1a")

import torch
from oracle.utils.device import get_device

DATA_DIR = Path("data")
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)

device = get_device()
logger.info(f"Device: {device}")

# ── Load models ───────────────────────────────────────────────────────────────
from oracle.models.cancer_score_mlp import CancerScoreFunction
from oracle.models.grn_transformer import GRNTransformer
from oracle.training.master_trainer import Stage1BiologicalTrainer

N_GENES = 19_331  # HVGs in Census (standard ORACLE gene set)

cancer_score_model = CancerScoreFunction(n_genes=N_GENES)
grn_transformer = GRNTransformer(n_genes=N_GENES)

n_cs = sum(p.numel() for p in cancer_score_model.parameters())
n_grn = sum(p.numel() for p in grn_transformer.parameters())
logger.info(f"CancerScoreFunction: {n_cs/1e6:.1f}M  |  GRNTransformer: {n_grn/1e6:.1f}M  |  Total: {(n_cs+n_grn)/1e6:.1f}M")

# Resume from checkpoint if available
ckpt_path = CKPT_DIR / "stage1a_census.pt"
if ckpt_path.exists():
    logger.info(f"Resuming from {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "cancer_score_state_dict" in state:
        cancer_score_model.load_state_dict(state["cancer_score_state_dict"])  # CancerScoreFunction
    if "grn_state_dict" in state:
        grn_transformer.load_state_dict(state["grn_state_dict"])

trainer = Stage1BiologicalTrainer(
    cancer_score_model=cancer_score_model,
    grn_transformer=grn_transformer,
    checkpoint_dir=CKPT_DIR,
    device=device,
    n_epochs_per_source=3,
    batch_size=64,
    lr=1e-4,
)

# ── Load CellxGeneCensusLoader from build_biological_pretrain_dataset.py ──────
_bio_script = ROOT / "scripts" / "build_biological_pretrain_dataset.py"
_spec = importlib.util.spec_from_file_location("build_bio", _bio_script)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CellxGeneCensusLoader = _mod.CellxGeneCensusLoader

logger.info("Initializing CellxGENE Census loader...")
census_loader = CellxGeneCensusLoader(cache_dir=DATA_DIR / "raw/census")

# ── Run Stage 1a ──────────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("Stage 1a: CellxGENE Census — ~53M cells × ~30k genes ≈ 1.6T pairs")
logger.info("=" * 60)

trainer.train_on_census(census_loader)

# Save final checkpoint
torch.save({
    "cancer_score_state_dict": cancer_score_model.state_dict(),
    "grn_state_dict": grn_transformer.state_dict(),
    "global_step": trainer.global_step,
}, CKPT_DIR / "stage1a_complete.pt")

logger.info(f"Stage 1a complete — {trainer.global_step:,} steps | checkpoint saved")
