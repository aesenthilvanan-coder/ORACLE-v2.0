#!/usr/bin/env python3
"""Build Stage 0 pretraining dataset and run foundation training."""
import sys, os, json, logging
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
        logging.FileHandler("logs/stage0_training.log"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger("oracle.stage0")

DATA_DIR  = Path("data")
# data/pretrain_shards so builder resolves output_dir.parent/"raw"/"molecules" → data/raw/molecules
SHARD_DIR = DATA_DIR / "pretrain_shards"
CKPT_DIR  = Path("checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
SHARD_DIR.mkdir(parents=True, exist_ok=True)

# Step 1 — Build pretraining dataset from downloaded sources
manifest = SHARD_DIR / "manifest.json"
if not manifest.exists():
    logger.info("Building Stage 0 pretraining dataset...")
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_pretrain_dataset import PretrainingDatasetBuilder

    builder = PretrainingDatasetBuilder(
        output_dir=SHARD_DIR,
        n_workers=8,
        generate_3d=False,   # 3D gen is 500× slower; SMILES augmentation gives scale
        n_smiles_aug=10,
        n_conformers=0,
        n_se3_aug=0,
    )
    n_total = builder.build()
    logger.info(f"Dataset built: {n_total:,} examples")
else:
    with open(manifest) as f:
        info = json.load(f)
    logger.info(f"Existing dataset: {info.get('n_examples',0):,} examples across {info.get('n_shards',0)} shards")

# Step 2 — Stage 0 foundation training
with open(manifest) as f:
    info = json.load(f)

if info.get("n_shards", 0) == 0:
    logger.error("No shards built — check molecule source files in data/raw/molecules/")
    sys.exit(1)

logger.info("Starting Stage 0 foundation training...")
import torch
from oracle.utils.device import get_device
from oracle.models.tcip_diffusion import TCIPDiffusionModel
from oracle.training.master_trainer import Stage0FoundationTrainer

device = get_device()
logger.info(f"Device: {device}")

model = TCIPDiffusionModel()
model.to(device)

n_params = sum(p.numel() for p in model.parameters())
logger.info(f"TCIPDiffusionModel: {n_params/1e6:.1f}M parameters")

trainer = Stage0FoundationTrainer(
    model=model,
    shard_dir=SHARD_DIR,
    checkpoint_dir=CKPT_DIR,
    device=device,
    n_epochs=3,              # 3 epochs × ~1.2h each ≈ 3.5h total — fits before interview
    batch_size=32,           # reduced from 256 — 450M model needs more headroom on 16GB
    lr=1e-4,
    lr_warmup_steps=10_000,
    ema_decay=0.9999,
    log_every=500,
    checkpoint_every_steps=10_000,
    n_dataloader_workers=0,  # 0 = main process (avoids macOS spawn env issues)
    max_shards=55,           # cap at original 5.5M examples; ignore zinc20 expansion
)

# Resume from latest step checkpoint if available
start_epoch = 0
step_ckpts = sorted(CKPT_DIR.glob("stage0_step_*.pt"))
if step_ckpts:
    latest_ckpt = step_ckpts[-1]
    logger.info(f"Resuming from checkpoint: {latest_ckpt}")
    state = torch.load(latest_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    trainer.ema_model.load_state_dict(state["ema_state_dict"])
    trainer.optimizer.load_state_dict(state["optimizer_state_dict"])
    trainer.scheduler.load_state_dict(state["scheduler_state_dict"])
    trainer.global_step = state["global_step"]
    trainer.loss_history = state.get("loss_history", [])
    trainer.best_loss = min(trainer.loss_history) if trainer.loss_history else float("inf")
    start_epoch = state.get("epoch", 0)
    logger.info(f"Resumed at global_step={trainer.global_step:,}, epoch={start_epoch}")

trainer.train(start_epoch=start_epoch)
torch.save({"ema_state_dict": trainer.ema_model.state_dict()}, CKPT_DIR / "stage0_complete.pt")
logger.info("Stage 0 complete — checkpoint saved.")
