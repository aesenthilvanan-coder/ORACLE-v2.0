<div align="center">

```
 ██████╗ ██████╗  █████╗  ██████╗██╗     ███████╗
██╔═══██╗██╔══██╗██╔══██╗██╔════╝██║     ██╔════╝
██║   ██║██████╔╝███████║██║     ██║     █████╗  
██║   ██║██╔══██╗██╔══██║██║     ██║     ██╔══╝  
╚██████╔╝██║  ██║██║  ██║╚██████╗███████╗███████╗
 ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝╚══════╝
                        v 2.0
```

**Oncogenic Reversion via Attractor-guided Computational Landscape Engineering**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![RDKit](https://img.shields.io/badge/RDKit-2023%2B-009CE3?style=flat-square)](https://rdkit.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Research-F59E0B?style=flat-square)]()
[![Models](https://img.shields.io/badge/Target%20Hardware-M1%2016GB%20%E2%86%92%20512M%20params-8B5CF6?style=flat-square)]()

*Can cancer be persuaded to forget it is cancer?*

</div>

---

## The Hypothesis

Every cancer cell is trapped in a **pathological attractor** — a stable gene expression state maintained by self-reinforcing epigenetic circuits. Normal tissue identity is a different attractor in the same cell's regulatory landscape, separated by an epigenetic barrier written by enzymes like EZH2 and BRD4.

ORACLE asks: **what is the minimal set of transcription factor perturbations that, when applied simultaneously via bifunctional epigenetic molecules, would push a cancer cell over that barrier and lock it into the normal attractor?**

This is not killing cancer. It is **reverting** it.

The approach is grounded in the KAIST REVERT proof-of-concept (Shin et al. 2025), which demonstrated transcriptional identity reversion in colorectal cancer. ORACLE is a computational framework to generalize that result to any cancer type, any patient, and produce drug-like molecule designs (TCIPs) to execute it.

---

## What ORACLE Does

```
scRNA-seq / scATAC-seq                         Drug-like TCIP molecules
       ↓                                               ↑
[ MODULE 0: Preprocessing ]     →    [ MODULE 2: TCD (Molecule Design) ]
       ↓                                               ↑
[ MODULE 1A: CAM ] → Cancer Attractor              RSP Output
[ MODULE 1B: RSP ] → Reversion Switch Set  ────────────┘
```

| Module | Full Name | What It Does |
|--------|-----------|--------------|
| **CAM** | Cancer Attractor Mapper | Infers the GRN topology and identifies the gene expression attractor state from scRNA-seq. Uses GNN + pseudotime + Boolean ODE simulation. |
| **RSP** | Reversion Switch Predictor | Finds the minimum set of TF activations/repressions that drive the cancer attractor to the normal attractor. Combinatorial search over GRN perturbations. |
| **TCD** | Transcriptional CIP Designer | Designs PROTAC-like bifunctional molecules (TCIPs) that recruit epigenetic writers/erasers to each TF locus. Assembles warhead + linker + recruiter. |

---

## Architecture

```
oracle/
├── cam/                        # Cancer Attractor Mapping
│   ├── cam_pipeline.py         # Orchestrator: scRNA → attractor state
│   ├── grn_inference.py        # GRN reconstruction (GENIE3 / correlation)
│   ├── attractor_finder.py     # Energy landscape + basin identification
│   ├── attractor_classifier.py # ML classifier: cancer vs. normal cell state
│   ├── boolean_network.py      # Boolean ODE network simulation
│   ├── continuous_ode.py       # Continuous ODE for attractor convergence
│   ├── landscape_computer.py   # Waddington potential surface computation
│   ├── pseudotime.py           # Diffusion pseudotime / trajectory inference
│   └── preprocessing.py        # CAM-specific data prep
│
├── rsp/                        # Reversion Switch Prediction
│   ├── rsp_pipeline.py         # Orchestrator: attractor → switch set
│   ├── cancer_score.py         # Differentiability scoring (cancer vs. normal)
│   ├── combinatorial_search.py # Beam search over TF perturbation combos
│   ├── gnn_predictor.py        # GNN: predicts reversion probability
│   ├── perturbation_sim.py     # In-silico perturbation & trajectory sim
│   ├── switch_optimizer.py     # Optimizes the switch set size vs. efficacy
│   ├── druggability_filter.py  # Filters for TF druggability (pocket score)
│   └── trajectory_tracker.py  # Tracks cell trajectory under perturbation
│
├── tcd/                        # TCIP Molecule Design
│   ├── tcd_pipeline.py         # Orchestrator: switch set → TCIP molecules
│   ├── tcip_assembler.py       # 5-tier amide coupling (warhead+linker+recruiter)
│   ├── linker_designer.py      # PEG/alkyl linker library + scoring
│   ├── writer_selector.py      # Writer/eraser selection (p300/BRD4/HDAC2/EZH2/PRMT5)
│   ├── tf_structurer.py        # TF pocket prediction (PDB/AlphaFold)
│   ├── molecule_generator.py   # EGNN-based 3D molecule generation (DDPM)
│   ├── tcip_scorer.py          # Multi-objective molecule scoring
│   ├── ternary_validator.py    # Ternary complex clash/geometry validation
│   └── hard_constraints.py     # Lipinski/Veber/PAINS/Brenk/Ames hard gates
│
├── models/                     # Neural network architectures
│   ├── grn_transformer.py      # Transformer for GRN inference
│   ├── attractor_gnn.py        # GNN: attractor state classification
│   ├── switch_predictor_gnn.py # GNN: switch set prediction
│   ├── cancer_score_mlp.py     # MLP: cancer score function
│   ├── affinity_predictor.py   # MPNN: TF-warhead binding affinity
│   ├── tcip_diffusion.py       # EGNN-DDPM: 3D TCIP generation
│   ├── ternary_complex_predictor.py  # Ternary complex geometry model
│   └── shared/                 # Shared layers (attention, GAT, transformer, SE3)
│
├── preprocessing/              # scRNA-seq preprocessing pipeline
│   ├── scrna_preprocessor.py   # 13-step scRNA pipeline (QC→normalization→HVG)
│   ├── cnv_inference.py        # CNV scoring from expression
│   └── cell_annotator.py       # Cell type annotation
│
├── data/                       # Data loading & fetching
│   ├── fetchers/               # GEO, CellxGene, PDB, AlphaFold, ZINC, TCGA, ENCODE
│   ├── datasets.py             # PyTorch datasets
│   ├── loaders.py              # DataLoader wrappers
│   ├── collators.py            # Batch collation
│   └── samplers.py             # Stratified / balanced samplers
│
├── training/                   # Training infrastructure
│   ├── master_trainer.py       # End-to-end training orchestrator
│   ├── cam_trainer.py          # CAM training loop
│   ├── rsp_trainer.py          # RSP training loop
│   ├── tcd_trainer.py          # TCD training loop
│   ├── losses.py               # All loss functions
│   ├── callbacks.py            # Checkpointing, LR scheduling, early stop
│   └── data_leakage_protocols.py # Patient-level train/test split enforcement
│
├── evaluation/                 # Evaluation & benchmarking
│   ├── benchmarks.py           # End-to-end pipeline benchmarks
│   ├── cam_eval.py             # Attractor accuracy, ARI, NMI
│   ├── rsp_eval.py             # Switch set reversion probability
│   └── tcd_eval.py             # TCIP validity, QED, SA, docking proxy
│
├── visualization/              # Output visualization
│   ├── landscape_viz.py        # 3D Waddington surface
│   ├── trajectory_viz.py       # Cell trajectory plots (UMAP, diffusion map)
│   ├── network_viz.py          # GRN network visualization
│   ├── molecule_viz.py         # TCIP 2D/3D structure rendering
│   └── report_generator.py     # Full HTML/PDF report generation
│
└── interfaces.py               # Frozen dataclass contracts (all module I/O)

scripts/
├── run_inference.py            # Full pipeline: GSE accession → TCIP SMILES
├── run_gbm_pipeline.py         # GBM flagship demo (no training required)
├── run_luad_pipeline.py        # LUAD pipeline (GSE131907)
├── run_aml_pipeline.py         # AML pipeline
├── run_atc_pipeline.py         # Anaplastic thyroid carcinoma pipeline
├── train_all.py                # Train all three modules end-to-end
├── fetch_data.py               # Download & cache GEO/Census datasets
├── preprocess_all.py           # Preprocess all cached datasets
├── plot_gbm_landscape_3d.py    # 3D Waddington landscape visualization
└── plot_gbm_attractor.py       # 2D attractor map with TCIP table
```

---

## Quick Start

### Installation

```bash
# Clone
git clone https://github.com/aesenthilvanan-coder/ORACLE-v2.0.git
cd ORACLE-v2.0

# Create environment (conda recommended)
conda create -n oracle python=3.11 -y
conda activate oracle

# Core dependencies
pip install -r requirements.txt

# Biology stack (scanpy, anndata, cellxgene-census)
pip install -r requirements-bio.txt

# Verify installation
make smoke-test
```

### Run the GBM Demo (No Training Required)

The GBM pipeline is fully self-contained and runs off literature-curated biology. It produces all 8 TCIP SMILES with properties and a 3D Waddington landscape in under 60 seconds.

```bash
# Design GBM TCIPs
python scripts/run_gbm_pipeline.py

# Render 3D Waddington landscape
python scripts/plot_gbm_landscape_3d.py

# Outputs appear in outputs/
ls outputs/
# gbm_landscape_3d.png   ← 3D epigenetic landscape
# gbm_attractor_map.png  ← attractor gene grid + TCIP table
```

### Full Pipeline from GEO Accession

```bash
# Download + preprocess + infer + design — one command
python scripts/run_inference.py \
    --gse GSE131928 \
    --cancer_type GBM \
    --normal_gse GSE67835 \
    --output_dir outputs/gbm_full/

# With a trained model checkpoint
python scripts/run_inference.py \
    --gse GSE131928 \
    --cancer_type GBM \
    --checkpoint checkpoints/stage1a_best.pt \
    --output_dir outputs/gbm_full/
```

---

## The GBM Demo: End-to-End Example

ORACLE ships with a complete Glioblastoma (GBM) analysis grounded in Neftel et al. 2019 (*Cell*, GSE131928). GBM was chosen as the flagship cancer because:

- Median survival: **14.6 months**. No second-line standard of care after recurrence.
- Cannot be fully resected — a *reversion* approach is uniquely compelling vs. a kill approach.
- The SOX2/NES stem-cell axis and GFAP/NEUROD1 mature-identity axis are the most well-validated in cancer biology.

### Cancer Attractor vs. Normal Attractor

| State | GBM (MES+NPC stem) | Normal Brain |
|-------|-------------------|--------------|
| **HIGH** | SOX2, NES, MYC, TWIST1, EZH2, BRD4, CDK4, EGFR, STAT3, VIM, CDH2 | GFAP, NEUROD1, RBFOX1, TUBB3, MAP2, S100B |
| **LOW** | NEUROD1, RBFOX1, GFAP, TUBB3, MAP2, CDKN2A | SOX2, NES, MYC, TWIST1, EZH2, BRD4 |

GBM oscillates between two co-dominant sub-attractors inside the cancer basin:

```
  MES sub-state ←─────────── GBM Cancer Basin ───────────→ NPC sub-state
  TWIST1↑ VIM↑ CDH2↑ ZEB1↑                               SOX2↑ NES↑ MYC↑ CDK4↑
```

Both must be targeted simultaneously — hitting only one allows the other to re-seed.

### Reversion Switch Set (RSP Output)

```
ACTIVATE  →  NEUROD1  RBFOX1  GFAP
REPRESS   →  SOX2  MYC  TWIST1  EZH2  BRD4
```

**Why these 8 genes:** EZH2 and BRD4 form a self-reinforcing epigenetic loop that keeps the cancer attractor stable:

```
BRD4 ──reads──→ H3K27ac at SOX2/MYC SE ──drives──→ SOX2/MYC expression
  ↑                                                        ↓
EZH2 expression ←── MYC transcription ←──────────────────┘
  ↓
H3K27me3 at NEUROD1/GFAP ←── EZH2 activity (silences normal identity)
```

ORACLE attacks this loop at three points: repressing EZH2/BRD4 expression, flipping the super-enhancers from H3K27ac to H3K27me3, and activating NEUROD1/RBFOX1/GFAP to pull the cell into the normal basin.

### TCIP Molecules (TCD Output) — 8/8 Pass · All Tier-1 Amide Assembly

| Gene | Effect | Recruiter | Ki (nM) | MW | logP | QED | SMILES |
|------|--------|-----------|---------|-----|------|-----|--------|
| NEUROD1 | ACTIVATE | p300 A-485 | 10 | 664 | 5.06 | 0.116 | `CC(CNC(=O)CCOCCOCCNC(=O)Cc1cc2ccccc2[nH]1)COC(=O)N[C@H]1CC[C@@H](c2nc3ccccc3s2)CC1` |
| RBFOX1 | ACTIVATE | p300 A-485 | 10 | 662 | 5.20 | 0.139 | `CC(CNC(=O)CCOCCOCCNC(=O)c1ccc2cnccc2c1)COC(=O)N[C@H]1CC[C@@H](c2nc3ccccc3s2)CC1` |
| GFAP | ACTIVATE | BRD4 JQ1 | 77 | 684 | 2.68 | 0.168 | `Cc1nc2c(-c3cc(C(=O)N4CCCC4)nn3C)c(NC(=O)CCOCCOCCNC(=O)c3ccc(S(N)(=O)=O)cc3)ccc2s1` |
| SOX2 | REPRESS | PRMT5 GSK | 3 | 686 | 4.24 | 0.138 | `O=C(CCOCCOCCNC(=O)c1ccc(-c2nc3ccccc3[nH]2)cc1)Nc1ccc(S(=O)(=O)N2CCC(n3ccnc3)CC2)cc1` |
| MYC | REPRESS | PRMT5 GSK | 3 | 713 | 4.39 | 0.125 | `O=C(CCOCCOCCNC(=O)c1ccc(Nc2ncnc3ccccc23)cc1)Nc1ccc(S(=O)(=O)N2CCC(n3ccnc3)CC2)cc1` |
| TWIST1 | REPRESS | HDAC2 entinostat | 1.5 | 642 | 3.88 | 0.048 | `O=C(CCOCCOCCNC(=O)c1ccc2[nH]ccc2c1)Nc1ccc2[nH]c(C(=O)NCc3ccc(NC(=O)NO)cc3)cc2c1` |
| EZH2 | REPRESS | PRMT5 GSK | 3 | 669 | 3.24 | 0.193 | `O=C(CCOCCOCCNC(=O)c1cnc(NC2CCCCC2)nc1)Nc1ccc(S(=O)(=O)N2CCC(n3ccnc3)CC2)cc1` |
| BRD4 | REPRESS | EZH2 EPZ-6438 | 2.5 | 681 | 2.07 | 0.193 | `O=C(CCOCCOCCNC(=O)Cc1nc2ccccc2s1)NCC(=O)Nc1ccc(C(=O)N2CCC(N3CCOCC3)CC2)cc1` |

All molecules: MW 640–713 Da · SA ≤ 3.5 · logP 2.1–5.2 · Connected single fragment · bRo5 PROTAC space

---

## Module Deep Dive

### Module 1A: CAM

```python
from oracle.cam.cam_pipeline import CAMPipeline

cam = CAMPipeline(config={
    "cancer_type": "GBM",
    "n_hvg": 3000,
    "n_pcs": 50,
    "grn_method": "genie3",
    "n_attractors": 2,
})

cam_output = cam.run(adata, normal_adata=normal_adata)

print(cam_output.cancer_genes_on)   # genes HIGH in cancer attractor
print(cam_output.cancer_genes_off)  # genes LOW in cancer attractor
print(cam_output.attractor_score)   # per-cell cancer attractor score [0, 1]
```

**Internally:** 13-step scRNA preprocessing → GRN inference → pseudotime → Boolean ODE simulation → energy landscape → basin identification → normal attractor comparison.

### Module 1B: RSP

```python
from oracle.rsp.rsp_pipeline import RSPPipeline

rsp = RSPPipeline(config={
    "max_switch_size": 10,
    "reversion_threshold": 0.7,
    "beam_width": 50,
    "n_simulations": 1000,
})

rsp_output = rsp.run(cam_output)

print(rsp_output.genes_to_activate)          # ['NEUROD1', 'RBFOX1', 'GFAP']
print(rsp_output.genes_to_repress)           # ['SOX2', 'MYC', 'TWIST1', 'EZH2', 'BRD4']
print(rsp_output.validated_reversion_fraction)  # 0.83
```

**Internally:** Cancer score gradient → beam search over TF combos → 1000× ODE/Boolean simulation per candidate → reversion fraction scoring → druggability filter → minimal switch set.

### Module 2: TCD

```python
from oracle.tcd.tcd_pipeline import TCDPipeline

tcd = TCDPipeline(config={
    "linker_library": "full",
    "max_mw": 1000,
    "require_connected": True,
})

tcd_output = tcd.run(rsp_output)

for tcip in tcd_output.tcip_molecules:
    print(f"{tcip.target_tf:10s} | {tcip.perturbation_type:8s} | "
          f"MW={tcip.molecular_weight:.0f} | QED={tcip.qed:.3f} | "
          f"{'PASS' if tcip.validation_result.passed else 'FAIL'}")
    print(f"  {tcip.full_smiles}")
```

**TCIP assembly architecture:**

```
[TF warhead]─── amide ───[H₂N-Linker-COOH]─── amide ───[Epigenetic recruiter]
     ↑                          ↑                               ↑
Binds TF protein         PEG₂/PEG₃/alkyl             Writer (p300/BRD4/CDK9) or
(HMG/bHLH/SET/RRM)      5–20 heavy atoms             Eraser (HDAC2/EZH2/PRMT5)
```

**5-tier assembly (no fallback for Tiers 1–4):**

| Tier | Strategy |
|------|----------|
| 1 | warhead-COOH + linker-NH₂ → product-COOH + recruiter-NH₂ |
| 2 | linker-COOH + warhead-NH₂ → try both second-step orientations |
| 3 | Brute-force all 6 permutations of the three fragments |
| 4 | Add acetic-acid COOH arm to warhead, then retry Tiers 1–3 |
| 5 | Force terminal-atom single bond (last resort) |

**Epigenetic recruiters:**

| Recruiter | Scaffold | Ki (nM) | Mark | Effect |
|-----------|----------|---------|------|--------|
| p300 | A-485 | 10 | H3K27ac write | Activation |
| BRD4 | JQ1 | 77 | H3K27ac amplify | Activation |
| CDK9 | AT7519 | 47 | pSer2 RNAPII | Activation |
| MED1 | Cortistatin A | 300 | Super-enhancer | Activation |
| HDAC2 | Entinostat | 1.5 | −H3K27ac | Repression |
| EZH2 | EPZ-6438 | 2.5 | H3K27me3 write | Repression |
| PRMT5 | GSK3326595 | 3.0 | H4R3me2s | Repression |
| LSD1 | Tranylcypromine | 243 | −H3K4me1 | Repression |
| DNMT3A | RG108 | 115 | 5mC CpG | Repression |

### Hard Constraints Filter

```python
from oracle.tcd.hard_constraints import TCIPHardConstraints

hc = TCIPHardConstraints()
result = hc.check(smiles, linker_smiles=linker_smiles)

if result.passed:
    print("PASS")
else:
    for v in result.violations:
        print(f"  FAIL: {v}")
```

| Constraint | Threshold (bRo5 PROTAC space) |
|-----------|-------------------------------|
| Lipinski MW | ≤ 1000 Da |
| Lipinski logP | ≤ 6.0 |
| HBD / HBA | ≤ 6 / ≤ 15 |
| Veber RotBonds / TPSA | ≤ 25 / ≤ 250 Å² |
| QED | ≥ 0.04 |
| SA Score | ≤ 7.0 |
| PAINS / Brenk / Ames | 0 alerts |

---

## Dataset Guide

### Recommended GBM Datasets

| GSE | Type | Description | Use |
|-----|------|-------------|-----|
| GSE131928 | scRNA | Neftel 2019 — 28 tumors, canonical MES/NPC/AC/OPC state map | Primary GBM attractor |
| GSE84465 | scRNA | Darmanis 2017 — first GBM single-cell atlas | GRN inference |
| GSE182109 | scRNA | Richards 2021 — stem cell hierarchy, 53K cells | Sub-basin resolution |
| GSE162631 | scRNA | 8 samples, 120K cells (10x) | High-density landscape |
| **GSE67835** | scRNA | **Zhang 2016 — normal human brain cell types** | **Normal attractor (required)** |
| GSE163120 | scATAC | Mack 2022 — GBM chromatin accessibility | Enhancer targeting |
| GSE194329 | Spatial | GBM Visium — primary + recurrent, IDH-wt | Tumor geography |
| GSE121719 | Bulk | Primary vs. recurrent paired | Attractor shift under SOC |

### Fetching Data

```bash
# Fetch a GEO dataset
python scripts/fetch_data.py --gse GSE131928 --output data/raw/

# Stream from CellxGene Census (no download)
python scripts/fetch_data.py --census --disease "glioblastoma" --n_cells 50000

# Fetch normal brain reference
python scripts/fetch_data.py --gse GSE67835 --output data/raw/normal/
```

---

## Training

```bash
# Stage 0: Molecular pretraining (ZINC22 + PubChem, ~5.6M molecules)
python scripts/run_stage0_only.py --config configs/base_config.yaml

# Stage 1a: Biological pretraining (CellxGene Census, ~2M cancer cells)
python scripts/run_stage1a_census.py --config configs/cam_config.yaml

# Stage 1b + 2: RSP and TCD training
python scripts/train_all.py --checkpoint checkpoints/stage1a_best.pt
```

**Hardware requirements:**

| Stage | Min RAM | Runtime (M1 16GB) |
|-------|---------|-------------------|
| Stage 0 (mol pretrain) | 8 GB | ~12h |
| Stage 1a (bio pretrain) | 16 GB | ~24h |
| Stage 1b (RSP) | 8 GB | ~4h |
| Stage 2 (TCD) | 8 GB | ~3h |
| Inference only | 4 GB | < 2 min |

---

## Cancer Types Supported

| Cancer | Script | Primary Dataset | Key Reversion Targets |
|--------|--------|----------------|-----------------------|
| GBM | `run_gbm_pipeline.py` | GSE131928 | SOX2/MYC/TWIST1/EZH2/BRD4 → NEUROD1/RBFOX1/GFAP |
| LUAD | `run_luad_pipeline.py` | GSE131907 | MYC/YAP1/ZEB1/EZH2 → FOXA2/NKX2-1 |
| AML | `run_aml_pipeline.py` | CellxGene Census | MYC/FLT3/EZH2 → CEBPA/PU.1 |
| ATC | `run_atc_pipeline.py` | CellxGene Census | BRAF effectors/SOX2 → PAX8/FOXE1 |

To add a new cancer type:

```bash
python scripts/run_inference.py \
    --gse YOUR_GSE_ID \
    --cancer_type YOUR_CANCER \
    --normal_gse MATCHED_NORMAL_GSE \
    --output_dir outputs/your_cancer/
```

---

## Interfaces

All modules communicate through frozen dataclass contracts in `oracle/interfaces.py`:

```python
from oracle.interfaces import (
    CAMOutput,         # Cancer attractor state + GRN + AnnData
    RSPOutput,         # Switch set (activate/repress) + reversion fraction
    TCDOutput,         # TCIP molecules + validation
    TCIPMolecule,      # Single TCIP: SMILES + properties + validation result
    ValidationResult,  # Hard constraint pass/fail + per-property scores
)
```

---

## Configuration

```yaml
# configs/base_config.yaml (excerpt)
model:
  max_params: 514_000_000    # M1 16GB upper bound
  hidden_dim: 512
  n_layers: 8

training:
  batch_size: 64
  learning_rate: 3e-4
  n_epochs: 50
  gradient_clip: 1.0

data:
  n_hvg: 3000
  min_cells: 200

tcd:
  linker_max_mw: 300
  require_connected: true
  max_assembly_tiers: 5      # 1 = strict Tier-1 amide only
```

---

## Visualizations

```bash
# 3D Waddington epigenetic landscape
python scripts/plot_gbm_landscape_3d.py
# → outputs/gbm_landscape_3d.png

# 2D attractor gene grid + TCIP summary table
python scripts/plot_gbm_attractor.py
# → outputs/gbm_attractor_map.png
```

The 3D landscape renders:
- GBM cancer basin (left well) with MES and NPC sub-basins
- Normal brain basin (right well) with astrocyte and neuron sub-basins
- Epigenetic barrier ridge (EZH2/BRD4 lock)
- TCIP reversion trajectory (purple) with per-intervention markers

---

## Project Background

ORACLE was built to explore a single scientific question: **can we compute a drug-like molecule that forces a cancer cell to re-adopt normal tissue identity?**

The key insight from KAIST REVERT (Shin et al. 2025) is that cancer is not only a genetic disease — it is a *cellular identity* disease. The cancer cell has not lost the code for normal function; it has been pushed into a different attractor in regulatory space and is held there by epigenetic locks. ORACLE's hypothesis is that those locks can be picked computationally.

The TCIP (Transcriptional Cancer Identity Perturbagen) concept is a PROTAC-like bifunctional molecule: one end binds a transcription factor, the other end recruits an epigenetic writer or eraser. Instead of degrading the TF (as PROTACs do), TCIPs **rewrite chromatin state** at TF-bound loci — converting active enhancers to silenced regions or vice versa, permanently redirecting cell identity.

---

## Citation

```bibtex
@software{oracle2025,
  author  = {Senthilvanan, Aravind E.},
  title   = {{ORACLE}: Oncogenic Reversion via Attractor-guided
             Computational Landscape Engineering},
  year    = 2025,
  version = {2.0},
  url     = {https://github.com/aesenthilvanan-coder/ORACLE-v2.0},
}
```

**Key references:**

- Shin et al. 2025 — REVERT: transcriptional identity reversion in colorectal cancer (KAIST)
- Neftel et al. 2019 — Integrative model of cellular states in GBM (*Cell*)
- Darmanis et al. 2017 — Single-cell characterization of GBM (*Nature Neuroscience*)
- Zhang et al. 2016 — Purification of progenitor and mature cells from human brain (*Neuron*)
- Filippakopoulos et al. 2010 — Selective inhibition of BET bromodomains (*Nature*)
- Konze et al. 2013 — An orally bioavailable chemical probe of EZH2 (*ACS Chemical Biology*)

---

## License

MIT License — see [LICENSE](LICENSE).

---

<div align="center">

*"The cancer cell already knows how to be normal. We just need to remind it."*

**Built with** Python · PyTorch · RDKit · scanpy · CellxGene Census

</div>
