"""
oracle/tcd/__init__.py
----------------------
Transcriptional CIP Designer (TCD) — Module 3 of ORACLE.

Designs Transcriptional CIP (TCIP) molecules that recruit epigenetic writers
or erasers to transcription factor binding sites, enabling targeted epigenetic
reprogramming of cancer cell states toward normal phenotypes.

Pipeline
--------
1. TFStructurer     — Prepare 3-D structural target for each TF
2. WriterEraserSelector — Choose the optimal epigenetic recruiter
3. MoleculeGenerator   — Diffusion-based warhead generation
4. LinkerDesigner       — Design the connecting linker
5. TernaryComplexValidator — Validate ternary complex geometry
6. TCIPScorer           — Rank and select top TCIPs
"""

from oracle.tcd.tf_structurer import TFStructurer, TFStructureResult
from oracle.tcd.writer_selector import WriterEraserSelector, WriterEraserSelection
from oracle.tcd.molecule_generator import MoleculeGenerator
from oracle.tcd.linker_designer import LinkerDesigner, LinkerInfo
from oracle.tcd.tcip_assembler import TCIPAssembler, AssembledTCIP
from oracle.tcd.ternary_validator import TernaryValidator, TernaryValidationResult
from oracle.tcd.tcip_scorer import TCIPScorer
from oracle.tcd.tcd_pipeline import TCDPipeline

__all__ = [
    "TFStructurer",
    "TFStructureResult",
    "WriterEraserSelector",
    "WriterEraserSelection",
    "MoleculeGenerator",
    "LinkerDesigner",
    "LinkerInfo",
    "TCIPAssembler",
    "AssembledTCIP",
    "TernaryValidator",
    "TernaryValidationResult",
    "TCIPScorer",
    "TCDPipeline",
]
