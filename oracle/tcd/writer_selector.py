from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class WriterEraserInfo:
    name: str
    mechanism: str
    recruiter_scaffold: str
    recruiter_smiles: str
    recruiter_ki_nM: float
    target_histone_mark: str
    transcriptional_effect: str


WRITERS: Dict[str, WriterEraserInfo] = {
    "BRD4": WriterEraserInfo(
        name="BRD4",
        mechanism="bromodomain_H3K27ac_reader",
        recruiter_scaffold="JQ1",
        recruiter_smiles="CC1=C(C2=CC=CC=C2S1)C3=NN4C(=C3)N=C(C)N(CC(=O)N5CCC[C@H]5C(=O)NC(C)(C)C4=O",
        recruiter_ki_nM=77.0,
        target_histone_mark="H3K27ac",
        transcriptional_effect="activation",
    ),
    "CDK9": WriterEraserInfo(
        name="CDK9",
        mechanism="P-TEFb_kinase_pSer2_RNAPII",
        recruiter_scaffold="AT7519",
        recruiter_smiles="CC1=C(NC(=O)NC2CCCCC2)SC=C1C(=O)N3CCC[C@@H]3CN4CCOCC4",
        recruiter_ki_nM=47.0,
        target_histone_mark="pSer2_RNAPII",
        transcriptional_effect="activation",
    ),
    "p300": WriterEraserInfo(
        name="p300",
        mechanism="HAT_H3K27ac_writer",
        recruiter_scaffold="A-485",
        recruiter_smiles="CC(C)COC(=O)N[C@@H]1CC[C@@H](CC1)c2nc3ccccc3s2",
        recruiter_ki_nM=10.0,
        target_histone_mark="H3K27ac",
        transcriptional_effect="activation",
    ),
    "MED1": WriterEraserInfo(
        name="MED1",
        mechanism="mediator_complex_anchor",
        recruiter_scaffold="cortistatin_A",
        recruiter_smiles="[H][C@@]12C[C@@H](OC(C)=O)[C@]3([H])C[C@@H](O)[C@]4(C)CC[C@@H]([C@@H]4[C@@H]3[C@@H](O)[C@]1(C)[C@H]1O2)C(=O)OC",
        recruiter_ki_nM=300.0,
        target_histone_mark="super_enhancer",
        transcriptional_effect="activation",
    ),
}

ERASERS: Dict[str, WriterEraserInfo] = {
    "HDAC1": WriterEraserInfo(
        name="HDAC1",
        mechanism="HDAC_zinc_deacetylase",
        recruiter_scaffold="vorinostat",
        recruiter_smiles="O=C(CCCCCCC(=O)Nc1ccccc1)NO",
        recruiter_ki_nM=10.0,
        target_histone_mark="H3K27ac",
        transcriptional_effect="repression",
    ),
    "HDAC2": WriterEraserInfo(
        name="HDAC2",
        mechanism="HDAC_zinc_deacetylase",
        recruiter_scaffold="entinostat",
        recruiter_smiles="O=C(Nc1ccc(cc1)CNC(=O)c2cc3ccccc3[nH]2)NO",
        recruiter_ki_nM=1.5,
        target_histone_mark="H3K27ac",
        transcriptional_effect="repression",
    ),
    "EZH2": WriterEraserInfo(
        name="EZH2",
        mechanism="PRC2_H3K27me3_writer",
        recruiter_scaffold="EPZ-6438",
        recruiter_smiles="CC(=O)Nc1ccc(cc1)C(=O)N2CC[C@@H](CC2)N3CCOCC3",
        recruiter_ki_nM=2.5,
        target_histone_mark="H3K27me3",
        transcriptional_effect="repression",
    ),
    "DNMT3A": WriterEraserInfo(
        name="DNMT3A",
        mechanism="DNA_methyltransferase",
        recruiter_scaffold="RG108",
        recruiter_smiles="c1ccc2c(c1)cc(c(c2)NC(=O)Cc3ccncc3)",
        recruiter_ki_nM=115.0,
        target_histone_mark="5mC_CpG",
        transcriptional_effect="repression",
    ),
    "LSD1": WriterEraserInfo(
        name="LSD1",
        mechanism="KDM1A_H3K4me1_eraser",
        recruiter_scaffold="tranylcypromine",
        recruiter_smiles="N[C@@H]1C[C@@H]1c1ccccc1",
        recruiter_ki_nM=243.0,
        target_histone_mark="H3K4me1",
        transcriptional_effect="repression",
    ),
    "PRMT5": WriterEraserInfo(
        name="PRMT5",
        mechanism="arginine_methyltransferase_H4R3me2s",
        recruiter_scaffold="GSK3326595",
        recruiter_smiles="Cc1ccc(cc1)S(=O)(=O)N2CC[C@@H](CC2)n3cncc3",
        recruiter_ki_nM=3.0,
        target_histone_mark="H4R3me2s",
        transcriptional_effect="repression",
    ),
}


@dataclass
class WriterEraserSelection:
    writer_eraser_name: str
    info: WriterEraserInfo
    selection_score: float
    expression_level: float
    encode_activity: float
    perturbation_type: str


class WriterEraserSelector:
    """Selects optimal epigenetic effector for each TF perturbation."""

    def select(
        self,
        tf_name: str,
        perturbation_type: str,
        cancer_expression: Dict[str, float],
        encode_data: Optional[Dict] = None,
    ) -> WriterEraserSelection:
        pt_lower = perturbation_type.lower()
        candidates = WRITERS if pt_lower == "activate" else ERASERS
        encode_data = encode_data or {}

        scored = []
        for name, info in candidates.items():
            score = self._score_candidate(info, tf_name, cancer_expression, encode_data)
            scored.append((score, name, info))

        scored.sort(reverse=True)
        best_score, best_name, best_info = scored[0]

        expr_level = cancer_expression.get(best_name, 0.5)
        encode_score = encode_data.get((best_name, tf_name), 0.5)

        logger.info(
            f"  Selected {best_name} for {tf_name} {perturbation_type} "
            f"(score={best_score:.3f}, expr={expr_level:.3f})"
        )

        return WriterEraserSelection(
            writer_eraser_name=best_name,
            info=best_info,
            selection_score=best_score,
            expression_level=expr_level,
            encode_activity=encode_score,
            perturbation_type=perturbation_type,
        )

    def _score_candidate(
        self,
        info: WriterEraserInfo,
        tf_name: str,
        cancer_expression: Dict[str, float],
        encode_data: Dict,
    ) -> float:
        expr = cancer_expression.get(info.name, 0.5)
        encode = encode_data.get((info.name, tf_name), 0.5)
        ki_score = np.exp(-info.recruiter_ki_nM / 500.0)
        return 0.4 * expr + 0.3 * encode + 0.3 * ki_score
