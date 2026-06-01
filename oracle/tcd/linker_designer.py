from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class LinkerInfo:
    name: str
    smiles: str
    length_A: float
    flexibility: str
    log_p: float
    h_bond_donors: int
    h_bond_acceptors: int
    rotatable_bonds: int
    linker_type: str


LINKER_LIBRARY: List[LinkerInfo] = [
    LinkerInfo(
        name="PEG2",
        smiles="OCCOCCO",
        length_A=7.0,
        flexibility="flexible",
        log_p=-1.2,
        h_bond_donors=2,
        h_bond_acceptors=3,
        rotatable_bonds=6,
        linker_type="peg",
    ),
    LinkerInfo(
        name="PEG3",
        smiles="OCCOCCOCCO",
        length_A=10.0,
        flexibility="flexible",
        log_p=-1.8,
        h_bond_donors=2,
        h_bond_acceptors=4,
        rotatable_bonds=9,
        linker_type="peg",
    ),
    LinkerInfo(
        name="alkyl4",
        smiles="CCCC",
        length_A=5.0,
        flexibility="flexible",
        log_p=1.2,
        h_bond_donors=0,
        h_bond_acceptors=0,
        rotatable_bonds=3,
        linker_type="alkyl",
    ),
    LinkerInfo(
        name="alkyl6",
        smiles="CCCCCC",
        length_A=7.5,
        flexibility="flexible",
        log_p=2.1,
        h_bond_donors=0,
        h_bond_acceptors=0,
        rotatable_bonds=5,
        linker_type="alkyl",
    ),
    LinkerInfo(
        name="rigid_piperazine",
        smiles="C1CNCCN1",
        length_A=5.5,
        flexibility="semi-rigid",
        log_p=0.1,
        h_bond_donors=2,
        h_bond_acceptors=2,
        rotatable_bonds=0,
        linker_type="rigid",
    ),
    LinkerInfo(
        name="triazole_peg",
        smiles="CCn1cc(COCCO)nn1",
        length_A=9.0,
        flexibility="semi-rigid",
        log_p=-0.3,
        h_bond_donors=1,
        h_bond_acceptors=4,
        rotatable_bonds=5,
        linker_type="triazole",
    ),
    LinkerInfo(
        name="amide_alkyl4",
        smiles="CCCC(=O)N",
        length_A=6.5,
        flexibility="semi-rigid",
        log_p=0.5,
        h_bond_donors=1,
        h_bond_acceptors=1,
        rotatable_bonds=4,
        linker_type="amide",
    ),
    LinkerInfo(
        name="cyclopropyl_peg",
        smiles="OCC1CC1COCCO",
        length_A=8.5,
        flexibility="semi-rigid",
        log_p=0.2,
        h_bond_donors=2,
        h_bond_acceptors=3,
        rotatable_bonds=5,
        linker_type="cyclopropyl",
    ),
    LinkerInfo(
        name="azetidine_peg",
        smiles="OCC1CNC1COCCO",
        length_A=8.0,
        flexibility="semi-rigid",
        log_p=-0.5,
        h_bond_donors=3,
        h_bond_acceptors=4,
        rotatable_bonds=5,
        linker_type="azetidine",
    ),
    LinkerInfo(
        name="H2N_PEG3_COOH",
        smiles="NCCOCCOCCC(=O)O",
        length_A=10.5,
        flexibility="flexible",
        log_p=-1.5,
        h_bond_donors=2,
        h_bond_acceptors=5,
        rotatable_bonds=10,
        linker_type="peg",
    ),
    LinkerInfo(
        name="H2N_alkyl5_COOH",
        smiles="NCCCCC(=O)O",
        length_A=6.5,
        flexibility="flexible",
        log_p=0.8,
        h_bond_donors=2,
        h_bond_acceptors=2,
        rotatable_bonds=5,
        linker_type="alkyl",
    ),
    LinkerInfo(
        name="H2N_PEG2_COOH",
        smiles="NCCOCCCC(=O)O",
        length_A=8.0,
        flexibility="flexible",
        log_p=-1.0,
        h_bond_donors=2,
        h_bond_acceptors=4,
        rotatable_bonds=7,
        linker_type="peg",
    ),
]


class LinkerDesigner:
    """Designs the chemical linker connecting TF warhead to epigenetic recruiter."""

    def __init__(
        self,
        max_mw_contribution: float = 300.0,
        target_log_p_range: Tuple[float, float] = (-2.0, 3.0),
    ):
        self.max_mw_contribution = max_mw_contribution
        self.target_log_p_range = target_log_p_range

    def design(
        self,
        required_distance_A: float,
        tf_exit_vector: Optional[np.ndarray] = None,
        recruiter_exit_vector: Optional[np.ndarray] = None,
        prefer_rigid: bool = False,
    ) -> LinkerInfo:
        scored = []
        for linker in LINKER_LIBRARY:
            score = self._score_linker(linker, required_distance_A, prefer_rigid)
            scored.append((score, linker))

        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            best_score, best_linker = scored[0]
            logger.info(
                f"Selected linker {best_linker.name} "
                f"(score={best_score:.3f}, length={best_linker.length_A:.1f}Å, "
                f"required={required_distance_A:.1f}Å)"
            )
            return best_linker

        return self._generate_custom_peg(required_distance_A)

    def _score_linker(
        self,
        linker: LinkerInfo,
        required_distance_A: float,
        prefer_rigid: bool,
    ) -> float:
        length_diff = abs(linker.length_A - required_distance_A)
        length_score = np.exp(-length_diff / 3.0)

        lp_lo, lp_hi = self.target_log_p_range
        lp = linker.log_p
        if lp_lo <= lp <= lp_hi:
            lp_score = 1.0
        else:
            lp_score = np.exp(-min(abs(lp - lp_lo), abs(lp - lp_hi)) / 1.0)

        flexibility_score = 0.5
        if prefer_rigid and linker.flexibility in ("semi-rigid", "rigid"):
            flexibility_score = 1.0
        elif not prefer_rigid and linker.flexibility == "flexible":
            flexibility_score = 1.0

        return 0.5 * length_score + 0.3 * lp_score + 0.2 * flexibility_score

    def _generate_custom_peg(self, required_distance_A: float) -> LinkerInfo:
        n_units = max(1, round(required_distance_A / 3.5))
        unit = "CCO"
        smiles = "O" + unit * n_units + "CO"
        return LinkerInfo(
            name=f"PEG{n_units}_custom",
            smiles=smiles,
            length_A=n_units * 3.5,
            flexibility="flexible",
            log_p=-0.5 * n_units,
            h_bond_donors=2,
            h_bond_acceptors=n_units + 1,
            rotatable_bonds=n_units * 3,
            linker_type="peg",
        )

    def calculate_required_distance(
        self,
        tf_pocket_center: np.ndarray,
        recruiter_binding_site_center: np.ndarray,
        tf_structure_source: str = "pdb",
    ) -> float:
        raw_dist = float(np.linalg.norm(tf_pocket_center - recruiter_binding_site_center))
        slack = 5.0 if tf_structure_source == "alphafold" else 3.0
        return raw_dist + slack
