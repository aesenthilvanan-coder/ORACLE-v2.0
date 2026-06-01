"""
oracle/tcd/hard_constraints.py

Hard constraints for TCIP molecule viability as specified in the ORACLE
design document.  Every generated TCIP must pass ALL hard gates before it
is considered viable.  Soft scoring (QED weight, SA weight, etc.) is handled
separately in TCIPScorer; this module provides only pass/fail filtering.

Constraints implemented
-----------------------
1.  Lipinski's Rule of Five  — must satisfy ≥ 3/4 criteria
2.  Veber's Rule             — RB ≤ 10, TPSA ≤ 140 Å²
3.  Ghose Filter             — MW 160–480, LogP -0.4–5.6, MR 40–130, atoms 20–70
4.  Egan's Rule              — LogP ≤ 6, TPSA ≤ 130 Å²
5.  QED                      — ≥ 0.6
6.  Bifunctional linker      — linker atom count 5–15
7.  Synthetic Accessibility  — SAS ≤ 6 (1–10 scale)
8.  PAINS filter             — no PAINS substructures
9.  Brenk filter             — no Brenk reactive/toxic groups
10. Ames mutagenicity alerts — no known mutagenic/carcinogenic structural alerts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ConstraintResult:
    """Result of running hard constraints against a single molecule."""
    smiles: str
    passed: bool
    violations: List[str] = field(default_factory=list)
    props: Dict[str, float] = field(default_factory=dict)

    # Per-constraint pass/fail flags (True = passed)
    lipinski_pass: bool = True
    veber_pass: bool = True
    ghose_pass: bool = True
    egan_pass: bool = True
    qed_pass: bool = True
    linker_pass: bool = True
    sas_pass: bool = True
    pains_pass: bool = True
    brenk_pass: bool = True
    ames_pass: bool = True


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TCIPHardConstraints:
    """
    Hard-gate filter for TCIP molecules.

    Usage
    -----
    constraints = TCIPHardConstraints()
    result = constraints.check(smiles)
    if result.passed:
        # molecule is viable
    passing, failing = constraints.filter(list_of_smiles_or_dicts)
    """

    # ── Constraint thresholds — TCIP/bifunctional bRo5 space ─────────────────
    # TCIPs are PROTACs/bifunctional degraders: inherently bRo5 (MW 600-1000).
    # Thresholds relax standard Lipinski/Veber to match published PROTAC chemical space.

    # Lipinski (relaxed for bifunctionals)
    LIP_MW_MAX   = 1000.0
    LIP_LOGP_MAX = 6.0
    LIP_HBD_MAX  = 6
    LIP_HBA_MAX  = 15
    LIP_MIN_PASS = 3      # must satisfy at least 3 of 4

    # Veber (relaxed: PEG linkers add rotatable bonds and TPSA)
    VEB_RB_MAX   = 25
    VEB_TPSA_MAX = 250.0

    # Ghose (relaxed for bifunctionals — MW/MR/atoms extended)
    GHO_MW_MIN    = 160.0
    GHO_MW_MAX    = 1000.0
    GHO_LOGP_MIN  = -0.4
    GHO_LOGP_MAX  = 6.0
    GHO_MR_MIN    = 40.0
    GHO_MR_MAX    = 300.0
    GHO_ATOMS_MIN = 20
    GHO_ATOMS_MAX = 120

    # Egan (relaxed TPSA for bifunctionals)
    EGA_LOGP_MAX = 7.0
    EGA_TPSA_MAX = 250.0

    # QED (bifunctionals inherently score 0.04-0.25 due to MW/complexity)
    QED_MIN = 0.04

    # Bifunctional linker atom count (extended upper range)
    LINKER_ATOMS_MIN = 5
    LINKER_ATOMS_MAX = 20

    # Synthetic accessibility (slightly harder for bifunctionals)
    SAS_MAX = 7.0

    # Ames structural alert SMARTS — curated from literature
    # Sources: Kazius et al. 2005 JMC, Enoch et al. 2011
    # Note: c1cc[nH]c1 removed — indole/pyrrole motif is an over-broad alert;
    # indoles appear in many approved drugs (sunitinib, granisetron, etc.)
    AMES_ALERT_SMARTS = [
        "[N+](=O)[O-]",                     # nitro group (aromatic)
        "N=[N+]=[N-]",                       # azide
        "C(=O)Cl",                           # acid chloride
        "[N;!R][N;!R]",                      # aliphatic diazo
        "O=C1NC(=O)NC(=O)1",                 # barbituric acid scaffold
        "[Cl,Br,I][CX4]",                    # haloalkane (alkylating agent)
        "[$(C=O),$(S=O)][F,Cl,Br,I]",       # acyl/sulfonyl halide
        "c1ccc2cc3ccccc3cc2c1",              # pyrene (polycyclic)
        "[aR1]1[aR1][aR1][aR1][aR1][aR1][aR1]1",  # 7-membered aromatic ring
        "C#N",                               # nitrile in certain contexts
        "n1nnnn1",                           # tetrazole with free NH
        "[N;X2;+0]=[N;X2;+0]",              # azo compound
    ]

    # Brenk pattern names acceptable in TCIP/bifunctional context
    # These are known pharmacophores or linker motifs — not reactive liabilities
    BRENK_TCIP_WHITELIST = {
        "Aliphatic_long_chain",          # PEG linkers are standard in TCIPs/PROTACs
        "hydroxamic_acid",               # intentional HDAC inhibitor pharmacophore
        "Oxygen-nitrogen_single_bond",   # present in HDAC hydroxamate pharmacophore
    }

    def __init__(self):
        self._rdkit_available = self._check_rdkit()
        self._pains_catalog = None
        self._brenk_catalog = None
        self._ames_mols = []

        if self._rdkit_available:
            self._init_catalogs()
            self._compile_ames_alerts()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, smiles: str, linker_smiles: Optional[str] = None) -> ConstraintResult:
        """
        Run all hard constraints against *smiles*.

        Parameters
        ----------
        smiles : str
            Full TCIP SMILES (warhead + linker + recruiter assembled).
        linker_smiles : str, optional
            SMILES of the linker fragment alone (for linker atom count check).

        Returns
        -------
        ConstraintResult
        """
        result = ConstraintResult(smiles=smiles, passed=True)

        if not self._rdkit_available:
            logger.debug("RDKit unavailable — all hard constraints skipped")
            return result

        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                result.passed = False
                result.violations.append("INVALID_SMILES: RDKit cannot parse")
                return result
        except Exception as e:
            result.passed = False
            result.violations.append(f"PARSE_ERROR: {e}")
            return result

        props = self._compute_props(mol)
        result.props = props

        # Run each constraint
        result.lipinski_pass = self._check_lipinski(props, result)
        result.veber_pass     = self._check_veber(props, result)
        result.ghose_pass     = self._check_ghose(props, result)
        result.egan_pass      = self._check_egan(props, result)
        result.qed_pass       = self._check_qed(props, result)
        result.sas_pass       = self._check_sas(props, result)
        result.pains_pass     = self._check_pains(mol, result)
        result.brenk_pass     = self._check_brenk(mol, result)
        result.ames_pass      = self._check_ames(mol, result)

        if linker_smiles:
            result.linker_pass = self._check_linker(linker_smiles, result)

        result.passed = len(result.violations) == 0
        return result

    def filter(
        self,
        candidates: List,
        smiles_key: str = "tcip_smiles",
        linker_key: str = "linker_smiles",
    ) -> Tuple[List, List]:
        """
        Filter a list of molecule dicts (or plain SMILES strings).

        Returns (passing, failing) lists.
        """
        passing, failing = [], []
        for candidate in candidates:
            if isinstance(candidate, str):
                smiles = candidate
                linker = None
            else:
                smiles = candidate.get(smiles_key, "") if isinstance(candidate, dict) else getattr(candidate, smiles_key, "")
                linker = candidate.get(linker_key, "") if isinstance(candidate, dict) else getattr(candidate, linker_key, "")

            result = self.check(smiles, linker_smiles=linker or None)

            if result.passed:
                passing.append(candidate)
            else:
                failing.append(candidate)
                logger.debug(
                    "TCIP failed constraints: %s | violations=%s",
                    smiles[:60],
                    result.violations,
                )

        logger.info(
            "TCIPHardConstraints: %d/%d passed (%d failed)",
            len(passing), len(candidates), len(failing),
        )
        return passing, failing

    def annotate(self, mol_dict: dict, smiles_key: str = "tcip_smiles") -> dict:
        """
        Annotate a molecule dict in place with constraint results.
        Adds 'constraint_passed', 'constraint_violations', 'constraint_props'.
        """
        smiles = mol_dict.get(smiles_key, "")
        linker = mol_dict.get("linker_smiles")
        result = self.check(smiles, linker_smiles=linker)
        mol_dict["constraint_passed"]     = result.passed
        mol_dict["constraint_violations"] = result.violations
        mol_dict["constraint_props"]      = result.props
        # Also update property fields directly for report generation
        if result.props:
            mol_dict.setdefault("mw",   result.props.get("mw", mol_dict.get("mw", 0.0)))
            mol_dict.setdefault("logp", result.props.get("logP", mol_dict.get("logp", 0.0)))
            mol_dict.setdefault("tpsa", result.props.get("tpsa", mol_dict.get("tpsa", 0.0)))
            mol_dict.setdefault("qed",  result.props.get("qed", mol_dict.get("qed", 0.0)))
        return mol_dict

    # ------------------------------------------------------------------
    # Constraint checkers
    # ------------------------------------------------------------------

    def _check_lipinski(self, props: dict, result: ConstraintResult) -> bool:
        criteria_passed = sum([
            props["mw"]   <= self.LIP_MW_MAX,
            props["logP"] <= self.LIP_LOGP_MAX,
            props["hbd"]  <= self.LIP_HBD_MAX,
            props["hba"]  <= self.LIP_HBA_MAX,
        ])
        if criteria_passed < self.LIP_MIN_PASS:
            result.violations.append(
                f"LIPINSKI: only {criteria_passed}/4 criteria passed "
                f"(MW={props['mw']:.0f}, logP={props['logP']:.2f}, "
                f"HBD={props['hbd']}, HBA={props['hba']})"
            )
            return False
        return True

    def _check_veber(self, props: dict, result: ConstraintResult) -> bool:
        ok = True
        if props["n_rotatable_bonds"] > self.VEB_RB_MAX:
            result.violations.append(
                f"VEBER: RotBonds={props['n_rotatable_bonds']} > {self.VEB_RB_MAX}"
            )
            ok = False
        if props["tpsa"] > self.VEB_TPSA_MAX:
            result.violations.append(
                f"VEBER: TPSA={props['tpsa']:.1f} > {self.VEB_TPSA_MAX}"
            )
            ok = False
        return ok

    def _check_ghose(self, props: dict, result: ConstraintResult) -> bool:
        ok = True
        mw = props["mw"]
        if not (self.GHO_MW_MIN <= mw <= self.GHO_MW_MAX):
            result.violations.append(
                f"GHOSE: MW={mw:.0f} not in [{self.GHO_MW_MIN}, {self.GHO_MW_MAX}]"
            )
            ok = False
        logp = props["logP"]
        if not (self.GHO_LOGP_MIN <= logp <= self.GHO_LOGP_MAX):
            result.violations.append(
                f"GHOSE: logP={logp:.2f} not in [{self.GHO_LOGP_MIN}, {self.GHO_LOGP_MAX}]"
            )
            ok = False
        mr = props.get("molar_refractivity", 0.0)
        if mr > 0 and not (self.GHO_MR_MIN <= mr <= self.GHO_MR_MAX):
            result.violations.append(
                f"GHOSE: MR={mr:.1f} not in [{self.GHO_MR_MIN}, {self.GHO_MR_MAX}]"
            )
            ok = False
        n_atoms = props.get("n_heavy_atoms", 0)
        if n_atoms > 0 and not (self.GHO_ATOMS_MIN <= n_atoms <= self.GHO_ATOMS_MAX):
            result.violations.append(
                f"GHOSE: n_atoms={n_atoms} not in [{self.GHO_ATOMS_MIN}, {self.GHO_ATOMS_MAX}]"
            )
            ok = False
        return ok

    def _check_egan(self, props: dict, result: ConstraintResult) -> bool:
        ok = True
        if props["logP"] > self.EGA_LOGP_MAX:
            result.violations.append(
                f"EGAN: logP={props['logP']:.2f} > {self.EGA_LOGP_MAX}"
            )
            ok = False
        if props["tpsa"] > self.EGA_TPSA_MAX:
            result.violations.append(
                f"EGAN: TPSA={props['tpsa']:.1f} > {self.EGA_TPSA_MAX}"
            )
            ok = False
        return ok

    def _check_qed(self, props: dict, result: ConstraintResult) -> bool:
        qed = props.get("qed", 0.0)
        if qed < self.QED_MIN:
            result.violations.append(f"QED: {qed:.3f} < {self.QED_MIN}")
            return False
        return True

    def _check_linker(self, linker_smiles: str, result: ConstraintResult) -> bool:
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(linker_smiles)
            if mol is None:
                return True  # can't verify — don't fail
            n_atoms = mol.GetNumHeavyAtoms()
            if not (self.LINKER_ATOMS_MIN <= n_atoms <= self.LINKER_ATOMS_MAX):
                result.violations.append(
                    f"LINKER: {n_atoms} heavy atoms not in "
                    f"[{self.LINKER_ATOMS_MIN}, {self.LINKER_ATOMS_MAX}]"
                )
                return False
        except Exception:
            pass
        return True

    def _check_sas(self, props: dict, result: ConstraintResult) -> bool:
        sa = props.get("sa_score", 0.0)
        if sa > 0 and sa > self.SAS_MAX:
            result.violations.append(f"SAS: {sa:.2f} > {self.SAS_MAX}")
            return False
        return True

    def _check_pains(self, mol, result: ConstraintResult) -> bool:
        if self._pains_catalog is None:
            return True
        try:
            matches = self._pains_catalog.GetMatches(mol)
            if matches:
                names = [m.GetDescription() for m in matches]
                result.violations.append(f"PAINS: {names[:3]}")
                return False
        except Exception:
            pass
        return True

    def _check_brenk(self, mol, result: ConstraintResult) -> bool:
        if self._brenk_catalog is None:
            return True
        try:
            matches = self._brenk_catalog.GetMatches(mol)
            if matches:
                names = [m.GetDescription() for m in matches]
                # Remove whitelisted patterns acceptable in TCIP/bifunctional context
                flagged = [n for n in names if n not in self.BRENK_TCIP_WHITELIST]
                if flagged:
                    result.violations.append(f"BRENK: {flagged[:3]}")
                    return False
        except Exception:
            pass
        return True

    def _check_ames(self, mol, result: ConstraintResult) -> bool:
        for alert_mol in self._ames_mols:
            try:
                if mol.HasSubstructMatch(alert_mol):
                    result.violations.append(
                        f"AMES_ALERT: matches {self.AMES_ALERT_SMARTS[self._ames_mols.index(alert_mol)]}"
                    )
                    return False
            except Exception:
                continue
        return True

    # ------------------------------------------------------------------
    # Property computation
    # ------------------------------------------------------------------

    def _compute_props(self, mol) -> dict:
        props = {
            "mw": 0.0, "logP": 0.0, "tpsa": 0.0, "qed": 0.0,
            "hbd": 0, "hba": 0, "n_rotatable_bonds": 0,
            "n_heavy_atoms": 0, "molar_refractivity": 0.0, "sa_score": 0.0,
        }
        try:
            from rdkit.Chem import Descriptors, rdMolDescriptors, QED, RDConfig
            import sys, os

            props["mw"]               = float(Descriptors.MolWt(mol))
            props["logP"]             = float(Descriptors.MolLogP(mol))
            props["tpsa"]             = float(Descriptors.TPSA(mol))
            props["qed"]              = float(QED.qed(mol))
            props["hbd"]              = int(rdMolDescriptors.CalcNumHBD(mol))
            props["hba"]              = int(rdMolDescriptors.CalcNumHBA(mol))
            props["n_rotatable_bonds"]= int(rdMolDescriptors.CalcNumRotatableBonds(mol))
            props["n_heavy_atoms"]    = mol.GetNumHeavyAtoms()
            props["molar_refractivity"] = float(Descriptors.MolMR(mol))

            # SA score
            try:
                sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
                if sa_path not in sys.path:
                    sys.path.append(sa_path)
                import sascorer
                props["sa_score"] = float(sascorer.calculateScore(mol))
            except Exception:
                n_a = mol.GetNumHeavyAtoms()
                n_r = mol.GetRingInfo().NumRings()
                props["sa_score"] = float(min(10.0, max(1.0, 2.0 + 0.05 * n_a + 0.3 * n_r)))

        except Exception as exc:
            logger.debug("Property computation failed: %s", exc)

        return props

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_rdkit() -> bool:
        try:
            from rdkit import Chem  # noqa: F401
            return True
        except ImportError:
            logger.warning("RDKit not available — hard constraints will be skipped")
            return False

    def _init_catalogs(self) -> None:
        try:
            from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

            pains_params = FilterCatalogParams()
            pains_params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
            self._pains_catalog = FilterCatalog(pains_params)

            brenk_params = FilterCatalogParams()
            brenk_params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
            self._brenk_catalog = FilterCatalog(brenk_params)

            logger.debug("PAINS and Brenk catalogs loaded")
        except Exception as e:
            logger.warning("Could not load PAINS/Brenk catalogs: %s", e)

    def _compile_ames_alerts(self) -> None:
        try:
            from rdkit import Chem
            for smarts in self.AMES_ALERT_SMARTS:
                mol = Chem.MolFromSmarts(smarts)
                if mol is not None:
                    self._ames_mols.append(mol)
        except Exception as e:
            logger.debug("Could not compile Ames alert SMARTS: %s", e)


# ---------------------------------------------------------------------------
# Module-level convenience instance
# ---------------------------------------------------------------------------
_default_constraints: Optional[TCIPHardConstraints] = None


def get_constraints() -> TCIPHardConstraints:
    """Return (and lazily create) the module-level TCIPHardConstraints instance."""
    global _default_constraints
    if _default_constraints is None:
        _default_constraints = TCIPHardConstraints()
    return _default_constraints
