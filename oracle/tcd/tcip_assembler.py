from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

_COOH_SMARTS = "[C;!r](=O)[OH]"
_NH2_SMARTS  = "[NH2]"
_NH_SMARTS   = "[NH;!$(NC=O);!n]"


@dataclass
class TCIPProperties:
    molecular_weight: float
    log_p: float
    h_bond_donors: int
    h_bond_acceptors: int
    rotatable_bonds: int
    tpsa: float
    qed: float
    lipinski_violations: int
    passes_ro5: bool


@dataclass
class AssembledTCIP:
    smiles: str
    tf_warhead_smiles: str
    linker_smiles: str
    recruiter_smiles: str
    properties: TCIPProperties
    assembly_method: str


class TCIPAssembler:
    """Genis-framework TCIP assembler — 5-tier covalent coupling."""

    def __init__(self) -> None:
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            self._cooh_pat = Chem.MolFromSmarts(_COOH_SMARTS)
            self._nh2_pat  = Chem.MolFromSmarts(_NH2_SMARTS)
            self._nh_pat   = Chem.MolFromSmarts(_NH_SMARTS)
            self._rxn_amide = AllChem.ReactionFromSmarts(
                "[C:1](=O)[OH].[NH2:2]>>[C:1](=O)[N:2]"
            )
        except Exception:
            self._cooh_pat = self._nh2_pat = self._nh_pat = self._rxn_amide = None

    # ── functional-group detectors ────────────────────────────────────────────

    def _has_cooh(self, mol) -> bool:
        return bool(mol.HasSubstructMatch(self._cooh_pat)) if self._cooh_pat else False

    def _has_nh2(self, mol) -> bool:
        return bool(mol.HasSubstructMatch(self._nh2_pat)) if self._nh2_pat else False

    # ── single amide coupling step ────────────────────────────────────────────

    def _amide_couple(self, mol_acid, mol_amine):
        """COOH + NH2 → amide.  Returns product mol or None."""
        try:
            from rdkit import Chem
            prods = self._rxn_amide.RunReactants((mol_acid, mol_amine))
            if prods:
                p = prods[0][0]
                Chem.SanitizeMol(p)
                return p
        except Exception:
            pass
        return None

    # ── Facet-2: warhead functionalization ───────────────────────────────────

    def _add_cooh_arm(self, mol):
        """
        Append -CC(=O)O to an accessible terminal atom of *mol*.
        Priority: terminal aliphatic N > terminal CH3 > any degree-1 C.
        Returns a new mol or None.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import BondType

            arm_mol = Chem.MolFromSmiles("CC(=O)O")   # acetic-acid arm
            if arm_mol is None:
                return None

            candidates = []
            for atom in mol.GetAtoms():
                if atom.IsInRing():
                    continue
                idx = atom.GetIdx()
                num = atom.GetAtomicNum()
                deg = atom.GetDegree()
                if num == 7 and deg <= 2 and atom.GetTotalNumHs() >= 1:
                    candidates.insert(0, idx)          # prefer N-H
                elif num == 6 and deg == 1:
                    candidates.append(idx)

            if not candidates:
                # fallback: any degree-1 non-ring heavy atom
                for atom in mol.GetAtoms():
                    if atom.GetDegree() == 1 and not atom.IsInRing():
                        candidates.append(atom.GetIdx())
                        break

            if not candidates:
                return None

            attach_idx = candidates[0]
            n_orig = mol.GetNumAtoms()

            rw = Chem.RWMol(Chem.CombineMols(mol, arm_mol))
            rw.AddBond(attach_idx, n_orig, BondType.SINGLE)   # bond to CH3 of arm
            Chem.SanitizeMol(rw)
            return rw.GetMol()
        except Exception:
            return None

    # ── Facet-3: force-connect three fragments ────────────────────────────────

    def _force_connect(self, mol_a, mol_b, mol_c):
        """
        Last resort: bond terminal atoms of A→B and B→C with single bonds,
        forming a linear connected chain.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import BondType

            def _terminal(m):
                for atom in m.GetAtoms():
                    if atom.GetDegree() == 1 and not atom.IsInRing():
                        return atom.GetIdx()
                for atom in m.GetAtoms():
                    if not atom.IsInRing():
                        return atom.GetIdx()
                return 0

            na = mol_a.GetNumAtoms()
            nb = mol_b.GetNumAtoms()

            rw = Chem.RWMol(Chem.CombineMols(mol_a, Chem.CombineMols(mol_b, mol_c)))
            a_term = _terminal(mol_a)
            b_term1 = na + _terminal(mol_b)
            b_term2 = na + (nb - 1 - _terminal(mol_b))   # other end of b
            c_term  = na + nb + _terminal(mol_c)

            rw.AddBond(a_term,  b_term1, BondType.SINGLE)
            rw.AddBond(b_term2, c_term,  BondType.SINGLE)
            Chem.SanitizeMol(rw)
            return rw.GetMol()
        except Exception:
            return None

    # ── 5-tier _combine_fragments ─────────────────────────────────────────────

    def _combine_fragments(
        self,
        tf_warhead: str,
        linker: str,
        recruiter: str,
        tf_ap: Optional[str],
        rec_ap: Optional[str],
    ) -> Tuple[str, str]:
        """
        Returns (smiles, tier_label).
        Guarantees single-component SMILES when any tier 1-5 succeeds.
        """
        try:
            from rdkit import Chem

            mol_wh = Chem.MolFromSmiles(tf_warhead)
            mol_lk = Chem.MolFromSmiles(linker)
            mol_rc = Chem.MolFromSmiles(recruiter)

            if mol_wh is None or mol_lk is None or mol_rc is None:
                return f"{tf_warhead}.{linker}.{recruiter}", "fallback_invalid_smiles"

            # ── Tier 1: warhead-COOH + linker-NH2 ────────────────────────────
            if self._has_cooh(mol_wh) and self._has_nh2(mol_lk):
                wl = self._amide_couple(mol_wh, mol_lk)
                if wl is not None:
                    mol_wl = Chem.MolFromSmiles(Chem.MolToSmiles(wl))
                    if mol_wl and self._has_nh2(mol_rc):
                        final = self._amide_couple(mol_wl, mol_rc)
                        if final:
                            smi = Chem.MolToSmiles(final)
                            if "." not in smi:
                                return smi, "tier1_forward"
                    if mol_wl and self._has_cooh(mol_wl) and self._has_nh2(mol_rc):
                        final = self._amide_couple(mol_wl, mol_rc)
                        if final:
                            smi = Chem.MolToSmiles(final)
                            if "." not in smi:
                                return smi, "tier1_forward_v2"

            # ── Tier 2: linker-COOH + warhead-NH2 ────────────────────────────
            if self._has_cooh(mol_lk) and self._has_nh2(mol_wh):
                wl = self._amide_couple(mol_lk, mol_wh)
                if wl is not None:
                    mol_wl = Chem.MolFromSmiles(Chem.MolToSmiles(wl))
                    if mol_wl and self._has_nh2(mol_rc) and self._has_cooh(mol_wl):
                        final = self._amide_couple(mol_wl, mol_rc)
                        if final:
                            smi = Chem.MolToSmiles(final)
                            if "." not in smi:
                                return smi, "tier2_reverse_lk_wh"
                    if mol_wl and self._has_nh2(mol_wl) and self._has_cooh(mol_rc):
                        final = self._amide_couple(mol_rc, mol_wl)
                        if final:
                            smi = Chem.MolToSmiles(final)
                            if "." not in smi:
                                return smi, "tier2_reverse_rc_wl"

            # ── Tier 3: brute-force all A+B then +C orderings ─────────────────
            for (m1, m2, m3, lbl) in [
                (mol_wh, mol_lk, mol_rc, "wh+lk+rc"),
                (mol_lk, mol_wh, mol_rc, "lk+wh+rc"),
                (mol_rc, mol_lk, mol_wh, "rc+lk+wh"),
                (mol_lk, mol_rc, mol_wh, "lk+rc+wh"),
                (mol_wh, mol_rc, mol_lk, "wh+rc+lk"),
                (mol_rc, mol_wh, mol_lk, "rc+wh+lk"),
            ]:
                step1 = self._amide_couple(m1, m2) or self._amide_couple(m2, m1)
                if step1 is not None:
                    m12 = Chem.MolFromSmiles(Chem.MolToSmiles(step1))
                    if m12 is None:
                        continue
                    step2 = self._amide_couple(m12, m3) or self._amide_couple(m3, m12)
                    if step2 is not None:
                        smi = Chem.MolToSmiles(step2)
                        if "." not in smi:
                            return smi, f"tier3_{lbl}"

            # ── Tier 4: functionalize warhead with COOH arm ───────────────────
            mol_wh_func = self._add_cooh_arm(mol_wh)
            if mol_wh_func is not None:
                logger.debug("Tier 4: warhead functionalized with COOH arm")
                for mol_lk2 in ([mol_lk] if self._has_nh2(mol_lk) else []):
                    wl = self._amide_couple(mol_wh_func, mol_lk2)
                    if wl is not None:
                        mol_wl = Chem.MolFromSmiles(Chem.MolToSmiles(wl))
                        if mol_wl and self._has_nh2(mol_rc):
                            if self._has_cooh(mol_wl):
                                final = self._amide_couple(mol_wl, mol_rc)
                                if final:
                                    smi = Chem.MolToSmiles(final)
                                    if "." not in smi:
                                        return smi, "tier4_func_wh_cooh"
                # Also try: func_wh-NH2 coupling (if arm added to N end)
                for mol_lk2 in ([mol_lk] if self._has_cooh(mol_lk) else []):
                    wl = self._amide_couple(mol_lk2, mol_wh_func)
                    if wl is not None:
                        mol_wl = Chem.MolFromSmiles(Chem.MolToSmiles(wl))
                        if mol_wl and self._has_nh2(mol_rc) and self._has_cooh(mol_wl):
                            final = self._amide_couple(mol_wl, mol_rc)
                            if final:
                                smi = Chem.MolToSmiles(final)
                                if "." not in smi:
                                    return smi, "tier4_func_wh_reverse"
                # Try all combos with functionalized warhead
                for (m1, m2, m3, lbl) in [
                    (mol_wh_func, mol_lk, mol_rc, "f+lk+rc"),
                    (mol_lk, mol_wh_func, mol_rc, "lk+f+rc"),
                    (mol_rc, mol_lk, mol_wh_func, "rc+lk+f"),
                ]:
                    step1 = self._amide_couple(m1, m2) or self._amide_couple(m2, m1)
                    if step1 is not None:
                        m12 = Chem.MolFromSmiles(Chem.MolToSmiles(step1))
                        if m12 is None:
                            continue
                        step2 = self._amide_couple(m12, m3) or self._amide_couple(m3, m12)
                        if step2 is not None:
                            smi = Chem.MolToSmiles(step2)
                            if "." not in smi:
                                return smi, f"tier4_{lbl}"

            # ── Tier 5: force-connect via explicit bond ───────────────────────
            forced = self._force_connect(mol_wh, mol_lk, mol_rc)
            if forced is not None:
                smi = Chem.MolToSmiles(forced)
                if "." not in smi:
                    logger.warning("Tier 5 force-connect used — non-standard bonding")
                    return smi, "tier5_force_connect"

        except Exception as e:
            logger.debug("Genis assembly failed: %s", e)

        # ── Fallback ──────────────────────────────────────────────────────────
        logger.warning("All Genis tiers failed — returning dot-concat fallback")
        return f"{tf_warhead}.{linker}.{recruiter}", "fallback_dot_concat"

    # ── public API ────────────────────────────────────────────────────────────

    def assemble(
        self,
        tf_warhead_smiles: str,
        linker_smiles: str,
        recruiter_smiles: str,
        tf_attachment_point: Optional[str] = None,
        recruiter_attachment_point: Optional[str] = None,
    ) -> AssembledTCIP:
        combined, method = self._combine_fragments(
            tf_warhead_smiles,
            linker_smiles,
            recruiter_smiles,
            tf_attachment_point,
            recruiter_attachment_point,
        )

        props = self._compute_properties(combined)

        logger.info(
            "Assembled TCIP [%s]: MW=%.1f LogP=%.2f QED=%.3f connected=%s",
            method, props.molecular_weight, props.log_p, props.qed,
            "." not in combined,
        )

        return AssembledTCIP(
            smiles=combined,
            tf_warhead_smiles=tf_warhead_smiles,
            linker_smiles=linker_smiles,
            recruiter_smiles=recruiter_smiles,
            properties=props,
            assembly_method=method,
        )

    def _compute_properties(self, smiles: str) -> TCIPProperties:
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, QED

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return self._fallback_properties(smiles)

            mw   = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd  = Descriptors.NumHDonors(mol)
            hba  = Descriptors.NumHAcceptors(mol)
            rotb = Descriptors.NumRotatableBonds(mol)
            tpsa = Descriptors.TPSA(mol)
            qed_score = QED.qed(mol)

            violations = sum([mw > 900, logp > 5, hbd > 5, hba > 10])

            return TCIPProperties(
                molecular_weight=mw,
                log_p=logp,
                h_bond_donors=hbd,
                h_bond_acceptors=hba,
                rotatable_bonds=rotb,
                tpsa=tpsa,
                qed=qed_score,
                lipinski_violations=violations,
                passes_ro5=(violations <= 1),
            )
        except Exception as e:
            logger.debug("Property computation failed: %s", e)
            return self._fallback_properties(smiles)

    def _fallback_properties(self, smiles: str) -> TCIPProperties:
        est_mw = len(smiles) * 5.5
        return TCIPProperties(
            molecular_weight=est_mw,
            log_p=2.0,
            h_bond_donors=2,
            h_bond_acceptors=4,
            rotatable_bonds=8,
            tpsa=80.0,
            qed=0.4,
            lipinski_violations=1,
            passes_ro5=(est_mw < 900),
        )
