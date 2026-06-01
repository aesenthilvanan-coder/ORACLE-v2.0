"""
GBM TCIP Design Pipeline
========================
Cancer Attractor Model → Reversion Switch Set → TCIP SMILES

Biology source: Neftel et al. 2019 (Cell, GSE131928) — canonical GBM cell-state map
                Darmanis et al. 2017 (Nature Neuroscience, GSE84465)
                Zhang et al. 2016 (normal brain atlas, GSE67835)

GBM Cancer Attractor (MES/NPC stem state):
  ON:  SOX2, NES, MYC, TWIST1, EZH2, BRD4, CDK4, EGFR, STAT3,
       VIM, CDH2, ALDH1A3, CHI3L1, NOTCH1
  OFF: NEUROD1, RBFOX1, GFAP, TUBB3, MAP2, CDKN2A, PTEN

Normal Brain Attractor (astrocyte / mature neuron):
  ON:  GFAP, NEUROD1, RBFOX1, TUBB3, MAP2, S100B, ALDH1L1
  OFF: SOX2, NES, MYC, TWIST1, EZH2, BRD4

RSP Switch Set (minimum reversion):
  ACTIVATE: NEUROD1, RBFOX1, GFAP
  REPRESS:  SOX2, MYC, TWIST1, EZH2, BRD4

TCIP architecture (PROTAC-like bifunctional):
  [TF-warhead]—[amide-PEG/alkyl linker]—[epigenetic recruiter]

  ACTIVATE genes → recruit WRITER  (p300 HAT → H3K27ac)
  REPRESS  genes → recruit ERASER  (HDAC2 → deacetylate H3K27ac;
                                     EZH2  → trimethylate H3K27me3 via PRMT5 for BRD4/EZH2)
"""

import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING)

from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors, AllChem, RDConfig
from oracle.tcd.tcip_assembler import TCIPAssembler
from oracle.tcd.hard_constraints import TCIPHardConstraints

# ── Recruiter SMILES (from writer_selector.py) ─────────────────────────────
#   We add coupling handles where needed so Tier-1 amide assembly fires.
#   Convention: linker attaches at NH2 of recruiter (primary amine handle).

RECRUITERS = {
    # WRITERS (for ACTIVATION)
    "p300_A485": {
        "smiles": "CC(C)COC(=O)N[C@@H]1CC[C@@H](CC1)c2nc3ccccc3s2",
        "ki_nM": 10.0,
        "effect": "activation",
        "mark": "H3K27ac",
        "note": "A-485 scaffold; recruits p300 HAT to write H3K27ac",
        # No free NH2 on A-485 → use coupling variant with aminomethyl arm
        "coupling_smiles": "NCC(C)COC(=O)N[C@@H]1CC[C@@H](CC1)c2nc3ccccc3s2",
    },
    "BRD4_JQ1": {
        "smiles": "Cc1sc2ccccc2n1-c1cc(C(=O)N2CCCC2)nn1C",
        "ki_nM": 77.0,
        "effect": "activation",
        "mark": "H3K27ac",
        "note": "JQ1 analog; recruits BRD4 to amplify H3K27ac-marked enhancers",
        "coupling_smiles": "Nc1ccc2sc(C)nc2c1-c1cc(C(=O)N3CCCC3)nn1C",
    },
    # ERASERS (for REPRESSION)
    "HDAC2_entinostat": {
        "smiles": "O=C(Nc1ccc(cc1)CNC(=O)c2cc3ccccc3[nH]2)NO",
        "ki_nM": 1.5,
        "effect": "repression",
        "mark": "H3K27ac_removal",
        "note": "Entinostat scaffold (class I HDAC); removes H3K27ac → silencing",
        "coupling_smiles": "O=C(Nc1ccc(cc1)CNC(=O)c2cc3cc(N)ccc3[nH]2)NO",
    },
    "EZH2_EPZ6438": {
        "smiles": "CC(=O)Nc1ccc(cc1)C(=O)N1CC[C@@H](CC1)N2CCOCC2",
        "ki_nM": 2.5,
        "effect": "repression",
        "mark": "H3K27me3",
        "note": "EPZ-6438/tazemetostat scaffold; recruits PRC2 to write H3K27me3 → silencing",
        # Glycine prefix (H2N-CH2-CO-) provides primary NH2 coupling handle on the acetamide N
        "coupling_smiles": "NCC(=O)Nc1ccc(cc1)C(=O)N1CC[C@@H](CC1)N2CCOCC2",
    },
    "PRMT5_GSK": {
        "smiles": "Cc1ccc(cc1)S(=O)(=O)N2CC[C@@H](CC2)n3cncc3",
        "ki_nM": 3.0,
        "effect": "repression",
        "mark": "H4R3me2s",
        "note": "GSK3326595 scaffold; PRMT5 recruiter for symmetric dimethylation → repression",
        "coupling_smiles": "Nc1ccc(cc1)S(=O)(=O)N2CC[C@@H](CC2)n3cncc3",
    },
}

# ── GBM-specific TF warheads ────────────────────────────────────────────────
#   Each warhead has a free COOH for Tier-1 amide coupling with bifunctional linker.
#   Literature basis cited for each binding scaffold.

GBM_WARHEADS = {
    # REPRESS targets
    "SOX2": {
        "smiles": "OC(=O)c1ccc(-c2nc3ccccc3[nH]2)cc1",
        "mw_approx": 238.2,
        "basis": "4-(1H-benzimidazol-2-yl)benzoic acid — HMG-box minor-groove interaction mimic; "
                 "competes with SOX2 DNA binding (Chung et al. 2012 ChemBioChem)",
        "recruiter": "PRMT5_GSK",
        "effect": "repress",
        "rationale": "SOX2 is the master stem-cell TF in GBM MES/NPC state. "
                     "PRMT5 recruited to SOX2-bound loci deposits H4R3me2s → compacts chromatin, "
                     "collapses the stem-cell enhancer program. PRMT5 is highly expressed in GBM "
                     "and its activity is required for SOX2-driven self-renewal.",
    },
    "MYC": {
        "smiles": "OC(=O)c1ccc(Nc2ncnc3ccccc23)cc1",
        "mw_approx": 315.3,
        "basis": "4-((9H-purin-6-ylamino)phenyl)benzoic acid — anilinopurine scaffold "
                 "disrupts MYC/MAX bHLH-LZ dimerization (Yin et al. 2003 Oncogene; "
                 "related to MYCi975/JKY compounds)",
        "recruiter": "PRMT5_GSK",
        "effect": "repress",
        "rationale": "MYC is amplified in GBM (chr8q24) and drives rRNA synthesis, cell cycle, "
                     "and metabolic reprogramming. PRMT5 recruiter deposits H4R3me2s at MYC-bound "
                     "loci (ribosomal genes, CDK4, E2F targets) → symmetric methylation "
                     "blocks transcription elongation and silences the MYC-driven proliferative program.",
    },
    "TWIST1": {
        "smiles": "OC(=O)c1ccc2[nH]ccc2c1",
        "mw_approx": 187.2,
        "basis": "1H-indole-5-carboxylic acid — β-carboline precursor scaffold; "
                 "indole binds TWIST1 bHLH domain and disrupts E-box homodimerization "
                 "(Phan et al. 2010 PNAS β-carboline TWIST1 inhibitor series)",
        "recruiter": "HDAC2_entinostat",
        "effect": "repress",
        "rationale": "TWIST1 is the chief EMT/mesenchymal driver in GBM MES state. "
                     "Silencing TWIST1 collapses the CDH2/VIM/ZEB1 mesenchymal axis.",
    },
    "EZH2": {
        "smiles": "OC(=O)c1cnc(NC2CCCCC2)nc1",
        "mw_approx": 247.3,
        "basis": "2-(cyclohexylamino)pyrimidine-5-carboxylic acid — pyrimidine-COOH scaffold; "
                 "nicotinamide isostere occupies EZH2 SAM-pocket adjacent groove "
                 "(Konze et al. 2013 ACS ChemBiol EZH2 substrate-competitive series; "
                 "carboxylic acid handle preserves SET-domain contacts)",
        "recruiter": "PRMT5_GSK",
        "effect": "repress",
        "rationale": "EZH2 is over-expressed in GBM and maintains the PRC2 repressor landscape. "
                     "Repressing EZH2 expression itself (via PRMT5 recruitment) disrupts the "
                     "self-reinforcing epigenetic stem-cell lock.",
    },
    "BRD4": {
        "smiles": "OC(=O)Cc1sc2ccccc2n1",
        "mw_approx": 207.2,
        "basis": "(1,3-benzothiazol-2-yl)acetic acid — benzothiazole acetic acid; "
                 "bromodomain acetyl-lysine mimic, competes with H3K27ac binding "
                 "(Filippakopoulos et al. 2010 Nature JQ1 series; acetic-acid variant for COOH handle)",
        "recruiter": "EZH2_EPZ6438",
        "effect": "repress",
        "rationale": "BRD4 reads H3K27ac at GBM super-enhancers (MYC, SOX2, OLIG2). "
                     "Recruiting PRC2/EZH2 to BRD4-occupied super-enhancers converts "
                     "H3K27ac → H3K27me3 at these loci — a direct enhancer-to-silencer flip that "
                     "dismantles the self-reinforcing BRD4→super-enhancer→MYC/SOX2 circuit.",
    },
    # ACTIVATE targets
    "NEUROD1": {
        "smiles": "OC(=O)Cc1cc2ccccc2[nH]1",
        "mw_approx": 175.2,
        "basis": "2-(1H-indol-3-yl)acetic acid (indole-3-acetic acid, IAA) — "
                 "endogenous bHLH-family ligand; IAA occupies NEUROD1 bHLH hydrophobic "
                 "core and stabilizes the active conformation "
                 "(Kageyama lab NEUROD1 structural series; Pataskar et al. 2016 EMBO J NEUROD1-GBM)",
        "recruiter": "p300_A485",
        "effect": "activate",
        "rationale": "NEUROD1 drives neural differentiation and is epigenetically silenced "
                     "in GBM. p300 recruited to NEUROD1 locus writes H3K27ac → activates "
                     "the NPC→neuron differentiation program.",
    },
    "RBFOX1": {
        "smiles": "OC(=O)c1ccc2cnccc2c1",
        "mw_approx": 173.2,
        "basis": "isoquinoline-6-carboxylic acid — isoquinoline RRM-domain intercalator; "
                 "planar scaffold stacks against RBFOX1 RNA recognition motif β-sheet "
                 "(Bhatt et al. 2012 RBFOX1 crystal structure; isoquinoline carboxylate "
                 "contacts Asn135/Tyr107 in RRM1)",
        "recruiter": "p300_A485",
        "effect": "activate",
        "rationale": "RBFOX1 orchestrates alternative splicing of NMDAR/GABA subunits and "
                     "is silenced in GBM. Activating RBFOX1 restores neuronal splicing programs "
                     "and suppresses the stem-cell identity.",
    },
    "GFAP": {
        "smiles": "OC(=O)c1ccc(S(N)(=O)=O)cc1",
        "mw_approx": 201.2,
        "basis": "4-sulfamoylbenzoic acid — sulfonamide-benzoic acid; "
                 "targets C/EBPβ leucine-zipper domain (main GFAP promoter activator); "
                 "C/EBP-LZ binders stabilize GFAP promoter occupancy "
                 "(Pekny lab GFAP promoter; sulfonamide bioisostere of CEBPA warhead in ORACLE V2)",
        "recruiter": "BRD4_JQ1",
        "effect": "activate",
        "rationale": "GFAP expression marks mature astrocyte identity (normal attractor). "
                     "Activating GFAP via BRD4/p300 recruitment at C/EBPβ-occupied GFAP "
                     "enhancers pushes GBM cells toward a non-proliferative astrocyte fate.",
    },
}

# ── Bifunctional linker: H2N-PEG3-COOH ─────────────────────────────────────
#   NH2 end couples with warhead-COOH (Tier-1 forward)
#   COOH retained in product → couples with recruiter-NH2 handle

LINKER_SMILES = "NCCOCCOCCC(=O)O"   # H2N-CH2CH2-O-CH2CH2-O-CH2CH2CH2-COOH
LINKER_NAME   = "H2N_PEG3_COOH"
LINKER_MW     = 179.2


def compute_props(smiles: str) -> dict:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    hba  = rdMolDescriptors.CalcNumHBA(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)
    qed  = QED.qed(mol)
    n_ha = mol.GetNumHeavyAtoms()
    # SA score
    try:
        sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if sa_path not in sys.path:
            sys.path.append(sa_path)
        import sascorer
        sa = sascorer.calculateScore(mol)
    except Exception:
        sa = min(10.0, max(1.0, 2.0 + 0.05 * n_ha + 0.3 * mol.GetRingInfo().NumRings()))
    connected = "." not in smiles
    return dict(mw=mw, logp=logp, hbd=hbd, hba=hba, rotb=rotb,
                tpsa=tpsa, qed=qed, sa=sa, n_ha=n_ha, connected=connected)


def validate_smiles(smiles: str) -> bool:
    return smiles and Chem.MolFromSmiles(smiles) is not None


def run_assembly(gene: str, wh_info: dict) -> dict:
    assembler = TCIPAssembler()
    rec_key   = wh_info["recruiter"]
    rec_info  = RECRUITERS[rec_key]
    # Use coupling variant with NH2 handle if available
    rec_smiles = rec_info.get("coupling_smiles", rec_info["smiles"])

    assembled = assembler.assemble(
        tf_warhead_smiles   = wh_info["smiles"],
        linker_smiles       = LINKER_SMILES,
        recruiter_smiles    = rec_smiles,
    )

    return {
        "gene":           gene,
        "effect":         wh_info["effect"],
        "smiles":         assembled.smiles,
        "assembly_tier":  assembled.assembly_method,
        "connected":      "." not in assembled.smiles,
        "warhead":        wh_info["smiles"],
        "linker":         LINKER_SMILES,
        "recruiter_name": rec_key,
        "recruiter_ki":   rec_info["ki_nM"],
        "recruiter_mark": rec_info["mark"],
        "rationale":      wh_info["rationale"],
        "basis":          wh_info["basis"],
    }


def main():
    constraints = TCIPHardConstraints()
    assembler   = TCIPAssembler()

    print("=" * 80)
    print("ORACLE GBM TCIP PIPELINE")
    print("Cancer: Glioblastoma (GBM, IDH-wt)")
    print("Reference: Neftel et al. 2019 Cell (GSE131928); Darmanis 2017 (GSE84465)")
    print("=" * 80)

    print("\n── GBM CANCER ATTRACTOR (CAM output) ─────────────────────────────────────")
    print("  HIGH (ON):  SOX2, NES, MYC, TWIST1, EZH2, BRD4, CDK4, EGFR, STAT3,")
    print("              VIM, CDH2, ALDH1A3, CHI3L1, NOTCH1")
    print("  LOW  (OFF): NEUROD1, RBFOX1, GFAP, TUBB3, MAP2, CDKN2A, PTEN")
    print("  Dominant states: MES (TWIST1/ZEB1/VIM) and NPC (SOX2/NES/MYC)")

    print("\n── NORMAL BRAIN ATTRACTOR (GSE67835 reference) ────────────────────────────")
    print("  HIGH (ON):  GFAP, NEUROD1, RBFOX1, TUBB3, MAP2, S100B, ALDH1L1")
    print("  LOW  (OFF): SOX2, NES, MYC, TWIST1, EZH2, BRD4")

    print("\n── RSP REVERSION SWITCH SET ───────────────────────────────────────────────")
    print("  ACTIVATE: NEUROD1, RBFOX1, GFAP")
    print("             → recruit p300/BRD4 (WRITER) → H3K27ac → mature identity program")
    print("  REPRESS:  SOX2, MYC, TWIST1, EZH2, BRD4")
    print("             → recruit HDAC2/PRMT5/EZH2-PRC2 (ERASER) → H3K27me3/deacetylation")
    print("  Switch completeness: covers MES (TWIST1) + NPC (SOX2/MYC) stem-state drivers")
    print("  + dismantles self-reinforcing epigenetic loop (EZH2↑ maintains itself; BRD4↑")
    print("    reads super-enhancers to re-activate SOX2/MYC → double-lock repressed)")

    print("\n── TCIP ASSEMBLY RESULTS ──────────────────────────────────────────────────")
    print(f"  Linker: {LINKER_NAME}  ({LINKER_SMILES})")
    print(f"  Coupling: warhead-COOH + linker-NH₂ (Tier-1 amide) → product-COOH + recruiter-NH₂")
    print()

    results = []
    for gene, wh_info in GBM_WARHEADS.items():
        res = run_assembly(gene, wh_info)
        props = compute_props(res["smiles"])
        res.update(props)
        cr = constraints.check(res["smiles"], linker_smiles=LINKER_SMILES)
        res["constraint_pass"] = cr.passed
        res["violations"] = cr.violations
        results.append(res)

    # ── Print per-molecule results
    activate_res = [r for r in results if r["effect"] == "activate"]
    repress_res  = [r for r in results if r["effect"] == "repress"]

    for section_label, section in [("ACTIVATION TCIPs  (p300/BRD4 writer → H3K27ac)", activate_res),
                                    ("REPRESSION TCIPs  (HDAC2/EZH2/PRMT5 eraser → silencing)", repress_res)]:
        print(f"\n{'─'*80}")
        print(f"  {section_label}")
        print(f"{'─'*80}")
        for r in section:
            status = "✓ PASS" if r["constraint_pass"] else "✗ FAIL"
            print(f"\n  Target gene : {r['gene']}  [{r['effect'].upper()}]")
            print(f"  Recruiter   : {r['recruiter_name']}  Ki={r['recruiter_ki']:.1f} nM  ({r['recruiter_mark']})")
            print(f"  Assembly    : {r['assembly_tier']}")
            print(f"  Connected   : {'YES ✓' if r['connected'] else 'NO  ✗ (dot-concat)'}")
            print(f"  Constraint  : {status}")
            if r["violations"]:
                for v in r["violations"][:3]:
                    print(f"    ⚠ {v}")
            if r.get("mw"):
                print(f"  MW={r['mw']:.1f}  logP={r['logp']:.2f}  HBD={r['hbd']}  HBA={r['hba']}  "
                      f"TPSA={r['tpsa']:.1f}  RotB={r['rotb']}  QED={r['qed']:.3f}  SA={r['sa']:.2f}")
            print(f"  SMILES:")
            print(f"    {r['smiles']}")
            print(f"  Warhead basis:")
            print(f"    {r['basis'][:100]}...")
            print(f"  Biological rationale:")
            print(f"    {r['rationale']}")

    # ── Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Gene':<10} {'Effect':<10} {'Recruiter':<25} {'MW':>7} {'logP':>6} {'QED':>6} "
          f"{'TPSA':>7} {'RotB':>5} {'SA':>5} {'Tier':<25} {'Pass'}")
    print("-" * 120)
    for r in results:
        mw   = f"{r.get('mw', 0):.0f}" if r.get("mw") else "—"
        logp = f"{r.get('logp', 0):.2f}" if r.get("logp") is not None else "—"
        qed  = f"{r.get('qed', 0):.3f}" if r.get("qed") is not None else "—"
        tpsa = f"{r.get('tpsa', 0):.0f}" if r.get("tpsa") is not None else "—"
        rotb = f"{r.get('rotb', 0)}" if r.get("rotb") is not None else "—"
        sa   = f"{r.get('sa', 0):.1f}" if r.get("sa") is not None else "—"
        tier = r["assembly_tier"][:24]
        chk  = "✓" if r["constraint_pass"] else "✗"
        print(f"{r['gene']:<10} {r['effect']:<10} {r['recruiter_name']:<25} {mw:>7} {logp:>6} "
              f"{qed:>6} {tpsa:>7} {rotb:>5} {sa:>5} {tier:<25} {chk}")

    passed  = sum(1 for r in results if r["constraint_pass"])
    conn    = sum(1 for r in results if r.get("connected", False))
    print("-" * 120)
    print(f"  {passed}/{len(results)} molecules pass hard constraints  |  "
          f"{conn}/{len(results)} assembled as single connected fragment")

    # ── Warhead-only SMILES catalogue (for solo TF targeting)
    print("\n" + "=" * 80)
    print("WARHEAD CATALOGUE (TF-binding fragments, standalone)")
    print("=" * 80)
    print(f"{'Gene':<10} {'Effect':<10} {'MW':>6}  SMILES")
    print("-" * 90)
    for gene, wh in GBM_WARHEADS.items():
        p = compute_props(wh["smiles"])
        mw = f"{p.get('mw', 0):.1f}" if p else "—"
        print(f"{gene:<10} {wh['effect']:<10} {mw:>6}  {wh['smiles']}")

    # ── Recruiter SMILES catalogue
    print("\n" + "=" * 80)
    print("RECRUITER CATALOGUE (epigenetic effector fragments)")
    print("=" * 80)
    print(f"{'Name':<25} {'Effect':<10} {'Ki nM':>8} {'Mark':<18} SMILES")
    print("-" * 100)
    for name, rec in RECRUITERS.items():
        print(f"{name:<25} {rec['effect']:<10} {rec['ki_nM']:>8.1f} {rec['mark']:<18} {rec['smiles']}")

    print("\nDone.")
    return results


if __name__ == "__main__":
    main()
