from typing import Dict, List, Optional, Set
import re
import logging

logger = logging.getLogger(__name__)

AA_1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
AA_3_TO_1 = {v: k for k, v in AA_1_TO_3.items()}

HISTONE_MARKS = {
    "H3K4me1": "active enhancer",
    "H3K4me3": "active promoter",
    "H3K27ac": "active enhancer/promoter",
    "H3K27me3": "polycomb repression",
    "H3K9me3": "constitutive heterochromatin",
    "H3K36me3": "transcribed gene body",
    "H4R3me2s": "transcriptional repression",
    "5mC_CpG": "DNA methylation / gene silencing",
    "pSer2_RNAPII": "transcription elongation",
}


def parse_uniprot_id(gene_name: str) -> Optional[str]:
    """Look up UniProt ID for a human gene symbol."""
    try:
        import requests
        resp = requests.get(
            "https://rest.uniprot.org/uniprotkb/search",
            params={
                "query": f"gene_exact:{gene_name} AND organism_id:9606 AND reviewed:true",
                "format": "json",
                "size": 1,
            },
            timeout=10,
        )
        results = resp.json().get("results", [])
        if results:
            return results[0]["primaryAccession"]
    except Exception as e:
        logger.debug(f"UniProt lookup failed for {gene_name}: {e}")
    return None


def canonical_smiles(smiles: str) -> Optional[str]:
    """Return RDKit canonical SMILES or None if invalid."""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def smiles_to_inchikey(smiles: str) -> Optional[str]:
    try:
        from rdkit import Chem
        from rdkit.Chem.inchi import MolToInchiKey
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return MolToInchiKey(mol)
    except Exception:
        return None


def load_human_tf_list() -> Set[str]:
    """Return set of known human TF gene symbols from a bundled list."""
    try:
        from pathlib import Path
        import json
        p = Path(__file__).parent.parent.parent / "data" / "raw" / "human_tfs.txt"
        if p.exists():
            return set(line.strip() for line in p.read_text().splitlines() if line.strip())
    except Exception:
        pass
    return set()


def gene_to_ensembl(gene_name: str) -> Optional[str]:
    """Map a gene symbol to Ensembl ID via MyGene.info."""
    try:
        import requests
        resp = requests.get(
            "https://mygene.info/v3/query",
            params={"q": gene_name, "species": "human", "fields": "ensembl.gene"},
            timeout=10,
        )
        hits = resp.json().get("hits", [])
        if hits:
            ensembl = hits[0].get("ensembl", {})
            if isinstance(ensembl, list):
                ensembl = ensembl[0]
            return ensembl.get("gene")
    except Exception as e:
        logger.debug(f"Ensembl lookup failed for {gene_name}: {e}")
    return None
