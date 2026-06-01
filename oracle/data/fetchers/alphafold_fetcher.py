import requests
from pathlib import Path
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

ALPHAFOLD_BASE = "https://alphafold.ebi.ac.uk/api"


class AlphaFoldFetcher:
    """Fetches predicted protein structures from the AlphaFold database."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("./.cache/alphafold")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._uniprot_cache: Dict[str, str] = {}

    def get_prediction(self, uniprot_id: str) -> Optional[Dict]:
        try:
            resp = requests.get(f"{ALPHAFOLD_BASE}/prediction/{uniprot_id}", timeout=10)
            resp.raise_for_status()
            predictions = resp.json()
            return predictions[0] if predictions else None
        except Exception as e:
            logger.debug(f"AlphaFold metadata fetch failed for {uniprot_id}: {e}")
            return None

    def download_pdb(self, uniprot_id: str, output_dir: Optional[Path] = None) -> Optional[Path]:
        out_dir = output_dir or self.cache_dir
        out_path = out_dir / f"{uniprot_id}_af.pdb"
        if out_path.exists():
            return out_path

        pred = self.get_prediction(uniprot_id)
        if not pred:
            return None

        pdb_url = pred.get("pdbUrl")
        if not pdb_url:
            return None

        try:
            resp = requests.get(pdb_url, timeout=60)
            resp.raise_for_status()
            out_path.write_text(resp.text)
            logger.info(f"Downloaded AlphaFold structure for {uniprot_id} to {out_path}")
            return out_path
        except Exception as e:
            logger.warning(f"AlphaFold PDB download failed for {uniprot_id}: {e}")
            return None

    def download_by_gene(self, gene_name: str, output_dir: Optional[Path] = None) -> Optional[Path]:
        from oracle.utils.bio_utils import parse_uniprot_id
        uniprot_id = self._uniprot_cache.get(gene_name) or parse_uniprot_id(gene_name)
        if not uniprot_id:
            logger.warning(f"No UniProt ID found for {gene_name}")
            return None
        self._uniprot_cache[gene_name] = uniprot_id
        return self.download_pdb(uniprot_id, output_dir)
