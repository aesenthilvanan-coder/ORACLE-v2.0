import json
import requests
from pathlib import Path
from typing import List, Optional, Dict
import logging

logger = logging.getLogger(__name__)

RCSB_BASE = "https://data.rcsb.org/rest/v1/core"
RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"


class PDBFetcher:
    """Fetches protein structures from the RCSB Protein Data Bank."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("./.cache/pdb")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def search_by_gene(self, gene_name: str, organism: str = "Homo sapiens") -> List[str]:
        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {"attribute": "rcsb_entity_source_organism.taxonomy_lineage.name", "operator": "exact_match", "value": organism},
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {"attribute": "rcsb_gene_name.value", "operator": "exact_match", "value": gene_name},
                    },
                ],
            },
            "return_type": "entry",
            "request_options": {"results_verbosity": "compact", "return_all_hits": False, "paginate": {"start": 0, "rows": 20}},
        }
        try:
            resp = requests.post(RCSB_SEARCH, json=query, timeout=15)
            data = resp.json()
            return [r["identifier"] for r in data.get("result_set", [])]
        except Exception as e:
            logger.debug(f"PDB search failed for {gene_name}: {e}")
            return []

    def download_structure(self, pdb_id: str, output_dir: Optional[Path] = None) -> Optional[Path]:
        out_dir = output_dir or self.cache_dir
        out_path = out_dir / f"{pdb_id.upper()}.pdb"
        if out_path.exists():
            return out_path

        try:
            url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            out_path.write_text(resp.text)
            logger.info(f"Downloaded PDB {pdb_id} to {out_path}")
            return out_path
        except Exception as e:
            logger.warning(f"Failed to download PDB {pdb_id}: {e}")
            return None

    def get_structure_metadata(self, pdb_id: str) -> Dict:
        try:
            resp = requests.get(f"{RCSB_BASE}/entry/{pdb_id.upper()}", timeout=10)
            return resp.json()
        except Exception as e:
            logger.debug(f"Metadata fetch failed for {pdb_id}: {e}")
            return {}
