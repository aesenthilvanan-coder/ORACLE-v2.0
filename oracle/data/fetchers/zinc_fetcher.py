import requests
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

ZINC_BASE = "https://zinc.docking.org"


class ZINCFetcher:
    """Fetches drug-like molecule SMILES from the ZINC database."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("./.cache/zinc")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def search_by_smiles(
        self,
        query_smiles: str,
        n_results: int = 20,
        similarity_threshold: float = 0.7,
    ) -> List[Dict]:
        try:
            resp = requests.get(
                f"{ZINC_BASE}/substances/subsets/fda.json",
                params={
                    "smiles": query_smiles,
                    "count": n_results,
                    "similarity": similarity_threshold,
                },
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.debug(f"ZINC similarity search failed: {e}")
            return []

    def get_drug_like_subset(
        self,
        subset: str = "drug-like",
        n_molecules: int = 1000,
        output_path: Optional[Path] = None,
    ) -> List[str]:
        cache_file = self.cache_dir / f"zinc_{subset}_{n_molecules}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)

        subset_map = {
            "drug-like": "druglike",
            "lead-like": "leads",
            "fda": "fda",
            "natural": "natural-products",
        }
        zinc_subset = subset_map.get(subset, subset)

        smiles_list = []
        try:
            page = 1
            while len(smiles_list) < n_molecules:
                resp = requests.get(
                    f"{ZINC_BASE}/substances/subsets/{zinc_subset}.json",
                    params={"page": page, "count": min(100, n_molecules - len(smiles_list))},
                    timeout=20,
                )
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    break
                for item in data:
                    if item.get("smiles"):
                        smiles_list.append(item["smiles"])
                page += 1
                if len(data) < 100:
                    break
        except Exception as e:
            logger.warning(f"ZINC subset download failed: {e}")

        if smiles_list:
            with open(cache_file, "w") as f:
                json.dump(smiles_list, f)

        return smiles_list[:n_molecules]

    def get_fragment_library(self, n_fragments: int = 500) -> List[str]:
        return self.get_drug_like_subset("lead-like", n_fragments)
