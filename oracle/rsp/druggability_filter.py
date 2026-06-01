import json
import requests
from pathlib import Path
from typing import Set, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DruggabilityFilter:
    """Filters TF candidates to those with druggable surfaces.

    Evidence sources:
    1. ChEMBL: known small molecule binders with Ki < 10 µM
    2. DGIdb: drug-gene interaction database
    3. PDB: experimental TF structures with co-crystallized ligands
    4. AlphaFold + fpocket: computed druggability score > threshold
    """

    def __init__(
        self,
        chembl_ki_cutoff: float = 10000.0,
        fpocket_score_cutoff: float = 0.3,
        cache_dir: Optional[Path] = None,
    ):
        self.chembl_ki_cutoff = chembl_ki_cutoff
        self.fpocket_score_cutoff = fpocket_score_cutoff
        self.cache_dir = cache_dir or Path("./.cache/druggability")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._chembl_binders: Dict[str, bool] = {}
        self._dgidb_targets: Set[str] = set()
        self._pdb_liganded: Set[str] = set()
        self._fpocket_scores: Dict[str, float] = {}

        self._load_dgidb()

    def filter_genes(self, gene_list: List[str]) -> Set[int]:
        druggable_indices = set()
        for i, gene in enumerate(gene_list):
            if self.is_druggable(gene):
                druggable_indices.add(i)
        logger.info(
            f"Druggability filter: {len(druggable_indices)}/{len(gene_list)} genes are druggable"
        )
        return druggable_indices

    def is_druggable(self, gene_name: str) -> bool:
        if self._check_chembl(gene_name):
            return True
        if gene_name in self._dgidb_targets:
            return True
        if gene_name in self._pdb_liganded:
            return True
        if self._fpocket_scores.get(gene_name, 0.0) >= self.fpocket_score_cutoff:
            return True
        return self._has_structural_data(gene_name)

    def _check_chembl(self, gene_name: str) -> bool:
        if gene_name in self._chembl_binders:
            return self._chembl_binders[gene_name]

        cache_file = self.cache_dir / f"chembl_{gene_name}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                data = json.load(f)
            result = data.get("has_binder", False)
            self._chembl_binders[gene_name] = result
            return result

        try:
            base = "https://www.ebi.ac.uk/chembl/api/data"
            resp = requests.get(
                f"{base}/target/search",
                params={"q": gene_name, "format": "json", "limit": 5},
                timeout=10,
            )
            targets = resp.json().get("targets", [])
            target_ids = [
                t["target_chembl_id"] for t in targets
                if t.get("pref_name", "").upper() == gene_name.upper()
            ]
            if not target_ids:
                self._chembl_binders[gene_name] = False
                with open(cache_file, "w") as f:
                    json.dump({"has_binder": False}, f)
                return False

            resp2 = requests.get(
                f"{base}/activity",
                params={
                    "target_chembl_id": target_ids[0],
                    "standard_type": "Ki",
                    "standard_value__lte": self.chembl_ki_cutoff,
                    "format": "json",
                    "limit": 1,
                },
                timeout=10,
            )
            has_binder = resp2.json().get("page_meta", {}).get("total_count", 0) > 0
            self._chembl_binders[gene_name] = has_binder
            with open(cache_file, "w") as f:
                json.dump({"has_binder": has_binder}, f)
            return has_binder
        except Exception as e:
            logger.debug(f"ChEMBL check failed for {gene_name}: {e}")
            return False

    def _load_dgidb(self) -> None:
        cache_file = self.cache_dir / "dgidb_targets.json"
        if cache_file.exists():
            with open(cache_file) as f:
                self._dgidb_targets = set(json.load(f))
            logger.info(f"Loaded {len(self._dgidb_targets)} DGIdb targets from cache")
            return
        try:
            resp = requests.get(
                "https://www.dgidb.org/api/v2/genes",
                params={"immunotherapy": "false", "anti_neoplastic": "true"},
                timeout=30,
            )
            data = resp.json()
            self._dgidb_targets = set(
                g["gene_name"] for g in data.get("genes", []) if g.get("gene_name")
            )
            with open(cache_file, "w") as f:
                json.dump(list(self._dgidb_targets), f)
            logger.info(f"Loaded {len(self._dgidb_targets)} DGIdb anti-neoplastic targets")
        except Exception as e:
            logger.warning(f"Could not load DGIdb: {e}")

    def _has_structural_data(self, gene_name: str) -> bool:
        from oracle.tcd.tf_structurer import TFStructurer
        af_path = Path(f"./data/raw/alphafold/{gene_name}_af.pdb")
        pdb_path = Path(f"./data/raw/pdb/{gene_name}.pdb")
        return af_path.exists() or pdb_path.exists()
