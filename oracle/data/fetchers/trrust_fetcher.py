import requests
import csv
import io
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)

TRRUST_URL = "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv"


class TRRUSTFetcher:
    """Fetches known TF-target regulatory interactions from TRRUST."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("./.cache/trrust")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._interactions: Optional[List[Tuple[str, str, str]]] = None

    def load(self) -> List[Tuple[str, str, str]]:
        """Return list of (TF, target, mode) tuples."""
        if self._interactions is not None:
            return self._interactions

        cache_file = self.cache_dir / "trrust_human.tsv"
        if not cache_file.exists():
            try:
                resp = requests.get(TRRUST_URL, timeout=30)
                resp.raise_for_status()
                cache_file.write_text(resp.text)
                logger.info(f"Downloaded TRRUST data to {cache_file}")
            except Exception as e:
                logger.warning(f"TRRUST download failed: {e}")
                self._interactions = []
                return []

        interactions = []
        with open(cache_file) as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 3:
                    tf, target, mode = row[0], row[1], row[2]
                    interactions.append((tf, target, mode))

        self._interactions = interactions
        logger.info(f"Loaded {len(interactions)} TRRUST interactions")
        return interactions

    def get_targets(self, tf_name: str) -> List[Tuple[str, str]]:
        interactions = self.load()
        return [(target, mode) for tf, target, mode in interactions if tf == tf_name]

    def get_regulators(self, gene_name: str) -> List[Tuple[str, str]]:
        interactions = self.load()
        return [(tf, mode) for tf, target, mode in interactions if target == gene_name]

    def build_tf_target_dict(self, tfs: Optional[Set[str]] = None) -> Dict[str, List[str]]:
        interactions = self.load()
        result: Dict[str, List[str]] = {}
        for tf, target, mode in interactions:
            if tfs is None or tf in tfs:
                result.setdefault(tf, []).append(target)
        return result

    def as_networkx_graph(self):
        import networkx as nx
        interactions = self.load()
        G = nx.DiGraph()
        for tf, target, mode in interactions:
            sign = 1 if mode in ("Activation", "+") else -1
            G.add_edge(tf, target, sign=sign, weight=1.0, source="TRRUST")
        logger.info(f"TRRUST graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
