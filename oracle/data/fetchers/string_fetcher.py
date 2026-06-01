import requests
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

STRING_BASE = "https://string-db.org/api"
HUMAN_TAXON = 9606


class STRINGFetcher:
    """Fetches protein interaction data from STRING database."""

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        score_threshold: int = 700,
    ):
        self.cache_dir = cache_dir or Path("./.cache/string")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.score_threshold = score_threshold

    def get_interactions(
        self,
        gene_names: List[str],
        species: int = HUMAN_TAXON,
    ) -> List[Tuple[str, str, float]]:
        cache_key = "_".join(sorted(gene_names[:20]))
        cache_file = self.cache_dir / f"string_{cache_key[:32]}.json"

        if cache_file.exists():
            with open(cache_file) as f:
                data = json.load(f)
            return [(d["preferredName_A"], d["preferredName_B"], d["score"] / 1000.0) for d in data]

        try:
            resp = requests.post(
                f"{STRING_BASE}/json/network",
                data={
                    "identifiers": "\r".join(gene_names),
                    "species": species,
                    "required_score": self.score_threshold,
                    "caller_identity": "oracle_pipeline",
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            with open(cache_file, "w") as f:
                json.dump(data, f)
            return [(d["preferredName_A"], d["preferredName_B"], d["score"] / 1000.0) for d in data]
        except Exception as e:
            logger.warning(f"STRING fetch failed: {e}")
            return []

    def build_interaction_graph(
        self,
        gene_names: List[str],
        species: int = HUMAN_TAXON,
    ):
        import networkx as nx
        interactions = self.get_interactions(gene_names, species)
        G = nx.Graph()
        G.add_nodes_from(gene_names)
        for a, b, score in interactions:
            G.add_edge(a, b, weight=score)
        logger.info(f"STRING interaction graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
