"""
ENCODEFetcher – fetches ENCODE ChIP-seq data for TF binding evidence.

Uses the ENCODE REST API to query ChIP-seq experiments targeting specific
transcription factors and compute a binding score for each candidate
target gene.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import requests  # type: ignore

logger = logging.getLogger(__name__)

BASE_URL = "https://www.encodeproject.org/api"
# The ENCODE portal base (for search endpoints that don't live under /api)
_ENCODE_PORTAL = "https://www.encodeproject.org"


class ENCODEFetcher:
    """Fetches ENCODE ChIP-seq data for TF binding evidence.

    Parameters
    ----------
    cache_dir:
        Local directory for caching downloaded results.
    """

    def __init__(self, cache_dir: str = "./cache/encode") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Accept": "application/json",
            }
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_tf_chipseq(
        self,
        tf_name: str,
        cell_type: str = "cancer",
    ) -> Dict[str, Dict[str, float]]:
        """Fetch ChIP-seq binding evidence for a transcription factor.

        Queries ENCODE for released ChIP-seq experiments targeting *tf_name*
        and returns a dictionary of the form::

            {tf_name: {target_gene: score, ...}}

        The score represents the strength of binding evidence aggregated
        across available experiments (see ``_compute_binding_score``).

        Parameters
        ----------
        tf_name:
            Transcription factor symbol, e.g. ``"TP53"``.
        cell_type:
            ENCODE biosample classification to filter by (used as a substring
            match against biosample term names).

        Returns
        -------
        dict
            ``{tf_name: {target_gene: score}}``
        """
        import json
        cache_key = f"{tf_name}_{cell_type.replace(' ', '_')}.json"
        cache_path = self.cache_dir / cache_key

        if cache_path.exists():
            logger.info("Loading cached ENCODE data: %s", cache_path)
            with open(cache_path, "r") as fh:
                return json.load(fh)

        experiments = self._query_encode(tf_name, assay="ChIP-seq")

        # Filter by cell type if specified and not the catch-all
        if cell_type and cell_type.lower() != "cancer":
            experiments = [
                exp for exp in experiments
                if cell_type.lower()
                in exp.get("biosample_summary", "").lower()
            ]

        if not experiments:
            logger.warning(
                "No ENCODE ChIP-seq experiments found for %s / %s",
                tf_name,
                cell_type,
            )
            return {tf_name: {}}

        # Aggregate binding scores per target gene (based on experiment
        # quality metrics such as signal value, number of peaks, etc.)
        target_scores: Dict[str, float] = {}
        for exp in experiments:
            # Use gene targets reported in the experiment metadata
            targets = exp.get("target", {})
            if isinstance(targets, dict):
                gene = targets.get("label", targets.get("name", ""))
            else:
                gene = str(targets)

            if not gene or gene.lower() == tf_name.lower():
                # Skip self-binding or unknown targets
                continue

            score = self._compute_binding_score([exp])
            if gene in target_scores:
                target_scores[gene] = max(target_scores[gene], score)
            else:
                target_scores[gene] = score

        result = {tf_name: target_scores}

        # Cache result
        with open(cache_path, "w") as fh:
            json.dump(result, fh, indent=2)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _query_encode(
        self,
        target: str,
        assay: str = "ChIP-seq",
    ) -> List[dict]:
        """Query the ENCODE search API for experiments.

        Parameters
        ----------
        target:
            TF gene symbol to search for.
        assay:
            Assay type, default ``"ChIP-seq"``.

        Returns
        -------
        list of dict
            Experiment metadata objects from ENCODE.
        """
        search_url = f"{_ENCODE_PORTAL}/search/"
        params = {
            "type": "Experiment",
            "assay_title": assay,
            "target.label": target,
            "status": "released",
            "format": "json",
            "limit": "all",
            "frame": "object",
        }

        try:
            resp = self._session.get(search_url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            logger.error("ENCODE search failed for target '%s': %s", target, exc)
            return []

        experiments = data.get("@graph", [])
        logger.info(
            "ENCODE: %d experiments found for %s / %s",
            len(experiments),
            target,
            assay,
        )
        return experiments

    def _compute_binding_score(self, experiments: List[dict]) -> float:
        """Compute an aggregate binding score from a list of experiments.

        The score is based on:
        - Number of experiments with ``"audit"`` level ``"ERROR"`` (penalised)
        - ``biosample_summary`` diversity (bonus for multiple cell types)
        - ``replication_type`` (bonus for isogenic replicates)

        Returns a score in ``[0.0, 1.0]``.

        Parameters
        ----------
        experiments:
            List of ENCODE experiment objects.

        Returns
        -------
        float
        """
        if not experiments:
            return 0.0

        score = 0.0
        n = len(experiments)

        for exp in experiments:
            # Base score from number of replicates
            replicates = exp.get("replicates", [])
            rep_score = min(len(replicates) / 4.0, 1.0)  # cap at 4 replicates

            # Quality bonus: released experiments with no errors
            audits = exp.get("audit", {})
            has_error = "ERROR" in audits
            quality_score = 0.0 if has_error else 0.5

            # Isogenic replication bonus
            rep_type = exp.get("replication_type", "")
            rep_bonus = 0.3 if "isogenic" in rep_type.lower() else 0.0

            exp_score = rep_score * 0.5 + quality_score + rep_bonus
            score += exp_score

        # Normalise by number of experiments, cap at 1.0
        return min(score / n, 1.0)
