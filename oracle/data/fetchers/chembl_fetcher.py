"""
ChEMBLFetcher – fetches known small-molecule binders for target TFs from ChEMBL.

ChEMBL REST API: https://www.ebi.ac.uk/chembl/api/data
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import requests  # type: ignore

logger = logging.getLogger(__name__)

BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

# Rate-limiting: ChEMBL public API allows ~5 requests/sec
_REQUEST_DELAY_S = 0.25


@dataclass
class Compound:
    """A small-molecule binder retrieved from ChEMBL.

    Attributes
    ----------
    chembl_id:
        ChEMBL compound identifier, e.g. ``"CHEMBL12345"``.
    smiles:
        Canonical SMILES string.
    target:
        Target TF name used to query ChEMBL.
    ki_nM:
        Inhibition constant in nM.
    assay_type:
        ChEMBL assay type string, e.g. ``"B"`` (binding), ``"F"``
        (functional).
    """

    chembl_id: str
    smiles: str
    target: str
    ki_nM: float
    assay_type: str


class ChEMBLFetcher:
    """Fetches known small-molecule binders for target TFs from ChEMBL.

    Parameters
    ----------
    cache_dir:
        Local directory for caching downloaded results.
    """

    def __init__(self, cache_dir: str = "./cache/chembl") -> None:
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

    def fetch_tf_binders(
        self,
        tf_name: str,
        max_ki_nM: float = 1000.0,
    ) -> List[Compound]:
        """Fetch known small-molecule binders for a transcription factor.

        Workflow:
        1. Look up the ChEMBL target ID for *tf_name*.
        2. Query bioactivities with ``standard_type='Ki'``,
           ``standard_units='nM'``, and value ≤ *max_ki_nM*.
        3. Retrieve SMILES for each molecule.
        4. Return a list of :class:`Compound` objects.

        Parameters
        ----------
        tf_name:
            Transcription factor gene symbol, e.g. ``"TP53"``.
        max_ki_nM:
            Maximum Ki value (in nM) to include.  Defaults to 1000 nM
            (i.e. 1 µM).

        Returns
        -------
        list of Compound
        """
        import json

        cache_key = f"{tf_name}_{int(max_ki_nM)}nM.json"
        cache_path = self.cache_dir / cache_key

        if cache_path.exists():
            logger.info("Loading cached ChEMBL data: %s", cache_path)
            with open(cache_path, "r") as fh:
                raw = json.load(fh)
            return [Compound(**c) for c in raw]

        # 1. Get ChEMBL target ID
        target_id = self._lookup_target(tf_name)
        if target_id is None:
            logger.warning("No ChEMBL target found for TF '%s'", tf_name)
            return []

        # 2. Query bioactivities
        activities = self._query_activities(target_id, max_ki_nM)
        if not activities:
            logger.info("No Ki activities found for %s (target=%s)", tf_name, target_id)
            return []

        # 3. Build Compound objects
        compounds: List[Compound] = []
        seen_ids: set = set()

        for act in activities:
            mol_id = act.get("molecule_chembl_id")
            if mol_id is None or mol_id in seen_ids:
                continue
            seen_ids.add(mol_id)

            ki_value = act.get("standard_value")
            if ki_value is None:
                continue
            try:
                ki = float(ki_value)
            except (TypeError, ValueError):
                continue

            if ki > max_ki_nM:
                continue

            smiles = self._get_smiles(mol_id)
            if smiles is None:
                continue

            assay_type = act.get("assay_type", "")
            compounds.append(
                Compound(
                    chembl_id=mol_id,
                    smiles=smiles,
                    target=tf_name,
                    ki_nM=ki,
                    assay_type=assay_type,
                )
            )
            time.sleep(_REQUEST_DELAY_S)

        # Cache results
        with open(cache_path, "w") as fh:
            import dataclasses
            json.dump(
                [dataclasses.asdict(c) for c in compounds],
                fh,
                indent=2,
            )

        logger.info(
            "Found %d binders for %s with Ki ≤ %.1f nM",
            len(compounds),
            tf_name,
            max_ki_nM,
        )
        return compounds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lookup_target(self, tf_name: str) -> Optional[str]:
        """Look up the ChEMBL target ID for a TF gene symbol.

        Searches the ChEMBL ``/target`` endpoint for single protein targets
        whose preferred name or gene symbol matches *tf_name*.

        Returns
        -------
        str or None
            ChEMBL target ID (e.g. ``"CHEMBL1907601"``), or ``None`` if not
            found.
        """
        url = f"{BASE_URL}/target.json"
        params = {
            "target_synonym__icontains": tf_name,
            "target_type": "SINGLE PROTEIN",
            "limit": 10,
        }

        try:
            resp = self._get(url, params=params)
            targets = resp.get("targets", [])
        except Exception as exc:
            logger.error("ChEMBL target lookup failed for '%s': %s", tf_name, exc)
            return None

        if not targets:
            # Retry with pref_name
            params2 = {
                "pref_name__icontains": tf_name,
                "target_type": "SINGLE PROTEIN",
                "limit": 10,
            }
            try:
                resp2 = self._get(url, params=params2)
                targets = resp2.get("targets", [])
            except Exception:
                return None

        if not targets:
            return None

        # Pick the first matching target
        return targets[0].get("target_chembl_id")

    def _query_activities(
        self,
        target_id: str,
        max_ki_nM: float,
    ) -> List[dict]:
        """Query ChEMBL for Ki bioactivities against a target.

        Parameters
        ----------
        target_id:
            ChEMBL target ID.
        max_ki_nM:
            Upper bound on Ki value.

        Returns
        -------
        list of dict
        """
        url = f"{BASE_URL}/activity.json"
        params = {
            "target_chembl_id": target_id,
            "standard_type": "Ki",
            "standard_units": "nM",
            "standard_value__lte": max_ki_nM,
            "assay_type": "B",
            "limit": 1000,
        }

        try:
            resp = self._get(url, params=params)
            return resp.get("activities", [])
        except Exception as exc:
            logger.error("ChEMBL activity query failed for %s: %s", target_id, exc)
            return []

    def _get_smiles(self, mol_id: str) -> Optional[str]:
        """Fetch the canonical SMILES for a ChEMBL molecule.

        Parameters
        ----------
        mol_id:
            ChEMBL molecule ID, e.g. ``"CHEMBL12345"``.

        Returns
        -------
        str or None
        """
        url = f"{BASE_URL}/molecule/{mol_id}.json"
        try:
            resp = self._get(url)
            struct = resp.get("molecule_structures") or {}
            return struct.get("canonical_smiles")
        except Exception as exc:
            logger.warning("Could not fetch SMILES for %s: %s", mol_id, exc)
            return None

    def _get(self, url: str, params: Optional[dict] = None) -> dict:
        """HTTP GET with basic error handling and rate-limiting."""
        time.sleep(_REQUEST_DELAY_S)
        response = self._session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
