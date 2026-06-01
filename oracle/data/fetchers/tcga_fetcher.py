import requests
import json
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

TCGA_BASE = "https://api.gdc.cancer.gov"

CANCER_TYPE_TO_PROJECT = {
    "colorectal": "TCGA-COAD",
    "aml": "TCGA-LAML",
    "breast": "TCGA-BRCA",
    "lung": "TCGA-LUAD",
    "glioblastoma": "TCGA-GBM",
    "melanoma": "TCGA-SKCM",
    "pancreatic": "TCGA-PAAD",
    "prostate": "TCGA-PRAD",
    "ovarian": "TCGA-OV",
    "hepatocellular": "TCGA-LIHC",
}


class TCGAFetcher:
    """Fetches gene expression and clinical data from TCGA via GDC API."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path("./.cache/tcga")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_project_files(
        self,
        project_id: str,
        data_category: str = "Transcriptome Profiling",
        data_type: str = "Gene Expression Quantification",
        max_files: int = 100,
    ) -> List[Dict]:
        filters = {
            "op": "and",
            "content": [
                {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
                {"op": "in", "content": {"field": "data_category", "value": [data_category]}},
                {"op": "in", "content": {"field": "data_type", "value": [data_type]}},
                {"op": "in", "content": {"field": "experimental_strategy", "value": ["RNA-Seq"]}},
            ],
        }

        try:
            resp = requests.post(
                f"{TCGA_BASE}/files",
                headers={"Content-Type": "application/json"},
                json={
                    "filters": filters,
                    "fields": "file_id,file_name,cases.case_id,cases.submitter_id",
                    "format": "json",
                    "size": max_files,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("data", {}).get("hits", [])
        except Exception as e:
            logger.warning(f"TCGA file listing failed for {project_id}: {e}")
            return []

    def get_cancer_expression_summary(self, cancer_type: str) -> Optional[Dict]:
        project_id = CANCER_TYPE_TO_PROJECT.get(cancer_type.lower())
        if not project_id:
            logger.warning(f"Unknown cancer type: {cancer_type}")
            return None

        cache_file = self.cache_dir / f"{project_id}_summary.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)

        files = self.get_project_files(project_id, max_files=5)
        summary = {"project_id": project_id, "cancer_type": cancer_type, "n_files": len(files)}

        with open(cache_file, "w") as f:
            json.dump(summary, f)

        return summary
