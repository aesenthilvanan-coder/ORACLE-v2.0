import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class OracleReportGenerator:
    """Generates HTML and JSON reports summarizing an ORACLE pipeline run."""

    def __init__(self, output_dir: str = "./outputs/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sections: List[Dict[str, Any]] = []

    def add_section(self, title: str, content: Any, section_type: str = "text") -> None:
        self.sections.append({"title": title, "content": content, "type": section_type})

    def add_cam_summary(self, cam_output) -> None:
        summary = {
            "sample_id": cam_output.sample_id,
            "cancer_type": cam_output.cancer_type,
            "n_genes": cam_output.n_genes,
            "n_attractors": len(cam_output.all_attractors),
            "attractor_labels": cam_output.attractor_labels,
            "basin_sizes": {str(k): v for k, v in cam_output.basin_sizes.items()},
        }
        self.add_section("CAM Module Output", summary, "json")

    def add_rsp_summary(self, rsp_output) -> None:
        switch = rsp_output.switch_set
        summary = {
            "genes_to_activate": switch.genes_to_activate,
            "genes_to_repress": switch.genes_to_repress,
            "predicted_reversion_probability": f"{switch.predicted_reversion_probability:.1%}",
            "validated_reversion_fraction": f"{switch.validated_reversion_fraction:.1%}",
            "predicted_cancer_score_after": f"{switch.predicted_cancer_score_after:.3f}",
            "top_genes_by_importance": sorted(
                switch.gene_importance_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10],
        }
        self.add_section("RSP Module Output", summary, "json")

    def add_tcd_summary(self, tcd_output) -> None:
        molecules = []
        for mol in tcd_output.tcip_molecules:
            molecules.append({
                "tf_name": mol.tf_name,
                "smiles": mol.smiles[:80] + "..." if len(mol.smiles) > 80 else mol.smiles,
                "molecular_weight": f"{mol.molecular_weight:.1f}",
                "predicted_affinity_nM": f"{mol.predicted_affinity_nM:.1f}",
                "passes_ro5": mol.passes_ro5,
            })
        self.add_section("TCD Module Output", {"molecules": molecules}, "json")

    def save_json(self, filename: str = "oracle_report.json") -> Path:
        report = {
            "generated_at": datetime.now().isoformat(),
            "sections": self.sections,
        }
        out_path = self.output_dir / filename
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"JSON report saved to {out_path}")
        return out_path

    def save_html(self, filename: str = "oracle_report.html") -> Path:
        html_parts = [
            "<!DOCTYPE html><html><head>",
            "<title>ORACLE Pipeline Report</title>",
            "<style>body{font-family:sans-serif;max-width:1200px;margin:auto;padding:20px}",
            "h2{color:#333}pre{background:#f4f4f4;padding:10px;overflow-x:auto}</style>",
            "</head><body>",
            f"<h1>ORACLE Pipeline Report</h1>",
            f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        ]

        for sec in self.sections:
            html_parts.append(f"<h2>{sec['title']}</h2>")
            if sec["type"] == "json":
                html_parts.append(f"<pre>{json.dumps(sec['content'], indent=2, default=str)}</pre>")
            else:
                html_parts.append(f"<p>{sec['content']}</p>")

        html_parts.append("</body></html>")
        out_path = self.output_dir / filename
        out_path.write_text("\n".join(html_parts))
        logger.info(f"HTML report saved to {out_path}")
        return out_path
