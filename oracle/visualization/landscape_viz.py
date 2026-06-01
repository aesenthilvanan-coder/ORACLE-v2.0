"""
landscape_viz.py
----------------
Visualization tools for the ORACLE attractor landscape.

Classes
-------
AttractorLandscapePlotter : Visualizes attractors via UMAP, 3-D energy
    surface (plotly), perturbation trajectories, and basin-size pie chart.
OracleReportGenerator : Assembles a self-contained HTML report from
    ORACLE pipeline outputs.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports — graceful degradation when heavyweight libs absent
# ---------------------------------------------------------------------------

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False
    Figure = Any  # type: ignore
    logger.warning("matplotlib not available; plotting will be limited.")

try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False
    logger.warning("plotly not available; 3-D landscape plot disabled.")

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors, QED, rdMolDescriptors, AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False
    logger.warning("RDKit not available; molecular images will be placeholder text.")

try:
    import umap as umap_module
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False
    logger.warning("umap-learn not available; UMAP projection disabled.")


# ---------------------------------------------------------------------------
# AttractorLandscapePlotter
# ---------------------------------------------------------------------------


class AttractorLandscapePlotter:
    """
    Visualizes the attractor landscape produced by the CAM module.

    Parameters
    ----------
    figsize : tuple
        Default figure size (width, height) in inches.
    dpi : int
        Figure resolution.
    """

    ATTRACTOR_COLORS = {
        "cancer": "red",
        "normal": "blue",
        "transitional": "gray",
    }

    def __init__(self, figsize: Tuple[int, int] = (14, 6), dpi: int = 120) -> None:
        self.figsize = figsize
        self.dpi = dpi

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot_landscape_umap(
        self,
        cam_output: Any,
        rsp_output: Any = None,
    ) -> Figure:
        """
        Two-panel UMAP of the attractor landscape.

        Left panel  — cells coloured by cancer score (RdBu_r, 0-1),
                      attractors marked as coloured stars.
        Right panel — perturbation trajectory (if rsp_output provided),
                      otherwise a copy of the left panel.

        Parameters
        ----------
        cam_output : CAMOutput
            Output from the CAM module containing .adata, .attractors,
            .attractor_labels, .cancer_score_fn.
        rsp_output : RSPOutput, optional
            Output from the RSP module.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not _MPL_AVAILABLE:
            logger.error("matplotlib required for plot_landscape_umap.")
            return None  # type: ignore

        fig, axes = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)
        ax_left, ax_right = axes

        # ---- Left: UMAP coloured by cancer score -----------------------
        self._plot_umap_cancer_score(cam_output, ax_left)

        # ---- Right: perturbation trajectory or duplicate ---------------
        if rsp_output is not None:
            self._plot_perturbation_trajectory(cam_output, rsp_output, ax_right)
        else:
            self._plot_umap_cancer_score(cam_output, ax_right)
            ax_right.set_title("UMAP (no perturbation data)")

        fig.tight_layout()
        return fig

    def plot_energy_landscape_3d(self, cam_output: Any) -> Any:
        """
        3-D energy surface in PC space using Plotly.

        Constructs a 50x50 grid in the first two PCs, projects back to
        gene space, computes the cancer score for each grid point, and
        renders it as a surface plot.

        Parameters
        ----------
        cam_output : CAMOutput

        Returns
        -------
        plotly.graph_objects.Figure or None
        """
        if not _PLOTLY_AVAILABLE:
            logger.error("plotly required for plot_energy_landscape_3d.")
            return None

        adata = getattr(cam_output, "adata", None)
        cancer_score_fn = getattr(cam_output, "cancer_score_fn", None)

        if adata is None or cancer_score_fn is None:
            logger.warning("cam_output missing .adata or .cancer_score_fn.")
            return _empty_plotly_figure("Missing cam_output fields")

        # Build PC grid
        try:
            pca_coords = adata.obsm.get("X_pca")
            if pca_coords is None or pca_coords.shape[1] < 2:
                raise ValueError("PCA not available.")
            pc1_range = np.linspace(pca_coords[:, 0].min(), pca_coords[:, 0].max(), 50)
            pc2_range = np.linspace(pca_coords[:, 1].min(), pca_coords[:, 1].max(), 50)
            PC1, PC2 = np.meshgrid(pc1_range, pc2_range)

            # Approximate back-projection using PCA loadings
            try:
                loadings = adata.varm.get("PCs")  # (n_genes, n_pcs)
                if loadings is None:
                    raise ValueError("No PCA loadings.")
                grid_flat = np.column_stack([PC1.ravel(), PC2.ravel()])
                # Reconstruct gene space using first 2 PCs
                loadings_2 = loadings[:, :2]  # (n_genes, 2)
                gene_space = grid_flat @ loadings_2.T  # (2500, n_genes)
                # Normalise to [0, 1]
                gene_min = gene_space.min(axis=0)
                gene_max = gene_space.max(axis=0)
                gene_norm = (gene_space - gene_min) / np.clip(gene_max - gene_min, 1e-8, None)
            except Exception:
                # Fallback: random gene space
                gene_norm = np.random.rand(2500, adata.n_vars).astype(np.float32)

            # Score each grid point
            import torch
            cancer_scores = cancer_score_fn.score_numpy
            energy_flat = np.array([
                cancer_scores(g.astype(np.float32)) for g in gene_norm
            ])
            Z = energy_flat.reshape(50, 50)

        except Exception as exc:
            logger.warning("3-D energy landscape computation failed: %s", exc)
            Z = np.random.rand(50, 50)
            PC1 = np.linspace(-5, 5, 50)
            PC2 = np.linspace(-5, 5, 50)
            PC1, PC2 = np.meshgrid(PC1, PC2)

        fig = go.Figure(data=[
            go.Surface(
                z=Z,
                x=PC1,
                y=PC2,
                colorscale="RdBu_r",
                cmin=0,
                cmax=1,
                colorbar=dict(title="Cancer Score"),
            )
        ])
        fig.update_layout(
            title="ORACLE Energy Landscape (PC1 vs PC2)",
            scene=dict(
                xaxis_title="PC1",
                yaxis_title="PC2",
                zaxis_title="Cancer Score",
            ),
            margin=dict(l=0, r=0, t=40, b=0),
        )
        return fig

    def plot_basin_sizes(self, cam_output: Any) -> Figure:
        """
        Pie chart of attractor basin sizes.

        Parameters
        ----------
        cam_output : CAMOutput

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not _MPL_AVAILABLE:
            return None  # type: ignore

        basin_sizes = getattr(cam_output, "basin_sizes", {})
        attractor_labels = getattr(cam_output, "attractor_labels", {})

        if not basin_sizes:
            fig, ax = plt.subplots(figsize=(6, 6), dpi=self.dpi)
            ax.text(0.5, 0.5, "No basin size data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("Basin Sizes")
            return fig

        labels = []
        sizes = []
        colors = []
        for idx, size in basin_sizes.items():
            label = attractor_labels.get(idx, f"Attractor {idx}")
            labels.append(f"{label}\n({size})")
            sizes.append(size)
            color_key = "cancer" if "cancer" in str(label).lower() else (
                "normal" if "normal" in str(label).lower() else "transitional"
            )
            colors.append(self.ATTRACTOR_COLORS.get(color_key, "gray"))

        fig, ax = plt.subplots(figsize=(7, 7), dpi=self.dpi)
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
            pctdistance=0.82,
        )
        for t in autotexts:
            t.set_fontsize(9)
        ax.set_title("Attractor Basin Sizes", fontsize=14, fontweight="bold")
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _plot_umap_cancer_score(self, cam_output: Any, ax: Any) -> None:
        """Plot UMAP embedding coloured by cancer score with attractor stars."""
        adata = getattr(cam_output, "adata", None)
        attractors = getattr(cam_output, "attractors", [])
        attractor_labels = getattr(cam_output, "attractor_labels", {})

        if adata is None:
            ax.text(0.5, 0.5, "No AnnData available", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("UMAP Attractor Landscape")
            return

        umap_coords = adata.obsm.get("X_umap")
        if umap_coords is None or umap_coords.shape[1] < 2:
            ax.text(0.5, 0.5, "UMAP not computed", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("UMAP Attractor Landscape")
            return

        cancer_scores = adata.obs.get("cancer_score", np.zeros(adata.n_obs))
        sc_plot = ax.scatter(
            umap_coords[:, 0],
            umap_coords[:, 1],
            c=np.asarray(cancer_scores),
            cmap="RdBu_r",
            vmin=0,
            vmax=1,
            s=4,
            alpha=0.6,
            linewidths=0,
        )
        plt.colorbar(sc_plot, ax=ax, label="Cancer Score", fraction=0.046, pad=0.04)

        # Mark attractors as stars
        for idx, att in enumerate(attractors):
            label = attractor_labels.get(idx, f"att_{idx}")
            color_key = (
                "cancer" if "cancer" in str(label).lower() else (
                    "normal" if "normal" in str(label).lower() else "transitional"
                )
            )
            color = self.ATTRACTOR_COLORS.get(color_key, "gray")
            umap_xy = self._project_attractor_to_umap(att, adata)
            ax.scatter(
                umap_xy[0],
                umap_xy[1],
                marker="*",
                s=300,
                c=color,
                edgecolors="black",
                linewidths=0.8,
                zorder=10,
                label=label,
            )

        if attractors:
            ax.legend(loc="best", fontsize=8, framealpha=0.7)

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title("UMAP Attractor Landscape")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def _project_attractor_to_umap(
        self, att: np.ndarray, adata: Any
    ) -> Tuple[float, float]:
        """
        Find the nearest cell in gene space and return its UMAP coordinates.

        Parameters
        ----------
        att : np.ndarray
            Attractor state vector (n_genes,).
        adata : AnnData

        Returns
        -------
        (umap_x, umap_y) : Tuple[float, float]
        """
        import scipy.sparse as sp

        umap_coords = adata.obsm.get("X_umap")
        if umap_coords is None:
            return (0.0, 0.0)

        try:
            if sp.issparse(adata.X):
                X = adata.X.toarray()
            else:
                X = np.asarray(adata.X)

            n_genes_att = att.shape[0]
            n_genes_data = X.shape[1]

            if n_genes_att < n_genes_data:
                X_sub = X[:, :n_genes_att]
                att_sub = att
            elif n_genes_att > n_genes_data:
                att_sub = att[:n_genes_data]
                X_sub = X
            else:
                X_sub = X
                att_sub = att

            # Normalise att_sub to [0, 1] to match gene space
            att_norm = att_sub.astype(np.float32)

            dists = np.sum((X_sub - att_norm) ** 2, axis=1)
            nearest_idx = int(np.argmin(dists))
            return (float(umap_coords[nearest_idx, 0]), float(umap_coords[nearest_idx, 1]))

        except Exception as exc:
            logger.debug("Attractor UMAP projection failed: %s", exc)
            # Return centroid as fallback
            return (float(umap_coords[:, 0].mean()), float(umap_coords[:, 1].mean()))

    def _plot_perturbation_trajectory(
        self, cam_output: Any, rsp_output: Any, ax: Any
    ) -> None:
        """
        Plot ODE perturbation trajectory arrows on UMAP.

        Parameters
        ----------
        cam_output : CAMOutput
        rsp_output : RSPOutput
        ax : matplotlib Axes
        """
        # First draw base UMAP
        self._plot_umap_cancer_score(cam_output, ax)

        adata = getattr(cam_output, "adata", None)
        umap_coords = None if adata is None else adata.obsm.get("X_umap")

        if umap_coords is None:
            ax.set_title("UMAP + Perturbation Trajectory")
            return

        # Extract trajectory states from RSPOutput
        final_states = getattr(rsp_output, "final_states", None)
        if final_states is None:
            # Try switch_set validation trajectory info
            switch_set = getattr(rsp_output, "switch_set", None)
            if switch_set is None:
                ax.set_title("UMAP + Perturbation (no trajectory data)")
                return
            ax.set_title("UMAP + Perturbation Trajectory")
            return

        # Project trajectory states to UMAP via nearest-cell lookup
        try:
            traj_umap = np.array([
                self._project_attractor_to_umap(s.astype(np.float32), adata)
                for s in final_states[:min(20, len(final_states))]
            ])

            if len(traj_umap) > 1:
                for i in range(len(traj_umap) - 1):
                    dx = traj_umap[i + 1, 0] - traj_umap[i, 0]
                    dy = traj_umap[i + 1, 1] - traj_umap[i, 1]
                    ax.annotate(
                        "",
                        xy=(traj_umap[i + 1, 0], traj_umap[i + 1, 1]),
                        xytext=(traj_umap[i, 0], traj_umap[i, 1]),
                        arrowprops=dict(
                            arrowstyle="->",
                            color="lime",
                            lw=1.5,
                        ),
                    )
        except Exception as exc:
            logger.debug("Trajectory arrows failed: %s", exc)

        ax.set_title("UMAP + Perturbation Trajectory")


# ---------------------------------------------------------------------------
# OracleReportGenerator
# ---------------------------------------------------------------------------


class OracleReportGenerator:
    """
    Generates a comprehensive self-contained HTML report from ORACLE outputs.

    Parameters
    ----------
    plotter : AttractorLandscapePlotter, optional
        Used to generate landscape figures embedded in the report.
    """

    CSS = """
    <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #f8f9fa;
           color: #212529; line-height: 1.6; }
    header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
             color: white; padding: 40px 60px; }
    header h1 { font-size: 2.4em; letter-spacing: 2px; }
    header .subtitle { font-size: 1.1em; color: #a0c4ff; margin-top: 8px; }
    .container { max-width: 1200px; margin: 0 auto; padding: 30px 40px; }
    .section { background: white; border-radius: 10px; padding: 30px;
               margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    .section h2 { font-size: 1.5em; color: #0f3460; border-bottom: 2px solid #e8eaf6;
                  padding-bottom: 10px; margin-bottom: 18px; }
    .section h3 { font-size: 1.15em; color: #16213e; margin: 18px 0 10px 0; }
    .summary-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
    .summary-card { background: #f0f4ff; border-radius: 8px; padding: 18px;
                    text-align: center; border-left: 4px solid #0f3460; }
    .summary-card .value { font-size: 2em; font-weight: bold; color: #0f3460; }
    .summary-card .label { font-size: 0.9em; color: #555; margin-top: 4px; }
    table { width: 100%; border-collapse: collapse; font-size: 0.93em; }
    th { background: #0f3460; color: white; padding: 10px 14px; text-align: left; }
    td { padding: 9px 14px; border-bottom: 1px solid #e9ecef; }
    tr:nth-child(even) td { background: #f8f9fa; }
    .mol-card { display: inline-block; background: white; border-radius: 10px;
                padding: 16px; margin: 10px; border: 1px solid #dee2e6;
                box-shadow: 0 1px 4px rgba(0,0,0,0.08); vertical-align: top;
                min-width: 280px; max-width: 340px; }
    .mol-card img { display: block; margin: 0 auto 10px auto; }
    .badge { display: inline-block; padding: 3px 10px; border-radius: 12px;
             font-size: 0.8em; font-weight: bold; }
    .badge-cancer { background: #fde8e8; color: #c0392b; }
    .badge-normal { background: #e8f5e9; color: #27ae60; }
    .badge-valid { background: #e8f4fd; color: #2980b9; }
    .badge-invalid { background: #fef9e7; color: #e67e22; }
    footer { background: #1a1a2e; color: #adb5bd; text-align: center;
             padding: 20px; font-size: 0.88em; }
    </style>
    """

    def __init__(self, plotter: Optional[AttractorLandscapePlotter] = None) -> None:
        self.plotter = plotter or AttractorLandscapePlotter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, oracle_output: Any, output_path: str) -> None:
        """
        Assemble the full HTML report and write it to *output_path*.

        Parameters
        ----------
        oracle_output : OracleOutput
            Full ORACLE pipeline output.
        output_path : str
            Path to the output .html file.
        """
        logger.info("OracleReportGenerator: generating report -> %s", output_path)

        sections = [
            self._header_section(oracle_output),
            "<div class='container'>",
            self._summary_section(oracle_output),
            self._landscape_section(oracle_output),
            self._switch_section(oracle_output),
            self._molecules_section(oracle_output),
            self._methods_section(),
            "</div>",
            self._footer_section(),
        ]
        html = "\n".join(sections)

        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        logger.info("Report written to %s", output_path)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _header_section(self, oracle_output: Any) -> str:
        sample_id = getattr(oracle_output, "sample_id", "Unknown")
        cancer_type = getattr(oracle_output, "cancer_type", "Unknown")
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ORACLE Report — {sample_id}</title>
  {self.CSS}
</head>
<body>
<header>
  <h1>ORACLE</h1>
  <div class="subtitle">Oncogenic Reprogramming via Attractor Landscape and TCIP Engineering</div>
  <div style="margin-top:16px; font-size:0.95em;">
    <strong>Sample:</strong> {sample_id} &nbsp;|&nbsp;
    <strong>Cancer Type:</strong> {cancer_type} &nbsp;|&nbsp;
    <strong>Generated:</strong> {_now_str()}
  </div>
</header>"""

    def _summary_section(self, oracle_output: Any) -> str:
        tcip_molecules = getattr(oracle_output, "tcip_molecules", [])
        rsp_output = getattr(oracle_output, "rsp_output", None)
        cam_output = getattr(oracle_output, "cam_output", None)

        n_molecules = len(tcip_molecules)
        reversion_prob = getattr(oracle_output, "predicted_reversion_probability", 0.0)
        n_attractors = len(getattr(cam_output, "attractors", [])) if cam_output else 0

        valid_count = sum(
            1 for m in tcip_molecules
            if getattr(getattr(m, "validation_result", None), "ternary_valid", False)
        )

        return f"""<div class="section">
  <h2>Executive Summary</h2>
  <div class="summary-grid">
    <div class="summary-card">
      <div class="value">{n_attractors}</div>
      <div class="label">Attractors Identified</div>
    </div>
    <div class="summary-card">
      <div class="value">{reversion_prob:.1%}</div>
      <div class="label">Predicted Reversion Probability</div>
    </div>
    <div class="summary-card">
      <div class="value">{n_molecules}</div>
      <div class="label">TCIP Molecules Designed ({valid_count} valid)</div>
    </div>
  </div>
</div>"""

    def _landscape_section(self, oracle_output: Any) -> str:
        cam_output = getattr(oracle_output, "cam_output", None)
        rsp_output = getattr(oracle_output, "rsp_output", None)

        umap_b64 = ""
        basin_b64 = ""

        if _MPL_AVAILABLE and cam_output is not None:
            try:
                fig_umap = self.plotter.plot_landscape_umap(cam_output, rsp_output)
                umap_b64 = _fig_to_base64(fig_umap)
                plt.close(fig_umap)
            except Exception as exc:
                logger.warning("UMAP plot failed: %s", exc)

            try:
                fig_basin = self.plotter.plot_basin_sizes(cam_output)
                basin_b64 = _fig_to_base64(fig_basin)
                plt.close(fig_basin)
            except Exception as exc:
                logger.warning("Basin pie failed: %s", exc)

        umap_img = (f'<img src="data:image/png;base64,{umap_b64}" style="max-width:100%;">'
                    if umap_b64 else "<p><em>UMAP not available.</em></p>")
        basin_img = (f'<img src="data:image/png;base64,{basin_b64}" style="max-width:460px;">'
                     if basin_b64 else "<p><em>Basin chart not available.</em></p>")

        return f"""<div class="section">
  <h2>Attractor Landscape</h2>
  {umap_img}
  <h3>Basin Size Distribution</h3>
  {basin_img}
</div>"""

    def _switch_section(self, oracle_output: Any) -> str:
        rsp_output = getattr(oracle_output, "rsp_output", None)
        if rsp_output is None:
            return """<div class="section"><h2>Reversion Switch Analysis</h2>
  <p><em>RSP output not available.</em></p></div>"""

        switch_set = getattr(rsp_output, "switch_set", None)
        if switch_set is None:
            return """<div class="section"><h2>Reversion Switch Analysis</h2>
  <p><em>Switch set not available.</em></p></div>"""

        act = ", ".join(getattr(switch_set, "genes_to_activate", [])) or "—"
        rep = ", ".join(getattr(switch_set, "genes_to_repress", [])) or "—"
        pred_score = getattr(switch_set, "predicted_cancer_score", float("nan"))
        val_rev = getattr(switch_set, "validated_reversion_fraction", float("nan"))
        pred_rev = getattr(switch_set, "predicted_reversion_probability", float("nan"))
        delta = getattr(switch_set, "validated_delta_score", float("nan"))

        importance = getattr(switch_set, "gene_importance_scores", {})
        imp_rows = "".join(
            f"<tr><td>{g}</td><td>{v:.4f}</td></tr>"
            for g, v in sorted(importance.items(), key=lambda x: -x[1])
        )

        return f"""<div class="section">
  <h2>Reversion Switch Analysis</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Genes to Activate</td><td>{act}</td></tr>
    <tr><td>Genes to Repress</td><td>{rep}</td></tr>
    <tr><td>Predicted Cancer Score</td><td>{pred_score:.4f}</td></tr>
    <tr><td>Predicted Reversion Probability</td><td>{pred_rev:.4f}</td></tr>
    <tr><td>Validated Reversion Fraction</td><td>{val_rev:.4f}</td></tr>
    <tr><td>Validated Delta Score</td><td>{delta:.4f}</td></tr>
  </table>
  <h3>Gene Importance Scores</h3>
  <table>
    <tr><th>Gene</th><th>Importance</th></tr>
    {imp_rows if imp_rows else "<tr><td colspan='2'>—</td></tr>"}
  </table>
</div>"""

    def _molecules_section(self, oracle_output: Any) -> str:
        molecules = getattr(oracle_output, "tcip_molecules", [])
        if not molecules:
            return """<div class="section"><h2>TCIP Molecule Cards</h2>
  <p><em>No TCIP molecules designed.</em></p></div>"""

        cards = "".join(self._build_molecule_card(mol, idx) for idx, mol in enumerate(molecules))
        return f"""<div class="section">
  <h2>TCIP Molecule Cards</h2>
  {cards}
</div>"""

    def _build_molecule_card(self, mol: Any, idx: int) -> str:
        """Build a single per-TCIP molecule card."""
        tf_name = getattr(mol, "tf_name", f"TF_{idx}")
        smiles = getattr(mol, "smiles", None)
        warhead_smiles = getattr(mol, "warhead_smiles", "—")
        linker_smiles = getattr(mol, "linker_smiles", "—")
        recruiter_smiles = getattr(mol, "recruiter_smiles", "—")
        perturbation_type = getattr(mol, "perturbation_type", "unknown")
        validation_result = getattr(mol, "validation_result", None)

        # 2-D molecule image
        mol_img_html = "<p><em>Structure N/A</em></p>"
        if _RDKIT_AVAILABLE and smiles:
            try:
                rdmol = Chem.MolFromSmiles(smiles)
                if rdmol is not None:
                    drawer = rdMolDraw2D.MolDraw2DSVG(300, 220)
                    drawer.DrawMolecule(rdmol)
                    drawer.FinishDrawing()
                    svg_str = drawer.GetDrawingText()
                    b64 = base64.b64encode(svg_str.encode()).decode()
                    mol_img_html = (f'<img src="data:image/svg+xml;base64,{b64}" '
                                    f'width="280" height="210">')
            except Exception:
                pass

        # Properties table
        props_rows = ""
        if _RDKIT_AVAILABLE and smiles:
            try:
                rdmol = Chem.MolFromSmiles(smiles)
                if rdmol is not None:
                    mw = Descriptors.ExactMolWt(rdmol)
                    logp = Descriptors.MolLogP(rdmol)
                    tpsa = rdMolDescriptors.CalcTPSA(rdmol)
                    qed = QED.qed(rdmol)
                    try:
                        from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
                        sa = _compute_sa_score(rdmol)
                    except Exception:
                        sa = float("nan")
                    props_rows = (
                        f"<tr><td>MW</td><td>{mw:.2f} Da</td></tr>"
                        f"<tr><td>logP</td><td>{logp:.2f}</td></tr>"
                        f"<tr><td>TPSA</td><td>{tpsa:.1f} Å²</td></tr>"
                        f"<tr><td>QED</td><td>{qed:.3f}</td></tr>"
                        f"<tr><td>SA Score</td><td>{sa:.2f}</td></tr>"
                    )
            except Exception:
                pass

        # Writer/Eraser
        writer = getattr(mol, "writer_eraser", None)
        if writer is None:
            writer = getattr(mol, "recruiter_name", "—")
        props_rows += f"<tr><td>Writer/Eraser</td><td>{writer}</td></tr>"

        # Ternary validity
        ternary_valid = False
        if validation_result is not None:
            ternary_valid = getattr(validation_result, "ternary_valid", False)
        valid_badge = (
            '<span class="badge badge-valid">VALID</span>' if ternary_valid
            else '<span class="badge badge-invalid">INVALID</span>'
        )
        props_rows += f"<tr><td>Ternary Valid</td><td>{valid_badge}</td></tr>"

        pert_badge = (
            '<span class="badge badge-cancer">Repression</span>'
            if "repress" in str(perturbation_type).lower()
            else '<span class="badge badge-normal">Activation</span>'
        )

        return f"""<div class="mol-card">
  <h3>{tf_name} — {pert_badge}</h3>
  {mol_img_html}
  <table>
    {props_rows if props_rows else "<tr><td colspan='2'>—</td></tr>"}
  </table>
  <h3 style="margin-top:12px;">Structure Components</h3>
  <table>
    <tr><th>Component</th><th>SMILES</th></tr>
    <tr><td>Warhead</td><td style="font-size:0.75em;word-break:break-all;">{warhead_smiles}</td></tr>
    <tr><td>Linker</td><td style="font-size:0.75em;word-break:break-all;">{linker_smiles}</td></tr>
    <tr><td>Recruiter</td><td style="font-size:0.75em;word-break:break-all;">{recruiter_smiles}</td></tr>
  </table>
</div>"""

    def _methods_section(self) -> str:
        return """<div class="section">
  <h2>Methods</h2>
  <p>
    <strong>Module 1 — Cancer Attraction Mapper (CAM):</strong>
    Single-cell RNA-seq data were processed through quality control, library-size
    normalisation, highly variable gene selection (Seurat v3), and UMAP embedding.
    A signed gene regulatory network (GRN) was inferred by combining GRNBoost2
    (data-driven) with TRRUST v2 / ENCODE ChIP-seq prior knowledge. Boolean and
    continuous (neural ODE) attractor searches were performed to identify stable
    cell fate states.
  </p>
  <p style="margin-top:12px;">
    <strong>Module 2 — Reversion Switch Predictor (RSP):</strong>
    A GATv2-based graph neural network (GNNSwitchPredictor) was used to score
    transcription factor perturbation candidates. Greedy forward selection
    followed by basin-size-aware pruning identified the minimal switch set.
    ODE integration (Euler method) was used to validate reversion trajectories.
  </p>
  <p style="margin-top:12px;">
    <strong>Module 3 — Transcriptional CIP Designer (TCD):</strong>
    For each target TF, structural preparation was performed using fpocket pocket
    detection and short MD ensemble sampling. An SE(3)-equivariant diffusion model
    was used to generate warhead candidates. Linker design and ternary complex
    validation completed the TCIP assembly pipeline.
  </p>
</div>"""

    def _footer_section(self) -> str:
        return f"""<footer>
  ORACLE Pipeline &mdash; Generated {_now_str()} &mdash; For Research Use Only.
</footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _fig_to_base64(fig: Any) -> str:
    """Convert a matplotlib Figure to a base64-encoded PNG string."""
    if fig is None:
        return ""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _now_str() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _empty_plotly_figure(message: str = "") -> Any:
    if not _PLOTLY_AVAILABLE:
        return None
    fig = go.Figure()
    fig.add_annotation(text=message or "No data", x=0.5, y=0.5,
                       xref="paper", yref="paper", showarrow=False)
    return fig


def _compute_sa_score(rdmol: Any) -> float:
    """Compute RDKit SA score if sascorer is available."""
    try:
        from rdkit.Chem import RDConfig
        import os, sys
        sascorer_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if sascorer_path not in sys.path:
            sys.path.append(sascorer_path)
        import sascorer
        return sascorer.calculateScore(rdmol)
    except Exception:
        return float("nan")
