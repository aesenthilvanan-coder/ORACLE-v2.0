"""
molecule_viz.py
---------------
Visualization tools for TCIP molecules and their structural components.

Class
-----
MoleculeVisualizer : 2-D structure drawings with component annotations,
    molecule grids, and property distribution violin plots.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False
    Figure = object  # type: ignore
    logger.warning("matplotlib not available; MoleculeVisualizer plots disabled.")

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, Descriptors, QED, rdMolDescriptors, AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False
    logger.warning("RDKit not available; 2-D molecule drawings disabled.")


class MoleculeVisualizer:
    """
    Visualizes TCIP molecules and their structural components.

    Parameters
    ----------
    dpi : int
        Figure resolution for matplotlib figures.
    """

    # Colors for the three components (warhead / linker / recruiter)
    COMPONENT_COLORS = {
        "warhead": (0.85, 0.2, 0.2),    # red-ish
        "linker": (0.2, 0.6, 0.85),     # blue-ish
        "recruiter": (0.2, 0.75, 0.35), # green-ish
    }

    def __init__(self, dpi: int = 120) -> None:
        self.dpi = dpi

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw_tcip_structure(self, tcip_molecule: object) -> Figure:
        """
        2-D RDKit drawing of a TCIP molecule with component annotations.

        The warhead, linker, and recruiter sub-structures are highlighted
        in distinct colours.  Falls back to a plain 2-D drawing if
        component SMILES are unavailable.

        Parameters
        ----------
        tcip_molecule : TCIPMolecule
            ORACLE TCIP molecule object with attributes:
            `smiles`, `warhead_smiles`, `linker_smiles`, `recruiter_smiles`,
            `tf_name`, `perturbation_type`.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not _MPL_AVAILABLE:
            logger.error("matplotlib required for draw_tcip_structure.")
            return None  # type: ignore

        smiles = getattr(tcip_molecule, "smiles", None)
        tf_name = getattr(tcip_molecule, "tf_name", "Unknown TF")
        ptype = getattr(tcip_molecule, "perturbation_type", "")

        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)

        if not _RDKIT_AVAILABLE or not smiles:
            ax.text(
                0.5, 0.5,
                f"TCIP for {tf_name}\n({ptype})\nSMILES: {smiles or 'N/A'}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=11, wrap=True,
            )
            ax.axis("off")
            ax.set_title(f"TCIP: {tf_name} ({ptype})", fontsize=13)
            fig.tight_layout()
            return fig

        try:
            rdmol = Chem.MolFromSmiles(smiles)
            if rdmol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")

            # Try to highlight sub-structure atoms by component
            highlight_atoms: dict = {}
            highlight_bonds: dict = {}
            highlight_atom_radius: dict = {}

            for comp_name, comp_attr in [
                ("warhead", "warhead_smiles"),
                ("linker", "linker_smiles"),
                ("recruiter", "recruiter_smiles"),
            ]:
                comp_smiles = getattr(tcip_molecule, comp_attr, None)
                if not comp_smiles:
                    continue
                try:
                    sub = Chem.MolFromSmiles(comp_smiles)
                    if sub is None:
                        continue
                    match = rdmol.GetSubstructMatch(sub)
                    color = self.COMPONENT_COLORS[comp_name]
                    for idx in match:
                        highlight_atoms[idx] = color
                        highlight_atom_radius[idx] = 0.4
                except Exception:
                    pass

            # Draw using RDKit SVG -> embed in matplotlib axes
            drawer = rdMolDraw2D.MolDraw2DSVG(600, 420)
            if highlight_atoms:
                drawer.drawOptions().addAtomIndices = False
                atom_list = list(highlight_atoms.keys())
                atom_colors = {k: v for k, v in highlight_atoms.items()}
                bond_colors: dict = {}
                # Highlight bonds between highlighted atoms
                for bond in rdmol.GetBonds():
                    a1 = bond.GetBeginAtomIdx()
                    a2 = bond.GetEndAtomIdx()
                    if a1 in atom_colors and a2 in atom_colors:
                        bond_colors[bond.GetIdx()] = atom_colors[a1]

                drawer.DrawMolecule(
                    rdmol,
                    highlightAtoms=atom_list,
                    highlightAtomColors=atom_colors,
                    highlightBonds=list(bond_colors.keys()),
                    highlightBondColors=bond_colors,
                    highlightAtomRadii=highlight_atom_radius,
                )
            else:
                drawer.DrawMolecule(rdmol)

            drawer.FinishDrawing()
            svg_str = drawer.GetDrawingText()

            # Convert SVG -> PNG via cairosvg if available, else display placeholder
            try:
                import cairosvg
                import io
                png_buf = io.BytesIO()
                cairosvg.svg2png(bytestring=svg_str.encode(), write_to=png_buf)
                png_buf.seek(0)
                img_arr = plt.imread(png_buf)
                ax.imshow(img_arr)
            except ImportError:
                # Embed SVG as text description
                ax.text(
                    0.5, 0.5,
                    f"TCIP for {tf_name}\n(install cairosvg for rendered image)\n{smiles[:60]}...",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, wrap=True,
                )

        except Exception as exc:
            logger.warning("draw_tcip_structure failed: %s", exc)
            ax.text(
                0.5, 0.5,
                f"Error rendering {tf_name}: {exc}",
                ha="center", va="center", transform=ax.transAxes, fontsize=9,
            )

        ax.axis("off")

        # Legend for component colours
        legend_handles = [
            plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor=rgb,
                markersize=10,
                label=name.capitalize(),
            )
            for name, rgb in self.COMPONENT_COLORS.items()
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower right",
            fontsize=9,
            framealpha=0.8,
        )

        ax.set_title(
            f"TCIP: {tf_name} — {ptype.title()}",
            fontsize=13,
            fontweight="bold",
        )
        fig.tight_layout()
        return fig

    def draw_molecule_grid(
        self,
        molecules: List,
        n_cols: int = 4,
    ) -> Figure:
        """
        Grid of 2-D molecule drawings.

        Parameters
        ----------
        molecules : List[TCIPMolecule]
            ORACLE TCIP molecule objects, each with a `.smiles` attribute.
        n_cols : int
            Number of columns in the grid.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not _MPL_AVAILABLE:
            logger.error("matplotlib required for draw_molecule_grid.")
            return None  # type: ignore

        n = len(molecules)
        if n == 0:
            fig, ax = plt.subplots(figsize=(6, 2), dpi=self.dpi)
            ax.text(0.5, 0.5, "No molecules to display", ha="center", va="center",
                    transform=ax.transAxes)
            ax.axis("off")
            return fig

        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(n_cols * 3.5, n_rows * 3.2),
            dpi=self.dpi,
        )

        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]

        axes_flat = [axes[r][c] for r in range(n_rows) for c in range(n_cols)]

        for i, mol in enumerate(molecules):
            ax = axes_flat[i]
            smiles = getattr(mol, "smiles", None)
            tf_name = getattr(mol, "tf_name", f"Mol {i}")
            ptype = getattr(mol, "perturbation_type", "")

            if _RDKIT_AVAILABLE and smiles:
                try:
                    rdmol = Chem.MolFromSmiles(smiles)
                    if rdmol is not None:
                        AllChem.Compute2DCoords(rdmol)
                        img = Draw.MolToImage(rdmol, size=(280, 210))
                        ax.imshow(np.array(img))
                        ax.axis("off")
                        ax.set_title(f"{tf_name}\n{ptype}", fontsize=8, pad=3)
                        continue
                except Exception:
                    pass

            ax.text(
                0.5, 0.5,
                f"{tf_name}\n{smiles[:40] if smiles else 'N/A'}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=7, wrap=True,
            )
            ax.axis("off")
            ax.set_title(f"{tf_name}", fontsize=8)

        # Hide empty axes
        for i in range(n, len(axes_flat)):
            axes_flat[i].axis("off")

        fig.suptitle("TCIP Molecule Gallery", fontsize=14, fontweight="bold", y=1.01)
        fig.tight_layout()
        return fig

    def plot_property_distributions(self, tcip_molecules: List) -> Figure:
        """
        Violin plots of MW, QED, SA Score, and logP across TCIP molecules.

        Parameters
        ----------
        tcip_molecules : List[TCIPMolecule]
            Collection of ORACLE TCIP molecule objects.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if not _MPL_AVAILABLE:
            logger.error("matplotlib required for plot_property_distributions.")
            return None  # type: ignore

        properties: dict = {"MW": [], "QED": [], "SA Score": [], "logP": []}

        for mol in tcip_molecules:
            smiles = getattr(mol, "smiles", None)
            if not smiles or not _RDKIT_AVAILABLE:
                continue
            try:
                rdmol = Chem.MolFromSmiles(smiles)
                if rdmol is None:
                    continue
                properties["MW"].append(Descriptors.ExactMolWt(rdmol))
                properties["QED"].append(QED.qed(rdmol))
                properties["logP"].append(Descriptors.MolLogP(rdmol))
                try:
                    from rdkit.Chem import RDConfig
                    import os, sys
                    sascorer_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
                    if sascorer_path not in sys.path:
                        sys.path.append(sascorer_path)
                    import sascorer
                    properties["SA Score"].append(sascorer.calculateScore(rdmol))
                except Exception:
                    properties["SA Score"].append(float("nan"))
            except Exception:
                continue

        n_props = 4
        fig, axes = plt.subplots(1, n_props, figsize=(14, 5), dpi=self.dpi)
        prop_names = list(properties.keys())
        colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

        for i, (prop, ax, color) in enumerate(zip(prop_names, axes, colors)):
            vals = [v for v in properties[prop] if not np.isnan(v)]
            if len(vals) >= 2:
                parts = ax.violinplot(vals, showmedians=True, showmeans=False)
                for pc in parts["bodies"]:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
                parts["cmedians"].set_color("black")
                ax.scatter(
                    np.ones(len(vals)) + np.random.uniform(-0.05, 0.05, len(vals)),
                    vals,
                    s=20,
                    color=color,
                    alpha=0.8,
                    zorder=5,
                )
            elif len(vals) == 1:
                ax.axhline(vals[0], color=color, lw=2)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)

            ax.set_title(prop, fontsize=12, fontweight="bold")
            ax.set_xticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Reference lines
            if prop == "MW":
                ax.axhline(500, color="gray", linestyle="--", lw=1, alpha=0.6,
                           label="Ro5 limit (500)")
                ax.legend(fontsize=7)
            elif prop == "logP":
                ax.axhline(5, color="gray", linestyle="--", lw=1, alpha=0.6,
                           label="Ro5 limit (5)")
                ax.legend(fontsize=7)
            elif prop == "SA Score":
                ax.axhline(6, color="gray", linestyle="--", lw=1, alpha=0.6,
                           label="Threshold (6)")
                ax.legend(fontsize=7)

        fig.suptitle(
            "TCIP Molecule Property Distributions",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig
