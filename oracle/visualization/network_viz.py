from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


def plot_grn(
    grn,
    genes: List[str],
    gene_importance: Optional[Dict[str, float]] = None,
    highlight_genes: Optional[List[str]] = None,
    max_edges: int = 200,
    output_path: Optional[str] = None,
) -> None:
    """Visualize a gene regulatory network with optional gene importance coloring."""
    try:
        import matplotlib.pyplot as plt
        import networkx as nx

        fig, ax = plt.subplots(figsize=(14, 10))

        edges_by_weight = sorted(grn.edges(data=True), key=lambda e: abs(e[2].get("weight", 0)), reverse=True)
        top_edges = edges_by_weight[:max_edges]
        sub = nx.DiGraph()
        sub.add_edges_from([(u, v) for u, v, _ in top_edges])

        pos = nx.spring_layout(sub, k=2.0, seed=42)

        node_colors = []
        for n in sub.nodes():
            if highlight_genes and n in highlight_genes:
                node_colors.append("red")
            elif gene_importance and n in gene_importance:
                imp = gene_importance[n]
                node_colors.append(plt.cm.YlOrRd(imp))
            else:
                node_colors.append("steelblue")

        edge_colors = []
        for u, v, d in top_edges:
            if u in sub and v in sub:
                edge_colors.append("green" if d.get("sign", 1) > 0 else "red")

        nx.draw_networkx_nodes(sub, pos, node_color=node_colors, node_size=300, ax=ax)
        nx.draw_networkx_labels(sub, pos, font_size=7, ax=ax)
        nx.draw_networkx_edges(
            sub, pos,
            edge_color=edge_colors[:len(list(sub.edges()))],
            arrows=True,
            arrowsize=10,
            ax=ax,
            alpha=0.6,
        )

        ax.set_title(f"Gene Regulatory Network ({sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges)")
        ax.axis("off")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"GRN plot saved to {output_path}")
        else:
            plt.show()
        plt.close()

    except ImportError:
        logger.warning("matplotlib/networkx not available for GRN visualization")


def plot_attractor_network(
    attractors: List[np.ndarray],
    labels: List[str],
    basin_sizes: Dict[tuple, float],
    genes: List[str],
    output_path: Optional[str] = None,
) -> None:
    """Plot attractor states as a heatmap."""
    try:
        import matplotlib.pyplot as plt

        if not attractors:
            return

        n_attractors = len(attractors)
        n_genes = min(len(genes), 50)

        data = np.stack([a[:n_genes] for a in attractors])
        fig, ax = plt.subplots(figsize=(max(12, n_genes // 3), max(4, n_attractors)))
        im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=0, vmax=1)
        ax.set_yticks(range(n_attractors))
        ax.set_yticklabels(labels)
        ax.set_xticks(range(n_genes))
        ax.set_xticklabels(genes[:n_genes], rotation=90, fontsize=7)
        ax.set_title("Attractor States (gene expression)")
        plt.colorbar(im, ax=ax, label="Expression level")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    except ImportError:
        logger.warning("matplotlib not available for attractor visualization")
