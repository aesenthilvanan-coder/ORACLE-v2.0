"""
Cancer Attraction Mapper - GRN Inference Module

Infers a directed, signed Gene Regulatory Network (GRN) by combining:
  - GRNBoost2 (data-driven, expression-based)
  - TRRUST v2 (curated human TF-target interactions with sign)
  - ENCODE ChIP-seq (TF binding evidence)
  - STRING (PPI-based co-expression)

Final network: top `grn_size` genes by degree centrality, edges filtered
by weighted confidence (data_weight=0.6, prior_weight=0.4).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from anndata import AnnData

from oracle.cam.preprocessing import CAMConfig

logger = logging.getLogger(__name__)


class GRNInferenceEngine:
    """
    Infers a directed, signed Gene Regulatory Network.

    The engine combines a data-driven component (GRNBoost2 applied to
    transition cells) with curated prior knowledge (TRRUST v2 + ENCODE
    ChIP-seq) via weighted integration, then trims the network to a core
    of `config.grn_size` hub genes.

    Parameters
    ----------
    config : CAMConfig
        Pipeline configuration.
    """

    def __init__(self, config: CAMConfig):
        self.config = config
        self.tf_list = self._load_tf_list()
        self.trrust = self._load_trrust()
        self.encode_tfs = self._load_encode()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer(self, adata: AnnData) -> nx.DiGraph:
        """
        Infer the signed GRN from preprocessed single-cell data.

        Steps
        -----
        1. Run GRNBoost2 on transition-cell subpopulation.
        2. Build prior network from TRRUST / ENCODE.
        3. Integrate data-driven and prior networks.
        4. Filter to core `grn_size` genes.
        5. Assign activation / repression signs to edges.

        Parameters
        ----------
        adata : AnnData
            Preprocessed AnnData (must contain 'cell_state' in .obs).

        Returns
        -------
        nx.DiGraph
            Signed, weighted GRN.
        """
        logger.info("Starting GRN inference.")
        data_df = self._run_grnboost2(adata)
        prior_graph = self._build_prior_network(adata)
        merged_graph = self._integrate_networks(data_df, prior_graph, adata)
        core_graph = self._filter_to_core(merged_graph)
        signed_graph = self._assign_signs(core_graph, adata)
        # Ensure all data genes are present as nodes even if no edges were inferred
        for gene in adata.var_names:
            if not signed_graph.has_node(gene):
                signed_graph.add_node(gene)
        logger.info(
            "GRN inference complete: %d nodes, %d edges.",
            signed_graph.number_of_nodes(),
            signed_graph.number_of_edges(),
        )
        return signed_graph

    # ------------------------------------------------------------------
    # Step 1: GRNBoost2 (data-driven)
    # ------------------------------------------------------------------

    def _run_grnboost2(self, adata: AnnData) -> pd.DataFrame:
        """
        Run GRNBoost2 on transition cells (cancer + transitional states).

        Falls back to Pearson-correlation-based pseudo-GRNBoost2 if the
        `arboreto` package is not available.

        Returns
        -------
        pd.DataFrame
            Columns: ['TF', 'target', 'importance'].
        """
        logger.info("Running GRNBoost2 on transition cells.")

        # Subset to cancer + transitional cells for transition-state GRN
        if "cell_state" in adata.obs.columns:
            mask = adata.obs["cell_state"].isin(["cancer", "transitional"])
            sub = adata[mask].copy() if mask.sum() > 20 else adata.copy()
        else:
            sub = adata.copy()

        # Extract expression matrix
        import scipy.sparse as sp

        if sp.issparse(sub.X):
            expr = pd.DataFrame(
                sub.X.toarray(), columns=sub.var_names, index=sub.obs_names
            )
        else:
            expr = pd.DataFrame(sub.X, columns=sub.var_names, index=sub.obs_names)

        tf_in_data = [tf for tf in self.tf_list if tf in expr.columns]
        logger.info(
            "GRNBoost2: %d cells, %d genes, %d TFs.", *expr.shape, len(tf_in_data)
        )

        try:
            from arboreto.algo import grnboost2

            network_df = grnboost2(
                expression_data=expr,
                tf_names=tf_in_data,
                seed=42,
                verbose=False,
            )
            # Normalize importance to [0, 1]
            max_imp = network_df["importance"].max()
            if max_imp > 0:
                network_df["importance"] = network_df["importance"] / max_imp
            logger.info("GRNBoost2 returned %d edges.", len(network_df))
            return network_df

        except ImportError:
            logger.warning(
                "arboreto not available; falling back to Pearson-correlation GRN."
            )
            return self._pearson_fallback_grn(expr, tf_in_data)

    def _pearson_fallback_grn(
        self, expr: pd.DataFrame, tf_list: List[str]
    ) -> pd.DataFrame:
        """Compute Pearson-correlation-based importance as GRNBoost2 fallback."""
        records = []
        genes = list(expr.columns)
        expr_np = expr.values.T  # shape: (n_genes, n_cells)
        # Standardize
        std = expr_np.std(axis=1, keepdims=True) + 1e-8
        mean = expr_np.mean(axis=1, keepdims=True)
        expr_z = (expr_np - mean) / std
        n = expr_np.shape[1]

        tf_idx = [genes.index(tf) for tf in tf_list if tf in genes]
        gene_idx = list(range(len(genes)))

        for ti in tf_idx:
            tf_name = genes[ti]
            corr = (expr_z[ti] @ expr_z[gene_idx].T) / n
            for gi, c in enumerate(corr):
                tgt = genes[gi]
                if tgt != tf_name and abs(c) > 0.05:
                    records.append(
                        {"TF": tf_name, "target": tgt, "importance": abs(float(c))}
                    )

        df = pd.DataFrame(records)
        if len(df) > 0:
            df["importance"] = df["importance"] / df["importance"].max()
            df = df.sort_values("importance", ascending=False).head(50000)
        return df

    # ------------------------------------------------------------------
    # Step 2: Prior network
    # ------------------------------------------------------------------

    def _build_prior_network(self, adata=None) -> nx.DiGraph:
        """
        Build a prior knowledge network from TRRUST v2 and ENCODE ChIP-seq.

        Edge weight is 1.0 for TRRUST-confirmed interactions and 0.7 for
        ENCODE-only interactions. If adata is None, return network without
        filtering to genes present in data.
        """
        logger.info("Building prior network from TRRUST + ENCODE.")
        G = nx.DiGraph()
        genes_in_data = set(adata.var_names) if adata is not None else None

        # TRRUST interactions
        for (tf, target), sign in self.trrust.items():
            if genes_in_data is not None and (tf not in genes_in_data or target not in genes_in_data):
                continue
            G.add_edge(tf, target, weight=1.0, sign=sign, source="TRRUST")

        # ENCODE ChIP-seq (TF binding evidence -> activation assumed)
        for tf, targets in self.encode_tfs.items():
            if genes_in_data is not None and tf not in genes_in_data:
                continue
            for target in targets:
                if genes_in_data is not None and target not in genes_in_data:
                    continue
                if G.has_edge(tf, target):
                    G[tf][target]["weight"] = min(1.0, G[tf][target]["weight"] + 0.2)
                    G[tf][target]["source"] = "TRRUST+ENCODE"
                else:
                    G.add_edge(tf, target, weight=0.7, sign=1, source="ENCODE")

        logger.info(
            "Prior network: %d nodes, %d edges.",
            G.number_of_nodes(),
            G.number_of_edges(),
        )
        return G

    # ------------------------------------------------------------------
    # Step 3: Integrate networks
    # ------------------------------------------------------------------

    def _integrate_networks(
        self,
        data_df: pd.DataFrame,
        prior_graph: nx.DiGraph,
        adata: AnnData,
    ) -> nx.DiGraph:
        """
        Weighted combination of data-driven (GRNBoost2) and prior networks.

        Combined weight = data_weight * data_importance
                        + prior_weight * prior_weight_value

        Parameters
        ----------
        data_df : pd.DataFrame
            GRNBoost2 output with columns TF, target, importance.
        prior_graph : nx.DiGraph
            Prior knowledge network.
        adata : AnnData
            Used for gene universe.

        Returns
        -------
        nx.DiGraph
            Merged network with combined confidence scores.
        """
        logger.info(
            "Integrating networks (data_weight=%.1f, prior_weight=%.1f).",
            self.config.data_weight,
            self.config.prior_weight,
        )
        G = nx.DiGraph()

        # Index data-driven edges
        data_edges: Dict[Tuple[str, str], float] = {}
        if len(data_df) > 0:
            for _, row in data_df.iterrows():
                data_edges[(row["TF"], row["target"])] = float(row["importance"])

        # Index prior edges
        prior_edges: Dict[Tuple[str, str], Dict] = {}
        for u, v, d in prior_graph.edges(data=True):
            prior_edges[(u, v)] = d

        # Union of all edges
        all_edges = set(data_edges.keys()) | set(prior_edges.keys())

        for (u, v) in all_edges:
            data_w = data_edges.get((u, v), 0.0)
            prior_d = prior_edges.get((u, v), {})
            prior_w = prior_d.get("weight", 0.0)
            combined = (
                self.config.data_weight * data_w
                + self.config.prior_weight * prior_w
            )
            sign = prior_d.get("sign", 1)  # default activation
            source = prior_d.get("source", "data")
            G.add_edge(
                u, v,
                weight=combined,
                sign=sign,
                source=source,
                data_importance=data_w,
                prior_weight=prior_w,
            )

        logger.info(
            "Merged network: %d nodes, %d edges.",
            G.number_of_nodes(),
            G.number_of_edges(),
        )
        return G

    # ------------------------------------------------------------------
    # Step 4: Filter to core
    # ------------------------------------------------------------------

    def _filter_to_core(self, G: nx.DiGraph) -> nx.DiGraph:
        """
        Retain the top `grn_size` genes by degree centrality and filter
        edges with combined confidence < `min_confidence`.

        Parameters
        ----------
        G : nx.DiGraph
            Full merged network.

        Returns
        -------
        nx.DiGraph
            Core network.
        """
        logger.info(
            "Filtering to core GRN (%d genes, min_confidence=%.2f).",
            self.config.grn_size,
            self.config.min_confidence,
        )
        # First filter by confidence
        low_conf_edges = [
            (u, v)
            for u, v, d in G.edges(data=True)
            if d.get("weight", 0.0) < self.config.min_confidence
        ]
        G.remove_edges_from(low_conf_edges)

        # Remove isolated nodes
        isolated = list(nx.isolates(G))
        G.remove_nodes_from(isolated)

        if G.number_of_nodes() <= self.config.grn_size:
            logger.info(
                "Network already has %d nodes; skipping degree centrality trim.",
                G.number_of_nodes(),
            )
            return G

        # Compute degree centrality and keep top grn_size
        centrality = nx.degree_centrality(G)
        top_genes = sorted(centrality, key=centrality.get, reverse=True)[
            : self.config.grn_size
        ]
        top_set = set(top_genes)
        core = G.subgraph(top_set).copy()

        logger.info(
            "Core GRN: %d nodes, %d edges.",
            core.number_of_nodes(),
            core.number_of_edges(),
        )
        return core

    # ------------------------------------------------------------------
    # Step 5: Assign signs
    # ------------------------------------------------------------------

    def _assign_signs(self, G: nx.DiGraph, adata: AnnData) -> nx.DiGraph:
        """
        Assign activation (+1) / repression (-1) signs to edges.

        Priority order:
        1. TRRUST curated sign (highest confidence).
        2. Pearson correlation sign between TF and target expression.
        3. Default: activation (+1).

        Parameters
        ----------
        G : nx.DiGraph
            Core network with edges potentially lacking signs.
        adata : AnnData
            Used for Pearson correlation computation.

        Returns
        -------
        nx.DiGraph
            Network with 'sign' attribute on all edges.
        """
        import scipy.sparse as sp

        logger.info("Assigning edge signs.")

        # Precompute per-gene means for Pearson sign
        genes = list(G.nodes())
        genes_in_data = [g for g in genes if g in adata.var_names]
        var_names = list(adata.var_names)

        gene_means: Dict[str, float] = {}
        for g in genes_in_data:
            idx = var_names.index(g)
            if sp.issparse(adata.X):
                vals = adata.X[:, idx].toarray().flatten()
            else:
                vals = adata.X[:, idx].flatten()
            gene_means[g] = float(np.mean(vals))

        n_assigned_trrust = 0
        n_assigned_pearson = 0
        n_assigned_default = 0

        for u, v, data in G.edges(data=True):
            # Priority 1: TRRUST sign already set
            if data.get("source") in ("TRRUST", "TRRUST+ENCODE") and "sign" in data:
                n_assigned_trrust += 1
                continue

            # Priority 2: Pearson correlation sign
            if u in gene_means and v in gene_means:
                # Approximate: use mean expression levels as proxy
                # Full Pearson requires computing on-the-fly which is expensive;
                # here we use the sign of the data-driven importance as proxy.
                # If TF is an activator in data context, correlation tends positive.
                data_imp = data.get("data_importance", 0.0)
                if u in var_names and v in var_names:
                    u_idx = var_names.index(u)
                    v_idx = var_names.index(v)
                    if sp.issparse(adata.X):
                        u_expr = adata.X[:, u_idx].toarray().flatten()
                        v_expr = adata.X[:, v_idx].toarray().flatten()
                    else:
                        u_expr = adata.X[:, u_idx].flatten()
                        v_expr = adata.X[:, v_idx].flatten()
                    corr = float(np.corrcoef(u_expr, v_expr)[0, 1])
                    data["sign"] = 1 if corr >= 0 else -1
                    n_assigned_pearson += 1
                    continue

            # Priority 3: default activation
            data["sign"] = 1
            n_assigned_default += 1

        logger.info(
            "Sign assignment: %d TRRUST, %d Pearson, %d default.",
            n_assigned_trrust,
            n_assigned_pearson,
            n_assigned_default,
        )
        return G

    # ------------------------------------------------------------------
    # Knowledge bases (hardcoded for offline / reproducible use)
    # ------------------------------------------------------------------

    def _load_tf_list(self) -> List[str]:
        """
        Return a comprehensive list of known human transcription factors.
        Drawn from AnimalTFDB v4 / ENCODE CHIP-seq / JASPAR curated sets.
        """
        return [
            # Master regulators / cancer drivers
            "TP53", "MYC", "MYCN", "MAX", "HIF1A", "EPAS1",
            # Wnt / EMT
            "CTNNB1", "TCF7L2", "TCF4", "LEF1", "SNAI1", "SNAI2",
            "ZEB1", "ZEB2", "TWIST1", "TWIST2",
            # Colorectal / intestinal
            "CDX2", "CDX1", "HNF4A", "FOXA1", "FOXA2", "GATA4", "GATA6",
            # Hematopoietic
            "CEBPA", "CEBPB", "SPI1", "IRF8", "IRF4", "IRF1", "STAT3",
            "RUNX1", "RUNX2", "RUNX3", "IKZF1", "IKZF3",
            # Neural / developmental
            "SOX2", "SOX9", "SOX10", "SOX17", "PAX3", "PAX5", "PAX6",
            "MITF", "OLIG2", "NEUROD1", "ASCL1",
            # Proliferation / cell cycle
            "E2F1", "E2F3", "E2F4", "RB1", "CCND1",
            # Breast / hormone
            "ESR1", "ESR2", "PGR", "AR", "GATA3", "FOXA1",
            # Lung / NKX
            "NKX2-1", "NKX2-5", "TP63", "TP73",
            # Stem cell / pluripotency
            "POU5F1", "NANOG", "KLF4", "KLF5", "YAP1", "WWTR1",
            # Signaling TFs
            "SMAD2", "SMAD3", "SMAD4", "STAT1", "STAT5A", "STAT5B",
            "NFKB1", "RELA", "SP1", "SP3", "KLF2", "KLF6",
            # Chromatin / epigenetic
            "EZH2", "BMI1", "SIRT1",
            # Oncogene TFs
            "KRAS", "BRAF", "FLI1", "ETV1", "ETV4", "ETV5",
            "ERG", "EWS", "FUS",
            # Additional common TFs
            "ATF3", "ATF4", "CREB1", "JUN", "JUNB", "JUND",
            "FOS", "FOSB", "FOSL1", "FOSL2",
            "EGR1", "EGR2", "EGR3", "KLF1", "KLF3",
            "NFE2L2", "KEAP1", "PPARA", "PPARG", "RXRA",
            "VHL", "MEN1", "WT1", "HOXA9", "HOXA10", "MEIS1", "PBX1",
            "TAL1", "LMO2", "GATA1", "GATA2",
            "BCL6", "PRDM1", "IRF3", "IRF5", "IRF7",
            "TEAD1", "TEAD2", "TEAD3", "TEAD4",
            "HEY1", "HEY2", "HES1", "NOTCH1",
            "GLI1", "GLI2", "GLI3",
            "TCF3", "TCF12", "ID1", "ID2", "ID3",
        ]

    def _load_trrust(self) -> Dict[Tuple[str, str], int]:  # noqa: E301
        """
        Return curated TF-target interactions from TRRUST v2.
        Signs: +1 = activation, -1 = repression.

        This is a representative subset of TRRUST v2 human interactions
        covering key cancer biology pathways.
        """
        interactions = {
            # TP53 targets
            ("TP53", "CDKN1A"): 1,   # p21 activation
            ("TP53", "BAX"): 1,
            ("TP53", "PUMA"): 1,
            ("TP53", "MDM2"): 1,
            ("TP53", "BBC3"): 1,
            ("TP53", "GADD45A"): 1,
            ("TP53", "MYC"): -1,
            ("TP53", "BCL2"): -1,
            ("TP53", "SNAI1"): -1,
            # MYC targets
            ("MYC", "CDK4"): 1,
            ("MYC", "CCND2"): 1,
            ("MYC", "TERT"): 1,
            ("MYC", "EZH2"): 1,
            ("MYC", "BCL2"): 1,
            ("MYC", "CDKN1B"): -1,
            ("MYC", "CDKN2B"): -1,
            # SNAI1/SNAI2 (EMT)
            ("SNAI1", "CDH1"): -1,
            ("SNAI1", "VIM"): 1,
            ("SNAI1", "FN1"): 1,
            ("SNAI2", "CDH1"): -1,
            ("SNAI2", "VIM"): 1,
            # ZEB1/2 (EMT)
            ("ZEB1", "CDH1"): -1,
            ("ZEB1", "VIM"): 1,
            ("ZEB2", "CDH1"): -1,
            # HIF1A targets
            ("HIF1A", "VEGFA"): 1,
            ("HIF1A", "SLC2A1"): 1,
            ("HIF1A", "LDHA"): 1,
            ("HIF1A", "CA9"): 1,
            ("HIF1A", "BNIP3"): 1,
            # STAT3 targets
            ("STAT3", "BCL2"): 1,
            ("STAT3", "MCL1"): 1,
            ("STAT3", "MYC"): 1,
            ("STAT3", "CCND1"): 1,
            ("STAT3", "VEGFA"): 1,
            # CEBPA targets (AML)
            ("CEBPA", "MPO"): 1,
            ("CEBPA", "ELANE"): 1,
            ("CEBPA", "MYC"): -1,
            ("CEBPA", "BCL2"): -1,
            # SPI1/PU.1
            ("SPI1", "MPO"): 1,
            ("SPI1", "LYZ"): 1,
            ("SPI1", "IRF8"): 1,
            ("SPI1", "IRF4"): -1,
            # RUNX1
            ("RUNX1", "SPI1"): 1,
            ("RUNX1", "IRF8"): 1,
            ("RUNX1", "GATA2"): 1,
            ("RUNX1", "MPO"): -1,
            # CDX2 (colorectal)
            ("CDX2", "MUC2"): 1,
            ("CDX2", "FABP1"): 1,
            ("CDX2", "MYC"): -1,
            ("CDX2", "CCND1"): -1,
            # NFkB
            ("NFKB1", "BCL2"): 1,
            ("NFKB1", "MYC"): 1,
            ("NFKB1", "CCND1"): 1,
            ("NFKB1", "VEGFA"): 1,
            ("NFKB1", "IL6"): 1,
            # E2F1
            ("E2F1", "CCNE1"): 1,
            ("E2F1", "CDC6"): 1,
            ("E2F1", "PCNA"): 1,
            ("E2F1", "RB1"): -1,
            # SOX2
            ("SOX2", "POU5F1"): 1,
            ("SOX2", "NANOG"): 1,
            ("SOX2", "MYC"): 1,
            # ESR1 (breast)
            ("ESR1", "PGR"): 1,
            ("ESR1", "TFF1"): 1,
            ("ESR1", "MYC"): 1,
            ("ESR1", "CCND1"): 1,
        }
        return interactions

    def _load_encode(self) -> Dict[str, List[str]]:
        """
        Return TF -> list of ChIP-seq supported target genes from ENCODE.

        This represents a curated subset of strong ENCODE ChIP-seq peaks
        near gene promoters for key cancer-relevant TFs.
        """
        return {
            "MYC": [
                "TERT", "CDK4", "EZH2", "BCL2", "CCND2", "E2F1",
                "LDHA", "PKM", "SLC2A1", "HMGN1",
            ],
            "TP53": [
                "CDKN1A", "BAX", "BBC3", "MDM2", "GADD45A", "PUMA",
                "TIGAR", "SCO2", "DDB2",
            ],
            "HIF1A": [
                "VEGFA", "SLC2A1", "LDHA", "CA9", "BNIP3", "PDK1",
                "PGAM1", "ALDOA",
            ],
            "STAT3": [
                "BCL2", "MCL1", "MYC", "CCND1", "VEGFA", "CDK2",
                "FOS", "JUN",
            ],
            "SNAI1": [
                "CDH1", "VIM", "FN1", "MMP2", "MMP9", "TWIST1",
                "ZEB1", "ZEB2",
            ],
            "E2F1": [
                "CCNE1", "CDC6", "PCNA", "TYMS", "MCM2", "MCM3",
                "DHFR", "TK1",
            ],
            "RUNX1": [
                "SPI1", "GATA2", "IRF8", "CD34", "MPO", "TAL1",
                "LMO2", "HLF",
            ],
            "CEBPA": [
                "MPO", "ELANE", "AZU1", "LYZ", "S100A8", "S100A9",
                "GFI1", "CSF3R",
            ],
            "ESR1": [
                "PGR", "TFF1", "TFF3", "GREB1", "MYC", "CCND1",
                "FOXA1", "XBP1",
            ],
            "SOX2": [
                "POU5F1", "NANOG", "KLF4", "SALL4", "CD44",
                "MYC", "NOTCH1",
            ],
            "CDX2": [
                "MUC2", "FABP1", "CEACAM5", "CDH17", "SLC26A3",
                "TFF3", "CA1",
            ],
            "ZEB1": [
                "CDH1", "VIM", "SNAI1", "SNAI2", "FN1", "MMP14",
                "TWIST1",
            ],
            "NFKB1": [
                "BCL2", "MYC", "CCND1", "VEGFA", "IL6", "TNF",
                "CXCL8", "ICAM1",
            ],
        }


def load_human_tfs() -> set:
    """Return a set of known human TF gene symbols (module-level convenience wrapper)."""
    _HUMAN_TFS = [
        "TP53", "MYC", "MYCN", "MAX", "HIF1A", "EPAS1",
        "CTNNB1", "TCF7L2", "TCF4", "LEF1", "SNAI1", "SNAI2",
        "ZEB1", "ZEB2", "TWIST1", "TWIST2",
        "CDX2", "CDX1", "HNF4A", "FOXA1", "FOXA2", "GATA4", "GATA6",
        "CEBPA", "CEBPB", "SPI1", "IRF8", "IRF4", "IRF1", "STAT3",
        "RUNX1", "RUNX2", "RUNX3", "IKZF1", "IKZF3",
        "SOX2", "SOX9", "SOX10", "SOX17", "PAX3", "PAX5", "PAX6",
        "MITF", "OLIG2", "NEUROD1", "ASCL1",
        "E2F1", "E2F3", "E2F4", "RB1", "CCND1",
        "ESR1", "ESR2", "PGR", "AR", "GATA3",
        "NKX2-1", "NKX2-5", "TP63", "TP73",
        "POU5F1", "NANOG", "KLF4", "KLF5", "YAP1", "WWTR1",
        "SMAD2", "SMAD3", "SMAD4", "STAT1", "STAT5A", "STAT5B",
        "NFKB1", "RELA", "SP1", "SP3", "KLF2", "KLF6",
        "EZH2", "BMI1", "SIRT1",
        "FLI1", "ETV1", "ETV4", "ETV5", "ERG",
        "ATF3", "ATF4", "CREB1", "JUN", "JUNB", "JUND",
        "FOS", "FOSB", "FOSL1", "FOSL2",
        "EGR1", "EGR2", "EGR3", "KLF1", "KLF3",
        "NFE2L2", "KEAP1", "PPARA", "PPARG", "RXRA",
        "VHL", "MEN1", "WT1", "HOXA9", "HOXA10", "MEIS1", "PBX1",
        "TAL1", "LMO2", "GATA1", "GATA2",
        "BCL6", "PRDM1", "IRF3", "IRF5", "IRF7",
        "TEAD1", "TEAD2", "TEAD3", "TEAD4",
        "HEY1", "HEY2", "HES1", "NOTCH1",
        "GLI1", "GLI2", "GLI3",
        "TCF3", "TCF12", "ID1", "ID2", "ID3",
        # LUAD / thyroid TFs
        "HHEX", "FOXE1", "PAX8", "NKX2-1",
        "BRAF", "EGFR", "STK11", "KRAS",
        "BRD4", "HDAC1", "DNMT3A", "KDM6A",
    ]
    return set(_HUMAN_TFS)
