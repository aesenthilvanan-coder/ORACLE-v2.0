"""Neural network models for the ORACLE pipeline.

Imports are guarded to allow the package to load even when optional
dependencies (torch_geometric, torch_scatter) are not installed.
"""

from __future__ import annotations

__all__ = [
    "GRNTransformer",
    "SwitchPredictorGNN",
    "TCIPDiffusionModel",
    "AttractorGNN",
    "TernaryComplexPredictor",
]


def __getattr__(name: str):
    if name == "GRNTransformer":
        from oracle.models.grn_transformer import GRNTransformer
        return GRNTransformer
    if name == "SwitchPredictorGNN":
        from oracle.models.switch_predictor_gnn import SwitchPredictorGNN
        return SwitchPredictorGNN
    if name == "TCIPDiffusionModel":
        from oracle.models.molecule_diffusion import TCIPDiffusionModel
        return TCIPDiffusionModel
    if name == "AttractorGNN":
        from oracle.models.attractor_gnn import AttractorGNN
        return AttractorGNN
    if name == "TernaryComplexPredictor":
        from oracle.models.ternary_complex_predictor import TernaryComplexPredictor
        return TernaryComplexPredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
