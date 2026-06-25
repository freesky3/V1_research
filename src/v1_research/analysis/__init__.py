"""Post-simulation analysis tools."""

from v1_research.analysis.communities import CommunityResult, LouvainConfig, identify_communities
from v1_research.analysis.diagnostics import graph_health_diagnostics, selection_funnel, unclassified_diagnostics
from v1_research.analysis.osi import compute_osi
from v1_research.analysis.overlap import LabelOverlapResult, compare_label_sets, overlap_significance
from v1_research.analysis.pipeline import AnalysisConfig, AnalysisInputs, AnalysisResult, run_analysis
from v1_research.analysis.robustness import run_louvain_parameter_grid, summarize_robustness
from v1_research.analysis.temporal import run_window_analysis

__all__ = [
    "AnalysisConfig",
    "AnalysisInputs",
    "AnalysisResult",
    "CommunityResult",
    "LabelOverlapResult",
    "LouvainConfig",
    "compare_label_sets",
    "compute_osi",
    "graph_health_diagnostics",
    "identify_communities",
    "overlap_significance",
    "run_analysis",
    "run_louvain_parameter_grid",
    "run_window_analysis",
    "selection_funnel",
    "summarize_robustness",
    "unclassified_diagnostics",
]
