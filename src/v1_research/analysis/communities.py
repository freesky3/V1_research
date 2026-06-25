"""Consensus Louvain community detection for activity traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import bct
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import pdist, squareform

from v1_research.analysis.clusters import relabel_consecutive

SimilarityKind = Literal["cosine", "pearson"]


@dataclass(frozen=True, slots=True)
class LouvainConfig:
    """Parameters interpreted by the Louvain community analysis."""

    thr_prop: float = 0.12
    gamma: float = 0.55
    num_runs: int = 50
    consensus_tau: float = 0.5
    consensus_reps: int = 100
    min_module_degree: float = 1.0
    min_cluster_size: int = 8
    similarity_kind: SimilarityKind = "cosine"


@dataclass(frozen=True, slots=True)
class CommunityResult:
    """Community labels and matrices from consensus Louvain analysis."""

    labels: NDArray[np.int64]
    similarity: NDArray[np.float64]
    agreement: NDArray[np.float64] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)
        similarity = np.asarray(self.similarity, dtype=float)
        if similarity.shape != (labels.size, labels.size):
            raise ValueError(f"similarity shape {similarity.shape} does not match labels length {labels.size}.")
        if np.any(labels < 0):
            raise ValueError("community labels must be non-negative; 0 is unclassified.")
        if not np.all(np.isfinite(similarity)):
            raise ValueError("similarity contains NaN or infinite values.")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "similarity", similarity)
        if self.agreement is not None:
            agreement = np.asarray(self.agreement, dtype=float)
            if agreement.shape != similarity.shape:
                raise ValueError("agreement shape must match similarity shape.")
            if not np.all(np.isfinite(agreement)):
                raise ValueError("agreement contains NaN or infinite values.")
            object.__setattr__(self, "agreement", agreement)

    @property
    def n_ensembles(self) -> int:
        """Number of non-zero communities."""

        return int(np.unique(self.labels[self.labels != 0]).size)

    @property
    def classified_neurons(self) -> int:
        """Number of neurons assigned to a non-zero community."""

        return int(np.sum(self.labels != 0))


def identify_communities(activity_trace: ArrayLike, cfg: LouvainConfig) -> CommunityResult:
    """Detects activity communities with repeated Louvain and consensus clustering."""

    trace = np.asarray(activity_trace, dtype=float)
    if trace.ndim != 2:
        raise ValueError("activity_trace must have shape (n_neurons, n_time).")
    if trace.shape[0] < 1:
        raise ValueError("activity_trace must contain at least one neuron.")
    if not np.all(np.isfinite(trace)):
        raise ValueError("activity_trace contains NaN or infinite values.")
    _validate_louvain_config(cfg)

    if cfg.similarity_kind == "cosine":
        similarity = cosine_similarity_matrix(trace)
    elif cfg.similarity_kind == "pearson":
        similarity = pearson_correlation_matrix(trace)
    else:
        raise ValueError("similarity_kind must be 'cosine' or 'pearson'.")

    graph_similarity = np.maximum(similarity, 0.0)
    graph = bct.threshold_proportional(graph_similarity, float(cfg.thr_prop))
    graph = bct.weight_conversion(graph, "normalize")

    partitions = np.empty((trace.shape[0], int(cfg.num_runs)), dtype=np.int64)
    for run_idx in range(int(cfg.num_runs)):
        labels, _ = bct.community_louvain(graph, gamma=float(cfg.gamma))
        partitions[:, run_idx] = np.asarray(labels, dtype=np.int64)

    agreement = agreement_matrix(partitions) / float(cfg.num_runs)
    consensus = np.asarray(
        bct.consensus_und(
            agreement,
            float(cfg.consensus_tau),
            max(2, int(cfg.consensus_reps)),
        ),
        dtype=float,
    )
    labels, cleanup_diagnostics = _drop_weak_or_small_clusters(
        consensus,
        graph > 0.0,
        min_module_degree=float(cfg.min_module_degree),
        min_cluster_size=int(cfg.min_cluster_size),
    )
    final_labels = relabel_consecutive(np.nan_to_num(labels, nan=0.0))
    return CommunityResult(
        labels=final_labels,
        similarity=similarity,
        agreement=agreement,
        diagnostics={
            "thr_prop": float(cfg.thr_prop),
            "gamma": float(cfg.gamma),
            "num_runs": int(cfg.num_runs),
            "consensus_tau": float(cfg.consensus_tau),
            "consensus_reps": int(cfg.consensus_reps),
            "min_module_degree": float(cfg.min_module_degree),
            "min_cluster_size": int(cfg.min_cluster_size),
            "similarity_kind": cfg.similarity_kind,
            "n_ensembles": int(np.unique(final_labels[final_labels != 0]).size),
            "classified_neurons": int(np.sum(final_labels != 0)),
            **cleanup_diagnostics,
        },
    )


def cosine_similarity_matrix(activity_trace: ArrayLike) -> NDArray[np.float64]:
    """Computes pairwise cosine similarity with a zero diagonal."""

    trace = np.asarray(activity_trace, dtype=float)
    if trace.ndim != 2:
        raise ValueError("activity_trace must have shape (n_neurons, n_time).")
    if trace.shape[0] == 1:
        return np.zeros((1, 1), dtype=float)
    distances = pdist(trace, metric="cosine")
    similarity = 1.0 - squareform(distances)
    similarity = np.nan_to_num(similarity, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(similarity, 0.0)
    return similarity.astype(float, copy=False)


def pearson_correlation_matrix(activity_trace: ArrayLike) -> NDArray[np.float64]:
    """Computes pairwise Pearson correlation with a zero diagonal."""

    trace = np.asarray(activity_trace, dtype=float)
    if trace.ndim != 2:
        raise ValueError("activity_trace must have shape (n_neurons, n_time).")
    if trace.shape[0] == 1:
        return np.zeros((1, 1), dtype=float)
    centered = trace - np.mean(trace, axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    normalized = np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 0.0)
    correlation = normalized @ normalized.T
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 0.0)
    return correlation.astype(float, copy=False)


def agreement_matrix(partitions: ArrayLike) -> NDArray[np.float64]:
    """Counts how often node pairs share a community across partition columns."""

    values = np.asarray(partitions)
    if values.ndim != 2:
        raise ValueError("partitions must have shape (n_nodes, n_partitions).")
    n_nodes, n_partitions = values.shape
    agreement = np.zeros((n_nodes, n_nodes), dtype=float)
    for run_idx in range(n_partitions):
        labels = values[:, run_idx]
        for label in np.unique(labels):
            members = np.flatnonzero(labels == label)
            if members.size:
                agreement[np.ix_(members, members)] += 1.0
    return agreement


def _drop_weak_or_small_clusters(
    labels: NDArray[np.float64],
    graph_binary: NDArray[np.bool_],
    *,
    min_module_degree: float,
    min_cluster_size: int,
) -> tuple[NDArray[np.float64], dict[str, int]]:
    cleaned = np.asarray(labels, dtype=float).copy()
    diagnostics = {"weak_module_degree_removed": 0, "small_cluster_removed": 0}

    for c_id in np.unique(cleaned[np.isfinite(cleaned)]):
        members = np.flatnonzero(cleaned == c_id)
        if 0 < members.size < int(min_cluster_size):
            diagnostics["small_cluster_removed"] += int(members.size)
            cleaned[members] = np.nan

    for c_id in np.unique(cleaned[np.isfinite(cleaned)]):
        members = np.flatnonzero(cleaned == c_id)
        if members.size == 0:
            continue
        degree = graph_binary[np.ix_(members, members)].sum(axis=1)
        weak = members[degree < float(min_module_degree)]
        diagnostics["weak_module_degree_removed"] += int(weak.size)
        cleaned[weak] = np.nan

    for c_id in np.unique(cleaned[np.isfinite(cleaned)]):
        members = np.flatnonzero(cleaned == c_id)
        if 0 < members.size < int(min_cluster_size):
            diagnostics["small_cluster_removed"] += int(members.size)
            cleaned[members] = np.nan
    return cleaned, diagnostics


def _validate_louvain_config(cfg: LouvainConfig) -> None:
    if not 0.0 < float(cfg.thr_prop) <= 1.0:
        raise ValueError("thr_prop must be in (0, 1].")
    if float(cfg.gamma) <= 0.0:
        raise ValueError("gamma must be positive.")
    if int(cfg.num_runs) <= 0:
        raise ValueError("num_runs must be positive.")
    if not 0.0 <= float(cfg.consensus_tau) <= 1.0:
        raise ValueError("consensus_tau must be in [0, 1].")
    if int(cfg.consensus_reps) <= 0:
        raise ValueError("consensus_reps must be positive.")
    if float(cfg.min_module_degree) < 0.0:
        raise ValueError("min_module_degree must be non-negative.")
    if int(cfg.min_cluster_size) <= 0:
        raise ValueError("min_cluster_size must be positive.")
