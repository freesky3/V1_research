"""Community-label overlap metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment

from v1_research.runs import json_ready


@dataclass(frozen=True, slots=True)
class LabelOverlapResult:
    """Overlap summary after matching coordinates between two analyses."""

    matched_count: int
    reference_unmatched_count: int
    query_unmatched_count: int
    adjusted_rand_index: float | None
    contingency: dict[str, Any]
    best_matches: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        """Returns a JSON-ready representation."""

        return json_ready(
            {
                "matched_count": self.matched_count,
                "reference_unmatched_count": self.reference_unmatched_count,
                "query_unmatched_count": self.query_unmatched_count,
                "adjusted_rand_index": self.adjusted_rand_index,
                "contingency": self.contingency,
                "best_matches": self.best_matches,
            }
        )


def compare_label_sets(
    *,
    reference_labels: ArrayLike,
    reference_coords: ArrayLike,
    query_labels: ArrayLike,
    query_coords: ArrayLike,
    coord_atol: float = 1.0e-10,
) -> LabelOverlapResult:
    """Matches coordinates and compares non-zero community labels."""

    ref_labels = _labels(reference_labels, "reference_labels")
    query_label_values = _labels(query_labels, "query_labels")
    ref_coords = _coords(reference_coords, "reference_coords")
    qry_coords = _coords(query_coords, "query_coords")
    if ref_labels.size != ref_coords.shape[0]:
        raise ValueError("reference label and coordinate counts must match.")
    if query_label_values.size != qry_coords.shape[0]:
        raise ValueError("query label and coordinate counts must match.")

    ref_idx, query_idx = match_coords(ref_coords, qry_coords, atol=coord_atol)
    matched_ref = ref_labels[ref_idx]
    matched_query = query_label_values[query_idx]
    contingency = contingency_table(matched_ref, matched_query)
    return LabelOverlapResult(
        matched_count=int(ref_idx.size),
        reference_unmatched_count=int(ref_labels.size - ref_idx.size),
        query_unmatched_count=int(query_label_values.size - query_idx.size),
        adjusted_rand_index=_finite_float(adjusted_rand_index(matched_ref, matched_query)),
        contingency=contingency,
        best_matches=best_label_matches(contingency),
    )


def overlap_significance(
    reference_labels: ArrayLike,
    query_labels: ArrayLike,
    *,
    num_surrogates: int,
) -> dict[str, Any]:
    """Compares observed ARI to shuffled query-label surrogates."""

    ref = _labels(reference_labels, "reference_labels")
    query = _labels(query_labels, "query_labels")
    if ref.size != query.size:
        raise ValueError("label arrays must have the same length.")
    actual = adjusted_rand_index(ref, query)
    surrogates = np.empty(int(num_surrogates), dtype=float)
    for index in range(int(num_surrogates)):
        shuffled = np.array(query, copy=True)
        np.random.shuffle(shuffled)
        surrogates[index] = adjusted_rand_index(ref, shuffled)
    return json_ready(
        {
            "actual_adjusted_rand_index": _finite_float(actual),
            "num_surrogates": int(num_surrogates),
            "surrogate_mean": _finite_float(np.mean(surrogates)) if surrogates.size else None,
            "p_ge": _finite_float(np.mean(surrogates >= actual)) if surrogates.size else None,
        }
    )


def match_coords(reference: ArrayLike, query: ArrayLike, *, atol: float = 1.0e-10) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Matches each reference coordinate to at most one query coordinate."""

    ref = _coords(reference, "reference")
    qry = _coords(query, "query")
    ref_indices: list[int] = []
    query_indices: list[int] = []
    used_query: set[int] = set()
    for ref_i, coord in enumerate(ref):
        diff = np.max(np.abs(qry - coord[np.newaxis, :]), axis=1)
        query_i = int(np.argmin(diff))
        if float(diff[query_i]) <= float(atol) and query_i not in used_query:
            ref_indices.append(ref_i)
            query_indices.append(query_i)
            used_query.add(query_i)
    return np.asarray(ref_indices, dtype=np.int64), np.asarray(query_indices, dtype=np.int64)


def contingency_table(reference_labels: ArrayLike, query_labels: ArrayLike) -> dict[str, Any]:
    """Builds a non-zero label contingency matrix."""

    ref = _labels(reference_labels, "reference_labels")
    query = _labels(query_labels, "query_labels")
    if ref.size != query.size:
        raise ValueError("label arrays must have the same length.")
    ref_ids = np.unique(ref[ref != 0]).astype(np.int64)
    query_ids = np.unique(query[query != 0]).astype(np.int64)
    matrix = np.zeros((ref_ids.size, query_ids.size), dtype=np.int64)
    for i, ref_id in enumerate(ref_ids):
        ref_mask = ref == ref_id
        for j, query_id in enumerate(query_ids):
            matrix[i, j] = int(np.sum(ref_mask & (query == query_id)))
    return {
        "reference_labels": ref_ids.tolist(),
        "query_labels": query_ids.tolist(),
        "matrix": matrix.tolist(),
    }


def best_label_matches(contingency: dict[str, Any]) -> list[dict[str, Any]]:
    """Finds one-to-one best matching non-zero communities."""

    matrix = np.asarray(contingency["matrix"], dtype=np.int64)
    ref_ids = np.asarray(contingency["reference_labels"], dtype=np.int64)
    query_ids = np.asarray(contingency["query_labels"], dtype=np.int64)
    if matrix.size == 0:
        return []
    row_ind, col_ind = linear_sum_assignment(-matrix)
    ref_sizes = matrix.sum(axis=1)
    query_sizes = matrix.sum(axis=0)
    rows: list[dict[str, Any]] = []
    for row, col in zip(row_ind, col_ind, strict=True):
        overlap = int(matrix[row, col])
        if overlap <= 0:
            continue
        union = int(ref_sizes[row] + query_sizes[col] - overlap)
        rows.append(
            {
                "reference_label": int(ref_ids[row]),
                "query_label": int(query_ids[col]),
                "overlap_count": overlap,
                "reference_count": int(ref_sizes[row]),
                "query_count": int(query_sizes[col]),
                "reference_fraction": _fraction(overlap, ref_sizes[row]),
                "query_fraction": _fraction(overlap, query_sizes[col]),
                "jaccard": _fraction(overlap, union),
            }
        )
    return sorted(rows, key=lambda item: (-float(item["jaccard"]), -int(item["overlap_count"])))


def adjusted_rand_index(labels_a: ArrayLike, labels_b: ArrayLike) -> float:
    """Computes adjusted Rand index without an sklearn dependency."""

    left = _labels(labels_a, "labels_a")
    right = _labels(labels_b, "labels_b")
    if left.size != right.size:
        raise ValueError("ARI inputs must have the same length.")
    n = int(left.size)
    if n < 2:
        return float("nan")
    _, inv_a = np.unique(left, return_inverse=True)
    _, inv_b = np.unique(right, return_inverse=True)
    matrix = np.zeros((int(inv_a.max()) + 1, int(inv_b.max()) + 1), dtype=np.int64)
    np.add.at(matrix, (inv_a, inv_b), 1)
    sum_comb = float(np.sum(_comb2(matrix)))
    sum_a = float(np.sum(_comb2(matrix.sum(axis=1))))
    sum_b = float(np.sum(_comb2(matrix.sum(axis=0))))
    total = float(_comb2(np.asarray([n], dtype=np.int64))[0])
    if total == 0.0:
        return float("nan")
    expected = (sum_a * sum_b) / total
    max_index = 0.5 * (sum_a + sum_b)
    denom = max_index - expected
    if denom == 0.0:
        return 1.0 if sum_comb == max_index else 0.0
    return float((sum_comb - expected) / denom)


def _labels(values: ArrayLike, name: str) -> NDArray[np.int64]:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D label array.")
    labels = arr.astype(np.int64, copy=False)
    if not np.all(np.asarray(arr, dtype=float) == labels):
        raise ValueError(f"{name} must contain integer labels.")
    return labels.astype(np.int64, copy=True)


def _coords(values: ArrayLike, name: str) -> NDArray[np.float64]:
    coords = np.asarray(values, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2).")
    if not np.all(np.isfinite(coords)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return coords


def _comb2(values: NDArray[np.int64]) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64)
    return arr * (arr - 1.0) / 2.0


def _fraction(numerator: Any, denominator: Any) -> float:
    denom = float(denominator)
    return 0.0 if denom == 0.0 else float(numerator) / denom


def _finite_float(value: Any) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


__all__ = [
    "LabelOverlapResult",
    "adjusted_rand_index",
    "best_label_matches",
    "compare_label_sets",
    "contingency_table",
    "match_coords",
    "overlap_significance",
]
