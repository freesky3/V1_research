"""Geometry and population-label helpers for the V1 model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class L4Config:
    """Layer 4 sheet geometry and orientation tuning configuration."""

    n_side: int = 40
    region_size: float = 2.0
    z_pos: float = 0.0
    all_tuned: bool = True
    n_orientations: int = 8


@dataclass(frozen=True, slots=True)
class L23Config:
    """Layer 2/3 sheet geometry and inhibitory-cell layout configuration."""

    n_side: int | None = None
    inhibitory_fraction: float | None = None
    region_size: float = 2.0
    z_pos: float = 0.1
    random_inhibitory: bool = False


@dataclass(frozen=True, slots=True)
class L4Tuning:
    """Orientation-tuning labels and preferred orientations for L4 cells."""

    tuning_labels: NDArray[np.str_]
    preferred_orientations: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SheetGeometry:
    """Regular square sheet of neurons embedded at a fixed z position."""

    n_side: int
    region_size: float
    z_pos: float

    @property
    def n_cells(self) -> int:
        """Total number of cells on this sheet."""

        return self.n_side * self.n_side

    @property
    def coords(self) -> NDArray[np.float64]:
        """Cell-center coordinates as an ``(n_cells, 2)`` array."""

        spacing = self.region_size / self.n_side
        axis = (np.arange(self.n_side, dtype=float) + 0.5) * spacing
        axis -= self.region_size / 2.0
        x, y = np.meshgrid(axis, axis, indexing="xy")
        return np.column_stack((x.ravel(), y.ravel()))

    def distance_matrix(self, *, periodic: bool = True) -> NDArray[np.float64]:
        """Pairwise distances within this sheet.

        Args:
            periodic: Whether to wrap 2D distances on the sheet boundary.

        Returns:
            Dense pairwise Euclidean distances.
        """

        delta = self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :]
        if periodic:
            delta = periodic_delta(delta, self.region_size)
        return np.linalg.norm(delta, axis=2)

    def distance_to(self, other: "SheetGeometry", *, periodic: bool = True) -> NDArray[np.float64]:
        """Pairwise distances from this sheet to another sheet.

        Args:
            other: Source or target sheet to compare against.
            periodic: Whether to wrap 2D distances on the sheet boundary.

        Returns:
            Dense Euclidean distances including z separation.
        """

        if periodic and not np.isclose(self.region_size, other.region_size):
            raise ValueError(
                "Periodic cross-layer distance requires matching region_size: "
                f"{self.region_size} vs {other.region_size}."
            )
        delta_2d = self.coords[:, np.newaxis, :] - other.coords[np.newaxis, :, :]
        if periodic:
            delta_2d = periodic_delta(delta_2d, self.region_size)
        dist_2d_sq = np.sum(delta_2d**2, axis=2)
        return np.sqrt(dist_2d_sq + (self.z_pos - other.z_pos) ** 2)


def periodic_delta(delta: NDArray[np.float64], box_size: float) -> NDArray[np.float64]:
    """Wrap coordinate deltas into the nearest periodic image."""

    return (delta + box_size / 2.0) % box_size - box_size / 2.0


def uniform_grid_indices(*, n_side: int, count: int) -> NDArray[np.int64]:
    """Choose approximately uniform grid indices from a square sheet.

    Args:
        n_side: Number of cells along one side of the sheet.
        count: Number of grid locations to select.

    Returns:
        Sorted unique flat indices.
    """

    n_side = int(n_side)
    n_cells = n_side * n_side
    count = _bounded_count(count, n_cells, "count")
    if count == 0:
        return np.array([], dtype=np.int64)
    if count == n_cells:
        return np.arange(n_cells, dtype=np.int64)

    n_cols = min(n_side, int(np.ceil(np.sqrt(count))))
    n_rows = min(n_side, int(np.ceil(count / n_cols)))
    while n_rows * n_cols < count:
        if n_cols < n_side:
            n_cols += 1
        elif n_rows < n_side:
            n_rows += 1
        else:
            break

    rows = np.rint(np.linspace(0, n_side - 1, n_rows)).astype(int)
    cols = np.rint(np.linspace(0, n_side - 1, n_cols)).astype(int)
    candidates = np.array([r * n_side + c for r in rows for c in cols], dtype=np.int64)
    candidates = np.unique(candidates)

    if candidates.size > count:
        keep = np.linspace(0, candidates.size - 1, count, dtype=int)
        candidates = candidates[keep]
    if candidates.size < count:
        all_indices = np.arange(n_cells, dtype=np.int64)
        remaining = np.setdiff1d(all_indices, candidates, assume_unique=True)
        extra = remaining[np.linspace(0, remaining.size - 1, count - candidates.size, dtype=int)]
        candidates = np.concatenate([candidates, extra])

    return candidates.astype(np.int64, copy=False)


def assign_l23_cell_types(
    *,
    n_side: int,
    n_inhibitory: int,
    random_inhibitory: bool,
) -> NDArray[np.str_]:
    """Assign E/I labels to layer 2/3 cells.

    Args:
        n_side: Number of cells along one side of the L2/3 sheet.
        n_inhibitory: Number of inhibitory cells to place.
        random_inhibitory: If true, sample inhibitory positions randomly.

    Returns:
        A one-dimensional array of ``"E"`` and ``"I"`` labels.
    """

    n_cells = int(n_side) * int(n_side)
    n_inhibitory = _bounded_count(n_inhibitory, n_cells, "n_inhibitory")
    labels = np.full(n_cells, "E", dtype="<U1")
    if n_inhibitory == 0:
        return labels
    if random_inhibitory:
        inhibitory_idx = np.random.choice(n_cells, size=n_inhibitory, replace=False)
    else:
        inhibitory_idx = uniform_grid_indices(n_side=n_side, count=n_inhibitory)
    labels[inhibitory_idx] = "I"
    return labels


def assign_l4_tuning(cfg: L4Config, *, tuned_fraction: float) -> L4Tuning:
    """Assign L4 tuning labels and preferred orientations.

    Args:
        cfg: L4 geometry and orientation configuration.
        tuned_fraction: Empirical fraction of L4 cells that are tuned when not all tuned.

    Returns:
        L4 tuning labels and preferred orientations in radians.
    """

    n_cells = cfg.n_side * cfg.n_side
    n_tuned = n_cells if cfg.all_tuned else int(round(n_cells * tuned_fraction))
    n_tuned = _bounded_count(n_tuned, n_cells, "n_tuned")

    labels = np.full(n_cells, "U", dtype="<U1")
    preferred = np.full(n_cells, np.nan, dtype=float)
    if n_tuned == 0:
        return L4Tuning(labels, preferred)

    tuned_idx = np.random.choice(n_cells, size=n_tuned, replace=False)
    labels[tuned_idx] = "T"
    theta_values = np.linspace(0.0, 2.0 * np.pi, cfg.n_orientations, endpoint=False)
    assigned = np.resize(theta_values, n_tuned)
    np.random.shuffle(assigned)
    preferred[tuned_idx] = assigned
    return L4Tuning(labels, preferred)


def _bounded_count(count: int, total: int, name: str) -> int:
    count = int(count)
    if count < 0 or count > total:
        raise ValueError(f"{name} must be between 0 and {total}, got {count}.")
    return count
