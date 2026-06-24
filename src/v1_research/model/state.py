"""Immutable model state objects for V1 network construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from v1_research.model.geometry import SheetGeometry


def _readonly_1d_str(values: NDArray[np.str_] | list[str], *, name: str, length: int) -> NDArray[np.str_]:
    labels = np.asarray(values, dtype="<U1").reshape(-1).copy()
    if labels.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {labels.shape}.")
    labels.setflags(write=False)
    return labels


def _readonly_1d_float(
    values: NDArray[np.float64] | list[float] | None,
    *,
    name: str,
    length: int,
) -> NDArray[np.float64] | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float).reshape(-1).copy()
    if arr.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {arr.shape}.")
    arr.setflags(write=False)
    return arr


@dataclass(frozen=True, slots=True)
class PopulationLayout:
    """Geometry and immutable population labels for the model.

    L2/3 rows are dynamic target cells. L4 cells are external input columns
    appended after L2/3 source columns.
    """

    l23: SheetGeometry
    l4: SheetGeometry
    l23_cell_types: NDArray[np.str_]
    l4_tuning_labels: NDArray[np.str_] | None = None
    l4_preferred_orientations: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        l23_labels = _readonly_1d_str(self.l23_cell_types, name="l23_cell_types", length=self.l23.n_cells)
        invalid = set(np.unique(l23_labels)) - {"E", "I"}
        if invalid:
            raise ValueError(f"Unsupported L2/3 cell types: {sorted(invalid)}.")
        object.__setattr__(self, "l23_cell_types", l23_labels)

        if self.l4_tuning_labels is not None:
            l4_labels = _readonly_1d_str(self.l4_tuning_labels, name="l4_tuning_labels", length=self.l4.n_cells)
            invalid_l4 = set(np.unique(l4_labels)) - {"T", "U"}
            if invalid_l4:
                raise ValueError(f"Unsupported L4 tuning labels: {sorted(invalid_l4)}.")
            object.__setattr__(self, "l4_tuning_labels", l4_labels)
        object.__setattr__(
            self,
            "l4_preferred_orientations",
            _readonly_1d_float(
                self.l4_preferred_orientations,
                name="l4_preferred_orientations",
                length=self.l4.n_cells,
            ),
        )

    @property
    def exc_idx(self) -> NDArray[np.int64]:
        """Indices of excitatory L2/3 target/source cells."""

        return np.flatnonzero(self.l23_cell_types == "E")

    @property
    def inh_idx(self) -> NDArray[np.int64]:
        """Indices of inhibitory L2/3 target/source cells."""

        return np.flatnonzero(self.l23_cell_types == "I")

    @property
    def input_idx(self) -> NDArray[np.int64]:
        """Column indices of L4 input cells in model matrices."""

        return np.arange(self.l23.n_cells, self.l23.n_cells + self.l4.n_cells, dtype=np.int64)

    @property
    def n_exc(self) -> int:
        """Number of excitatory L2/3 cells."""

        return int(np.sum(self.l23_cell_types == "E"))

    @property
    def n_inh(self) -> int:
        """Number of inhibitory L2/3 cells."""

        return int(np.sum(self.l23_cell_types == "I"))

    @property
    def n_input(self) -> int:
        """Number of L4 input cells."""

        return self.l4.n_cells

    @property
    def shape(self) -> tuple[int, int]:
        """Matrix shape: L2/3 targets by L2/3 plus L4 sources."""

        return self.l23.n_cells, self.l23.n_cells + self.l4.n_cells


@dataclass(frozen=True, slots=True)
class ModelState:
    """Built model topology and signed synaptic weights."""

    layout: PopulationLayout
    connection_mask: sparse.csr_matrix
    weights: sparse.csr_matrix | NDArray[np.float64]

    def __post_init__(self) -> None:
        mask = sparse.csr_matrix(self.connection_mask, dtype=bool)
        if mask.shape != self.layout.shape:
            raise ValueError(f"connection_mask shape {mask.shape} does not match layout shape {self.layout.shape}.")

        if sparse.issparse(self.weights):
            weights = sparse.csr_matrix(self.weights, dtype=float)
        else:
            weights = np.asarray(self.weights, dtype=float)
            if weights.ndim != 2:
                raise ValueError("weights must be a 2D matrix.")
        if weights.shape != self.layout.shape:
            raise ValueError(f"weights shape {weights.shape} does not match layout shape {self.layout.shape}.")

        object.__setattr__(self, "connection_mask", mask)
        object.__setattr__(self, "weights", weights)

    @property
    def shape(self) -> tuple[int, int]:
        """Matrix shape shared by the connection mask and weights."""

        return self.layout.shape
