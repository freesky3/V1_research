"""Gabor receptive fields for L4 visual inputs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from v1_research.model.state import PopulationLayout


@dataclass(frozen=True, slots=True)
class GaborConfig:
    """Spatial Gabor receptive-field parameters."""

    sigma: float = 0.085
    gamma: float = 1.0
    spatial_frequency: float = 14.137166941154069
    phase: float = 0.0

    def __post_init__(self) -> None:
        if self.sigma <= 0.0:
            raise ValueError("Gabor sigma must be positive.")
        if self.gamma <= 0.0:
            raise ValueError("Gabor gamma must be positive.")


@dataclass(frozen=True, slots=True)
class ReceptiveFieldConfig:
    """Visual grid and shared Gabor parameters for L4 receptive fields."""

    stimulus_size: float = 2.0
    resolution: int = 300
    gabor: GaborConfig = field(default_factory=GaborConfig)

    def __post_init__(self) -> None:
        if self.stimulus_size <= 0.0:
            raise ValueError("stimulus_size must be positive.")
        if self.resolution <= 1:
            raise ValueError("resolution must be greater than 1.")


@dataclass(frozen=True, slots=True)
class VisualGrid:
    """Centered 2D midpoint grid for visual-space integration."""

    x_axis: NDArray[np.float64]
    y_axis: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    dx: float
    dy: float

    @classmethod
    def centered_midpoint(cls, *, size: float, resolution: int) -> "VisualGrid":
        """Builds a centered square visual grid.

        Args:
            size: Physical side length of the visual field.
            resolution: Number of grid samples per side.

        Returns:
            A square midpoint grid centered at zero.
        """

        if size <= 0.0:
            raise ValueError("size must be positive.")
        if resolution <= 1:
            raise ValueError("resolution must be greater than 1.")
        dx = float(size) / int(resolution)
        axis = (np.arange(int(resolution), dtype=float) + 0.5) * dx
        axis -= float(size) / 2.0
        x, y = np.meshgrid(axis, axis, indexing="xy")
        return cls(x_axis=axis, y_axis=axis, x=x, y=y, dx=dx, dy=dx)

    @property
    def area_element(self) -> float:
        """Area represented by one visual grid sample."""

        return self.dx * self.dy


def gabor_kernel(
    grid: VisualGrid,
    cfg: GaborConfig,
    *,
    preferred_orientation: float,
    tuned: bool,
) -> NDArray[np.float64]:
    """Evaluates one L4 spatial receptive field on a visual grid.

    Args:
        grid: Visual grid coordinates relative to the RF center.
        cfg: Shared Gabor parameters.
        preferred_orientation: Preferred orientation in radians.
        tuned: Whether to use an oriented Gabor or an untuned Gaussian RF.

    Returns:
        A 2D receptive-field kernel.
    """

    theta = float(preferred_orientation) if tuned else 0.0
    x_prime = grid.x * np.cos(theta) + grid.y * np.sin(theta)
    y_prime = -grid.x * np.sin(theta) + grid.y * np.cos(theta)

    gamma = float(cfg.gamma) if tuned else 1.0
    gaussian = np.exp(-(x_prime**2 + gamma * y_prime**2) / (2.0 * float(cfg.sigma) ** 2))
    if not tuned:
        return gaussian
    return gaussian * np.cos(float(cfg.spatial_frequency) * x_prime - float(cfg.phase))


def gabor_bank(
    grid: VisualGrid,
    cfg: GaborConfig,
    *,
    preferred_orientations: NDArray[np.float64],
    tuned: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Builds a stack of Gabor kernels for all L4 cells.

    Args:
        grid: Visual grid shared by all cells.
        cfg: Shared Gabor parameters.
        preferred_orientations: Preferred orientation per L4 cell.
        tuned: Tuning mask per L4 cell.

    Returns:
        Array with shape ``(n_l4, resolution, resolution)``.
    """

    if preferred_orientations.shape != tuned.shape:
        raise ValueError("preferred_orientations and tuned must have matching shapes.")
    kernels = [
        gabor_kernel(grid, cfg, preferred_orientation=float(theta), tuned=bool(is_tuned))
        for theta, is_tuned in zip(preferred_orientations, tuned, strict=True)
    ]
    return np.stack(kernels, axis=0)


class L4GaborBank:
    """Lazy Gabor RF bank bound to a V1 ``PopulationLayout``."""

    def __init__(self, cfg: ReceptiveFieldConfig, layout: PopulationLayout) -> None:
        self.cfg = cfg
        self.layout = layout
        self.grid = VisualGrid.centered_midpoint(size=cfg.stimulus_size, resolution=cfg.resolution)
        self.tuned, self.preferred_orientations = l4_tuning_arrays(layout)
        self._filters: NDArray[np.float64] | None = None

    @property
    def filters(self) -> NDArray[np.float64]:
        """Read-only RF stack with shape ``(n_l4, resolution, resolution)``."""

        if self._filters is None:
            filters = gabor_bank(
                self.grid,
                self.cfg.gabor,
                preferred_orientations=self.preferred_orientations,
                tuned=self.tuned,
            )
            filters.setflags(write=False)
            self._filters = filters
        return self._filters


def l4_tuning_arrays(layout: PopulationLayout) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Returns L4 tuning mask and finite preferred orientations from a layout."""

    if layout.l4_tuning_labels is None or layout.l4_preferred_orientations is None:
        raise ValueError("PopulationLayout must include L4 tuning labels and preferred orientations.")
    tuned = np.asarray(layout.l4_tuning_labels == "T", dtype=bool).reshape(-1)
    preferred = np.nan_to_num(np.asarray(layout.l4_preferred_orientations, dtype=float).reshape(-1), nan=0.0)
    if tuned.shape != (layout.n_input,) or preferred.shape != (layout.n_input,):
        raise ValueError("L4 tuning arrays must match layout.n_input.")
    return tuned, preferred
