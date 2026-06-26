"""Drifting-grating inputs projected onto L4 receptive fields."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from v1_research.inputs.gabor import GaborConfig, ReceptiveFieldConfig, VisualGrid, gabor_kernel, l4_tuning_arrays
from v1_research.model.state import PopulationLayout


@dataclass(frozen=True, slots=True)
class DriftingGratingConfig:
    """Drifting-grating stimulus and L4 RF projection parameters."""

    receptive_field: ReceptiveFieldConfig = field(default_factory=ReceptiveFieldConfig)
    baseline_rate: float = 0.0
    visual_gain: float = 400.0
    luminance: float = 1.0
    contrast: float = 1.0
    temporal_frequency: float = 2.0 * np.pi
    n_orientations: int = 8
    visual_gain_ramp_duration: float = 0.0


class DriftingGratingInput:
    """Analytic L4 drive for drifting gratings."""

    def __init__(self, cfg: DriftingGratingConfig, layout: PopulationLayout) -> None:
        self.cfg = cfg
        self.layout = layout
        self.grid = VisualGrid.centered_midpoint(
            size=cfg.receptive_field.stimulus_size,
            resolution=cfg.receptive_field.resolution,
        )
        self.gabor_cfg: GaborConfig = cfg.receptive_field.gabor
        self._tuned, self._preferred_orientations = l4_tuning_arrays(layout)
        self._coords = np.asarray(layout.l4.coords, dtype=float)
        self._integral_cache: dict[float, tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]] = {}

    @property
    def orientation_angles(self) -> NDArray[np.float64]:
        """Default evenly spaced stimulus orientations."""

        return np.linspace(0.0, 2.0 * np.pi, int(self.cfg.n_orientations), endpoint=False)

    def external_drive(self, theta_stim: float, t: float) -> NDArray[np.float64]:
        """Computes L4 rates for one grating orientation and time.

        Args:
            theta_stim: Stimulus orientation in radians.
            t: Time in seconds.

        Returns:
            One L4 rate per input cell.
        """

        base, cos_coeff, sin_coeff = self._precompute_integrals(theta_stim)
        phase = float(self.cfg.temporal_frequency) * float(t)
        integral = base + cos_coeff * np.cos(phase) + sin_coeff * np.sin(phase)
        return np.maximum(0.0, float(self.cfg.baseline_rate) + integral) * self._visual_gain(t)

    def make_drive_func(self, theta_stim: float) -> Callable[[float], NDArray[np.float64]]:
        """Returns a continuous-time drive function for one orientation."""

        def drive(t: float) -> NDArray[np.float64]:
            return self.external_drive(theta_stim, t)

        return drive

    def make_batched_drive_func(
        self,
        theta_angles: ArrayLike,
        phase_offsets: ArrayLike | None = None,
    ) -> Callable[[float], NDArray[np.float64]]:
        """Returns a continuous-time drive matrix function for many orientations."""

        theta = np.asarray(theta_angles, dtype=float).reshape(-1)
        if phase_offsets is None:
            offsets = np.zeros_like(theta)
        else:
            offsets = np.asarray(phase_offsets, dtype=float).reshape(-1)
            if offsets.shape != theta.shape:
                raise ValueError(f"phase_offsets must have shape {theta.shape}, got {offsets.shape}.")
            if not np.all(np.isfinite(offsets)):
                raise ValueError("phase_offsets must contain only finite values.")

        terms = [self._precompute_integrals(float(angle)) for angle in theta]
        base = np.column_stack([item[0] for item in terms])
        cos_coeff = np.column_stack([item[1] for item in terms])
        sin_coeff = np.column_stack([item[2] for item in terms])

        def drive(t: float) -> NDArray[np.float64]:
            phase = float(self.cfg.temporal_frequency) * float(t) + offsets[np.newaxis, :]
            integral = base + cos_coeff * np.cos(phase) + sin_coeff * np.sin(phase)
            return np.maximum(0.0, float(self.cfg.baseline_rate) + integral) * self._visual_gain(t)

        return drive

    def stimulus_frame(self, theta_stim: float, t: float) -> NDArray[np.float64]:
        """Renders the visual grating around every L4 RF center."""

        x = self._coords[:, 0, np.newaxis, np.newaxis] + self.grid.x[np.newaxis, :, :]
        y = self._coords[:, 1, np.newaxis, np.newaxis] + self.grid.y[np.newaxis, :, :]
        k = float(self.gabor_cfg.spatial_frequency)
        phase = k * (x * np.cos(theta_stim) + y * np.sin(theta_stim))
        phase -= float(self.cfg.temporal_frequency) * float(t)
        return float(self.cfg.luminance) * (1.0 + float(self.cfg.contrast) * np.cos(phase))

    def _visual_gain(self, t: float) -> float:
        duration = float(self.cfg.visual_gain_ramp_duration)
        gain = float(self.cfg.visual_gain)
        if duration <= 0.0:
            return gain
        if t <= 0.0:
            return 0.0
        if t >= duration:
            return gain
        s = float(t) / duration
        return gain * s * s * (3.0 - 2.0 * s)

    def _precompute_integrals(
        self,
        theta_stim: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        key = float(theta_stim)
        if key in self._integral_cache:
            return self._integral_cache[key]

        n_l4 = self.layout.n_input
        base = np.zeros(n_l4, dtype=float)
        cos_coeff = np.zeros(n_l4, dtype=float)
        sin_coeff = np.zeros(n_l4, dtype=float)
        area = self.grid.area_element
        k = float(self.gabor_cfg.spatial_frequency)

        cos_theta = np.cos(key)
        sin_theta = np.sin(key)
        grid_phase = k * (self.grid.x * cos_theta + self.grid.y * sin_theta)
        grid_cos = np.cos(grid_phase)
        grid_sin = np.sin(grid_phase)

        cell_phase = k * (self._coords[:, 0] * cos_theta + self._coords[:, 1] * sin_theta)
        cell_cos = np.cos(cell_phase)
        cell_sin = np.sin(cell_phase)

        for tuned in (False, True):
            tuned_mask = self._tuned == tuned
            if not np.any(tuned_mask):
                continue
            theta_values = np.unique(self._preferred_orientations[tuned_mask]) if tuned else np.array([0.0])
            for theta_pref in theta_values:
                group = tuned_mask & (self._preferred_orientations == theta_pref)
                kernel = gabor_kernel(
                    self.grid,
                    self.gabor_cfg,
                    preferred_orientation=float(theta_pref),
                    tuned=tuned,
                )
                luminance = float(self.cfg.luminance)
                contrast = float(self.cfg.contrast)
                group_base = np.sum(kernel * luminance) * area
                group_cos = np.sum(kernel * luminance * contrast * grid_cos) * area
                group_sin = np.sum(kernel * luminance * contrast * grid_sin) * area
                base[group] = group_base
                cos_coeff[group] = cell_cos[group] * group_cos - cell_sin[group] * group_sin
                sin_coeff[group] = cell_sin[group] * group_cos + cell_cos[group] * group_sin

        self._integral_cache[key] = (base, cos_coeff, sin_coeff)
        return self._integral_cache[key]
