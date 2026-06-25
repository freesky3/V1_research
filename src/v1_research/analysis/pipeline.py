"""Main analysis data flow from simulation arrays to community metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from v1_research.analysis.communities import CommunityResult, LouvainConfig, identify_communities
from v1_research.analysis.direction_tuning import DirectionTuningConfig, summarize_direction_tuning
from v1_research.analysis.metrics import activity_health_metrics, summarize_communities
from v1_research.analysis.osi import compute_osi
from v1_research.runs import load_model_state


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Local configuration interpreted by the analysis pipeline."""

    osi_threshold: float = 0.4
    active_threshold: float = 1.0e-6
    filter_by_osi: bool = True
    random_sample_fraction: float = 1.0
    center_side_fraction: float = 1.0
    trace_source_hz: float = 100.0
    trace_target_hz: float = 4.0
    louvain: LouvainConfig = field(default_factory=LouvainConfig)
    direction_tuning: DirectionTuningConfig = field(default_factory=DirectionTuningConfig)


@dataclass(frozen=True, slots=True)
class AnalysisInputs:
    """Arrays required by the post-simulation analysis."""

    responses: NDArray[np.float64]
    coords: NDArray[np.float64]
    distance: NDArray[np.float64]
    orientation_angles: NDArray[np.float64]
    trial_responses: NDArray[np.float64] | None = None
    trial_direction_indices: NDArray[np.int64] | None = None

    def validate(self) -> None:
        """Checks shape invariants that would otherwise silently corrupt analysis."""

        responses = np.asarray(self.responses, dtype=float)
        coords = np.asarray(self.coords, dtype=float)
        distance = np.asarray(self.distance, dtype=float)
        orientation_angles = np.asarray(self.orientation_angles, dtype=float)
        if responses.ndim != 3:
            raise ValueError("responses must have shape (n_neurons, n_orientation, n_time).")
        n_neurons, n_orientation, n_time = responses.shape
        if n_neurons < 1 or n_orientation < 1 or n_time < 1:
            raise ValueError("responses must contain at least one neuron, orientation, and time point.")
        if coords.shape != (n_neurons, 2):
            raise ValueError(f"coords must have shape ({n_neurons}, 2), got {coords.shape}.")
        if distance.shape != (n_neurons, n_neurons):
            raise ValueError(f"distance must have shape ({n_neurons}, {n_neurons}), got {distance.shape}.")
        if orientation_angles.shape != (n_orientation,):
            raise ValueError(
                f"orientation_angles must have shape ({n_orientation},), got {orientation_angles.shape}."
            )
        trial_responses = None if self.trial_responses is None else np.asarray(self.trial_responses, dtype=float)
        trial_direction_indices = (
            None if self.trial_direction_indices is None else np.asarray(self.trial_direction_indices, dtype=np.int64)
        )
        if trial_responses is not None:
            if trial_responses.ndim != 3:
                raise ValueError("trial_responses must have shape (n_neurons, n_trial, n_time).")
            if trial_responses.shape[0] != n_neurons or trial_responses.shape[2] != n_time:
                raise ValueError("trial_responses must share neuron and time dimensions with responses.")
            if not np.all(np.isfinite(trial_responses)):
                raise ValueError("trial_responses contains NaN or infinite values.")
            if trial_direction_indices is None:
                raise ValueError("trial_direction_indices is required when trial_responses is provided.")
            if trial_direction_indices.shape != (trial_responses.shape[1],):
                raise ValueError(
                    f"trial_direction_indices must have shape ({trial_responses.shape[1]},), "
                    f"got {trial_direction_indices.shape}."
                )
            if np.any((trial_direction_indices < 0) | (trial_direction_indices >= n_orientation)):
                raise ValueError("trial_direction_indices contains directions outside orientation_angles.")
        for name, values in (
            ("responses", responses),
            ("coords", coords),
            ("distance", distance),
            ("orientation_angles", orientation_angles),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} contains NaN or infinite values.")


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Aggregated analysis result for one simulation bundle."""

    status: str
    selected_indices: NDArray[np.int64]
    osi: NDArray[np.float64]
    preferred_orientation: NDArray[np.float64]
    responses_mean: NDArray[np.float64]
    steady_state_responses: NDArray[np.float64]
    coords: NDArray[np.float64]
    distance: NDArray[np.float64]
    communities: CommunityResult | None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected_indices", np.asarray(self.selected_indices, dtype=np.int64))
        object.__setattr__(self, "osi", np.asarray(self.osi, dtype=float))
        object.__setattr__(self, "preferred_orientation", np.asarray(self.preferred_orientation, dtype=float))
        object.__setattr__(self, "responses_mean", np.asarray(self.responses_mean, dtype=float))
        object.__setattr__(self, "steady_state_responses", np.asarray(self.steady_state_responses, dtype=float))
        object.__setattr__(self, "coords", np.asarray(self.coords, dtype=float))
        object.__setattr__(self, "distance", np.asarray(self.distance, dtype=float))


def run_analysis(cfg: AnalysisConfig, inputs: AnalysisInputs) -> AnalysisResult:
    """Executes OSI, neuron selection, Louvain, and summary metrics."""

    inputs.validate()
    responses = np.asarray(inputs.responses, dtype=float)
    steady_start = int(responses.shape[2] * 2 / 3)
    steady_state = responses[:, :, steady_start:]
    responses_mean = np.mean(steady_state, axis=2)

    osi, preferred = compute_osi(responses_mean, inputs.orientation_angles, min_osi=cfg.osi_threshold)
    per_neuron_activity = np.mean(responses_mean, axis=1)
    selected = select_analysis_neuron_indices(osi, cfg=cfg, activity=per_neuron_activity)
    osi_candidates = np.flatnonzero(np.isfinite(osi) & (osi >= float(cfg.osi_threshold)))
    active_candidates = np.flatnonzero(per_neuron_activity > float(cfg.active_threshold))

    diagnostics: dict[str, Any] = {
        "steady_state_start": int(steady_start),
        "candidate_neurons_before_osi_filter": int(osi.size),
        "candidate_neurons_after_osi_filter": int(osi_candidates.size),
        "candidate_neurons_after_activity_filter": int(active_candidates.size),
        "filter_by_osi": bool(cfg.filter_by_osi),
        "selection_filter": "osi" if cfg.filter_by_osi else "activity",
        "selected_neurons_for_analysis": int(selected.size),
        "osi_threshold": float(cfg.osi_threshold),
        "active_threshold": float(cfg.active_threshold),
        "random_sample_fraction": float(cfg.random_sample_fraction),
    }
    diagnostics.update(activity_health_metrics(responses_mean, active_threshold=cfg.active_threshold))

    selected_distance = np.asarray(inputs.distance, dtype=float)[np.ix_(selected, selected)]
    selected_coords = np.asarray(inputs.coords, dtype=float)[selected]
    if selected.size < 2:
        summary, rows = summarize_communities(
            np.zeros(selected.size, dtype=np.int64),
            distance=selected_distance,
            coords=selected_coords,
            osi=osi[selected],
            preferred_orientation=preferred[selected],
        )
        _add_direction_tuning_diagnostics(
            diagnostics,
            labels=np.zeros(selected.size, dtype=np.int64),
            responses_mean=responses_mean[selected],
            orientation_angles=inputs.orientation_angles,
            cfg=cfg.direction_tuning,
        )
        diagnostics["metrics_summary"] = summary
        diagnostics["ensemble_metrics"] = rows
        return AnalysisResult(
            status="not_enough_neurons",
            selected_indices=selected,
            osi=osi[selected],
            preferred_orientation=preferred[selected],
            responses_mean=responses_mean[selected],
            steady_state_responses=steady_state[selected],
            coords=selected_coords,
            distance=selected_distance,
            communities=None,
            diagnostics=diagnostics,
        )

    selected_steady = steady_state[selected]
    louvain_steady = (
        np.asarray(inputs.trial_responses, dtype=float)[selected, :, steady_start:]
        if inputs.trial_responses is not None
        else selected_steady
    )
    activity_trace, decimation_factor = ensemble_activity_trace(
        louvain_steady,
        source_hz=cfg.trace_source_hz,
        target_hz=cfg.trace_target_hz,
    )
    diagnostics["activity_decimation_factor"] = int(decimation_factor)
    communities = identify_communities(activity_trace, cfg.louvain)
    summary, rows = summarize_communities(
        communities.labels,
        similarity=communities.similarity,
        distance=selected_distance,
        coords=selected_coords,
        osi=osi[selected],
        preferred_orientation=preferred[selected],
    )
    _add_direction_tuning_diagnostics(
        diagnostics,
        labels=communities.labels,
        responses_mean=responses_mean[selected],
        orientation_angles=inputs.orientation_angles,
        cfg=cfg.direction_tuning,
    )
    diagnostics["metrics_summary"] = summary
    diagnostics["ensemble_metrics"] = rows
    return AnalysisResult(
        status="ok",
        selected_indices=selected,
        osi=osi[selected],
        preferred_orientation=preferred[selected],
        responses_mean=responses_mean[selected],
        steady_state_responses=selected_steady,
        coords=selected_coords,
        distance=selected_distance,
        communities=communities,
        diagnostics=diagnostics,
    )


def select_analysis_neuron_indices(
    osi: NDArray[np.float64],
    *,
    cfg: AnalysisConfig,
    activity: NDArray[np.float64] | None = None,
) -> NDArray[np.int64]:
    """Selects cells for community analysis from OSI or activity filters."""

    if not 0.0 <= float(cfg.osi_threshold) <= 1.0:
        raise ValueError("osi_threshold must be in [0, 1].")
    if not 0.0 < float(cfg.random_sample_fraction) <= 1.0:
        raise ValueError("random_sample_fraction must be in (0, 1].")
    values = np.asarray(osi, dtype=float)
    if cfg.filter_by_osi:
        candidates = np.flatnonzero(np.isfinite(values) & (values >= float(cfg.osi_threshold)))
    else:
        if activity is None:
            raise ValueError("activity is required when filter_by_osi is false.")
        activity_values = np.asarray(activity, dtype=float)
        if activity_values.shape != values.shape:
            raise ValueError(f"activity must have shape {values.shape}, got {activity_values.shape}.")
        candidates = np.flatnonzero(np.isfinite(activity_values) & (activity_values > float(cfg.active_threshold)))
    if candidates.size == 0:
        return candidates.astype(np.int64, copy=False)

    sample_size = max(1, int(round(candidates.size * float(cfg.random_sample_fraction))))
    sample_size = min(sample_size, candidates.size)
    if sample_size == candidates.size:
        return candidates.astype(np.int64, copy=True)
    selected = np.random.choice(candidates, size=sample_size, replace=False)
    selected.sort()
    return selected.astype(np.int64, copy=False)


def ensemble_activity_trace(
    steady_state_responses: NDArray[np.float64],
    *,
    source_hz: float = 100.0,
    target_hz: float = 4.0,
) -> tuple[NDArray[np.float64], int]:
    """Decimates and flattens orientation-time responses for Louvain input."""

    responses = np.asarray(steady_state_responses, dtype=float)
    if responses.ndim != 3:
        raise ValueError("steady_state_responses must have shape (n_neurons, n_orientation, n_time).")
    if float(source_hz) <= 0.0 or float(target_hz) <= 0.0:
        raise ValueError("source_hz and target_hz must be positive.")
    factor = max(1, int(round(float(source_hz) / float(target_hz))))
    if factor > 1 and responses.shape[2] > 3 * factor:
        filtered = signal.decimate(responses, factor, axis=2)
        return filtered.reshape(responses.shape[0], -1), factor
    return responses.reshape(responses.shape[0], -1), 1


def load_analysis_inputs_from_simulation(run_dir: str | Path, *, center_side_fraction: float = 1.0) -> AnalysisInputs:
    """Loads analysis arrays from a new-format grating simulation run bundle."""

    root = Path(run_dir)
    arrays = root / "arrays"
    orientation_path = arrays / "orientation_angles.npy"
    trajectory_path = arrays / "excitatory_trajectory.npy"
    rates_path = arrays / "excitatory_rates.npy"
    trial_direction_path = arrays / "trial_direction_indices.npy"
    if not orientation_path.exists():
        raise FileNotFoundError(f"Missing orientation angles: {orientation_path}")
    orientation_angles = np.asarray(np.load(orientation_path), dtype=float)
    trial_responses = None
    trial_direction_indices = None
    if trajectory_path.exists():
        trajectory = np.load(trajectory_path)
        batch_responses = np.transpose(np.asarray(trajectory, dtype=float), (2, 1, 0))
        if trial_direction_path.exists():
            trial_direction_indices = np.asarray(np.load(trial_direction_path), dtype=np.int64).reshape(-1)
            trial_responses = batch_responses
            responses = direction_average_trial_responses(
                trial_responses,
                trial_direction_indices,
                n_orientations=orientation_angles.size,
            )
        else:
            responses = batch_responses
    elif rates_path.exists():
        rates = np.asarray(np.load(rates_path), dtype=float)
        if rates.ndim != 2:
            raise ValueError("excitatory_rates.npy must have shape (n_batch, n_exc).")
        batch_responses = np.transpose(rates, (1, 0))[:, :, np.newaxis]
        if trial_direction_path.exists():
            trial_direction_indices = np.asarray(np.load(trial_direction_path), dtype=np.int64).reshape(-1)
            trial_responses = batch_responses
            responses = direction_average_trial_responses(
                trial_responses,
                trial_direction_indices,
                n_orientations=orientation_angles.size,
            )
        else:
            responses = batch_responses
    else:
        raise FileNotFoundError(f"Missing excitatory trajectory or rates under {arrays}.")

    model = load_model_state(root / "model")
    exc_idx = model.layout.exc_idx
    coords = model.layout.l23.coords[exc_idx]
    distance = model.layout.l23.distance_matrix(periodic=True)[np.ix_(exc_idx, exc_idx)]
    if not 0.0 < float(center_side_fraction) <= 1.0:
        raise ValueError("center_side_fraction must be in (0, 1].")
    if float(center_side_fraction) < 1.0:
        half_side = model.layout.l23.region_size * float(center_side_fraction) / 2.0
        keep = (np.abs(coords[:, 0]) <= half_side) & (np.abs(coords[:, 1]) <= half_side)
        coords = coords[keep]
        distance = distance[np.ix_(keep, keep)]
        responses = responses[keep]
        if trial_responses is not None:
            trial_responses = trial_responses[keep]

    return AnalysisInputs(
        responses=np.asarray(responses, dtype=float),
        coords=np.asarray(coords, dtype=float),
        distance=np.asarray(distance, dtype=float),
        orientation_angles=orientation_angles,
        trial_responses=None if trial_responses is None else np.asarray(trial_responses, dtype=float),
        trial_direction_indices=trial_direction_indices,
    )


def direction_average_trial_responses(
    trial_responses: NDArray[np.float64],
    trial_direction_indices: NDArray[np.int64],
    *,
    n_orientations: int,
) -> NDArray[np.float64]:
    """Averages repeated trials into one response trace per direction."""

    trials = np.asarray(trial_responses, dtype=float)
    indices = np.asarray(trial_direction_indices, dtype=np.int64).reshape(-1)
    if trials.ndim != 3:
        raise ValueError("trial_responses must have shape (n_neurons, n_trial, n_time).")
    if indices.shape != (trials.shape[1],):
        raise ValueError(f"trial_direction_indices must have shape ({trials.shape[1]},), got {indices.shape}.")
    if np.any((indices < 0) | (indices >= int(n_orientations))):
        raise ValueError("trial_direction_indices contains directions outside orientation_angles.")
    responses = np.empty((trials.shape[0], int(n_orientations), trials.shape[2]), dtype=float)
    for direction in range(int(n_orientations)):
        mask = indices == direction
        if not np.any(mask):
            raise ValueError(f"No trials found for direction index {direction}.")
        responses[:, direction, :] = np.mean(trials[:, mask, :], axis=1)
    return responses


def _add_direction_tuning_diagnostics(
    diagnostics: dict[str, Any],
    *,
    labels: NDArray[np.int64],
    responses_mean: NDArray[np.float64],
    orientation_angles: NDArray[np.float64],
    cfg: DirectionTuningConfig,
) -> None:
    if not cfg.enabled:
        return
    summary, rows = summarize_direction_tuning(labels, responses_mean, orientation_angles, cfg)
    diagnostics["direction_tuning_summary"] = summary
    diagnostics["direction_tuning_rows"] = rows
