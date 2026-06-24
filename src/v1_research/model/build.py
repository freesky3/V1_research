"""Build complete V1 model states from local model configuration."""

from __future__ import annotations

from dataclasses import dataclass

from v1_research.data.experimental import ExperimentalData, PopulationCounts, derive_population_counts
from v1_research.model.connectivity import (
    ConnectivityConfig,
    ConnectionProbabilities,
    SpatialKernelConfig,
    derive_connection_probabilities,
    sample_connectivity,
)
from v1_research.model.geometry import (
    L23Config,
    L4Config,
    SheetGeometry,
    assign_l23_cell_types,
    assign_l4_tuning,
)
from v1_research.model.state import ModelState, PopulationLayout
from v1_research.model.weights import BlockWeightScales, WeightConfig, sample_weights


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Local configuration for V1 model construction."""

    l4: L4Config = L4Config()
    l23: L23Config = L23Config()
    p_ee: float = 0.12
    connectivity_kernel: SpatialKernelConfig = SpatialKernelConfig()
    equalize_indegree: bool = True
    periodic: bool = True
    weight: WeightConfig = WeightConfig()


@dataclass(frozen=True, slots=True)
class ModelBuildSpec:
    """Derived model quantities used before topology and weight sampling."""

    counts: PopulationCounts
    probabilities: ConnectionProbabilities
    connectivity: ConnectivityConfig
    weights: WeightConfig


def build_population_layout(cfg: ModelConfig, empirical: ExperimentalData) -> PopulationLayout:
    """Build sheet geometries and population labels.

    Args:
        cfg: Local model configuration.
        empirical: Experimental constraints.

    Returns:
        Population geometry and labels.
    """

    l4 = SheetGeometry(cfg.l4.n_side, cfg.l4.region_size, cfg.l4.z_pos)
    counts = derive_population_counts(
        n_input=l4.n_cells,
        empirical=empirical,
        l23_n_side=cfg.l23.n_side,
        inhibitory_fraction=cfg.l23.inhibitory_fraction,
    )
    l23 = SheetGeometry(counts.l23_n_side, cfg.l23.region_size, cfg.l23.z_pos)
    l23_cell_types = assign_l23_cell_types(
        n_side=counts.l23_n_side,
        n_inhibitory=counts.n_inh,
        random_inhibitory=cfg.l23.random_inhibitory,
    )
    l4_tuning = assign_l4_tuning(cfg.l4, tuned_fraction=empirical.eta_t_x)
    return PopulationLayout(
        l23=l23,
        l4=l4,
        l23_cell_types=l23_cell_types,
        l4_tuning_labels=l4_tuning.tuning_labels,
        l4_preferred_orientations=l4_tuning.preferred_orientations,
    )


def build_model_spec(cfg: ModelConfig, empirical: ExperimentalData, layout: PopulationLayout) -> ModelBuildSpec:
    """Derive counts and block probabilities from layout and empirical data.

    Args:
        cfg: Local model configuration.
        empirical: Experimental constraints.
        layout: Built population layout.

    Returns:
        Derived model construction spec.
    """

    counts = PopulationCounts(
        l23_n_side=layout.l23.n_side,
        n_exc=layout.n_exc,
        n_inh=layout.n_inh,
        n_input=layout.n_input,
    )
    probabilities = derive_connection_probabilities(counts=counts, empirical=empirical, p_ee=cfg.p_ee)
    connectivity = ConnectivityConfig(
        probabilities=probabilities,
        kernel=cfg.connectivity_kernel,
        periodic=cfg.periodic,
        equalize_indegree=cfg.equalize_indegree,
    )
    return ModelBuildSpec(counts=counts, probabilities=probabilities, connectivity=connectivity, weights=cfg.weight)


def build_model(cfg: ModelConfig, empirical: ExperimentalData) -> ModelState:
    """Build a complete V1 model state.

    Args:
        cfg: Local model construction configuration.
        empirical: Experimental constraints and weight samples.

    Returns:
        Model state containing layout, topology, and signed sparse weights.
    """

    layout = build_population_layout(cfg, empirical)
    spec = build_model_spec(cfg, empirical, layout)
    connection_mask = sample_connectivity(layout, spec.connectivity)
    weights = sample_weights(layout, connection_mask, spec.weights, empirical.weights)
    return ModelState(layout=layout, connection_mask=connection_mask, weights=weights)


__all__ = [
    "BlockWeightScales",
    "ConnectivityConfig",
    "L23Config",
    "L4Config",
    "ModelBuildSpec",
    "ModelConfig",
    "ModelState",
    "PopulationLayout",
    "SheetGeometry",
    "SpatialKernelConfig",
    "WeightConfig",
    "build_model",
    "build_model_spec",
    "build_population_layout",
]
