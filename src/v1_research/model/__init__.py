"""Model construction primitives for V1 research workflows."""

from v1_research.model.build import (
    ModelBuildSpec,
    ModelConfig,
    build_model,
    build_model_spec,
    build_population_layout,
)
from v1_research.model.connectivity import (
    ConnectivityConfig,
    ConnectionBlock,
    ConnectionProbabilities,
    SpatialKernel,
    SpatialKernelConfig,
    derive_connection_probabilities,
    probability_block,
    probability_matrix,
    sample_connectivity,
)
from v1_research.model.geometry import (
    L23Config,
    L4Config,
    L4Tuning,
    SheetGeometry,
    assign_l23_cell_types,
    assign_l4_tuning,
    uniform_grid_indices,
)
from v1_research.model.state import ModelState, PopulationLayout
from v1_research.model.weights import (
    BlockWeightScales,
    WeightConfig,
    as_connection_mask,
    as_dense_weights,
    limit_row_sums,
    row_sums,
    sample_weights,
    validate_indices,
)

__all__ = [
    "BlockWeightScales",
    "ConnectionBlock",
    "ConnectionProbabilities",
    "ConnectivityConfig",
    "L23Config",
    "L4Config",
    "L4Tuning",
    "ModelBuildSpec",
    "ModelConfig",
    "ModelState",
    "PopulationLayout",
    "SheetGeometry",
    "SpatialKernel",
    "SpatialKernelConfig",
    "WeightConfig",
    "as_connection_mask",
    "as_dense_weights",
    "assign_l23_cell_types",
    "assign_l4_tuning",
    "build_model",
    "build_model_spec",
    "build_population_layout",
    "derive_connection_probabilities",
    "limit_row_sums",
    "probability_block",
    "probability_matrix",
    "row_sums",
    "sample_connectivity",
    "sample_weights",
    "uniform_grid_indices",
    "validate_indices",
]
