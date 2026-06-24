"""Spatial connectivity probabilities and topology sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from v1_research.data.experimental import ExperimentalData, PopulationCounts
from v1_research.model.state import PopulationLayout

PopulationName = Literal["E", "I", "X"]


def _require_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")


@dataclass(frozen=True, slots=True)
class SpatialKernelConfig:
    """Two-Gaussian spatial kernel parameters for connection sampling."""

    sigma_narrow: float = 0.075
    sigma_broad: float = 0.225
    kappa: float = 0.45

    def __post_init__(self) -> None:
        if self.sigma_narrow <= 0.0 or self.sigma_broad <= 0.0:
            raise ValueError("Kernel sigmas must be positive.")
        if self.sigma_narrow >= self.sigma_broad:
            raise ValueError("sigma_narrow must be smaller than sigma_broad.")
        _require_probability(self.kappa, "kappa")


@dataclass(frozen=True, slots=True)
class SpatialKernel:
    """Spatial score kernel used before probability normalization."""

    cfg: SpatialKernelConfig

    def evaluate(self, distance: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the two-Gaussian kernel at pairwise distances.

        Args:
            distance: Pairwise distance matrix.

        Returns:
            Non-negative spatial scores with the same shape as ``distance``.
        """

        distance = np.asarray(distance, dtype=float)
        d2 = distance * distance
        narrow = np.exp(-d2 / (2.0 * self.cfg.sigma_narrow**2))
        broad = np.exp(-d2 / (2.0 * self.cfg.sigma_broad**2))
        return self.cfg.kappa * narrow + (1.0 - self.cfg.kappa) * broad


@dataclass(frozen=True, slots=True)
class ConnectionProbabilities:
    """Connection probabilities by target/source block."""

    ee: float
    ei: float
    ex: float
    ie: float
    ii: float
    ix: float

    def __post_init__(self) -> None:
        for name in ("ee", "ei", "ex", "ie", "ii", "ix"):
            _require_probability(getattr(self, name), f"p_{name}")


@dataclass(frozen=True, slots=True)
class ConnectionBlock:
    """One directed block, named by target population then source population."""

    target: PopulationName
    source: PopulationName
    probability: float
    sign: int

    @property
    def key(self) -> str:
        """Lowercase target/source block key, such as ``ei``."""

        return f"{self.target.lower()}{self.source.lower()}"


@dataclass(frozen=True, slots=True)
class ConnectivityConfig:
    """Connectivity probabilities and spatial sampling configuration."""

    probabilities: ConnectionProbabilities
    kernel: SpatialKernelConfig = SpatialKernelConfig()
    periodic: bool = True
    equalize_indegree: bool = True

    @property
    def blocks(self) -> tuple[ConnectionBlock, ...]:
        """Directed blocks in target/source matrix order."""

        p = self.probabilities
        return (
            ConnectionBlock("E", "E", p.ee, +1),
            ConnectionBlock("E", "I", p.ei, -1),
            ConnectionBlock("E", "X", p.ex, +1),
            ConnectionBlock("I", "E", p.ie, +1),
            ConnectionBlock("I", "I", p.ii, -1),
            ConnectionBlock("I", "X", p.ix, +1),
        )

    def probability_for(self, target: PopulationName, source: PopulationName) -> float:
        """Return the configured probability for a target/source block."""

        return float(getattr(self.probabilities, f"{target.lower()}{source.lower()}"))


def derive_connection_probabilities(
    *,
    counts: PopulationCounts,
    empirical: ExperimentalData,
    p_ee: float,
) -> ConnectionProbabilities:
    """Derive block probabilities from final counts and empirical ratios.

    Args:
        counts: Final population counts.
        empirical: Experimental ratios loaded from sample data.
        p_ee: Base recurrent E-to-E connection probability.

    Returns:
        Probabilities for each target/source block.
    """

    _require_probability(float(p_ee), "p_ee")
    k_ee = float(p_ee) * counts.n_exc
    k_e_total = k_ee / empirical.gamma_ee

    k_ei = empirical.gamma_ei * k_e_total
    k_ex = empirical.gamma_ex * k_e_total
    k_i_total = empirical.chi * k_e_total
    k_ie = empirical.gamma_ie * k_i_total
    k_ii = empirical.gamma_ii * k_i_total
    k_ix = empirical.gamma_ix * k_i_total

    return ConnectionProbabilities(
        ee=float(p_ee),
        ei=_divide_expected_count(k_ei, counts.n_inh, "p_ei"),
        ex=_divide_expected_count(k_ex, counts.n_input, "p_ex"),
        ie=_divide_expected_count(k_ie, counts.n_exc, "p_ie"),
        ii=_divide_expected_count(k_ii, counts.n_inh, "p_ii"),
        ix=_divide_expected_count(k_ix, counts.n_input, "p_ix"),
    )


def probability_matrix(layout: PopulationLayout, cfg: ConnectivityConfig) -> NDArray[np.float64]:
    """Build dense Bernoulli connection probabilities without sampling.

    Args:
        layout: Population geometry and labels.
        cfg: Connectivity configuration.

    Returns:
        Dense probability matrix with shape ``layout.shape``.
    """

    kernel = SpatialKernel(cfg.kernel)
    probabilities = np.zeros(layout.shape, dtype=float)
    for target_idx, source_idx, distance, block in _iter_distance_blocks(layout, cfg):
        valid = np.ones(distance.shape, dtype=bool)
        if block.target == block.source and block.target in {"E", "I"}:
            valid &= target_idx[:, None] != source_idx[None, :]
        block_prob = probability_block(
            kernel.evaluate(distance),
            block.probability,
            valid_mask=valid,
            equalize_rows=cfg.equalize_indegree,
        )
        probabilities[np.ix_(target_idx, source_idx)] = block_prob
    return probabilities


def sample_connectivity(
    layout: PopulationLayout,
    cfg: ConnectivityConfig,
) -> sparse.csr_matrix:
    """Sample a sparse boolean connectivity mask from configured probabilities.

    Args:
        layout: Population geometry and labels.
        cfg: Connectivity configuration.

    Returns:
        Boolean CSR matrix with shape ``layout.shape``.
    """

    probabilities = probability_matrix(layout, cfg)
    return sparse.csr_matrix(np.random.random(probabilities.shape) < probabilities)


def probability_block(
    score: NDArray[np.float64],
    target_probability: float,
    *,
    valid_mask: NDArray[np.bool_] | None = None,
    equalize_rows: bool = True,
) -> NDArray[np.float64]:
    """Scale a spatial score matrix to match a target probability.

    Args:
        score: Non-negative spatial score matrix.
        target_probability: Mean probability over valid entries.
        valid_mask: Boolean mask of allowed connections.
        equalize_rows: If true, each target row is scaled independently.

    Returns:
        Probability matrix with invalid entries set to zero.
    """

    _require_probability(float(target_probability), "target_probability")
    score = np.asarray(score, dtype=float)
    if valid_mask is None:
        valid = np.ones(score.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != score.shape:
            raise ValueError(f"valid_mask shape {valid.shape} does not match score shape {score.shape}.")

    out = np.zeros(score.shape, dtype=float)
    if target_probability == 0.0 or not np.any(valid):
        return out

    base = np.where(valid, score, 0.0)
    if np.any(base[valid] < 0.0):
        raise ValueError("Connectivity score must be non-negative.")
    if equalize_rows:
        return _row_equalized_probability(base, valid, target_probability)
    return _globally_scaled_probability(base, valid, target_probability)


def _iter_distance_blocks(layout: PopulationLayout, cfg: ConnectivityConfig):
    dist_l23 = layout.l23.distance_matrix(periodic=cfg.periodic)
    dist_l23_to_l4 = layout.l23.distance_to(layout.l4, periodic=cfg.periodic)
    local_input = np.arange(layout.l4.n_cells, dtype=np.int64)
    blocks = cfg.blocks

    yield layout.exc_idx, layout.exc_idx, dist_l23[np.ix_(layout.exc_idx, layout.exc_idx)], blocks[0]
    yield layout.exc_idx, layout.inh_idx, dist_l23[np.ix_(layout.exc_idx, layout.inh_idx)], blocks[1]
    yield layout.exc_idx, layout.input_idx, dist_l23_to_l4[np.ix_(layout.exc_idx, local_input)], blocks[2]
    yield layout.inh_idx, layout.exc_idx, dist_l23[np.ix_(layout.inh_idx, layout.exc_idx)], blocks[3]
    yield layout.inh_idx, layout.inh_idx, dist_l23[np.ix_(layout.inh_idx, layout.inh_idx)], blocks[4]
    yield layout.inh_idx, layout.input_idx, dist_l23_to_l4[np.ix_(layout.inh_idx, local_input)], blocks[5]


def _row_equalized_probability(
    base: NDArray[np.float64],
    valid: NDArray[np.bool_],
    target_probability: float,
) -> NDArray[np.float64]:
    valid_counts = valid.sum(axis=1, keepdims=True)
    positive_rows = (valid_counts[:, 0] > 0) & (base.sum(axis=1) > 0.0)
    if np.any((valid_counts[:, 0] > 0) & ~positive_rows):
        raise ValueError("Cannot assign positive probability to a row with zero connectivity score.")

    lo = np.zeros((base.shape[0], 1), dtype=float)
    hi = np.ones((base.shape[0], 1), dtype=float)
    active = positive_rows[:, None]

    for _ in range(80):
        means = _row_valid_mean(np.clip(hi * base, 0.0, 1.0), valid, valid_counts)
        needs_more = active & (means < target_probability)
        if not np.any(needs_more):
            break
        hi = np.where(needs_more, hi * 2.0, hi)
    else:
        raise ValueError(f"Could not scale rows to target probability {target_probability}.")

    for _ in range(48):
        mid = (lo + hi) / 2.0
        means = _row_valid_mean(np.clip(mid * base, 0.0, 1.0), valid, valid_counts)
        hi = np.where(active & (means > target_probability), mid, hi)
        lo = np.where(active & (means <= target_probability), mid, lo)

    out = np.clip(((lo + hi) / 2.0) * base, 0.0, 1.0)
    out[~valid] = 0.0
    return out


def _globally_scaled_probability(
    base: NDArray[np.float64],
    valid: NDArray[np.bool_],
    target_probability: float,
) -> NDArray[np.float64]:
    if np.sum(base[valid]) == 0.0:
        raise ValueError("Cannot assign positive probability to a zero connectivity score matrix.")

    lo = 0.0
    hi = 1.0
    for _ in range(80):
        if np.mean(np.clip(hi * base[valid], 0.0, 1.0)) >= target_probability:
            break
        hi *= 2.0
    else:
        raise ValueError(f"Could not scale block to target probability {target_probability}.")

    for _ in range(48):
        mid = (lo + hi) / 2.0
        if np.mean(np.clip(mid * base[valid], 0.0, 1.0)) > target_probability:
            hi = mid
        else:
            lo = mid

    out = np.clip(((lo + hi) / 2.0) * base, 0.0, 1.0)
    out[~valid] = 0.0
    return out


def _row_valid_mean(
    probability: NDArray[np.float64],
    valid: NDArray[np.bool_],
    valid_counts: NDArray[np.int64],
) -> NDArray[np.float64]:
    row_sums = np.sum(np.where(valid, probability, 0.0), axis=1, keepdims=True)
    return np.divide(row_sums, valid_counts, out=np.zeros_like(row_sums), where=valid_counts > 0)


def _divide_expected_count(expected_count: float, n_source: int, name: str) -> float:
    if n_source <= 0:
        if expected_count == 0.0:
            return 0.0
        raise ValueError(f"Cannot derive {name}: source population has size {n_source}.")
    probability = float(expected_count) / float(n_source)
    _require_probability(probability, name)
    return probability

