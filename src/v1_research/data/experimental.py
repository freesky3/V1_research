"""Experimental constraints used to build V1 model populations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


def _scalar_float(data: Mapping[str, Any], key: str) -> float:
    try:
        value = data[key]
    except KeyError as exc:
        raise KeyError(f"Missing experimental data key {key!r}.") from exc
    return float(np.asarray(value).item())


def _readonly_samples(data: Mapping[str, Any], key: str) -> NDArray[np.float64]:
    try:
        value = data[key]
    except KeyError as exc:
        raise KeyError(f"Missing empirical weight sample key {key!r}.") from exc
    samples = np.asarray(value, dtype=float).reshape(-1).copy()
    if samples.size == 0:
        raise ValueError(f"Empirical weight sample {key!r} must be non-empty.")
    samples.setflags(write=False)
    return samples


def _require_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")


@dataclass(frozen=True, slots=True)
class EmpiricalWeightSamples:
    """Empirical synaptic weight magnitudes by target/source block.

    Block suffixes are target/source labels, so ``ei`` means target E cells
    receiving from source I cells.
    """

    ee: NDArray[np.float64]
    ei: NDArray[np.float64]
    ex: NDArray[np.float64]
    ie: NDArray[np.float64]
    ii: NDArray[np.float64]
    ix: NDArray[np.float64]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EmpiricalWeightSamples":
        """Read empirical weight samples from a loaded data mapping.

        Args:
            data: Mapping containing the ``sampled_J_*`` arrays.

        Returns:
            Read-only block-wise empirical samples.
        """

        return cls(
            ee=_readonly_samples(data, "sampled_J_EE"),
            ei=_readonly_samples(data, "sampled_J_EI"),
            ex=_readonly_samples(data, "sampled_J_EX"),
            ie=_readonly_samples(data, "sampled_J_IE"),
            ii=_readonly_samples(data, "sampled_J_II"),
            ix=_readonly_samples(data, "sampled_J_IX"),
        )


@dataclass(frozen=True, slots=True)
class ExperimentalData:
    """Empirical ratios and weight samples for constructing the V1 model."""

    eta_i: float
    eta_x: float
    gamma_ee: float
    gamma_ei: float
    gamma_ex: float
    gamma_ie: float
    gamma_ii: float
    gamma_ix: float
    chi: float
    eta_t_e: float
    eta_t_x: float
    weights: EmpiricalWeightSamples

    def __post_init__(self) -> None:
        if self.eta_i < 0.0:
            raise ValueError(f"eta_i must be non-negative, got {self.eta_i}.")
        if self.eta_x <= 0.0:
            raise ValueError(f"eta_x must be positive, got {self.eta_x}.")
        if self.gamma_ee <= 0.0:
            raise ValueError(f"gamma_ee must be positive, got {self.gamma_ee}.")
        for name in ("gamma_ei", "gamma_ex", "gamma_ie", "gamma_ii", "gamma_ix", "chi"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}.")
        _require_probability(self.eta_t_e, "eta_t_e")
        _require_probability(self.eta_t_x, "eta_t_x")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ExperimentalData":
        """Build experimental constraints from a loaded mapping.

        Args:
            data: Mapping with empirical ratios and ``sampled_J_*`` arrays.

        Returns:
            Immutable experimental data for model construction.
        """

        return cls(
            eta_i=_scalar_float(data, "eta_I"),
            eta_x=_scalar_float(data, "eta_X"),
            gamma_ee=_scalar_float(data, "gamma_EE"),
            gamma_ei=_scalar_float(data, "gamma_EI"),
            gamma_ex=_scalar_float(data, "gamma_EX"),
            gamma_ie=_scalar_float(data, "gamma_IE"),
            gamma_ii=_scalar_float(data, "gamma_II"),
            gamma_ix=_scalar_float(data, "gamma_IX"),
            chi=_scalar_float(data, "chi"),
            eta_t_e=_scalar_float(data, "etaT_E"),
            eta_t_x=_scalar_float(data, "etaT_X"),
            weights=EmpiricalWeightSamples.from_mapping(data),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "ExperimentalData":
        """Load experimental constraints from a NumPy archive path.

        Args:
            path: File containing the empirical sample-data mapping.

        Returns:
            Immutable experimental data for model construction.
        """

        loaded = np.load(Path(path), allow_pickle=True)
        try:
            if isinstance(loaded, np.lib.npyio.NpzFile):
                data = {key: loaded[key] for key in loaded.files}
            elif isinstance(loaded, Mapping):
                data = loaded
            elif isinstance(loaded, np.ndarray) and loaded.shape == () and isinstance(loaded.item(), Mapping):
                data = loaded.item()
            else:
                raise TypeError(f"Expected experimental data at {path!s} to load as a mapping.")
            return cls.from_mapping(data)
        finally:
            if hasattr(loaded, "close"):
                loaded.close()


@dataclass(frozen=True, slots=True)
class PopulationCounts:
    """Final neuron counts used to derive block connection probabilities."""

    l23_n_side: int
    n_exc: int
    n_inh: int
    n_input: int

    @property
    def n_l23(self) -> int:
        """Total number of layer 2/3 cells."""

        return self.l23_n_side * self.l23_n_side

    def __post_init__(self) -> None:
        if self.l23_n_side <= 0:
            raise ValueError(f"l23_n_side must be positive, got {self.l23_n_side}.")
        if self.n_input <= 0:
            raise ValueError(f"n_input must be positive, got {self.n_input}.")
        if self.n_exc < 0 or self.n_inh < 0:
            raise ValueError(f"n_exc and n_inh must be non-negative, got {self.n_exc}, {self.n_inh}.")
        if self.n_exc + self.n_inh != self.n_l23:
            raise ValueError(
                "n_exc + n_inh must match L2/3 sheet size, "
                f"got n_exc={self.n_exc}, n_inh={self.n_inh}, n_l23={self.n_l23}."
            )


def derive_population_counts(
    *,
    n_input: int,
    empirical: ExperimentalData,
    l23_n_side: int | None = None,
    inhibitory_fraction: float | None = None,
) -> PopulationCounts:
    """Derive L2/3 and input population counts from empirical ratios.

    Args:
        n_input: Number of L4 input cells.
        empirical: Experimental ratios loaded from sample data.
        l23_n_side: Optional fixed L2/3 side length.
        inhibitory_fraction: Optional fixed inhibitory fraction in L2/3.

    Returns:
        Final population counts used by model construction.
    """

    n_input = int(n_input)
    if l23_n_side is None:
        exact_n_exc = n_input / empirical.eta_x
        exact_n_inh = exact_n_exc * empirical.eta_i
        l23_n_side = int(np.ceil(np.sqrt(exact_n_exc + exact_n_inh)))
    else:
        l23_n_side = int(l23_n_side)

    n_l23 = l23_n_side * l23_n_side
    if inhibitory_fraction is None:
        n_exc = int(n_l23 / (1.0 + empirical.eta_i))
        n_inh = n_l23 - n_exc
    else:
        _require_probability(float(inhibitory_fraction), "inhibitory_fraction")
        n_inh = int(round(n_l23 * float(inhibitory_fraction)))
        n_exc = n_l23 - n_inh

    return PopulationCounts(l23_n_side=l23_n_side, n_exc=n_exc, n_inh=n_inh, n_input=n_input)
