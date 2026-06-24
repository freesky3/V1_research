"""Shared learning-rule protocol for training workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

import numpy as np
from numpy.typing import NDArray

from v1_research.model.state import ModelState

LearningStateT = TypeVar("LearningStateT")


@dataclass(frozen=True, slots=True)
class RateBatch:
    """Batch-first E/I rates consumed by learning rules."""

    exc: NDArray[np.float64]
    inh: NDArray[np.float64]
    external: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        exc = _batch_matrix("exc", self.exc)
        inh = _batch_matrix("inh", self.inh)
        if exc.shape[0] != inh.shape[0]:
            raise ValueError("exc and inh rates must have the same batch size.")
        object.__setattr__(self, "exc", exc)
        object.__setattr__(self, "inh", inh)

        if self.external is not None:
            external = _batch_matrix("external", self.external)
            if external.shape[0] != exc.shape[0]:
                raise ValueError("external rates must have the same batch size as exc.")
            object.__setattr__(self, "external", external)

    @property
    def batch_size(self) -> int:
        """Number of samples in this rate batch."""

        return int(self.exc.shape[0])


@dataclass(frozen=True, slots=True)
class LearningUpdate(Generic[LearningStateT]):
    """Model and learning state after applying one rule step."""

    model: ModelState
    state: LearningStateT
    updated: bool


class LearningRule(Protocol[LearningStateT]):
    """Minimal protocol used by training workflows."""

    def initialize(self, model: ModelState, rates: RateBatch) -> LearningStateT:
        """Creates rule-local state from the first observed rate batch."""

        ...

    def step(self, model: ModelState, rates: RateBatch, state: LearningStateT) -> LearningUpdate[LearningStateT]:
        """Applies one plasticity update."""

        ...

    def stats(self, model: ModelState, state: LearningStateT) -> dict[str, float]:
        """Returns compact scalar diagnostics for logging."""

        ...


def _batch_matrix(name: str, values: NDArray[np.float64]) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} rates must be a 2D batch matrix.")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} rates batch must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} rates contain NaN or infinite values.")
    return arr.astype(float, copy=True)
