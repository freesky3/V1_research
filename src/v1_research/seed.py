"""Process-wide random seeding helpers for workflow entrypoints."""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np


_seed_preserve_depth = 0


def set_global_seed(seed: int | None) -> None:
    """Sets all supported process-wide random seeds.

    Torch is optional for this project. If it is installed, its CPU/CUDA seeds
    and cuDNN deterministic switches are set; otherwise the helper only affects
    Python and NumPy random state.
    """

    if seed is None or _seed_preserve_depth > 0:
        return
    seed_value = int(seed)
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    torch = _try_import_torch()
    if torch is None:
        return
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    if cudnn is not None:
        cudnn.benchmark = False
        cudnn.deterministic = True


def _try_import_torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


@contextmanager
def preserve_global_seed() -> Iterator[None]:
    """Temporarily prevents nested workflow calls from reseeding the process."""

    global _seed_preserve_depth
    _seed_preserve_depth += 1
    try:
        yield
    finally:
        _seed_preserve_depth -= 1


__all__ = ["preserve_global_seed", "set_global_seed"]
