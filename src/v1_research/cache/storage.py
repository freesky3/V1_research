"""Tiny disk storage helpers for projection caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


class CacheStore:
    """Stores NumPy arrays and JSON metadata under content-addressed names."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)

    def array_path(self, key: str) -> Path:
        """Path for a cached array."""

        return self.cache_dir / f"{key}.npy"

    def metadata_path(self, key: str) -> Path:
        """Path for cache metadata."""

        return self.cache_dir / f"{key}.json"

    def has(self, key: str) -> bool:
        """Whether both array and metadata files exist."""

        return self.array_path(key).exists() and self.metadata_path(key).exists()

    def load_array(self, key: str) -> NDArray[np.float64]:
        """Loads a cached array as read-only."""

        array = np.load(self.array_path(key))
        array.setflags(write=False)
        return array

    def save_array(self, key: str, array: NDArray[np.float64], metadata: dict[str, Any]) -> None:
        """Saves an array and sidecar JSON metadata."""

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.array_path(key), np.asarray(array))
        self.metadata_path(key).write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    def load_metadata(self, key: str) -> dict[str, Any]:
        """Loads sidecar JSON metadata."""

        return json.loads(self.metadata_path(key).read_text(encoding="utf-8"))
