"""Run-bundle and checkpoint IO for workflow orchestration."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
from omegaconf import OmegaConf
from scipy import sparse

from v1_research.model.state import ModelState, PopulationLayout
from v1_research.model.geometry import SheetGeometry


RUN_SUBDIRS = ("model", "arrays", "tables", "analysis", "figures")


def create_run_dir(run_root: str | Path, workflow: str) -> Path:
    """Creates a timestamped run directory with standard workflow subfolders.

    Args:
        run_root: Root directory for all workflow runs.
        workflow: Workflow name such as ``"train"`` or ``"simulate"``.

    Returns:
        Newly created run directory.
    """

    root = Path(run_root) / workflow
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = root / stamp
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{stamp}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir()
    for name in RUN_SUBDIRS:
        (run_dir / name).mkdir()
    return run_dir


def write_config(run_dir: str | Path, cfg: Any) -> Path:
    """Writes a workflow config snapshot as YAML."""

    path = Path(run_dir) / "config.yaml"
    OmegaConf.save(OmegaConf.create(json_ready(cfg)), path)
    return path


def write_manifest(run_dir: str | Path, payload: Mapping[str, Any]) -> Path:
    """Writes the run manifest JSON."""

    return write_json(Path(run_dir) / "manifest.json", payload)


def write_json(path: str | Path, payload: Any) -> Path:
    """Writes JSON after converting dataclasses and arrays to simple values."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")
    return path


def write_csv_rows(path: str | Path, rows: list[Mapping[str, Any]]) -> Path:
    """Writes rows to CSV using keys from the first row."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    return path


def save_model_state(
    checkpoint_dir: str | Path,
    model: ModelState,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Saves a model checkpoint under ``checkpoint_dir``.

    The checkpoint keeps sparse matrices in CSR component form inside
    ``state.npz`` so workflow IO does not densify large models.
    """

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    mask = model.connection_mask.tocsr()
    weights = sparse.csr_matrix(model.weights).tocsr()
    layout = model.layout
    np.savez_compressed(
        checkpoint_dir / "state.npz",
        l23_n_side=np.array(layout.l23.n_side, dtype=np.int64),
        l23_region_size=np.array(layout.l23.region_size, dtype=float),
        l23_z_pos=np.array(layout.l23.z_pos, dtype=float),
        l4_n_side=np.array(layout.l4.n_side, dtype=np.int64),
        l4_region_size=np.array(layout.l4.region_size, dtype=float),
        l4_z_pos=np.array(layout.l4.z_pos, dtype=float),
        l23_cell_types=np.asarray(layout.l23_cell_types),
        l4_tuning_labels=np.asarray(layout.l4_tuning_labels if layout.l4_tuning_labels is not None else []),
        l4_preferred_orientations=np.asarray(
            layout.l4_preferred_orientations if layout.l4_preferred_orientations is not None else []
        ),
        mask_data=mask.data,
        mask_indices=mask.indices,
        mask_indptr=mask.indptr,
        mask_shape=np.asarray(mask.shape, dtype=np.int64),
        weights_data=weights.data,
        weights_indices=weights.indices,
        weights_indptr=weights.indptr,
        weights_shape=np.asarray(weights.shape, dtype=np.int64),
    )
    write_json(checkpoint_dir / "metadata.json", dict(metadata or {}))
    return checkpoint_dir


def load_model_state(checkpoint_dir: str | Path) -> ModelState:
    """Loads a ``ModelState`` saved by :func:`save_model_state`."""

    checkpoint_dir = Path(checkpoint_dir)
    with np.load(checkpoint_dir / "state.npz", allow_pickle=False) as data:
        l23 = SheetGeometry(
            n_side=int(data["l23_n_side"]),
            region_size=float(data["l23_region_size"]),
            z_pos=float(data["l23_z_pos"]),
        )
        l4 = SheetGeometry(
            n_side=int(data["l4_n_side"]),
            region_size=float(data["l4_region_size"]),
            z_pos=float(data["l4_z_pos"]),
        )
        l4_labels = data["l4_tuning_labels"]
        l4_pref = data["l4_preferred_orientations"]
        layout = PopulationLayout(
            l23=l23,
            l4=l4,
            l23_cell_types=data["l23_cell_types"],
            l4_tuning_labels=l4_labels if l4_labels.size else None,
            l4_preferred_orientations=l4_pref if l4_pref.size else None,
        )
        mask = _csr_from_npz(data, "mask")
        weights = _csr_from_npz(data, "weights")
    return ModelState(layout=layout, connection_mask=mask, weights=weights)


def model_summary(model: ModelState) -> dict[str, int | list[int]]:
    """Returns compact model dimensions for manifests."""

    return {
        "n_exc": model.layout.n_exc,
        "n_inh": model.layout.n_inh,
        "n_input": model.layout.n_input,
        "shape": list(model.shape),
        "connections": int(model.connection_mask.nnz),
    }


def json_ready(value: Any) -> Any:
    """Converts common scientific Python objects into JSON/YAML values."""

    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if sparse.issparse(value):
        return {"format": value.getformat(), "shape": list(value.shape), "nnz": int(value.nnz)}
    return value


def _csr_from_npz(data: Mapping[str, NDArray[Any]], prefix: str) -> sparse.csr_matrix:
    shape = tuple(int(x) for x in data[f"{prefix}_shape"])
    return sparse.csr_matrix(
        (data[f"{prefix}_data"], data[f"{prefix}_indices"], data[f"{prefix}_indptr"]),
        shape=shape,
    )


def _csv_value(value: Any) -> Any:
    value = json_ready(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return value
