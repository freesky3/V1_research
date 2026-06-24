from __future__ import annotations

import json

import numpy as np
from scipy import sparse

from v1_research.model import ModelState, PopulationLayout, SheetGeometry
from v1_research.runs import create_run_dir, load_model_state, save_model_state, write_manifest


def _one_cell_model() -> ModelState:
    layout = PopulationLayout(
        l23=SheetGeometry(n_side=1, region_size=1.0, z_pos=0.0),
        l4=SheetGeometry(n_side=1, region_size=1.0, z_pos=0.1),
        l23_cell_types=np.array(["E"]),
        l4_tuning_labels=np.array(["T"]),
        l4_preferred_orientations=np.array([0.0]),
    )
    weights = sparse.csr_matrix(np.array([[0.0, 0.25]], dtype=float))
    return ModelState(layout=layout, connection_mask=weights != 0.0, weights=weights)


def test_model_checkpoint_roundtrips_layout_mask_and_weights(tmp_path) -> None:
    model = _one_cell_model()
    checkpoint_dir = save_model_state(tmp_path / "model", model, metadata={"step": 3})

    loaded = load_model_state(checkpoint_dir)

    assert loaded.layout.l23.n_side == 1
    assert loaded.layout.l4.n_side == 1
    np.testing.assert_array_equal(loaded.layout.l23_cell_types, model.layout.l23_cell_types)
    np.testing.assert_array_equal(loaded.layout.l4_tuning_labels, model.layout.l4_tuning_labels)
    np.testing.assert_allclose(loaded.layout.l4_preferred_orientations, model.layout.l4_preferred_orientations)
    assert (loaded.connection_mask != model.connection_mask).nnz == 0
    np.testing.assert_allclose(loaded.weights.toarray(), model.weights.toarray())

    metadata = json.loads((checkpoint_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["step"] == 3


def test_create_run_dir_and_manifest_use_workflow_layout(tmp_path) -> None:
    run_dir = create_run_dir(tmp_path, "simulate")
    manifest_path = write_manifest(run_dir, {"workflow": "simulate", "summary": {"n_batch": 2}})

    assert run_dir.parent == tmp_path / "simulate"
    assert (run_dir / "arrays").is_dir()
    assert (run_dir / "tables").is_dir()
    assert (run_dir / "analysis").is_dir()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["summary"]["n_batch"] == 2
