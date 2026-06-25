# Analyze Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add default single-run analysis diagnostics and optional robustness outputs to the `analyze` workflow.

**Architecture:** Pure diagnostic calculations live in `v1_research.analysis`, file writing stays in `analysis.artifacts`, Matplotlib output uses a small workflow-facing figure module, and `workflows.analyze` only wires config, IO, and manifest paths. The default path writes a result atlas and failure diagnosis without rerunning analysis; robustness reruns analysis only when explicitly enabled.

**Tech Stack:** Python, NumPy, SciPy/BCT already in use, Matplotlib with `Agg`, Typer/OmegaConf config loading, pytest.

---

## File Map

- Create `src/v1_research/analysis/diagnostics.py`: selection funnel, graph diagnostics, unclassified diagnostics, table row helpers.
- Create `src/v1_research/analysis/robustness.py`: optional window and Louvain-grid robustness row generation, baseline overlap rows.
- Create `src/v1_research/workflows/analysis_figures.py`: Matplotlib figure writers for result atlas, failure diagnosis, and robustness summary.
- Modify `src/v1_research/analysis/communities.py`: expose weak-node and small-cluster cleanup counts in diagnostics.
- Modify `src/v1_research/analysis/artifacts.py`: write inspection JSON/CSV outputs and merge returned paths.
- Modify `src/v1_research/workflows/analyze.py`: add inspection config dataclasses, call diagnostics/figures/robustness, update manifest.
- Modify `src/v1_research/analysis/__init__.py`: export new helper entrypoints where useful.
- Modify `configs/analyze_louvain.yaml`: add example inspection config.
- Modify `docs/analysis.md`, `docs/quickstart.md`, and `docs/parameters.md`: document new outputs and config.
- Add tests:
  - `tests/test_analysis_diagnostics.py`
  - `tests/test_analysis_robustness.py`
  - extend `tests/test_workflows_analyze.py`

## Task 1: Community Cleanup Diagnostics

**Files:**
- Modify: `src/v1_research/analysis/communities.py`
- Test: `tests/test_analysis_clusters.py`

- [ ] **Step 1: Write the failing cleanup diagnostics test**

Add this test to `tests/test_analysis_clusters.py`:

```python
def test_louvain_cleanup_reports_weak_degree_and_small_cluster_counts() -> None:
    labels = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
    graph_binary = np.array(
        [
            [False, True, False, False, False],
            [True, False, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
        ]
    )

    cleaned, diagnostics = _drop_weak_or_small_clusters(
        labels,
        graph_binary,
        min_module_degree=1.0,
        min_cluster_size=2,
    )

    assert np.array_equal(np.nan_to_num(cleaned, nan=0.0), np.array([1.0, 1.0, 0.0, 0.0, 0.0]))
    assert diagnostics["weak_module_degree_removed"] == 2
    assert diagnostics["small_cluster_removed"] == 1
```

- [ ] **Step 2: Run the test and verify RED**

Run: `uv run pytest tests/test_analysis_clusters.py::test_louvain_cleanup_reports_weak_degree_and_small_cluster_counts -q`

Expected: FAIL because `_drop_weak_or_small_clusters(...)` currently returns only one array.

- [ ] **Step 3: Implement cleanup diagnostics**

Change `_drop_weak_or_small_clusters(...)` to return `(cleaned, diagnostics)`. Track two counters:

```python
diagnostics = {"weak_module_degree_removed": 0, "small_cluster_removed": 0}
```

Increment `weak_module_degree_removed` by the number of members removed by the degree rule. Increment `small_cluster_removed` by the number of members removed by the size rule. Update `identify_communities(...)` to unpack the tuple and include the counters in `CommunityResult.diagnostics`.

- [ ] **Step 4: Run the focused test and existing community tests**

Run:

```bash
uv run pytest tests/test_analysis_clusters.py tests/test_analysis_overlap_temporal.py -q
```

Expected: PASS.

## Task 2: Pure Analysis Diagnostics

**Files:**
- Create: `src/v1_research/analysis/diagnostics.py`
- Test: `tests/test_analysis_diagnostics.py`

- [ ] **Step 1: Write failing tests for funnel and graph diagnostics**

Create `tests/test_analysis_diagnostics.py` with:

```python
from __future__ import annotations

import numpy as np
import pytest

from v1_research.analysis.communities import LouvainConfig
from v1_research.analysis.diagnostics import (
    graph_health_diagnostics,
    selection_funnel,
    unclassified_diagnostics,
)
from v1_research.analysis.pipeline import AnalysisConfig


def test_selection_funnel_counts_activity_osi_sampling_and_classification() -> None:
    cfg = AnalysisConfig(osi_threshold=0.4, active_threshold=0.5, filter_by_osi=True, random_sample_fraction=1.0)
    osi = np.array([0.1, 0.4, 0.8, np.nan])
    activity = np.array([0.0, 1.0, 2.0, 3.0])
    selected = np.array([1, 2])
    labels = np.array([1, 0])

    rows, summary = selection_funnel(
        osi=osi,
        activity=activity,
        selected_indices=selected,
        labels=labels,
        cfg=cfg,
    )

    assert summary["total_candidates"] == 4
    assert summary["active_candidates"] == 3
    assert summary["finite_osi_candidates"] == 3
    assert summary["osi_pass_candidates"] == 2
    assert summary["selected_neurons"] == 2
    assert summary["classified_neurons"] == 1
    assert summary["unclassified_selected_neurons"] == 1
    assert rows[0]["stage"] == "total_candidates"


def test_graph_health_diagnostics_reports_density_and_degree_distribution() -> None:
    similarity = np.array(
        [
            [0.0, 0.9, 0.2],
            [0.9, 0.0, 0.0],
            [0.2, 0.0, 0.0],
        ]
    )

    metrics = graph_health_diagnostics(similarity, LouvainConfig(thr_prop=0.5, min_module_degree=1.0))

    assert metrics["selected_neurons"] == 3
    assert metrics["positive_similarity_fraction"] == pytest.approx(2.0 / 3.0)
    assert metrics["thresholded_edge_density"] == pytest.approx(1.0 / 3.0)
    assert metrics["degree_min"] == 0
    assert metrics["isolated_node_count"] == 1


def test_unclassified_diagnostics_explains_population_and_louvain_dropouts() -> None:
    cfg = AnalysisConfig(osi_threshold=0.5, active_threshold=0.5, filter_by_osi=True)
    reasons = unclassified_diagnostics(
        osi=np.array([0.1, 0.7, np.nan, 0.8]),
        activity=np.array([0.0, 1.0, 2.0, 3.0]),
        selected_indices=np.array([1, 3]),
        labels=np.array([0, 2]),
        cfg=cfg,
        community_diagnostics={"weak_module_degree_removed": 1, "small_cluster_removed": 0},
    )

    assert reasons["not_active_enough"] == 1
    assert reasons["finite_osi_unavailable"] == 1
    assert reasons["below_osi_threshold"] == 1
    assert reasons["selected_but_louvain_unclassified"] == 1
    assert reasons["weak_module_degree_removed"] == 1
```

- [ ] **Step 2: Run diagnostics tests and verify RED**

Run: `uv run pytest tests/test_analysis_diagnostics.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'v1_research.analysis.diagnostics'`.

- [ ] **Step 3: Implement `analysis.diagnostics`**

Create the module with these public functions:

```python
def selection_funnel(*, osi, activity, selected_indices, labels, cfg) -> tuple[list[dict[str, object]], dict[str, object]]:
    ...

def graph_health_diagnostics(similarity, cfg: LouvainConfig) -> dict[str, object]:
    ...

def unclassified_diagnostics(*, osi, activity, selected_indices, labels, cfg, community_diagnostics=None) -> dict[str, object]:
    ...
```

Use `json_ready(...)` before returning. For graph thresholding, mirror `identify_communities(...)`: positive similarities, `bct.threshold_proportional(...)`, `bct.weight_conversion(...)`, then count undirected edges from the upper triangle.

- [ ] **Step 4: Run diagnostics tests**

Run: `uv run pytest tests/test_analysis_diagnostics.py -q`

Expected: PASS.

## Task 3: Artifact Writing and Workflow Wiring

**Files:**
- Modify: `src/v1_research/analysis/artifacts.py`
- Modify: `src/v1_research/workflows/analyze.py`
- Test: `tests/test_workflows_analyze.py`

- [ ] **Step 1: Write failing workflow output test**

Extend `test_analysis_workflow_reads_simulation_bundle_and_writes_outputs` with assertions:

```python
assert (run_dir / "analysis" / "selection_funnel.json").is_file()
assert (run_dir / "analysis" / "graph_diagnostics.json").is_file()
assert (run_dir / "analysis" / "unclassified_diagnostics.json").is_file()
assert (run_dir / "tables" / "selection_funnel.csv").is_file()
manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
assert "selection_funnel" in manifest["analysis_outputs"]
```

Add a config test:

```python
def test_cli_analyze_loads_inspection_config(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "analyze.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"simulation_run: {tmp_path / 'runs' / 'simulate' / 'demo'}",
                "inspection:",
                "  enabled: true",
                "  save_plots: false",
                "  robustness:",
                "    enabled: true",
                "    tail_fractions: [0.5]",
            ]
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_run_analysis_workflow(cfg):
        captured["enabled"] = cfg.inspection.enabled
        captured["save_plots"] = cfg.inspection.save_plots
        captured["robustness_enabled"] = cfg.inspection.robustness.enabled
        captured["tail_fractions"] = tuple(cfg.inspection.robustness.tail_fractions)

        class Result:
            run_dir = tmp_path / "runs" / "simulate" / "demo"

        return Result()

    monkeypatch.setattr(cli, "run_analysis_workflow", fake_run_analysis_workflow)
    result = CliRunner().invoke(cli.app, ["analyze", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert captured == {
        "enabled": True,
        "save_plots": False,
        "robustness_enabled": True,
        "tail_fractions": (0.5,),
    }
```

- [ ] **Step 2: Run workflow tests and verify RED**

Run: `uv run pytest tests/test_workflows_analyze.py -q`

Expected: FAIL because inspection config and new artifacts do not exist.

- [ ] **Step 3: Add config dataclasses and inspection artifact writing**

Add `AnalysisRobustnessConfig` and `AnalysisInspectionConfig` to `workflows/analyze.py`, with `inspection: AnalysisInspectionConfig = field(default_factory=AnalysisInspectionConfig)` on `AnalysisWorkflowConfig`.

In `run_analysis_workflow(...)`, after `write_analysis_result(...)`, compute `activity = np.mean(result.responses_mean, axis=1)` for selected cells and call diagnostics with full unselected population values from the pipeline diagnostics where available. Write JSON via `write_json(...)` and CSV via `write_csv_rows(...)`.

- [ ] **Step 4: Run workflow tests**

Run: `uv run pytest tests/test_workflows_analyze.py tests/test_analysis_diagnostics.py -q`

Expected: PASS.

## Task 4: Default Analysis Figures

**Files:**
- Create: `src/v1_research/workflows/analysis_figures.py`
- Modify: `src/v1_research/workflows/analyze.py`
- Test: `tests/test_workflows_analyze.py`

- [ ] **Step 1: Write failing figure test**

Add assertions to the workflow test when default inspection is enabled:

```python
assert (run_dir / "figures" / "analysis_summary.png").is_file()
assert (run_dir / "figures" / "analysis_cortical_map.png").is_file()
assert (run_dir / "figures" / "analysis_similarity.png").is_file()
assert (run_dir / "figures" / "analysis_tuning.png").is_file()
assert (run_dir / "figures" / "analysis_failure_diagnosis.png").is_file()
```

Add a disabled-plots test:

```python
def test_analysis_workflow_can_disable_inspection_plots(tmp_path) -> None:
    run_dir = _simulation_bundle(tmp_path)
    result = run_analysis_workflow(
        AnalysisWorkflowConfig(
            simulation_run=run_dir,
            inspection=AnalysisInspectionConfig(save_plots=False),
            analysis=AnalysisConfig(
                osi_threshold=0.0,
                filter_by_osi=False,
                louvain=LouvainConfig(num_runs=1, consensus_reps=1, min_cluster_size=1),
            ),
        )
    )

    assert "analysis_summary_figure" not in result.output_paths
    assert not (run_dir / "figures" / "analysis_summary.png").exists()
```

- [ ] **Step 2: Run workflow tests and verify RED**

Run: `uv run pytest tests/test_workflows_analyze.py -q`

Expected: FAIL because figure files are not written.

- [ ] **Step 3: Implement figure writers**

Create `save_analysis_figures(run_dir, result, selection_rows, graph_diagnostics, unclassified_diagnostics, orientation_angles) -> dict[str, Path]`. Use `matplotlib.use("Agg")`. Create no-data panels for missing community result or fewer than two selected cells. Return keys:

- `analysis_summary_figure`
- `analysis_cortical_map_figure`
- `analysis_similarity_figure`
- `analysis_tuning_figure`
- `analysis_failure_diagnosis_figure`

- [ ] **Step 4: Run workflow tests**

Run: `uv run pytest tests/test_workflows_analyze.py -q`

Expected: PASS.

## Task 5: Optional Robustness Tables and Figure

**Files:**
- Create: `src/v1_research/analysis/robustness.py`
- Modify: `src/v1_research/workflows/analyze.py`
- Modify: `src/v1_research/workflows/analysis_figures.py`
- Test: `tests/test_analysis_robustness.py`
- Test: `tests/test_workflows_analyze.py`

- [ ] **Step 1: Write failing robustness tests**

Create `tests/test_analysis_robustness.py` with a small `AnalysisInputs` fixture and tests that:

```python
rows = run_louvain_parameter_grid(
    cfg,
    inputs,
    parameter_grid={"louvain.gamma": [0.5, 0.7]},
)
assert [row["louvain.gamma"] for row in rows] == [0.5, 0.7]
assert all("n_ensembles" in row for row in rows)
```

Add a workflow test with `inspection=AnalysisInspectionConfig(robustness=AnalysisRobustnessConfig(enabled=True, tail_fractions=(0.5,), louvain_parameter_grid={"louvain.gamma": (0.5,)}))`, asserting:

```python
assert (run_dir / "tables" / "robustness_windows.csv").is_file()
assert (run_dir / "tables" / "robustness_louvain.csv").is_file()
assert (run_dir / "analysis" / "robustness_summary.json").is_file()
assert (run_dir / "figures" / "analysis_robustness.png").is_file()
```

- [ ] **Step 2: Run robustness tests and verify RED**

Run: `uv run pytest tests/test_analysis_robustness.py tests/test_workflows_analyze.py -q`

Expected: FAIL because `analysis.robustness` is missing.

- [ ] **Step 3: Implement robustness helpers**

Implement:

```python
def run_louvain_parameter_grid(cfg: AnalysisConfig, inputs: AnalysisInputs, *, parameter_grid: Mapping[str, Sequence[object]]) -> list[dict[str, object]]:
    ...

def summarize_robustness(*, window_rows: list[dict[str, object]], louvain_rows: list[dict[str, object]]) -> dict[str, object]:
    ...
```

Only support dot paths under `louvain.` in this first implementation. Copy dataclass configs with `dataclasses.replace(...)`. Use `run_window_analysis(...)` for window rows.

- [ ] **Step 4: Wire optional robustness outputs**

When `cfg.inspection.robustness.enabled` is true, write window and Louvain rows to tables, write `robustness_summary.json`, and write `analysis_robustness.png`. If no robustness rows are configured, write a summary with zero rows and skip empty CSVs.

- [ ] **Step 5: Run robustness tests**

Run: `uv run pytest tests/test_analysis_robustness.py tests/test_workflows_analyze.py -q`

Expected: PASS.

## Task 6: Docs, Config, and Full Verification

**Files:**
- Modify: `configs/analyze_louvain.yaml`
- Modify: `docs/analysis.md`
- Modify: `docs/quickstart.md`
- Modify: `docs/parameters.md`

- [ ] **Step 1: Update config and docs**

Add `inspection:` to `configs/analyze_louvain.yaml` with default enabled inspection and disabled robustness. Update docs output lists to mention:

- `analysis/selection_funnel.json`
- `analysis/graph_diagnostics.json`
- `analysis/unclassified_diagnostics.json`
- `tables/selection_funnel.csv`
- `figures/analysis_*.png`
- optional robustness outputs

- [ ] **Step 2: Run focused and full tests**

Run:

```bash
uv run pytest tests/test_analysis_diagnostics.py tests/test_analysis_robustness.py tests/test_workflows_analyze.py -q
uv run pytest -q
```

Expected: PASS.

- [ ] **Step 3: Commit implementation**

Stage only task-related files:

```bash
git add -f docs/superpowers/specs/2026-06-25-analyze-diagnostics-design.md docs/superpowers/plans/2026-06-25-analyze-diagnostics.md
git add src/v1_research/analysis/diagnostics.py src/v1_research/analysis/robustness.py src/v1_research/workflows/analysis_figures.py
git add src/v1_research/analysis/communities.py src/v1_research/analysis/artifacts.py src/v1_research/workflows/analyze.py src/v1_research/analysis/__init__.py
git add tests/test_analysis_diagnostics.py tests/test_analysis_robustness.py tests/test_workflows_analyze.py tests/test_analysis_clusters.py
git add configs/analyze_louvain.yaml docs/analysis.md docs/quickstart.md docs/parameters.md
git commit -m "feat: add analyze diagnostics"
```

Expected: commit succeeds without staging `.gitignore` or `.superpowers/`.

## Self-Review

- Spec coverage: Lane 1 is covered by Task 4; Lane 2 by Tasks 1-3; Lane 3 by Task 5; docs/config/testing by Task 6.
- Placeholder scan: no red-flag placeholder instructions are present.
- Type consistency: config types are introduced before workflow tests use them, diagnostics helpers return JSON-ready dict/list shapes, and figure path keys are named consistently.
