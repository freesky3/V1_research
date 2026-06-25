# Analyze Diagnostics Design

Date: 2026-06-25

## Goal

Extend the `analyze` workflow so each analysis run can answer three questions:

1. Does this single run look scientifically interpretable?
2. If the analysis fails or classifies too few neurons, why?
3. Are the detected ensembles stable under reasonable analysis choices?

The design keeps routine runs lightweight. A single-run result atlas and failure diagnosis are enabled by default. Robustness analysis is available behind an explicit config switch because it can rerun analysis many times.

## Current Behavior

`run_analysis_workflow(...)` loads a simulation run bundle, calls `run_analysis(...)`, writes compact artifacts with `write_analysis_result(...)`, and updates the source manifest when `output_run_root` is not set.

Current outputs include:

- `analysis/selected_indices.npy`
- `analysis/osi.npy`
- `analysis/preferred_orientation.npy`
- `analysis/responses_mean.npy`
- `analysis/steady_state_responses.npy`
- `analysis/coords.npy`
- `analysis/distance.npy`
- `analysis/diagnostics.json`
- `analysis/community_labels.npy` when Louvain runs
- `analysis/similarity.npy` when Louvain runs
- `analysis/agreement.npy` when consensus agreement is available
- `analysis/community_diagnostics.json` when Louvain runs
- `analysis/metrics.json`
- `tables/ensemble_metrics.csv`
- optional `analysis/inputs.json`

The workflow currently produces no analysis-specific figures. It also records useful scalar metrics, but it does not expose a clear selection funnel, graph-health counters, unclassified-cell reasons, or robustness summaries through the CLI workflow.

## Non-Goals

- Do not replace the existing OSI, Louvain, overlap, temporal, or metrics implementations.
- Do not migrate legacy plotting scripts wholesale.
- Do not make robustness sweeps mandatory for normal `analyze` runs.
- Do not add a new plotting dependency beyond Matplotlib and the scientific stack already used in the project.

## Proposed Configuration

Add an inspection config to the analysis workflow:

```yaml
inspection:
  enabled: true
  save_plots: true
  save_tables: true
  robustness:
    enabled: false
    tail_fractions: [0.25, 0.5, 0.75, 1.0]
    end_times: []
    louvain_parameter_grid: {}
    overlap_surrogates: 0
```

`inspection.enabled` controls all new diagnostics. `save_plots` controls PNG output. `save_tables` controls CSV/JSON diagnosis tables. `robustness.enabled` controls repeated analysis runs.

The default `analyze` path remains fast: it computes the normal analysis, writes the atlas figures, and writes failure-diagnosis counters without rerunning the pipeline.

## Output Layout

When `output_run_root` is `null`, outputs are written next to the current analysis artifacts:

```text
<simulation_run>/
  analysis/
    diagnostics.json
    metrics.json
    selection_funnel.json
    graph_diagnostics.json
    unclassified_diagnostics.json
    robustness_summary.json              # only when robustness.enabled=true
  tables/
    ensemble_metrics.csv
    selection_funnel.csv
    robustness_windows.csv               # only when configured
    robustness_louvain.csv               # only when configured
  figures/
    analysis_summary.png
    analysis_cortical_map.png
    analysis_similarity.png
    analysis_tuning.png
    analysis_failure_diagnosis.png
    analysis_robustness.png              # only when robustness.enabled=true
```

When `output_run_root` is set, the same relative structure is created under that output root, matching the current workflow behavior for tables.

Manifest output keys should include these paths when the files are created. Summary scalars should remain compact and sweep-friendly.

## Lane 1: Result Atlas

The result atlas is the default visual readout for a single run.

`analysis_summary.png` contains:

- selection funnel counts: total excitatory cells, active cells, OSI-pass cells, sampled cells, classified cells, unclassified cells
- OSI distribution and configured OSI threshold
- per-neuron mean activity distribution and active threshold
- ensemble size distribution and classified fraction

`analysis_cortical_map.png` contains:

- selected L2/3 coordinates colored by community label
- unclassified selected cells in neutral gray
- optional small panels for OSI, preferred orientation, and mean activity using the same coordinates

`analysis_similarity.png` contains:

- similarity matrix ordered by community label, with unclassified cells grouped last
- agreement matrix next to it when available
- community boundary markers on both axes

`analysis_tuning.png` contains:

- mean response by orientation for each ensemble
- optional variability ribbon across member neurons when ensemble size permits it
- member preferred-orientation coherence and mean OSI in the title or legend

These plots should tolerate small smoke runs by displaying "no data" panels instead of raising errors.

## Lane 2: Failure Diagnosis

Failure diagnosis should explain bad or sparse outputs without requiring a debugger.

`selection_funnel.json` and `selection_funnel.csv` record:

- total candidate excitatory cells after any center crop
- active candidates above `active_threshold`
- finite OSI cells
- cells at or above `osi_threshold`
- selected cells after `filter_by_osi` and `random_sample_fraction`
- Louvain-classified cells
- Louvain-unclassified cells

`graph_diagnostics.json` records graph health for the selected activity trace:

- similarity kind
- selected neuron count
- positive similarity fraction
- thresholded edge density after `thr_prop`
- degree mean, median, p05, p95, min, max
- isolated node count
- weak-module-degree candidate count using `min_module_degree`

`unclassified_diagnostics.json` records why cells are unavailable or unclassified:

- not active enough
- finite OSI unavailable
- below OSI threshold
- removed by random sampling
- not selected by current filter mode
- selected but assigned Louvain label `0`
- clusters removed for `min_cluster_size`
- nodes removed for `min_module_degree`

To make the last two reasons precise, `identify_communities(...)` should expose cleanup counts from the weak-node and small-cluster pruning step. This can be done by adding fields to community diagnostics without changing the public label semantics.

`analysis_failure_diagnosis.png` visualizes the same data as:

- funnel bar chart
- OSI/activity threshold histograms
- thresholded graph degree histogram
- unclassified reason bar chart

## Lane 3: Robustness Tracker

Robustness analysis is optional and reruns parts of the pipeline.

Window robustness reuses `run_window_analysis(...)` and writes `tables/robustness_windows.csv`. Each row includes:

- window kind and value
- time index range and time-point count
- status
- selected-neuron count
- `n_ensembles`
- `classified_fraction`
- `osi_mean`
- `within_similarity_mean`
- `between_similarity_mean`

Louvain robustness runs the same loaded inputs over a configured grid of Louvain parameters. It writes `tables/robustness_louvain.csv`. Each row includes:

- varied parameter values
- status
- selected-neuron count
- `n_ensembles`
- `classified_fraction`
- ensemble size statistics
- within and between similarity summaries

Overlap robustness compares non-zero labels between the baseline analysis and robustness variants when coordinates can be matched. It records:

- matched selected-neuron count
- adjusted Rand index
- best community matches
- optional surrogate p-value when `overlap_surrogates > 0`

`analysis_robustness.png` contains:

- metrics over window choices
- metrics over Louvain parameter choices
- ARI/overlap heatmap against the baseline
- warning panel when variants have too few matched cells for meaningful comparison

## Data Flow

The new workflow remains orchestration-only:

```text
run_analysis_workflow
  -> load_analysis_inputs_from_simulation
  -> run_analysis
  -> write_analysis_result
  -> compute_analysis_inspection
  -> write_analysis_inspection
  -> save_analysis_figures
  -> optional run_analysis_robustness
  -> update manifest
```

Pure calculations live under `v1_research.analysis`. Workflow-specific IO and Matplotlib figure creation live under `v1_research.workflows` or `analysis.artifacts`, following the existing `simulate` inspection pattern.

Suggested modules:

- `analysis.diagnostics`: selection funnel, graph diagnostics, unclassified diagnostics
- `analysis.robustness`: window, Louvain-grid, and overlap robustness rows
- `workflows.analysis_figures`: Matplotlib figure writers for analysis outputs
- `workflows.analyze`: config wiring, IO, manifest updates

## Error Handling

- Missing trajectory remains supported through the existing mean-rate fallback. Result atlas still works; robustness plots should report that temporal robustness has only one time point when applicable.
- If selected neurons are fewer than two, figure writers still create summary/failure figures and skip similarity/tuning panels that require pairs or ensembles.
- If Louvain does not run, community-specific files are not written, and failure diagnosis records `not_enough_neurons`.
- If robustness variants fail validation, their row records the status and error message while the baseline analysis output remains usable.
- Figure writers should never mutate analysis arrays and should close Matplotlib figures after saving.

## Testing

Add focused tests for:

- `selection_funnel` counts for OSI-filtered and activity-filtered modes
- graph diagnostics on a small known similarity matrix
- unclassified diagnostics when clusters are removed for weak degree or small size
- workflow writes inspection JSON/CSV and figure paths when enabled
- workflow omits robustness files when `robustness.enabled=false`
- robustness window rows reuse existing `run_window_analysis(...)`
- figure writers handle small smoke runs without raising
- manifest includes new outputs only when the files exist

Existing tests for OSI, Louvain, metrics, workflow analyze, temporal windows, and overlap should continue to pass unchanged.

## Recommended Build Order

1. Add inspection config dataclasses and wire them through `AnalysisWorkflowConfig`.
2. Add pure diagnostic helpers for selection funnel, graph health, and unclassified reasons.
3. Write JSON/CSV inspection artifacts and update manifest paths.
4. Add the four default result-atlas figures plus failure diagnosis figure.
5. Add optional robustness tables using existing temporal and overlap helpers.
6. Add the robustness summary figure.
7. Update docs and quickstart output lists.

This order gives useful single-run diagnostics before the heavier repeated-analysis machinery.
