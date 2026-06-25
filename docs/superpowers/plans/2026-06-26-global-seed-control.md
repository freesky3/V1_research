# Global Seed Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single top-level `seed` knob for CLI/YAML workflow runs and make `full`/`sweep` consume one continuous random stream per command.

**Architecture:** Add one shared seed helper that initializes Python, NumPy, and optional Torch state. Expose `seed` on each top-level workflow config so `--config` and `-o seed=...` work uniformly. Apply the seed once per command at workflow entry; for composed workflows, clear child seeds before dispatch so the stream is not reset mid-command.

**Tech Stack:** Python dataclasses, Typer, OmegaConf, NumPy, optional Torch import.

---

### Task 1: Add seed helper and config fields

**Files:**
- Create: `src/v1_research/seed.py`
- Modify: `src/v1_research/workflows/train.py`
- Modify: `src/v1_research/workflows/simulate.py`
- Modify: `src/v1_research/workflows/analyze.py`
- Modify: `src/v1_research/workflows/full.py`
- Modify: `src/v1_research/workflows/sweep.py`
- Modify: `src/v1_research/workflows/__init__.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_set_global_seed_repeats_numpy_and_random_streams():
    ...

def test_workflow_config_loads_seed_from_yaml_and_override():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_seed_control.py -v`

- [ ] **Step 3: Write minimal implementation**

```python
def set_global_seed(seed: int | None) -> None:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_seed_control.py -v`

### Task 2: Apply seeds inside workflow entrypoints

**Files:**
- Modify: `src/v1_research/workflows/train.py`
- Modify: `src/v1_research/workflows/simulate.py`
- Modify: `src/v1_research/workflows/analyze.py`
- Modify: `src/v1_research/workflows/full.py`
- Modify: `src/v1_research/workflows/sweep.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_full_workflow_uses_one_seed_across_train_and_simulate():
    ...

def test_sweep_uses_one_seed_across_all_grid_points():
    ...

def test_simulation_seed_makes_trial_schedule_reproducible():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_seed_control.py -v`

- [ ] **Step 3: Write minimal implementation**

```python
set_global_seed(cfg.seed)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_seed_control.py -v`

### Task 3: Record seed in run manifests and update docs

**Files:**
- Modify: `src/v1_research/workflows/train.py`
- Modify: `src/v1_research/workflows/simulate.py`
- Modify: `src/v1_research/workflows/analyze.py`
- Modify: `src/v1_research/workflows/full.py`
- Modify: `src/v1_research/workflows/sweep.py`
- Modify: `docs/quickstart.md`
- Modify: `docs/parameters.md`
- Modify: `docs/workflows.md`
- Modify: `docs/sweeps.md`

- [ ] **Step 1: Write the failing tests**

```python
def test_run_manifest_records_seed():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_seed_control.py -v`

- [ ] **Step 3: Write minimal implementation**

```python
write_manifest(..., {"seed": cfg.seed, ...})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_seed_control.py -v`

