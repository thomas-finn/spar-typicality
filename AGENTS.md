# AGENTS.md

## Project

This repository contains code and experiments for investigating how typicality may be internally represented in LLMs.

## Environment

- Python version: see `.python-version`
- Package manager: `uv`
- Install/sync environment with:

    uv sync

- Run Python with:

    uv run python ...

## Repository structure

- `src/` — reusable project code
- `scripts/` — command-line entry points
- `experiments/` — experiment configurations and notes
- `notebooks/` — exploratory analysis only
- `data/raw/` — immutable source data
- `data/processed/` — generated datasets
- `outputs/` — generated results, never committed

## Rules

1. Put reusable logic in `src/`, not notebooks.
2. Keep notebooks thin; import code from `src/`.
3. Never commit datasets, model checkpoints, or generated outputs unless explicitly requested.
4. Never modify files in `data/raw/`.
5. Every experiment must have a configuration and a short README.
6. Make experiments reproducible: record the config, random seed, git commit, and relevant package/environment information.
7. Prefer small, composable functions over large scripts.
8. Add tests when changing reusable functionality.
9. Do not silently change experiment configurations or data-processing assumptions.
10. Before making a large structural change, explain the proposed change.

## Running things

Example:

    uv run python scripts/run_experiment.py experiments/001_baseline/config.yaml

Tests:

    uv run pytest

Formatting/linting:

    uv run ruff check .
    uv run ruff format .

## Experiment conventions

An experiment should contain:

- `config.yaml`
- `README.md`
- results should go under `outputs/`, not inside the experiment directory

Each experiment must have a row in `experiments/experiments.md`. Add the row
when the experiment is created and update it after each run.

The experiment README should record:

- hypothesis/question
- dataset
- configuration
- command used
- result
- interpretation
- next experiment

## Colab

Code intended to run on Colab should use the same `src/` package as local experiments.

Do not create a separate implementation specifically for Colab unless necessary.

## Artifacts and storage

Large experiment artifacts must not be committed to Git.

Colab uses local disk as temporary scratch space only.

Persistent experiment artifacts live in the configured cloud artifact
store, under:

    <bucket>/<project>/experiments/<run_id>/

Each run must contain metadata describing:
- git commit
- experiment/config
- random seed
- environment
- run status

Small metrics and experiment summaries should be easy to retrieve without
downloading large artifacts.

Activations and other large tensors should only be persisted when required
by the experiment.

## Code conventions

 - Use simplified technical English by the ASD-STE100 standard
 - Lean towards human readable code rather than pythonic 
 - Lean towards being less verbose in code comments