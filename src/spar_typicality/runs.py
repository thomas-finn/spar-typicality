"""Config, run directory and metadata helpers for experiment scripts."""

import json
import time

import yaml

from spar_typicality.activations import model_slug
from spar_typicality.suites import REPO_ROOT


def read_config(path, overrides):
    """Read a YAML config. Values in `overrides` that are not None replace it."""
    with open(path) as file:
        config = yaml.safe_load(file)
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def make_run_dir(config):
    """Create outputs/.../<experiment>_<model>_<suite>_<timestamp>/."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_id = (
        config["experiment"]
        + "_"
        + model_slug(config["model"])
        + "_"
        + config["suite"]
        + "_"
        + timestamp
    )
    run_dir = REPO_ROOT / config["output_dir"] / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def write_metadata(run_dir, metadata):
    with open(run_dir / "metadata.json", "w") as file:
        json.dump(metadata, file, indent=2)
