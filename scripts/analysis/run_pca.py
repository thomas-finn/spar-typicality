"""Run PCA on cached activations of a dataset suite and write 3D HTML plots.

Activations that are not in the cache are extracted first. PCA is fitted
separately for each dataset and each layer.

Usage:
    uv run python scripts/analysis/run_pca.py experiments/pca/pca-01/config.yaml
    uv run python scripts/analysis/run_pca.py experiments/pca/pca-01/config.yaml \
        --model Qwen/Qwen2.5-0.5B --suite rosch
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import yaml

from spar_typicality.activations import (
    model_num_layers,
    model_slug,
    select_layers,
    suite_activations,
)
from spar_typicality.pca import fit_pca
from spar_typicality.pca_plots import pca_figure, write_html
from spar_typicality.provenance import provenance
from spar_typicality.suites import REPO_ROOT, load_suite


def read_config(path, overrides):
    with open(path) as file:
        config = yaml.safe_load(file)
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def make_run_dir(config):
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


def save_pca_results(path, layers, results_by_layer):
    """Save projections, components and explained variance of each layer."""
    np.savez(
        path,
        layers=np.array(layers),
        projections=np.stack([results_by_layer[layer].projections for layer in layers]),
        components=np.stack([results_by_layer[layer].components for layer in layers]),
        explained_variance_ratio=np.stack(
            [results_by_layer[layer].explained_variance_ratio for layer in layers]
        ),
        mean=np.stack([results_by_layer[layer].mean for layer in layers]),
    )


def run(config, run_dir):
    random.seed(config["seed"])
    np.random.seed(config["seed"])

    layers = select_layers(
        model_num_layers(config["model"]), config["layer_start"], config["layer_step"]
    )
    print("Layers:", layers)
    datasets = load_suite(config["suite"], dataset_names=config.get("datasets"))
    activations = suite_activations(
        config["model"],
        config["suite"],
        datasets,
        layers,
        batch_size=config["batch_size"],
        device=config["device"],
        dtype=config["dtype"],
        extra_metadata=provenance(),
    )

    summary_rows = []
    for dataset in datasets:
        results_by_layer = {}
        for position, layer in enumerate(layers):
            matrix = activations[dataset.name][:, position, :]
            result = fit_pca(matrix, config["n_components"])
            results_by_layer[layer] = result
            row = {"dataset": dataset.name, "layer": layer}
            for index, ratio in enumerate(result.explained_variance_ratio):
                row["pc" + str(index + 1)] = float(ratio)
            summary_rows.append(row)

        save_pca_results(
            run_dir / "pca" / (dataset.name + ".npz"), layers, results_by_layer
        )
        title = dataset.name + " - " + config["model"]
        figure = pca_figure(title, dataset.rows, results_by_layer)
        write_html(figure, run_dir / "html" / (dataset.name + ".html"))
        print("Wrote plot for", dataset.name)

    with open(run_dir / "explained_variance.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    return layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--model", default=None)
    parser.add_argument("--suite", default=None)
    args = parser.parse_args()

    config = read_config(args.config, {"model": args.model, "suite": args.suite})
    if config["n_components"] != 3:
        raise ValueError("The 3D plots need n_components = 3.")
    run_id, run_dir = make_run_dir(config)
    (run_dir / "pca").mkdir()
    with open(run_dir / "config.yaml", "w") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    metadata = {
        "run_id": run_id,
        "config_path": str(args.config),
        "config": config,
        "seed": config["seed"],
        "status": "running",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    metadata.update(provenance())
    write_metadata(run_dir, metadata)
    print("Run directory:", run_dir)

    try:
        metadata["layers"] = run(config, run_dir)
        metadata["status"] = "completed"
    except BaseException:
        metadata["status"] = "failed"
        raise
    finally:
        metadata["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        write_metadata(run_dir, metadata)


if __name__ == "__main__":
    main()
