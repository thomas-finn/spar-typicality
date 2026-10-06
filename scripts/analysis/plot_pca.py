"""Write the PCA HTML plot of an existing run from its saved PCA results.

This does not load a model.

Usage:
    uv run python scripts/analysis/plot_pca.py outputs/experiments/<run_id>
"""

import argparse
from pathlib import Path

import yaml

from spar_typicality.pca import load_pca_results
from spar_typicality.pca_plots import plot_payload, write_html
from spar_typicality.suites import load_suite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    with open(args.run_dir / "config.yaml") as file:
        config = yaml.safe_load(file)
    datasets = load_suite(config["suite"], dataset_names=config.get("datasets"))

    plot_datasets = []
    layers = None
    for dataset in datasets:
        results_by_layer = load_pca_results(
            args.run_dir / "pca" / (dataset.name + ".npz")
        )
        layers = list(results_by_layer.keys())
        plot_datasets.append((dataset.name, dataset.rows, results_by_layer))

    title = config["experiment"] + " - " + config["model"] + " - " + config["suite"]
    path = args.run_dir / "html" / "pca.html"
    write_html(plot_payload(title, layers, plot_datasets), path)
    print("Wrote plot to", path)


if __name__ == "__main__":
    main()
