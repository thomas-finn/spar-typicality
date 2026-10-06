"""Fit category-level typicality probes and project all datasets onto them.

For each train dataset, layer, probe type and category, a probe is fitted
to the member rows of that category only (non-members are not used). Then
every dataset in the suite is projected onto the probe direction, and the
projections of the member rows of each category are correlated with the
typicality scores. This shows:

- cross-category generalisation: other categories, same dataset
- cross-dataset generalisation: same category, other datasets

Cached activations are used when possible. Activations that are not in the
cache are extracted first.

Usage:
    uv run python scripts/analysis/run_probes.py experiments/probes/probe-01/config.yaml
    uv run python scripts/analysis/run_probes.py experiments/probes/probe-01/config.yaml \
        --model Qwen/Qwen2.5-0.5B --suite rosch
"""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import yaml

from spar_typicality.activations import (
    config_layers,
    model_num_layers,
    suite_activations,
)
from spar_typicality.probe_plots import probe_payload, write_html
from spar_typicality.probes import correlation_metrics, fit_probe, typicality_scores
from spar_typicality.provenance import provenance
from spar_typicality.runs import make_run_dir, read_config, write_metadata
from spar_typicality.suites import load_suite


def reference_rows(datasets):
    """Return rows that give the category and membership of each row index.

    Some datasets (for example rosch_neutral) have no category in the prompt.
    Their rows get the category of the same row in a dataset that has one.
    All datasets must have the same items and ratings in the same order.
    """
    reference = None
    for dataset in datasets:
        if any(row["category"] is not None for row in dataset.rows):
            reference = dataset
            break
    if reference is None:
        raise ValueError("No dataset in the suite has categories.")
    for dataset in datasets:
        same = len(dataset.rows) == len(reference.rows)
        if same:
            for row, reference_row in zip(dataset.rows, reference.rows):
                if (
                    row["item"] != reference_row["item"]
                    or row["typicality_rating_normalised"]
                    != reference_row["typicality_rating_normalised"]
                ):
                    same = False
                    break
        if not same:
            raise ValueError(
                "Rows of " + dataset.name + " do not match rows of " + reference.name
            )
    return reference.rows


def member_indices_by_category(rows):
    """Return {category: indices of member rows}, with sorted categories."""
    result = {}
    for index, row in enumerate(rows):
        if row["is_member"] == 1:
            result.setdefault(row["category"], []).append(index)
    return {category: np.array(result[category]) for category in sorted(result)}


def relation_name(train_dataset, train_category, eval_dataset, eval_category):
    same_dataset = train_dataset == eval_dataset
    same_category = train_category == eval_category
    if same_dataset and same_category:
        return "in_sample"
    if same_category:
        return "cross_dataset"
    if same_dataset:
        return "cross_category"
    return "cross_both"


def write_csv(path, rows):
    """Write rows to CSV. The columns are all keys, in order of first use."""
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarise(metric_rows):
    """Return the mean metrics for each train dataset, probe, layer and relation."""
    groups = {}
    for row in metric_rows:
        key = (row["train_dataset"], row["probe"], row["layer"], row["relation"])
        groups.setdefault(key, []).append(row)
    summary = []
    for key, rows in groups.items():
        summary_row = {
            "train_dataset": key[0],
            "probe": key[1],
            "layer": key[2],
            "relation": key[3],
            "n_cells": len(rows),
        }
        for metric in ["pearson", "spearman", "kendall"]:
            values = np.array([row[metric] for row in rows])
            summary_row[metric + "_mean"] = float(np.nanmean(values))
            summary_row[metric + "_std"] = float(np.nanstd(values))
        summary.append(summary_row)
    return summary


def print_summary(summary):
    print()
    print("Mean Spearman correlation of projection with typicality score")
    header = "train_dataset    probe       layer  " + "  ".join(
        name.rjust(14)
        for name in ["in_sample", "cross_category", "cross_dataset", "cross_both"]
    )
    print(header)
    table = {}
    for row in summary:
        key = (row["train_dataset"], row["probe"], row["layer"])
        table.setdefault(key, {})[row["relation"]] = row["spearman_mean"]
    for key, values in table.items():
        cells = "  ".join(
            format(values[name], ".3f").rjust(14)
            for name in ["in_sample", "cross_category", "cross_dataset", "cross_both"]
        )
        print(key[0].ljust(16), key[1].ljust(11), str(key[2]).rjust(5), "", cells)


def run(config, run_dir, cache_only=False):
    random.seed(config["seed"])
    np.random.seed(config["seed"])

    layers = config_layers(config, model_num_layers(config["model"]))
    print("Layers:", layers)
    datasets = load_suite(config["suite"], dataset_names=config.get("eval_datasets"))
    dataset_names = [dataset.name for dataset in datasets]
    for name in config["train_datasets"]:
        if name not in dataset_names:
            raise ValueError("Train dataset " + name + " is not an eval dataset.")
    rows = reference_rows(datasets)
    members = member_indices_by_category(rows)
    categories = list(members.keys())
    scores = typicality_scores(
        [
            np.nan
            if row["typicality_rating_normalised"] is None
            else row["typicality_rating_normalised"]
            for row in rows
        ]
    )

    activations = suite_activations(
        config["model"],
        config["suite"],
        datasets,
        layers,
        batch_size=config["batch_size"],
        device=config["device"],
        dtype=config["dtype"],
        extra_metadata=provenance(),
        storage_dtype=config.get("activation_storage_dtype", "float32"),
        cache_only=cache_only,
    )

    train_datasets = config["train_datasets"]
    probe_types = config["probe_types"]
    d_model = activations[dataset_names[0]].shape[2]
    shape = (len(train_datasets), len(probe_types), len(layers), len(categories))
    directions = np.zeros(shape + (d_model,), dtype=np.float32)
    projections = np.zeros(shape + (len(datasets), len(rows)), dtype=np.float32)

    probe_rows = []
    metric_rows = []
    for t, train_dataset in enumerate(train_datasets):
        for p, probe_type in enumerate(probe_types):
            print("Fitting", probe_type, "probes on", train_dataset)
            for l, layer in enumerate(layers):
                for c, category in enumerate(categories):
                    fit_rows = members[category]
                    matrix = activations[train_dataset][fit_rows, l, :]
                    probe = fit_probe(probe_type, matrix, scores[fit_rows], config)
                    directions[t, p, l, c] = probe.direction
                    probe_row = {
                        "train_dataset": train_dataset,
                        "probe": probe_type,
                        "layer": layer,
                        "category": category,
                        "n_fit": len(fit_rows),
                    }
                    probe_row.update(probe.details)
                    probe_rows.append(probe_row)

                    for e, eval_dataset in enumerate(dataset_names):
                        projected = activations[eval_dataset][:, l, :] @ probe.direction
                        projections[t, p, l, c, e] = projected
                        for eval_category in categories:
                            eval_rows = members[eval_category]
                            metric_row = {
                                "train_dataset": train_dataset,
                                "probe": probe_type,
                                "layer": layer,
                                "train_category": category,
                                "eval_dataset": eval_dataset,
                                "eval_category": eval_category,
                                "relation": relation_name(
                                    train_dataset, category, eval_dataset, eval_category
                                ),
                                "n": len(eval_rows),
                            }
                            metric_row.update(
                                correlation_metrics(
                                    projected[eval_rows], scores[eval_rows]
                                )
                            )
                            metric_rows.append(metric_row)

    np.savez(
        run_dir / "probes.npz",
        directions=directions,
        projections=projections,
        train_datasets=np.array(train_datasets),
        probe_types=np.array(probe_types),
        layers=np.array(layers),
        categories=np.array(categories),
        eval_datasets=np.array(dataset_names),
    )
    write_csv(run_dir / "probes.csv", probe_rows)
    write_csv(run_dir / "metrics.csv", metric_rows)
    summary = summarise(metric_rows)
    write_csv(run_dir / "summary.csv", summary)
    print_summary(summary)

    title = config["experiment"] + " - " + config["model"] + " - " + config["suite"]
    payload = probe_payload(
        title,
        layers,
        train_datasets,
        probe_types,
        categories,
        dataset_names,
        rows,
        metric_rows,
        projections,
    )
    write_html(payload, run_dir / "html" / "probes.html")
    print("Wrote plot to", run_dir / "html" / "probes.html")
    return layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--model", default=None)
    parser.add_argument("--suite", default=None)
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Fail if activations are not cached. Never load the model.",
    )
    args = parser.parse_args()

    config = read_config(args.config, {"model": args.model, "suite": args.suite})
    run_id, run_dir = make_run_dir(config)
    with open(run_dir / "config.yaml", "w") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    metadata = {
        "run_id": run_id,
        "config_path": str(args.config),
        "config": config,
        "seed": config["seed"],
        "status": "running",
        "cache_only": args.cache_only,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    metadata.update(provenance())
    write_metadata(run_dir, metadata)
    print("Run directory:", run_dir)

    try:
        metadata["layers"] = run(config, run_dir, cache_only=args.cache_only)
        metadata["status"] = "completed"
    except BaseException:
        metadata["status"] = "failed"
        raise
    finally:
        metadata["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        write_metadata(run_dir, metadata)


if __name__ == "__main__":
    main()
