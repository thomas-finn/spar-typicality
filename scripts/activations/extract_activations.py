"""Extract and cache activations at the final period of each prompt in a suite.

Activations are cached in outputs/activations/<model>/<suite>/<dataset>.npz,
or <dataset>.float16.npz with --storage-dtype float16.

With --config, the model, suite, datasets, layers, storage dtype and run
settings come from an experiment config. Command line options replace
config values.

Usage:
    uv run python scripts/activations/extract_activations.py \
        --model Qwen/Qwen2.5-1.5B --suite rosch
    uv run python scripts/activations/extract_activations.py \
        --config experiments/probes/probe-02/config.yaml
"""

import argparse

from spar_typicality.activations import (
    config_layers,
    model_num_layers,
    suite_activations,
)
from spar_typicality.provenance import provenance
from spar_typicality.runs import read_config
from spar_typicality.suites import load_suite

DEFAULTS = {
    "datasets": None,
    "layers": None,
    "layer_start": 5,
    "layer_step": 5,
    "batch_size": 16,
    "device": "auto",
    "dtype": "auto",
    "activation_storage_dtype": "float32",
}


def extraction_settings(args):
    """Return the settings: defaults, then the config, then the options."""
    settings = dict(DEFAULTS)
    if args.config is not None:
        config = read_config(args.config, {})
        settings.update(config)
        # Probe configs give the datasets as eval_datasets. null means all.
        if "eval_datasets" in config:
            settings["datasets"] = config["eval_datasets"]
    options = {
        "model": args.model,
        "suite": args.suite,
        "datasets": args.datasets,
        "layers": args.layers,
        "layer_start": args.layer_start,
        "layer_step": args.layer_step,
        "batch_size": args.batch_size,
        "device": args.device,
        "dtype": args.dtype,
        "activation_storage_dtype": args.storage_dtype,
    }
    for key, value in options.items():
        if value is not None:
            settings[key] = value
    # --layer-start or --layer-step replace a layer list from the config.
    start_or_step = args.layer_start is not None or args.layer_step is not None
    if start_or_step and args.layers is None:
        settings["layers"] = None
    for key in ["model", "suite"]:
        if settings.get(key) is None:
            raise ValueError("Give --" + key + " or a --config that has it.")
    return settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--suite", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--layer-start", default=None, type=int)
    parser.add_argument("--layer-step", default=None, type=int)
    parser.add_argument("--batch-size", default=None, type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--storage-dtype", default=None, choices=["float32", "float16"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    settings = extraction_settings(args)
    layers = config_layers(settings, model_num_layers(settings["model"]))
    print("Model:", settings["model"])
    print("Layers:", layers)
    print("Storage dtype:", settings["activation_storage_dtype"])
    datasets = load_suite(settings["suite"], dataset_names=settings["datasets"])
    suite_activations(
        settings["model"],
        settings["suite"],
        datasets,
        layers,
        batch_size=settings["batch_size"],
        device=settings["device"],
        dtype=settings["dtype"],
        overwrite=args.overwrite,
        extra_metadata=provenance(),
        storage_dtype=settings["activation_storage_dtype"],
    )


if __name__ == "__main__":
    main()
