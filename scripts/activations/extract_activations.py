"""Extract and cache activations at the final period of each prompt in a suite.

Activations are cached in outputs/activations/<model>/<suite>/<dataset>.npz.

Usage:
    uv run python scripts/activations/extract_activations.py \
        --model Qwen/Qwen2.5-1.5B --suite rosch
"""

import argparse

from spar_typicality.activations import (
    model_num_layers,
    select_layers,
    suite_activations,
)
from spar_typicality.provenance import provenance
from spar_typicality.suites import load_suite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--layer-start", default=5, type=int)
    parser.add_argument("--layer-step", default=5, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    layers = select_layers(
        model_num_layers(args.model), args.layer_start, args.layer_step
    )
    print("Layers:", layers)
    datasets = load_suite(args.suite, dataset_names=args.datasets)
    suite_activations(
        args.model,
        args.suite,
        datasets,
        layers,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
        overwrite=args.overwrite,
        extra_metadata=provenance(),
    )


if __name__ == "__main__":
    main()
