"""Extract and cache residual stream activations at the final period of a prompt.

Layer convention: layer L is the output of transformer block L (1-indexed),
which is `hidden_states[L]` in Hugging Face models. `hidden_states[0]` is the
embedding output. In most Hugging Face models the last entry of
`hidden_states` also has the final norm applied.
"""

import json
from pathlib import Path

import numpy as np

from spar_typicality.suites import REPO_ROOT

ACTIVATIONS_DIR = REPO_ROOT / "outputs" / "activations"


def select_layers(num_layers, start, step):
    """Return the layers start, start + step, ... that are <= num_layers."""
    if start < 1 or start > num_layers:
        raise ValueError(
            "Start layer " + str(start) + " is not in 1.." + str(num_layers)
        )
    return list(range(start, num_layers + 1, step))


def final_period_token_index(prompt, offsets):
    """Return the index of the token that contains the last '.' of the prompt.

    `offsets` is the list of (start, end) character spans of each token,
    from a fast tokenizer. Special tokens have the span (0, 0).
    """
    period_index = prompt.rfind(".")
    if period_index == -1:
        raise ValueError("Prompt has no period: " + repr(prompt))
    for token_index, (start, end) in enumerate(offsets):
        if start <= period_index < end:
            return token_index
    raise ValueError("No token contains the final period of: " + repr(prompt))


def model_slug(model_name):
    """Return a model name that is safe to use as a directory name."""
    return model_name.replace("/", "__")


def cache_dir(model_name, suite, activations_dir=ACTIVATIONS_DIR):
    return Path(activations_dir) / model_slug(model_name) / suite


def cache_path(model_name, suite, dataset_name, activations_dir=ACTIVATIONS_DIR):
    return cache_dir(model_name, suite, activations_dir) / (dataset_name + ".npz")


def save_activations(path, activations, layers, prompts, token_indices):
    """Save activations with shape (n_prompts, n_layers, d_model)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        activations=activations,
        layers=np.array(layers),
        prompts=np.array(prompts),
        token_indices=np.array(token_indices),
    )


def load_activations(path):
    """Return a dict with the keys activations, layers, prompts, token_indices."""
    with np.load(path) as data:
        return {
            "activations": data["activations"],
            "layers": [int(layer) for layer in data["layers"]],
            "prompts": [str(prompt) for prompt in data["prompts"]],
            "token_indices": [int(index) for index in data["token_indices"]],
        }


def load_cached_layers(path, prompts, layers):
    """Return cached activations for `layers`, or None if the cache is not usable.

    The cache is usable if it has the same prompts and has all `layers`.
    The result has shape (n_prompts, len(layers), d_model).
    """
    if not Path(path).exists():
        return None
    cached = load_activations(path)
    if cached["prompts"] != list(prompts):
        return None
    positions = []
    for layer in layers:
        if layer not in cached["layers"]:
            return None
        positions.append(cached["layers"].index(layer))
    return cached["activations"][:, positions, :]


def pick_device(device):
    import torch

    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(dtype, device):
    import torch

    if dtype == "auto":
        if device == "cuda":
            return torch.bfloat16
        return torch.float32
    return getattr(torch, dtype)


def load_model(model_name, device="auto", dtype="auto"):
    """Return (model, tokenizer, device) for a Hugging Face causal LM."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is necessary to find token offsets.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=pick_dtype(dtype, device)
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device


def extract_activations(model, tokenizer, device, prompts, layers, batch_size=16):
    """Return (activations, token_indices) for the final period of each prompt.

    `activations` has shape (n_prompts, len(layers), d_model) and dtype float32.
    """
    import torch

    all_activations = []
    all_token_indices = []
    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start : batch_start + batch_size]
        encoding = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            return_offsets_mapping=True,
        )
        offsets = encoding.pop("offset_mapping").tolist()
        token_indices = []
        for prompt, prompt_offsets in zip(batch_prompts, offsets):
            token_indices.append(final_period_token_index(prompt, prompt_offsets))

        encoding = encoding.to(device)
        with torch.no_grad():
            output = model(**encoding, output_hidden_states=True)

        rows = torch.arange(len(batch_prompts), device=device)
        columns = torch.tensor(token_indices, device=device)
        per_layer = []
        for layer in layers:
            hidden = output.hidden_states[layer]
            per_layer.append(hidden[rows, columns, :].float().cpu())
        batch_activations = torch.stack(per_layer, dim=1)

        all_activations.append(batch_activations.numpy())
        all_token_indices.extend(token_indices)
    return np.concatenate(all_activations, axis=0), all_token_indices


def write_cache_metadata(directory, metadata):
    """Write metadata.json for an activation cache directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "metadata.json", "w") as file:
        json.dump(metadata, file, indent=2)


def suite_activations(
    model_name,
    suite,
    datasets,
    layers,
    batch_size=16,
    device="auto",
    dtype="auto",
    overwrite=False,
    activations_dir=ACTIVATIONS_DIR,
    extra_metadata=None,
):
    """Return {dataset name: activations} and use the cache when possible.

    The model is only loaded if a dataset is not in the cache.
    Each array has shape (n_prompts, len(layers), d_model).
    """
    results = {}
    missing = []
    for dataset in datasets:
        path = cache_path(model_name, suite, dataset.name, activations_dir)
        cached = None
        if not overwrite:
            cached = load_cached_layers(path, dataset.prompts(), layers)
        if cached is None:
            missing.append(dataset)
        else:
            print("Cache hit:", path)
            results[dataset.name] = cached

    if missing:
        model, tokenizer, device = load_model(model_name, device, dtype)
        for dataset in missing:
            path = cache_path(model_name, suite, dataset.name, activations_dir)
            print("Extracting", dataset.name, "to", path)
            activations, token_indices = extract_activations(
                model, tokenizer, device, dataset.prompts(), layers, batch_size
            )
            save_activations(
                path, activations, layers, dataset.prompts(), token_indices
            )
            results[dataset.name] = activations

        metadata = {
            "model": model_name,
            "suite": suite,
            "layers": layers,
            "layer_convention": "layer L is hidden_states[L], the output of block L",
            "token_position": "token that contains the final '.' of the prompt",
            "num_hidden_layers": model.config.num_hidden_layers,
            "hidden_size": model.config.hidden_size,
            "device": device,
            "dtype": str(model.dtype),
            "datasets_extracted_last": [dataset.name for dataset in missing],
        }
        if extra_metadata is not None:
            metadata.update(extra_metadata)
        write_cache_metadata(cache_dir(model_name, suite, activations_dir), metadata)
    return results


def model_num_layers(model_name):
    """Return the number of transformer blocks without loading the weights."""
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_name).num_hidden_layers
