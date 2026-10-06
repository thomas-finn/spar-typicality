import numpy as np
import pytest

from spar_typicality.activations import (
    cache_path,
    config_layers,
    load_activations,
    load_cached_layers,
    save_activations,
    suite_activations,
)
from spar_typicality.suites import Dataset


def test_config_layers():
    assert config_layers({"layer_start": 5, "layer_step": 5}, 28) == [5, 10, 15, 20, 25]
    assert config_layers({"layers": [20], "layer_start": 5, "layer_step": 5}, 28) == [
        20
    ]
    assert config_layers({"layers": [25, 10]}, 28) == [10, 25]
    with pytest.raises(ValueError):
        config_layers({"layers": [30]}, 28)


def test_cache_path_has_dtype_suffix(tmp_path):
    path32 = cache_path("Qwen/Qwen2.5-7B", "rosch", "rosch_label", tmp_path)
    path16 = cache_path("Qwen/Qwen2.5-7B", "rosch", "rosch_label", tmp_path, "float16")
    assert path32.name == "rosch_label.npz"
    assert path16.name == "rosch_label.float16.npz"
    assert path16.parent == path32.parent
    with pytest.raises(ValueError):
        cache_path("m", "s", "d", tmp_path, "bfloat16")


def test_float16_cache_round_trip(tmp_path):
    path = tmp_path / "cache.float16.npz"
    activations = np.random.default_rng(0).normal(size=(3, 2, 8)).astype(np.float32)
    save_activations(
        path, activations, [5, 20], ["a.", "b.", "c."], [1, 1, 1], "float16"
    )

    assert load_activations(path)["activations"].dtype == np.float16
    loaded = load_cached_layers(path, ["a.", "b.", "c."], [20])
    assert loaded.dtype == np.float32
    assert loaded.shape == (3, 1, 8)
    assert np.allclose(loaded[:, 0], activations[:, 1], atol=1e-2)


def test_float16_overflow_is_an_error(tmp_path):
    activations = np.full((1, 1, 2), 1e6, dtype=np.float32)
    with pytest.raises(ValueError, match="float16"):
        save_activations(tmp_path / "x.npz", activations, [5], ["a."], [1], "float16")


def test_cache_only_uses_cache_and_never_loads_model(tmp_path):
    dataset = Dataset("toy_set", [{"prompt": "a."}, {"prompt": "b."}])
    activations = np.ones((2, 1, 4), dtype=np.float32)
    path = cache_path("org/model", "suite", "toy_set", tmp_path, "float16")
    save_activations(path, activations, [20], dataset.prompts(), [1, 1], "float16")

    results = suite_activations(
        "org/model",
        "suite",
        [dataset],
        [20],
        activations_dir=tmp_path,
        storage_dtype="float16",
        cache_only=True,
    )
    assert results["toy_set"].dtype == np.float32
    assert np.array_equal(results["toy_set"], activations)

    # The float32 cache does not exist, and a missing layer is a miss too.
    # Without cache_only these would load "org/model", which does not exist.
    with pytest.raises(FileNotFoundError, match="cache_only"):
        suite_activations(
            "org/model",
            "suite",
            [dataset],
            [20],
            activations_dir=tmp_path,
            cache_only=True,
        )
    with pytest.raises(FileNotFoundError, match="toy_set.float16.npz"):
        suite_activations(
            "org/model",
            "suite",
            [dataset],
            [5, 20],
            activations_dir=tmp_path,
            storage_dtype="float16",
            cache_only=True,
        )
