import numpy as np
import pytest

from spar_typicality.activations import (
    final_period_token_index,
    load_cached_layers,
    save_activations,
    select_layers,
)
from spar_typicality.pca import fit_pca
from spar_typicality.pca_plots import pca_figure
from spar_typicality.suites import load_suite


def test_select_layers():
    assert select_layers(28, 5, 5) == [5, 10, 15, 20, 25]
    assert select_layers(30, 5, 5) == [5, 10, 15, 20, 25, 30]
    with pytest.raises(ValueError):
        select_layers(4, 5, 5)


def test_final_period_token_index():
    # Tokens: "A", " doll", " is", " a", " toy", "."
    prompt = "A doll is a toy."
    offsets = [(0, 1), (1, 6), (6, 9), (9, 11), (11, 15), (15, 16)]
    assert final_period_token_index(prompt, offsets) == 5


def test_final_period_token_index_uses_last_period_and_merged_tokens():
    # Tokens: "<s>", "toy", ":", " doll", ".", " 21."
    prompt = "toy: doll. 21."
    offsets = [(0, 0), (0, 3), (3, 4), (4, 9), (9, 10), (10, 14)]
    assert final_period_token_index(prompt, offsets) == 5


def test_final_period_token_index_without_period():
    with pytest.raises(ValueError):
        final_period_token_index("no period", [(0, 2), (2, 9)])


def test_cache_round_trip(tmp_path):
    path = tmp_path / "cache.npz"
    activations = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    save_activations(path, activations, [5, 10, 15], ["a.", "b."], [1, 1])

    selected = load_cached_layers(path, ["a.", "b."], [15, 5])
    assert np.array_equal(selected, activations[:, [2, 0], :])
    assert load_cached_layers(path, ["a.", "b."], [20]) is None
    assert load_cached_layers(path, ["a.", "c."], [5]) is None
    assert load_cached_layers(tmp_path / "missing.npz", ["a."], [5]) is None


def test_fit_pca_finds_main_direction():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=(200, 1)) * 10
    matrix = signal @ np.array([[1.0, 0.0, 0.0, 0.0]]) + rng.normal(size=(200, 4))
    result = fit_pca(matrix, 3)
    assert result.projections.shape == (200, 3)
    assert result.components.shape == (3, 4)
    assert abs(result.components[0, 0]) > 0.99
    assert result.components[0, 0] > 0
    ratios = result.explained_variance_ratio
    assert ratios[0] > 0.9
    assert ratios[0] >= ratios[1] >= ratios[2]


def test_load_suite_and_figure():
    datasets = load_suite("rosch", dataset_names=["rosch_example", "rosch_neutral"])
    assert [dataset.name for dataset in datasets] == ["rosch_example", "rosch_neutral"]
    rng = np.random.default_rng(0)
    for dataset in datasets:
        assert dataset.prompts()[0].endswith(".")
        matrix = rng.normal(size=(len(dataset.rows), 8))
        results = {5: fit_pca(matrix), 10: fit_pca(matrix)}
        figure = pca_figure(dataset.name, dataset.rows, results)
        visible = [trace for trace in figure.data if trace.visible]
        assert len(visible) > 0
        assert len(figure.layout.updatemenus[0].buttons) == 4
