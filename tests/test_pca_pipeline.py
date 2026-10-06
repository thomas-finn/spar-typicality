import numpy as np
import pytest

from spar_typicality.activations import (
    final_period_token_index,
    load_cached_layers,
    save_activations,
    select_layers,
)
from spar_typicality.pca import fit_pca, load_pca_results
from spar_typicality.pca_plots import plot_html, plot_payload
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


def test_load_suite_and_plot():
    datasets = load_suite("rosch", dataset_names=["rosch_example", "rosch_neutral"])
    assert [dataset.name for dataset in datasets] == ["rosch_example", "rosch_neutral"]
    rng = np.random.default_rng(0)
    plot_datasets = []
    for dataset in datasets:
        assert dataset.prompts()[0].endswith(".")
        matrix = rng.normal(size=(len(dataset.rows), 8))
        results = {5: fit_pca(matrix), 10: fit_pca(matrix)}
        plot_datasets.append((dataset.name, dataset.rows, results))

    payload = plot_payload("test", [5, 10], plot_datasets)
    assert payload["layers"] == [5, 10]
    assert "bird" in payload["categories"]
    assert None not in payload["categories"]
    assert [dataset["name"] for dataset in payload["datasets"]] == [
        "rosch_example",
        "rosch_neutral",
    ]
    example = payload["datasets"][0]
    assert len(example["layers"]["5"]["projections"]) == len(example["rows"])

    page = plot_html(payload)
    assert page.count("<script>") == 1
    assert '"rosch_neutral"' in page


def test_plot_payload_rejects_wrong_row_count():
    datasets = load_suite("rosch", dataset_names=["rosch_example"])
    matrix = np.random.default_rng(0).normal(size=(3, 8))
    with pytest.raises(ValueError):
        plot_payload(
            "test", [5], [("rosch_example", datasets[0].rows, {5: fit_pca(matrix)})]
        )


def test_pca_results_round_trip(tmp_path):
    matrix = np.random.default_rng(0).normal(size=(20, 6))
    result = fit_pca(matrix)
    path = tmp_path / "pca.npz"
    np.savez(
        path,
        layers=np.array([5]),
        projections=result.projections[None],
        components=result.components[None],
        explained_variance_ratio=result.explained_variance_ratio[None],
        mean=result.mean[None],
    )
    loaded = load_pca_results(path)
    assert list(loaded.keys()) == [5]
    assert np.allclose(loaded[5].projections, result.projections)
