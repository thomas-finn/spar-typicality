import numpy as np

from spar_typicality.probe_plots import probe_html, probe_payload, quantise_vectors
from spar_typicality.probes import (
    DIFF_MEANS,
    RIDGE,
    fit_diff_means,
    fit_probe,
    fit_ridge,
    kendall_tau_b,
    pearson,
    ranks,
    spearman,
    typicality_scores,
)


def synthetic_data(n_rows=60, d_model=200, seed=0):
    """Rows where the score is coded along the first axis, plus noise."""
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=n_rows)
    matrix = rng.normal(size=(n_rows, d_model))
    matrix[:, 0] += 5 * scores
    return matrix, scores


def test_typicality_scores_flip_sign():
    assert np.array_equal(typicality_scores([-1.5, 0.0, 2.0]), [1.5, 0.0, -2.0])


def test_ranks_with_ties():
    assert np.array_equal(ranks([10, 30, 20, 20]), [1, 4, 2.5, 2.5])


def test_correlations():
    first = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(pearson(first, 2 * first + 1), 1)
    assert np.isclose(spearman(first, first**3), 1)
    assert np.isclose(kendall_tau_b(first, -first), -1)
    # One swapped pair of six: (5 - 1) / 6.
    assert np.isclose(kendall_tau_b(first, [1.0, 3.0, 2.0, 4.0]), 4 / 6)


def test_diff_means_points_to_typical():
    matrix, scores = synthetic_data()
    probe = fit_diff_means(matrix, scores)
    assert probe.probe_type == DIFF_MEANS
    assert np.isclose(np.linalg.norm(probe.direction), 1)
    assert probe.direction[0] > 0.8
    assert probe.details["n_top"] == 20
    assert spearman(matrix @ probe.direction, scores) > 0.8


def test_ridge_points_to_typical_and_loo_is_exact():
    matrix, scores = synthetic_data()
    probe = fit_ridge(matrix, scores, [1.0])
    assert probe.probe_type == RIDGE
    assert probe.direction[0] > 0.5

    # Compare the hat matrix leave-one-out error with explicit refits.
    alpha = probe.details["alpha"]
    errors = []
    for held_out in range(len(scores)):
        keep = np.arange(len(scores)) != held_out
        x_mean = matrix[keep].mean(axis=0)
        y_mean = scores[keep].mean()
        centred = matrix[keep] - x_mean
        weights = np.linalg.solve(
            centred.T @ centred + alpha * np.eye(matrix.shape[1]),
            centred.T @ (scores[keep] - y_mean),
        )
        prediction = (matrix[held_out] - x_mean) @ weights + y_mean
        errors.append((scores[held_out] - prediction) ** 2)
    assert np.isclose(probe.details["loo_mse"], np.mean(errors))


def test_ridge_selects_alpha_from_grid():
    matrix, scores = synthetic_data()
    config = {"ridge_alpha_factors": [0.001, 0.1, 10.0], "diff_means_fraction": 1 / 3}
    probe = fit_probe(RIDGE, matrix, scores, config)
    assert probe.details["alpha_factor"] in [0.001, 0.1, 10.0]


def test_quantise_keeps_order():
    vectors = np.random.default_rng(0).normal(size=(3, 50))
    encoded = quantise_vectors(vectors)
    import base64

    levels = np.frombuffer(base64.b64decode(encoded["data"]), dtype="<u2")
    levels = levels.reshape(3, 50)
    for index in range(3):
        assert np.array_equal(np.argsort(levels[index]), np.argsort(vectors[index]))


def test_probe_payload_and_html():
    rows = []
    for category in ["bird", "toy"]:
        for k in range(4):
            rows.append(
                {
                    "item": category + str(k),
                    "category": category,
                    "typicality_rating_normalised": float(k),
                    "is_member": 1,
                }
            )
    metric_rows = []
    for train_category in ["bird", "toy"]:
        for eval_dataset in ["a", "b"]:
            for eval_category in ["bird", "toy"]:
                metric_rows.append(
                    {
                        "train_dataset": "a",
                        "probe": RIDGE,
                        "layer": 5,
                        "train_category": train_category,
                        "eval_dataset": eval_dataset,
                        "eval_category": eval_category,
                        "pearson": 0.5,
                        "spearman": 0.4,
                        "kendall": 0.3,
                    }
                )
    projections = np.zeros((1, 1, 1, 2, 2, len(rows)))
    payload = probe_payload(
        "test",
        [5],
        ["a"],
        [RIDGE],
        ["bird", "toy"],
        ["a", "b"],
        rows,
        metric_rows,
        projections,
    )
    assert len(payload["metrics"]["spearman"]) == 8
    assert payload["metrics"]["kendall"][0] == 0.3
    page = probe_html(payload)
    assert page.count("<script>") == 1
    assert "__DATA__" not in page
