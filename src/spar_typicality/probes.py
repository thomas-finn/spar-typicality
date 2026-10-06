"""Linear typicality probes and correlation metrics, with numpy.

Typicality score convention: the score is the negative of the typicality
z-score, so a higher score means a more typical item. Each probe direction
is a unit vector that points towards more typical items. A projection that
keeps the typicality order thus has a positive correlation with the score.
"""

import numpy as np

DIFF_MEANS = "diff_means"
RIDGE = "ridge"
PROBE_TYPES = [DIFF_MEANS, RIDGE]


def typicality_scores(z_scores):
    """Return scores where higher means more typical."""
    return -np.asarray(z_scores, dtype=np.float64)


def unit_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("The probe direction has zero length.")
    return vector / norm


class Probe:
    """A fitted probe. `direction` is a unit vector of length d_model.

    `details` holds values that are specific to the probe type.
    """

    def __init__(self, probe_type, direction, details):
        self.probe_type = probe_type
        self.direction = direction
        self.details = details


def fit_diff_means(matrix, scores, fraction=1 / 3):
    """Return the direction from the mean of the least typical rows to the
    mean of the most typical rows.

    Each group has int(n_rows * fraction) rows. A stable sort decides ties.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    count = int(len(scores) * fraction)
    if count < 1:
        raise ValueError("Too few rows for a difference-of-means probe.")
    order = np.argsort(scores, kind="stable")
    bottom = order[:count]
    top = order[-count:]
    difference = matrix[top].mean(axis=0) - matrix[bottom].mean(axis=0)
    details = {"n_top": count, "n_bottom": count}
    return Probe(DIFF_MEANS, unit_vector(difference), details)


def fit_ridge(matrix, scores, alpha_factors):
    """Fit ridge regression of `scores` on the rows of `matrix`.

    The intercept is not penalised. The penalty is alpha = factor * mean
    squared row norm of the centred matrix, so the same factors are usable
    at each layer. The factor with the lowest leave-one-out mean squared
    error is selected. The leave-one-out errors are exact and come from the
    hat matrix, so no refit is necessary.

    The solution uses the n x n Gram matrix, because n_rows << d_model.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    n_rows = len(scores)
    centred = matrix - matrix.mean(axis=0)
    centred_scores = scores - scores.mean()

    gram = centred @ centred.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.clip(eigenvalues, 0, None)
    scale = np.trace(gram) / n_rows
    rotated_scores = eigenvectors.T @ centred_scores

    best = None
    for factor in alpha_factors:
        alpha = factor * scale
        shrink = eigenvalues / (eigenvalues + alpha)
        fitted = eigenvectors @ (shrink * rotated_scores)
        hat_diagonal = (eigenvectors**2) @ shrink + 1 / n_rows
        leverage_gap = np.maximum(1 - hat_diagonal, 1e-12)
        loo_residuals = (centred_scores - fitted) / leverage_gap
        loo_mse = float(np.mean(loo_residuals**2))
        if best is None or loo_mse < best["loo_mse"]:
            best = {
                "alpha_factor": float(factor),
                "alpha": float(alpha),
                "loo_mse": loo_mse,
                "loo_predictions": scores - loo_residuals,
            }

    dual = eigenvectors @ (rotated_scores / (eigenvalues + best["alpha"]))
    weights = centred.T @ dual
    details = {
        "alpha_factor": best["alpha_factor"],
        "alpha": best["alpha"],
        "loo_mse": best["loo_mse"],
        "loo_spearman": spearman(best["loo_predictions"], scores),
        "weight_norm": float(np.linalg.norm(weights)),
    }
    return Probe(RIDGE, unit_vector(weights), details)


def fit_probe(probe_type, matrix, scores, config):
    if probe_type == DIFF_MEANS:
        return fit_diff_means(matrix, scores, config["diff_means_fraction"])
    if probe_type == RIDGE:
        return fit_ridge(matrix, scores, config["ridge_alpha_factors"])
    raise ValueError("Unknown probe type: " + probe_type)


def ranks(values):
    """Return ranks from 1 to n. Tied values get the mean of their ranks."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    result = np.empty(len(values))
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        result[order[start : end + 1]] = (start + end) / 2 + 1
        start = end + 1
    return result


def pearson(first, second):
    first = np.asarray(first, dtype=np.float64) - np.mean(first)
    second = np.asarray(second, dtype=np.float64) - np.mean(second)
    denominator = np.sqrt(np.sum(first**2) * np.sum(second**2))
    if denominator == 0:
        return float("nan")
    return float(np.sum(first * second) / denominator)


def spearman(first, second):
    return pearson(ranks(first), ranks(second))


def kendall_tau_b(first, second):
    """Kendall's tau-b. It is +1 if all pairs have the same order in both."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first_signs = np.sign(first[:, None] - first[None, :])
    second_signs = np.sign(second[:, None] - second[None, :])
    upper = np.triu_indices(len(first), k=1)
    first_signs = first_signs[upper]
    second_signs = second_signs[upper]
    numerator = np.sum(first_signs * second_signs)
    denominator = np.sqrt(np.sum(first_signs != 0) * np.sum(second_signs != 0))
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def correlation_metrics(projections, scores):
    return {
        "pearson": pearson(projections, scores),
        "spearman": spearman(projections, scores),
        "kendall": kendall_tau_b(projections, scores),
    }
