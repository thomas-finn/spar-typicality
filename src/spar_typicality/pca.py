"""Principal component analysis with numpy."""

import numpy as np


class PcaResult:
    """The result of fitting PCA to a matrix of shape (n_samples, n_features).

    `projections` has shape (n_samples, n_components).
    `components` has shape (n_components, n_features).
    """

    def __init__(self, mean, components, explained_variance_ratio, projections):
        self.mean = mean
        self.components = components
        self.explained_variance_ratio = explained_variance_ratio
        self.projections = projections


def fit_pca(matrix, n_components=3):
    """Fit PCA to the rows of `matrix` and project the rows.

    The sign of each component is fixed so that its largest absolute
    loading is positive. This makes the result deterministic.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    mean = matrix.mean(axis=0)
    centred = matrix - mean
    _, singular_values, right_vectors = np.linalg.svd(centred, full_matrices=False)

    components = right_vectors[:n_components]
    for index in range(len(components)):
        largest = np.argmax(np.abs(components[index]))
        if components[index, largest] < 0:
            components[index] = -components[index]

    variances = singular_values**2
    explained_variance_ratio = variances[:n_components] / variances.sum()
    projections = centred @ components.T
    return PcaResult(mean, components, explained_variance_ratio, projections)


def load_pca_results(path):
    """Load a file written by `save_pca_results` in scripts/analysis/run_pca.py.

    Returns a dict that maps a layer to a PcaResult.
    """
    data = np.load(path)
    results_by_layer = {}
    for position, layer in enumerate(data["layers"]):
        results_by_layer[int(layer)] = PcaResult(
            data["mean"][position],
            data["components"][position],
            data["explained_variance_ratio"][position],
            data["projections"][position],
        )
    return results_by_layer
