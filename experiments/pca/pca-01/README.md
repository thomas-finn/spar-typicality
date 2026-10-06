# pca-01: PCA of final-period activations

## Question

Do the top three principal components of the residual stream at the final
period of a prompt show structure that relates to category membership or
typicality? How does this structure change with depth?

## Dataset

All prompt sets of the `rosch` suite in `data/processed/rosch/` (11 sets, 665
prompts each: 565 Rosch (1975) member pairs and 100 random non-member pairs).

## Configuration

See `config.yaml`.

- Model: `Qwen/Qwen2.5-1.5B` (28 layers).
- Token: the token that contains the last `.` of the prompt.
- Layers: 5, 10, 15, 20, 25. Layer L is `hidden_states[L]`, the output of
  transformer block L.
- PCA: fitted separately for each dataset and each layer, 3 components.
- Seed: 0. PCA uses a deterministic SVD; the seed is recorded for consistency.

## Command

    uv run python scripts/analysis/run_pca.py experiments/pca/pca-01/config.yaml

Use `--model` and `--suite` to change the model or suite without editing the
config. To only extract and cache activations:

    uv run python scripts/activations/extract_activations.py --model Qwen/Qwen2.5-1.5B --suite rosch

## Outputs

- Activation cache (reused by later experiments):
  `outputs/activations/<model>/<suite>/<dataset>.npz` with the arrays
  `activations` (n_prompts, n_layers, d_model), `layers`, `prompts`,
  `token_indices`, and a `metadata.json`.
- Run directory: `outputs/experiments/<run_id>/`
  - `metadata.json`: config, seed, git commit, environment, status
  - `explained_variance.csv`: explained variance ratio of PC1-3 per dataset and layer
  - `pca/<dataset>.npz`: projections, components, means per layer
  - `html/<dataset>.html`: 3D plot; the menu selects the layer and the colour
    mode (category or typicality z-score). The plots load plotly from a CDN.

## Result

Not run yet.

## Interpretation

## Next experiment
