# pca-02: PCA of final-period activations on Qwen2.5-7B, layer 15

## Question

Does the pca-01 structure (category and typicality in the top three PCs) also
show in a larger model of the same family?

## Dataset

As pca-01: all 11 prompt sets of the `rosch` suite (665 prompts each).

## Configuration

See `config.yaml`. The method is the same as pca-01 (see
`experiments/pca/pca-01/README.md`). Differences from pca-01:

- Model: `Qwen/Qwen2.5-7B` (28 layers, d_model 3584).
- Layers: 15 only (`layers: [15]`).
- Activation cache stored as float16 (`<dataset>.float16.npz`). This is the
  same cache as probe-02, so one extraction serves both experiments.
- `batch_size: 8` for GPU memory.

## Command

Extract the activations on a GPU first (see probe-02 README). Then, locally,
without the model:

    uv run python scripts/analysis/run_pca.py \
        experiments/pca/pca-02/config.yaml --cache-only

## Result

Not run yet.

## Interpretation

## Next experiment
