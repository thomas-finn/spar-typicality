# probe-02: category-level typicality probes on Qwen2.5-7B, layer 15

## Question

Do the probe-01 results hold for a larger model of the same family? Each probe
is fitted on one category, at one middle-to-late layer. Does it keep the
typicality order of other categories (cross-category) and of the same category
in other prompt sets (cross-dataset)?

## Dataset

As probe-01: all 11 prompt sets of the `rosch` suite (665 prompts each). Train
sets: `rosch_label` and `rosch_example`.

## Configuration

See `config.yaml`. The probe method is the same as probe-01 (see
`experiments/probes/probe-01/README.md`). Differences from probe-01:

- Model: `Qwen/Qwen2.5-7B` (28 layers, d_model 3584).
- Layers: 15 only (`layers: [15]`). In probe-01, cross-category transfer was
  highest at layers 10-20.
- Activation cache stored as float16 (`<dataset>.float16.npz`, about 52 MB for
  the suite). The analysis converts to float32. float16 caches do not replace
  float32 caches.
- `batch_size: 8` for GPU memory.

## Command

Extraction needs a GPU. On Colab, use an L4 or A100 (bf16); a T4 has too little
memory for the 7B model in 16-bit.

    git clone <repo-url> && cd spar-typicality && git checkout <commit>
    pip install uv && uv sync
    uv run python scripts/activations/extract_activations.py \
        --config experiments/probes/probe-02/config.yaml

Copy `outputs/activations/Qwen__Qwen2.5-7B/rosch/` to the same path locally
(or to the artifact store). Then, locally, without the model:

    uv run python scripts/analysis/run_probes.py \
        experiments/probes/probe-02/config.yaml --cache-only

`--cache-only` stops with an error if a dataset has no cache with layer 15 in
float16. The local and Colab checkouts must have the same
`data/processed/rosch/` prompts, or the cache does not match.

Alternative: run both commands on Colab and only keep the run directory (about
10 MB).

## Result

Not run yet.

## Interpretation

## Next experiment
