# probe-01: category-level typicality probes

## Question

Is there a linear direction in the residual stream that orders the members of
a category by typicality? If a probe is fitted on one category, does it keep
the typicality order of:

- other categories in the same prompt set (cross-category)?
- the same category in other prompt sets (cross-dataset)?

## Dataset

All prompt sets of the `rosch` suite in `data/processed/rosch/` (11 sets, 665
prompts each). The rows of all sets are aligned (same items and ratings in the
same order). `rosch_neutral` has no category in the prompt; its rows get the
category of the aligned row.

- Train sets: `rosch_label` (`{category}: {item}.`) and `rosch_example`
  (`A {item} is a {category}.`), fitted separately.
- Eval sets: all 11 sets.

## Configuration

See `config.yaml`.

- Model: `Qwen/Qwen2.5-1.5B`. Activations are the cache from pca-01: the token
  that contains the last `.`, layers 5, 10, 15, 20, 25.
- One probe per train set x layer x probe type x category (200 probes). Each
  probe is fitted on all member rows of the category (50-60 rows). Non-member
  rows are not used. There is no train/test split.
- Typicality score = minus the Rosch z-score (z-scores are per category), so a
  higher score means more typical. Probe directions point to more typical.
- `diff_means`: mean of the most typical third minus mean of the least typical
  third (int(n / 3) rows each), normalised to unit length.
- `ridge`: ridge regression of the score on the activations, intercept not
  penalised, no feature scaling. alpha = factor x mean squared row norm of the
  centred activations. The factor is selected per probe from
  `ridge_alpha_factors` by exact leave-one-out (LOO) error. The direction is
  the unit weight vector.
- Evaluation: project every row of every eval set onto the direction. For each
  eval category, correlate the projections of its member rows with the score
  (Pearson, Spearman, Kendall tau-b).
- Relations: `in_sample` (same set and category, the fit data),
  `cross_category` (same set, other category), `cross_dataset` (other set, same
  category), `cross_both`.
- Seed: 0. All steps are deterministic.

## Command

    uv run python scripts/analysis/run_probes.py experiments/probes/probe-01/config.yaml

Use `--model` and `--suite` for another model or suite. Activations that are
not in the cache are extracted first (this loads the model).

## Outputs

`outputs/experiments/<run_id>/`:

- `metadata.json`: config, seed, git commit, environment, status
- `summary.csv`: mean/std of each metric per train set, probe, layer, relation
- `metrics.csv`: one row per probe x eval set x eval category
- `probes.csv`: per probe: n_fit, n_top/n_bottom, or the selected alpha, LOO MSE
  and LOO Spearman for ridge
- `probes.npz`: `directions` (train set, probe, layer, category, d_model) and
  `projections` (train set, probe, layer, category, eval set, row)
- `html/probes.html`: menus for train set, probe, layer and metric. It shows the
  mean metric by layer for each relation, a heatmap of probe category x eval
  category (with a menu for the eval set), a heatmap of category x eval set,
  and an ordering detail: projection vs typicality score (with the non-member
  pairs for reference) and a slope chart of the Rosch order vs the probe
  order. A click on a heatmap cell opens it in the detail view.

## Result

Run `probe-01_Qwen__Qwen2.5-1.5B_rosch_20261006-112301`. Mean Spearman over
cells:

| train set | probe | layer | in_sample | cross_category | cross_dataset | cross_both |
|:-|:-|-:|-:|-:|-:|-:|
| label   | diff_means | 15 | 0.61 | 0.20 | 0.31 | 0.12 |
| label   | ridge      | 15 | 0.93 | 0.31 | 0.28 | 0.14 |
| label   | ridge      | 25 | 0.95 | 0.16 | 0.55 | 0.09 |
| example | diff_means | 15 | 0.71 | 0.33 | 0.45 | 0.16 |
| example | ridge      | 10 | 0.95 | 0.36 | 0.55 | 0.13 |
| example | ridge      | 15 | 0.96 | 0.39 | 0.48 | 0.14 |
| example | ridge      | 25 | 0.94 | 0.25 | 0.67 | 0.12 |

All layers are in `summary.csv`.

- Ridge `in_sample` values are overfit (d_model = 1536 > n = 50-60). The ridge
  LOO Spearman (in `probes.csv`) is a fairer same-category, same-set number:
  0.35-0.66 (label) and 0.40-0.68 (example), with the highest values at layers
  15-25.
- Cross-category generalisation is low at layer 5 (~0) and is highest at layers
  10-20. The best mean is 0.39 (example, ridge, layer 15). It changes much
  between categories: for example, at layer 15 the mean over the other probe
  categories is 0.57 for clothing but 0.04 for tool.
- Cross-dataset generalisation (same category) is larger than cross-category.
  Probes from `rosch_example` transfer best to `rosch_neg` (0.78, ridge, layer
  15) and to the conjunction sets (~0.5). Probes from `rosch_label` transfer
  best to `rosch_story` (0.62) and `rosch_neutral` (0.50).
- For `rosch_label` probes, cross-dataset generalisation is lowest at layer 20
  (0.24 diff_means, 0.20 ridge) and highest at layers 5 and 25. For
  `rosch_example` probes it changes less with layer (0.37-0.50 diff_means,
  0.47-0.67 ridge).
- Ridge selected the lowest alpha factor (0.0001) for 13 of 100 probes and the
  highest (100) for 4. For these probes the grid may be too narrow.

## Interpretation

There is a linear typicality signal inside each category that partly transfers
to other categories at middle layers. Thus some of the typicality signal is
shared between categories, but much of it is specific to the category.

Probes transfer to `rosch_neutral` (`Word: {item}.`, no category in the prompt).
Thus part of each probe can be a property of the item only (for example
frequency or familiarity, which correlate with typicality). It is not
necessarily a property of the item relative to the category. This is a
hypothesis; this experiment does not test it.

## Next experiment

- Remove a frequency or familiarity covariate (for example word frequency) from
  the score and test if the cross-category transfer stays.
- Fit one probe on all categories together (pooled) and test it on a held-out
  category (leave-one-category-out).
- Use a wider ridge alpha grid, or select alpha by cross-category error.
