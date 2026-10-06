# Experiments

One row per experiment. Add a row when you create an experiment, and update
it after each run. Details are in the README of each experiment.

| ID | Question | Model | Data | Method | Layers | Status | Run ID | Main result |
|:-|:-|:-|:-|:-|:-|:-|:-|:-|
| [pca-01](pca/pca-01/README.md) | Do the top 3 PCs of final-period activations show category or typicality structure? | Qwen/Qwen2.5-1.5B | `rosch`, all 11 sets | PCA (3 components) per set and layer | 5, 10, 15, 20, 25 | Run; result not written up | `pca-01_Qwen__Qwen2.5-1.5B_rosch_20261006-103714` | - |
| [pca-02](pca/pca-02/README.md) | Does the pca-01 structure hold for a 7B model? | Qwen/Qwen2.5-7B | `rosch`, all 11 sets | As pca-01; float16 activation cache (shared with probe-02) | 15 | Set up, not run | - | - |
| [probe-01](probes/probe-01/README.md) | Does a typicality probe fitted on one category transfer to other categories and to other prompt sets? | Qwen/Qwen2.5-1.5B | Train: `rosch_label`, `rosch_example`; eval: all 11 sets | Per-category diff-of-means (top vs bottom third) and ridge (LOO alpha); projection correlation | 5, 10, 15, 20, 25 | Done | `probe-01_Qwen__Qwen2.5-1.5B_rosch_20261006-112301` | Mean Spearman, other categories: up to 0.39 (example, ridge, layer 15). Same category, other sets: 0.2-0.67. Transfer to `rosch_neutral` suggests part of the signal is item-only. |
| [probe-02](probes/probe-02/README.md) | Do the probe-01 results hold for a 7B model? | Qwen/Qwen2.5-7B | As probe-01 | As probe-01; float16 activation cache | 15 | Set up, not run | - | - |
