# Extended experiments

These notebooks contain the post-dissertation experiments run in September–October 2026. They are kept separate from the original thesis pipeline so that the dissertation experiments remain easy to reproduce.

## Contents

- `model_selection/` — architecture search, five-seed confirmation, per-fold selection and locked nested-CV evaluation.
- `architecture_sensitivity/` — small, medium and full architecture runs plus the JHub worker scripts.
- `classical_baselines/` — logistic regression, RBF-SVM, random forest and XGBoost baselines, including MRI-embedding variants.
- `fusion/` — SVM/TMC and SVM/neural fusion experiments.
- `uncertainty_and_error_analysis/` — MC-dropout, participant difficulty and model-disagreement analyses.

The `archive/` notebooks are earlier versions retained for provenance; the non-archive version should be used for current results.

Notebook outputs were cleared before publication. This avoids committing participant-level ADNI data, local filesystem paths and large execution logs. Aggregate result tables and selected figures are stored under `results/extended_experiments/`.
