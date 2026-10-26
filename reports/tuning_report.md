# Fine-Tuning and Baseline Comparison

## Tuning method

- Hyperparameter selection metric: macro F1.
- Candidate grid: TF-IDF `ngram_range` in `[(1, 1), (1, 2)]`, `min_df` in `[1, 2]`; Logistic Regression `C` in `[0.5, 1.0, 2.0]`, and `class_weight` in `[None, balanced]` (24 candidates).
- Cross-validation: 4-fold `StratifiedGroupKFold`, with scenario families kept intact. Search used the 600 training rows only.
- Best grid CV macro F1: **0.7285**.
- Baseline validation metrics: accuracy **0.6700**, macro F1 **0.6602**.
- Best grid candidate validation metrics: accuracy **0.6700**, macro F1 **0.6602**.
- Selection: Baseline settings were retained because the tuned candidate did not improve validation macro F1.
- Selected parameters: `{'tfidf__ngram_range': [1, 2], 'tfidf__min_df': 1, 'classifier__C': 1.0, 'classifier__class_weight': None}`.
- Final pipeline fit used 900 train + validation rows after selection. The test data did not influence the choice of parameters.

## Final held-out test comparison

After model selection and final fitting were complete, both historical baseline results and tuned-model results are reported for the same held-out test set. The baseline is the Phase 4 model; tuned results use the final selected pipeline.

| Metric | Baseline | Selected final model |
|---|---:|---:|
| Accuracy | 0.9000 | 1.0000 |
| Suspicious precision | 0.8409 | 1.0000 |
| Suspicious recall | 0.9867 | 1.0000 |
| Suspicious F1 | 0.9080 | 1.0000 |
| Macro F1 | 0.8992 | 1.0000 |
| ROC-AUC | 0.9876 | 1.0000 |
| Average precision | 0.9886 | 1.0000 |

The tuned test evaluation is saved in `reports/tuned_evaluation_metrics.json`; detailed predictions and plots use the `tuned_` prefix. Baseline test metrics remain in `reports/evaluation_metrics.json`. The final model uses the same hyperparameters as the baseline and was refit with 900 train + validation messages instead of 600 train messages. The higher test scores therefore cannot be attributed to hyperparameter changes; they may reflect the additional synthetic examples and the authored scenario structure.

## Trade-offs and limitations

Validation macro F1 determined whether to keep the tuned configuration or retain the simpler baseline settings. The grouped CV prevents scenario-family overlap within CV folds. Results are based on fictional synthetic scenarios only and do not establish real-world detection performance. Repeated experimentation against this small validation set can still overfit the authored templates.

## Reproducibility

- Seed: `42`.
- Selected model artifact: `models/textshield_tuned.joblib`.
- Parameter and score metadata: `models/textshield_tuned_metadata.json`.
- All grid results: `reports/tuning_cv_results.csv`.
