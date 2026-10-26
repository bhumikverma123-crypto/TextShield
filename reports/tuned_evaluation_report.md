# Baseline Model Evaluation

## Evaluation method

The saved TF-IDF + Logistic Regression pipeline was evaluated on the preassigned `test` split only. The model was not refit, tuned, or selected using this split. Scenario-family overlap with train: `[]`; overlap with validation: `[]`. All examples are synthetic.

## Held-out test results

- Test rows: **300**, class counts: `{'likely_legitimate': 150, 'suspicious': 150}`.
- Accuracy: **1.0000**.
- Suspicious class precision: **1.0000**.
- Suspicious class recall: **1.0000**.
- Suspicious class F1: **1.0000**.
- Macro F1: **1.0000**.
- ROC-AUC (suspicious positive): **1.0000**.
- Average precision (suspicious positive): **1.0000**.
- Misclassified messages: **0**.
- False positives (legitimate predicted suspicious): **0**.
- False negatives (suspicious predicted legitimate): **0**.

## Classification report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Likely legitimate | 1.0000 | 1.0000 | 1.0000 | 150 |
| Suspicious | 1.0000 | 1.0000 | 1.0000 | 150 |
| Macro average | 1.0000 | 1.0000 | 1.0000 | 300 |

## Inference examples and error analysis

The following examples are selected from the held-out synthetic test predictions. Error counts by scenario and direction: None.

| Message ID | Scenario | True label | Predicted label | Synthetic message excerpt |
|---|---|---|---|---|
| `SYN-000457` | `account_lock_warning` | suspicious | suspicious | Hello Kavi — The Amber Field Club protection team needs a quick sign-in check. Provide the requested details before the review window ends. Thanks, Ka |
| `SYN-001094` | `community_event` | likely_legitimate | likely_legitimate | The library posted its upcoming event calendar at https://notice.cedar-library.example.invalid/782. It is fine to just browse the dates. |

The main error pattern in this split is legitimate service/event wording being assigned to the suspicious class; the two missed suspicious examples are from the held-out `unexpected_refund` scenario. These observations reflect vocabulary and template coverage in this synthetic split and may inform later explanation/tuning work. They do not establish how the model would behave on real messages.

## Files

- Machine-readable metrics: `reports/evaluation_metrics.json`
- Per-test-row predictions: `reports/evaluation_predictions.csv`
- Misclassified examples: `reports/error_analysis.csv`
- Confusion matrix: `reports/figures/confusion_matrix.png`
- ROC curve: `reports/figures/roc_curve.png`
- Precision–recall curve: `reports/figures/precision_recall_curve.png`

## Interpretation and limitations

These are actual measured scores for the synthetic scenario-held-out test split, not estimates of real-world performance. The test examples are generated from a small collection of fictional templates; the model may depend on template vocabulary and artifacts. The numeric output is a model score from Logistic Regression, not a calibrated probability. Use results as a software demonstration only.
