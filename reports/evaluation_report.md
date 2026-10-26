# Baseline Model Evaluation

## Evaluation method

The saved TF-IDF + Logistic Regression pipeline was evaluated on the preassigned `test` split only. The model was not refit, tuned, or selected using this split. Scenario-family overlap with train: `[]`; overlap with validation: `[]`. All examples are synthetic.

## Held-out test results

- Test rows: **300**, class counts: `{'likely_legitimate': 150, 'suspicious': 150}`.
- Accuracy: **0.9000**.
- Suspicious class precision: **0.8409**.
- Suspicious class recall: **0.9867**.
- Suspicious class F1: **0.9080**.
- Macro F1: **0.8992**.
- ROC-AUC (suspicious positive): **0.9876**.
- Average precision (suspicious positive): **0.9886**.
- Misclassified messages: **30**.
- False positives (legitimate predicted suspicious): **28**.
- False negatives (suspicious predicted legitimate): **2**.

## Classification report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Likely legitimate | 0.9839 | 0.8133 | 0.8905 | 150 |
| Suspicious | 0.8409 | 0.9867 | 0.9080 | 150 |
| Macro average | 0.9124 | 0.9000 | 0.8992 | 300 |

## Inference examples and error analysis

The following examples are selected from the held-out synthetic test predictions. Error counts by scenario and direction: `community_event`: 6 (likely_legitimate → suspicious), `routine_service_notice`: 22 (likely_legitimate → suspicious), `unexpected_refund`: 2 (suspicious → likely_legitimate).

| Message ID | Scenario | True label | Predicted label | Synthetic message excerpt |
|---|---|---|---|---|
| `SYN-000457` | `account_lock_warning` | suspicious | suspicious | Hello Kavi — The Amber Field Club protection team needs a quick sign-in check. Provide the requested details before the review window ends. Thanks, Ka |
| `SYN-001094` | `community_event` | likely_legitimate | likely_legitimate | The library posted its upcoming event calendar at https://notice.cedar-library.example.invalid/782. It is fine to just browse the dates. |
| `SYN-001127` | `routine_service_notice` | likely_legitimate | suspicious | Juniper Workshop service note: planned maintenance is scheduled for Thursday. Details are posted at https://events.willow-desk.example.invalid/262. —  |
| `SYN-000560` | `unexpected_refund` | suspicious | likely_legitimate | Hello Mira — The refund desk cannot complete reference TS-4629-75 until you verify the payment card information by message. Thanks, Mira. |

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
