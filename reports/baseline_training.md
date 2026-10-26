# Baseline Training Record

## Dataset and split

- Source: `data/processed/synthetic_messages.csv` (synthetic-only).
- Rows used for fitting: **600**, from `split=train` only.
- Training label counts: `{'likely_legitimate': 300, 'suspicious': 300}`.
- Training scenario families: `account_review`, `appointment_reminder`, `delivery_fee`, `parcel_tracking`, `prize_claim`, `school_update`, `support_impersonation`, `team_coordination`.
- Split strategy: preassigned scenario-family groups; no family crosses train, validation, or test.
- Held-out validation/test rows passed to model fitting: **0**.

## Pipeline and configuration

- Full fitted pipeline: whitespace-only normalization → `TfidfVectorizer` → `LogisticRegression`.
- TF-IDF: word-level default tokenization, lowercase, `ngram_range=(1, 2)`, `min_df=1`, `max_df=1.0`, `sublinear_tf=True`.
- Logistic Regression: solver `liblinear`, `max_iter=1000`, `class_weight=None`, `random_state=42`.
- Seed: **42**.
- Saved model: `models\textshield_baseline.joblib`.
- Configuration: `models/textshield_baseline_metadata.json`.

The entire preprocessing and vectorization pipeline is fitted on training rows, so its vocabulary is learned from training text only. Validation and test data were not used for fitting or model selection. No performance metrics are reported here; held-out evaluation is reserved for Phase 5.

## Inference smoke check

The training command reloads the saved artifact and predicts one hand-written fictional demonstration sentence that is not taken from the dataset. This checks serialization and inference only; it is not an evaluation example or performance result.

## Limitations

The baseline is trained on synthetic template messages and is intended for educational demonstration. Performance on this dataset cannot establish effectiveness on real messages.
