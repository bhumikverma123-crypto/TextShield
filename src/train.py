"""Train the reproducible TF-IDF + Logistic Regression baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.preprocess import DEFAULT_DATA_PATH, PROJECT_ROOT, build_text_pipeline, validate_input_frame

MODEL_PATH = PROJECT_ROOT / "models" / "textshield_baseline.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "textshield_baseline_metadata.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "baseline_training.md"
DEFAULT_SEED = 42


def train_baseline(
    data_path: Path = DEFAULT_DATA_PATH,
    seed: int = DEFAULT_SEED,
    model_path: Path = MODEL_PATH,
) -> dict[str, object]:
    """Fit the baseline pipeline on train rows only and persist the artifact."""
    frame = pd.read_csv(data_path)
    validate_input_frame(frame)
    if "split" not in frame or "scenario" not in frame:
        raise ValueError("Dataset must include scenario-group split columns")
    scenario_split_counts = frame.groupby("scenario")["split"].nunique()
    if (scenario_split_counts != 1).any():
        raise ValueError("At least one scenario family crosses data splits")

    # Only this subset contributes message text and labels to fitting.
    train_rows = frame.loc[frame["split"] == "train"].copy()
    if train_rows.empty:
        raise ValueError("No rows are assigned to the train split")
    if train_rows["label"].nunique() < 2:
        raise ValueError("Training rows must contain both target classes")

    classifier = LogisticRegression(
        random_state=seed,
        max_iter=1000,
        solver="liblinear",
        class_weight=None,
    )
    pipeline = build_text_pipeline(
        classifier,
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
        sublinear_tf=True,
    )
    pipeline.fit(train_rows["message_text"].tolist(), train_rows["label"].tolist())

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metadata: dict[str, object] = {
        "model_name": "TF-IDF + Logistic Regression baseline",
        "seed": seed,
        "training_rows": int(len(train_rows)),
        "training_class_counts": {
            str(label): int(count)
            for label, count in train_rows["label"].value_counts().sort_index().items()
        },
        "training_scenario_families": sorted(train_rows["scenario"].unique().tolist()),
        "split_strategy": "Preassigned scenario-family groups; only rows with split=train are passed to fit.",
        "held_out_rows_used_for_fit": 0,
        "vectorizer": {
            "class": "TfidfVectorizer",
            "ngram_range": [1, 2],
            "min_df": 1,
            "max_df": 1.0,
            "sublinear_tf": True,
            "lowercase": True,
            "token_pattern": "scikit-learn default; raw punctuation/URL text is preserved before vectorization",
        },
        "classifier": {
            "class": "LogisticRegression",
            "solver": "liblinear",
            "max_iter": 1000,
            "class_weight": None,
            "random_state": seed,
        },
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "synthetic_only": True,
        "evaluation_metrics": "Pending Phase 5; no validation/test metrics were computed during training.",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def write_training_report(metadata: dict[str, object], model_path: Path) -> None:
    """Record training configuration and the boundary between phases."""
    class_counts = metadata["training_class_counts"]
    families = metadata["training_scenario_families"]
    content = f"""# Baseline Training Record

## Dataset and split

- Source: `data/processed/synthetic_messages.csv` (synthetic-only).
- Rows used for fitting: **{metadata['training_rows']}**, from `split=train` only.
- Training label counts: `{class_counts}`.
- Training scenario families: {', '.join(f'`{name}`' for name in families)}.
- Split strategy: preassigned scenario-family groups; no family crosses train, validation, or test.
- Held-out validation/test rows passed to model fitting: **0**.

## Pipeline and configuration

- Full fitted pipeline: whitespace-only normalization → `TfidfVectorizer` → `LogisticRegression`.
- TF-IDF: word-level default tokenization, lowercase, `ngram_range=(1, 2)`, `min_df=1`, `max_df=1.0`, `sublinear_tf=True`.
- Logistic Regression: solver `liblinear`, `max_iter=1000`, `class_weight=None`, `random_state={metadata['seed']}`.
- Seed: **{metadata['seed']}**.
- Saved model: `{model_path.relative_to(PROJECT_ROOT)}`.
- Configuration: `models/textshield_baseline_metadata.json`.

The entire preprocessing and vectorization pipeline is fitted on training rows, so its vocabulary is learned from training text only. Validation and test data were not used for fitting or model selection. No performance metrics are reported here; held-out evaluation is reserved for Phase 5.

## Inference smoke check

The training command reloads the saved artifact and predicts one hand-written fictional demonstration sentence that is not taken from the dataset. This checks serialization and inference only; it is not an evaluation example or performance result.

## Limitations

The baseline is trained on synthetic template messages and is intended for educational demonstration. Performance on this dataset cannot establish effectiveness on real messages.
"""
    REPORT_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    """Train, save, reload, and smoke-check the baseline pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="synthetic CSV path")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="classifier random seed")
    parser.add_argument("--model-out", type=Path, default=MODEL_PATH, help="fitted pipeline output path")
    args = parser.parse_args()

    metadata = train_baseline(args.data, args.seed, args.model_out)
    loaded_pipeline = joblib.load(args.model_out)
    smoke_message = "A note from Maple Learning Circle: the meeting is at 4:00 pm in the reading room."
    prediction = loaded_pipeline.predict([smoke_message])[0]
    write_training_report(metadata, args.model_out)
    print(f"Baseline pipeline saved to: {args.model_out}")
    print(f"Metadata saved to: {METADATA_PATH}")
    print(f"Training rows: {metadata['training_rows']}")
    print(f"Training class counts: {metadata['training_class_counts']}")
    print(f"Held-out rows used for fit: {metadata['held_out_rows_used_for_fit']}")
    print(f"Reloaded artifact smoke prediction: {prediction} (not an evaluation metric)")
    print(f"Training record saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
