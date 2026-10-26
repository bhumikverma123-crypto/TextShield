"""Tune the baseline with grouped train-only CV and validation-based selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from src.evaluate import evaluate_model, write_evaluation_report
from src.preprocess import DEFAULT_DATA_PATH, PROJECT_ROOT, build_text_pipeline, validate_input_frame

BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "textshield_baseline.joblib"
TUNED_MODEL_PATH = PROJECT_ROOT / "models" / "textshield_tuned.joblib"
TUNED_METADATA_PATH = PROJECT_ROOT / "models" / "textshield_tuned_metadata.json"
BASELINE_METRICS_PATH = PROJECT_ROOT / "reports" / "evaluation_metrics.json"
TUNING_RESULTS_PATH = PROJECT_ROOT / "reports" / "tuning_cv_results.csv"
TUNING_REPORT_PATH = PROJECT_ROOT / "reports" / "tuning_report.md"
DEFAULT_SEED = 42
CLASS_ORDER = ["likely_legitimate", "suspicious"]


def build_candidate_pipeline(seed: int, classifier_options: dict[str, Any] | None = None, vectorizer_options: dict[str, Any] | None = None):
    """Build a fresh pipeline with specified tunable options."""
    classifier_options = classifier_options or {}
    vectorizer_options = vectorizer_options or {}
    classifier = LogisticRegression(
        random_state=seed,
        max_iter=1000,
        solver="liblinear",
        **classifier_options,
    )
    defaults = {
        "ngram_range": (1, 2),
        "min_df": 1,
        "max_df": 1.0,
        "sublinear_tf": True,
    }
    defaults.update(vectorizer_options)
    return build_text_pipeline(classifier, **defaults)


def validation_metrics(model: Any, frame: pd.DataFrame) -> dict[str, float]:
    """Measure a fixed candidate on validation rows for selection only."""
    truth = frame["label"].to_numpy()
    predicted = model.predict(frame["message_text"].tolist())
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=CLASS_ORDER, average="macro", zero_division=0)),
    }


def _plain_params(params: dict[str, Any]) -> dict[str, Any]:
    """Convert sklearn parameter values to JSON-friendly structures."""
    return {key: list(value) if isinstance(value, tuple) else value for key, value in params.items()}


def tune_and_evaluate(data_path: Path = DEFAULT_DATA_PATH, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Tune on training data, select with validation, then test once at the end."""
    frame = pd.read_csv(data_path)
    validate_input_frame(frame)
    train = frame.loc[frame["split"] == "train"].copy()
    validation = frame.loc[frame["split"] == "validation"].copy()
    if train.empty or validation.empty:
        raise ValueError("Both train and validation splits must be non-empty")
    if train["label"].nunique() != 2 or validation["label"].nunique() != 2:
        raise ValueError("Train and validation splits must contain both classes")
    if set(train["scenario"]) & set(validation["scenario"]):
        raise ValueError("Scenario families must not overlap between train and validation")

    x_train = train["message_text"].tolist()
    y_train = train["label"].to_numpy()
    groups = train["scenario"].to_numpy()
    cv_strategy = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    cv_splits = list(cv_strategy.split(x_train, y_train, groups))
    for fold_number, (fit_indices, validation_indices) in enumerate(cv_splits, start=1):
        fit_groups = set(groups[fit_indices])
        fold_groups = set(groups[validation_indices])
        if fit_groups & fold_groups:
            raise ValueError(f"Scenario leakage found in grouped CV fold {fold_number}")
        if len(set(y_train[validation_indices])) != 2:
            raise ValueError(f"Grouped CV fold {fold_number} does not contain both classes")

    estimator = build_candidate_pipeline(seed)
    search = GridSearchCV(
        estimator=estimator,
        param_grid={
            "tfidf__ngram_range": [(1, 1), (1, 2)],
            "tfidf__min_df": [1, 2],
            "classifier__C": [0.5, 1.0, 2.0],
            "classifier__class_weight": [None, "balanced"],
        },
        scoring="f1_macro",
        cv=cv_splits,
        refit=True,
        n_jobs=1,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(x_train, y_train)

    baseline = joblib.load(BASELINE_MODEL_PATH)
    baseline_validation = validation_metrics(baseline, validation)
    candidate_validation = validation_metrics(search.best_estimator_, validation)
    # Keep the baseline configuration when the validation macro-F1 is tied or
    # lower; this limits complexity and prevents tuning from being accepted on
    # cross-validation alone.
    tuned_selected = candidate_validation["macro_f1"] > baseline_validation["macro_f1"]
    if tuned_selected:
        selected_params = search.best_params_
        selection_reason = "Grid-search candidate had higher validation macro F1 than the baseline."
    else:
        selected_params = {
            "tfidf__ngram_range": (1, 2),
            "tfidf__min_df": 1,
            "classifier__C": 1.0,
            "classifier__class_weight": None,
        }
        selection_reason = "Baseline settings were retained because the tuned candidate did not improve validation macro F1."

    selected_vectorizer = {
        "ngram_range": selected_params["tfidf__ngram_range"],
        "min_df": selected_params["tfidf__min_df"],
        "max_df": 1.0,
        "sublinear_tf": True,
    }
    selected_classifier = {
        "C": selected_params["classifier__C"],
        "class_weight": selected_params["classifier__class_weight"],
    }
    final_model = build_candidate_pipeline(seed, selected_classifier, selected_vectorizer)
    train_and_validation = pd.concat([train, validation], ignore_index=True)
    final_model.fit(
        train_and_validation["message_text"].tolist(),
        train_and_validation["label"].tolist(),
    )
    TUNED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, TUNED_MODEL_PATH)

    cv_results = pd.DataFrame(search.cv_results_)
    result_columns = [
        "param_tfidf__ngram_range",
        "param_tfidf__min_df",
        "param_classifier__C",
        "param_classifier__class_weight",
        "mean_test_score",
        "std_test_score",
        "rank_test_score",
    ]
    cv_results[result_columns].sort_values("rank_test_score").to_csv(TUNING_RESULTS_PATH, index=False)

    metadata: dict[str, Any] = {
        "seed": seed,
        "selection_metric": "macro F1",
        "selection_strategy": "4-fold StratifiedGroupKFold on train scenarios; compare best CV candidate with baseline on validation; retain baseline unless candidate improves validation macro F1.",
        "cv_folds": len(cv_splits),
        "grid_candidate_count": len(search.cv_results_["params"]),
        "train_rows_for_cv": int(len(train)),
        "validation_rows_for_selection": int(len(validation)),
        "train_validation_rows_for_final_fit": int(len(train_and_validation)),
        "test_rows_used_before_final_evaluation": 0,
        "best_cross_validation_macro_f1": float(search.best_score_),
        "best_grid_parameters": _plain_params(search.best_params_),
        "best_grid_validation_metrics": candidate_validation,
        "baseline_validation_metrics": baseline_validation,
        "selected_candidate_was_tuned": bool(tuned_selected),
        "selection_reason": selection_reason,
        "selected_parameters": _plain_params(selected_params),
        "final_fit_splits": ["train", "validation"],
        "synthetic_only": True,
        "final_model_path": str(TUNED_MODEL_PATH.relative_to(PROJECT_ROOT)),
    }
    TUNED_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Hyperparameter selection and final fitting are now complete. Only now is
    # the held-out test split read for final tuned-model evaluation.
    tuned_metrics = evaluate_model(data_path, TUNED_MODEL_PATH, output_prefix="tuned_")
    write_evaluation_report(tuned_metrics, output_prefix="tuned_")
    baseline_metrics = json.loads(BASELINE_METRICS_PATH.read_text(encoding="utf-8"))
    metadata["tuned_test_metrics"] = {
        key: tuned_metrics[key]
        for key in (
            "accuracy", "precision_suspicious", "recall_suspicious", "f1_suspicious",
            "macro_f1", "roc_auc_suspicious_positive", "average_precision_suspicious_positive",
        )
    }
    TUNED_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_tuning_report(metadata, baseline_metrics, tuned_metrics)
    return metadata


def write_tuning_report(
    metadata: dict[str, Any], baseline_test: dict[str, Any], tuned_test: dict[str, Any]
) -> None:
    """Record the selection process and measured baseline/tuned comparison."""
    validation_base = metadata["baseline_validation_metrics"]
    validation_tuned = metadata["best_grid_validation_metrics"]
    test_fields = [
        ("Accuracy", "accuracy"),
        ("Suspicious precision", "precision_suspicious"),
        ("Suspicious recall", "recall_suspicious"),
        ("Suspicious F1", "f1_suspicious"),
        ("Macro F1", "macro_f1"),
        ("ROC-AUC", "roc_auc_suspicious_positive"),
        ("Average precision", "average_precision_suspicious_positive"),
    ]
    comparison_rows = "\n".join(
        f"| {name} | {baseline_test[key]:.4f} | {tuned_test[key]:.4f} |"
        for name, key in test_fields
    )
    content = f"""# Fine-Tuning and Baseline Comparison

## Tuning method

- Hyperparameter selection metric: macro F1.
- Candidate grid: TF-IDF `ngram_range` in `[(1, 1), (1, 2)]`, `min_df` in `[1, 2]`; Logistic Regression `C` in `[0.5, 1.0, 2.0]`, and `class_weight` in `[None, balanced]` ({metadata['grid_candidate_count']} candidates).
- Cross-validation: {metadata['cv_folds']}-fold `StratifiedGroupKFold`, with scenario families kept intact. Search used the {metadata['train_rows_for_cv']} training rows only.
- Best grid CV macro F1: **{metadata['best_cross_validation_macro_f1']:.4f}**.
- Baseline validation metrics: accuracy **{validation_base['accuracy']:.4f}**, macro F1 **{validation_base['macro_f1']:.4f}**.
- Best grid candidate validation metrics: accuracy **{validation_tuned['accuracy']:.4f}**, macro F1 **{validation_tuned['macro_f1']:.4f}**.
- Selection: {metadata['selection_reason']}
- Selected parameters: `{metadata['selected_parameters']}`.
- Final pipeline fit used {metadata['train_validation_rows_for_final_fit']} train + validation rows after selection. The test data did not influence the choice of parameters.

## Final held-out test comparison

After model selection and final fitting were complete, both historical baseline results and tuned-model results are reported for the same held-out test set. The baseline is the Phase 4 model; tuned results use the final selected pipeline.

| Metric | Baseline | Selected final model |
|---|---:|---:|
{comparison_rows}

The tuned test evaluation is saved in `reports/tuned_evaluation_metrics.json`; detailed predictions and plots use the `tuned_` prefix. Baseline test metrics remain in `reports/evaluation_metrics.json`.

## Trade-offs and limitations

Validation macro F1 determined whether to keep the tuned configuration or retain the simpler baseline settings. The grouped CV prevents scenario-family overlap within CV folds. Results are based on fictional synthetic scenarios only and do not establish real-world detection performance. Repeated experimentation against this small validation set can still overfit the authored templates.

## Reproducibility

- Seed: `{metadata['seed']}`.
- Selected model artifact: `models/textshield_tuned.joblib`.
- Parameter and score metadata: `models/textshield_tuned_metadata.json`.
- All grid results: `reports/tuning_cv_results.csv`.
"""
    TUNING_REPORT_PATH.write_text(content, encoding="utf-8")


def main() -> None:
    """Run tuning, validation-based selection, final fitting, and test evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="synthetic dataset CSV")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="reproducibility seed")
    args = parser.parse_args()
    metadata = tune_and_evaluate(args.data, args.seed)
    print(f"Best grouped-CV macro F1: {metadata['best_cross_validation_macro_f1']:.4f}")
    print(f"Baseline validation macro F1: {metadata['baseline_validation_metrics']['macro_f1']:.4f}")
    print(f"Tuned candidate validation macro F1: {metadata['best_grid_validation_metrics']['macro_f1']:.4f}")
    print(f"Selected tuned candidate: {metadata['selected_candidate_was_tuned']}")
    print(f"Selection reason: {metadata['selection_reason']}")
    print(f"Selected parameters: {metadata['selected_parameters']}")
    print(f"Final model: {TUNED_MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Tuning report: {TUNING_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print("Final tuned-model test results are in reports/tuned_evaluation_report.md")


if __name__ == "__main__":
    main()
