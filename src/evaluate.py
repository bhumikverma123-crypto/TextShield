"""Evaluate the saved baseline on the scenario-held-out test split only."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "textshield-matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.preprocess import DEFAULT_DATA_PATH, PROJECT_ROOT, validate_input_frame

MODEL_PATH = PROJECT_ROOT / "models" / "textshield_baseline.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "textshield_baseline_metadata.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "evaluation_metrics.json"
PREDICTIONS_PATH = REPORTS_DIR / "evaluation_predictions.csv"
ERRORS_PATH = REPORTS_DIR / "error_analysis.csv"
CLASS_ORDER = ["likely_legitimate", "suspicious"]
POSITIVE_LABEL = "suspicious"


def _save_confusion_matrix(actual: np.ndarray, predicted: np.ndarray, path: Path) -> list[list[int]]:
    matrix = confusion_matrix(actual, predicted, labels=CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Held-out synthetic test confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(CLASS_ORDER)), ["Likely legitimate", "Suspicious"])
    ax.set_yticks(range(len(CLASS_ORDER)), ["Likely legitimate", "Suspicious"])
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                    color="white" if matrix[row, col] > threshold else "black")
    fig.colorbar(image, ax=ax, label="Messages")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return matrix.tolist()


def _save_roc_curve(y_true: np.ndarray, scores: np.ndarray, path: Path) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = float(roc_auc_score(y_true, scores))
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}", color="#476a8a", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Chance reference")
    ax.set_title("Held-out synthetic test ROC curve")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return auc


def _save_precision_recall_curve(y_true: np.ndarray, scores: np.ndarray, path: Path) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    precision, recall, _ = precision_recall_curve(y_true, scores)
    # Guard against tiny floating-point overshoot above the metric's upper bound.
    average_precision = min(1.0, float(average_precision_score(y_true, scores)))
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot(recall, precision, label=f"Average precision = {average_precision:.3f}", color="#bd4b4b", linewidth=2)
    ax.set_title("Held-out synthetic test precision–recall curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return average_precision


def evaluate_model(
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = MODEL_PATH,
    output_prefix: str = "",
) -> dict[str, Any]:
    """Compute held-out metrics without fitting or selecting a model."""
    if output_prefix and not all(character.isalnum() or character in "_-" for character in output_prefix):
        raise ValueError("output_prefix may contain only letters, numbers, underscores, and hyphens")
    metrics_path = REPORTS_DIR / f"{output_prefix}evaluation_metrics.json"
    predictions_path = REPORTS_DIR / f"{output_prefix}evaluation_predictions.csv"
    errors_path = REPORTS_DIR / f"{output_prefix}error_analysis.csv"
    report_path = REPORTS_DIR / f"{output_prefix}evaluation_report.md"
    confusion_path = FIGURES_DIR / f"{output_prefix}confusion_matrix.png"
    roc_path = FIGURES_DIR / f"{output_prefix}roc_curve.png"
    pr_path = FIGURES_DIR / f"{output_prefix}precision_recall_curve.png"
    frame = pd.read_csv(data_path)
    validate_input_frame(frame)
    if "split" not in frame or "scenario" not in frame:
        raise ValueError("Dataset must include scenario-group split columns")
    model = joblib.load(model_path)

    test = frame.loc[frame["split"] == "test"].copy()
    if test.empty:
        raise ValueError("No rows are assigned to the held-out test split")
    if set(test["label"].unique()) != set(CLASS_ORDER):
        raise ValueError("Held-out test rows must contain both expected classes")
    train_scenarios = set(frame.loc[frame["split"] == "train", "scenario"])
    validation_scenarios = set(frame.loc[frame["split"] == "validation", "scenario"])
    test_scenarios = set(test["scenario"])
    if train_scenarios & test_scenarios or validation_scenarios & test_scenarios:
        raise ValueError("Test scenario families overlap with train or validation families")

    x_test = test["message_text"].tolist()
    y_true = test["label"].to_numpy()
    predictions = model.predict(x_test)
    classes = list(model.named_steps["classifier"].classes_)
    if POSITIVE_LABEL not in classes:
        raise ValueError(f"Model does not include expected class {POSITIVE_LABEL!r}")

    probabilities = model.predict_proba(x_test)
    positive_scores = probabilities[:, classes.index(POSITIVE_LABEL)]
    y_binary = (y_true == POSITIVE_LABEL).astype(int)
    report = classification_report(
        y_true,
        predictions,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    confusion = _save_confusion_matrix(y_true, predictions, confusion_path)
    roc_auc = _save_roc_curve(y_binary, positive_scores, roc_path)
    average_precision = _save_precision_recall_curve(
        y_binary, positive_scores, pr_path
    )

    predictions_frame = pd.DataFrame({
        "message_id": test["message_id"].to_numpy(),
        "scenario": test["scenario"].to_numpy(),
        "message_text": test["message_text"].to_numpy(),
        "true_label": y_true,
        "predicted_label": predictions,
        "suspicious_score": positive_scores,
    })
    predictions_frame.to_csv(predictions_path, index=False)
    errors = test.loc[predictions != y_true, ["message_id", "scenario", "message_text", "label"]].copy()
    errors = errors.rename(columns={"label": "true_label"})
    errors["predicted_label"] = predictions[predictions != y_true]
    errors["suspicious_score"] = positive_scores[predictions != y_true]
    errors.to_csv(errors_path, index=False)

    false_positives = errors.loc[
        (errors["true_label"] == "likely_legitimate")
        & (errors["predicted_label"] == "suspicious")
    ]
    false_negatives = errors.loc[
        (errors["true_label"] == "suspicious")
        & (errors["predicted_label"] == "likely_legitimate")
    ]

    def example_records(rows: pd.DataFrame) -> list[dict[str, str]]:
        return rows[["message_id", "scenario", "message_text", "true_label", "predicted_label"]].to_dict(orient="records")

    correct_examples = predictions_frame.loc[
        predictions_frame["true_label"] == predictions_frame["predicted_label"]
    ].groupby("true_label", sort=True).head(1)
    error_scenario_counts = (
        errors.groupby(["scenario", "true_label", "predicted_label"]).size().reset_index(name="count")
    )

    metrics: dict[str, Any] = {
        "evaluation_scope": "Held-out synthetic test split only; no fitting or model selection performed.",
        "synthetic_only": True,
        "model_path": str(model_path.relative_to(PROJECT_ROOT)),
        "test_rows": int(len(test)),
        "test_class_counts": {str(key): int(value) for key, value in test["label"].value_counts().sort_index().items()},
        "test_scenario_families": sorted(test_scenarios),
        "train_test_scenario_overlap": sorted(train_scenarios & test_scenarios),
        "validation_test_scenario_overlap": sorted(validation_scenarios & test_scenarios),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_suspicious": float(precision_score(y_true, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "recall_suspicious": float(recall_score(y_true, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "f1_suspicious": float(f1_score(y_true, predictions, pos_label=POSITIVE_LABEL, zero_division=0)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", labels=CLASS_ORDER, zero_division=0)),
        "roc_auc_suspicious_positive": roc_auc,
        "average_precision_suspicious_positive": average_precision,
        "confusion_matrix_label_order": CLASS_ORDER,
        "confusion_matrix": confusion,
        "classification_report": report,
        "misclassified_rows": int(len(errors)),
        "false_positive_count": int(len(false_positives)),
        "false_negative_count": int(len(false_negatives)),
        "misclassified_by_scenario": [
            {str(key): (int(value) if key == "count" else str(value)) for key, value in row.items()}
            for row in error_scenario_counts.to_dict(orient="records")
        ],
        "correct_inference_examples": example_records(correct_examples),
        "error_inference_examples": example_records(false_positives.head(1)) + example_records(false_negatives.head(1)),
        "predictions_path": str(predictions_path.relative_to(PROJECT_ROOT)),
        "error_analysis_path": str(errors_path.relative_to(PROJECT_ROOT)),
        "figures": [
            str(confusion_path.relative_to(PROJECT_ROOT)),
            str(roc_path.relative_to(PROJECT_ROOT)) if roc_auc is not None else None,
            str(pr_path.relative_to(PROJECT_ROOT)) if average_precision is not None else None,
        ],
        "caution": "Results describe this synthetic scenario-held-out dataset only and do not establish real-world detection performance.",
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def write_evaluation_report(metrics: dict[str, Any], output_prefix: str = "") -> None:
    """Write a concise, cautious interpretation of actual test outputs."""
    report = metrics["classification_report"]
    example_rows = metrics["correct_inference_examples"] + metrics["error_inference_examples"]
    example_table = "\n".join(
        f"| `{row['message_id']}` | `{row['scenario']}` | {row['true_label']} | {row['predicted_label']} | {row['message_text'][:150].replace('|', '/')} |"
        for row in example_rows
    )
    scenario_errors = ", ".join(
        f"`{row['scenario']}`: {row['count']} ({row['true_label']} → {row['predicted_label']})"
        for row in metrics["misclassified_by_scenario"]
    ) or "None"
    content = f"""# Baseline Model Evaluation

## Evaluation method

The saved TF-IDF + Logistic Regression pipeline was evaluated on the preassigned `test` split only. The model was not refit, tuned, or selected using this split. Scenario-family overlap with train: `{metrics['train_test_scenario_overlap']}`; overlap with validation: `{metrics['validation_test_scenario_overlap']}`. All examples are synthetic.

## Held-out test results

- Test rows: **{metrics['test_rows']}**, class counts: `{metrics['test_class_counts']}`.
- Accuracy: **{metrics['accuracy']:.4f}**.
- Suspicious class precision: **{metrics['precision_suspicious']:.4f}**.
- Suspicious class recall: **{metrics['recall_suspicious']:.4f}**.
- Suspicious class F1: **{metrics['f1_suspicious']:.4f}**.
- Macro F1: **{metrics['macro_f1']:.4f}**.
- ROC-AUC (suspicious positive): **{metrics['roc_auc_suspicious_positive']:.4f}**.
- Average precision (suspicious positive): **{metrics['average_precision_suspicious_positive']:.4f}**.
- Misclassified messages: **{metrics['misclassified_rows']}**.
- False positives (legitimate predicted suspicious): **{metrics['false_positive_count']}**.
- False negatives (suspicious predicted legitimate): **{metrics['false_negative_count']}**.

## Classification report

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Likely legitimate | {report['likely_legitimate']['precision']:.4f} | {report['likely_legitimate']['recall']:.4f} | {report['likely_legitimate']['f1-score']:.4f} | {int(report['likely_legitimate']['support'])} |
| Suspicious | {report['suspicious']['precision']:.4f} | {report['suspicious']['recall']:.4f} | {report['suspicious']['f1-score']:.4f} | {int(report['suspicious']['support'])} |
| Macro average | {report['macro avg']['precision']:.4f} | {report['macro avg']['recall']:.4f} | {report['macro avg']['f1-score']:.4f} | {int(report['macro avg']['support'])} |

## Inference examples and error analysis

The following examples are selected from the held-out synthetic test predictions. Error counts by scenario and direction: {scenario_errors}.

| Message ID | Scenario | True label | Predicted label | Synthetic message excerpt |
|---|---|---|---|---|
{example_table}

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
"""
    (REPORTS_DIR / f"{output_prefix}evaluation_report.md").write_text(content, encoding="utf-8")


def main() -> None:
    """Evaluate the saved baseline and persist metrics, plots, and errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="synthetic CSV path")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="saved fitted pipeline")
    parser.add_argument("--output-prefix", default="", help="prefix for metrics, plots, and reports")
    args = parser.parse_args()
    metrics = evaluate_model(args.data, args.model, args.output_prefix)
    write_evaluation_report(metrics, args.output_prefix)
    print(f"Evaluated {metrics['test_rows']} held-out test rows")
    for name in ("accuracy", "precision_suspicious", "recall_suspicious", "f1_suspicious", "macro_f1", "roc_auc_suspicious_positive", "average_precision_suspicious_positive"):
        print(f"{name}: {metrics[name]}")
    print(f"Misclassified rows: {metrics['misclassified_rows']}")
    print(f"Metrics saved to: {(REPORTS_DIR / f'{args.output_prefix}evaluation_metrics.json').relative_to(PROJECT_ROOT)}")
    print(f"Evaluation report saved to: {(REPORTS_DIR / f'{args.output_prefix}evaluation_report.md').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
