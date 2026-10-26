"""Preprocessing helpers and exploratory analysis for synthetic TextShield data.

Preprocessing is intentionally conservative: it normalizes whitespace only,
leaving message wording, punctuation, and URLs intact for vectorization.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
import tempfile
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CONFIG_DIR = Path(tempfile.gettempdir()) / "textshield-matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import ClassifierMixin

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "synthetic_messages.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FINDINGS_PATH = PROJECT_ROOT / "reports" / "eda_findings.md"
REQUIRED_COLUMNS = {"message_id", "message_text", "label", "message_type", "scenario", "split"}


def normalize_text(text: str) -> str:
    """Trim leading/trailing whitespace and collapse internal whitespace only."""
    if not isinstance(text, str):
        raise TypeError(f"Expected a string message, got {type(text).__name__}")
    return re.sub(r"\s+", " ", text).strip()


def normalize_texts(texts: Iterable[str]) -> list[str]:
    """Normalize a batch of messages while preserving text content and URLs."""
    return [normalize_text(text) for text in texts]


def build_text_pipeline(classifier: ClassifierMixin, **vectorizer_options: Any) -> Pipeline:
    """Build a preprocessing, TF-IDF, and classifier pipeline.

    Fit this full pipeline on training data only. Vectorizer vocabulary is
    learned during ``fit`` so validation and test data cannot leak into it.
    """
    return Pipeline(
        steps=[
            ("normalize", FunctionTransformer(normalize_texts, validate=False)),
            ("tfidf", TfidfVectorizer(**vectorizer_options)),
            ("classifier", classifier),
        ]
    )


def validate_input_frame(frame: pd.DataFrame) -> None:
    """Raise a readable error if the generated CSV does not match its schema."""
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Dataset has no rows")
    if frame["message_text"].isna().any() or frame["message_text"].astype(str).str.strip().eq("").any():
        raise ValueError("Dataset contains missing or empty messages")


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a normalized message column and length measures."""
    validate_input_frame(frame)
    prepared = frame.copy()
    prepared["message_text"] = prepared["message_text"].map(normalize_text)
    prepared["char_length"] = prepared["message_text"].str.len()
    prepared["word_count"] = prepared["message_text"].str.split().str.len()
    return prepared


def _save_class_distribution(frame: pd.DataFrame, path: Path) -> None:
    counts = frame["label"].value_counts().reindex(["likely_legitimate", "suspicious"], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(["Likely legitimate", "Suspicious"], counts.values, color=["#2b7a78", "#bd4b4b"])
    ax.bar_label(bars, padding=4)
    ax.set_title("Synthetic message class counts")
    ax.set_ylabel("Messages")
    ax.set_ylim(0, max(counts.max() * 1.15, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_length_distribution(frame: pd.DataFrame, path: Path) -> None:
    labels = ["likely_legitimate", "suspicious"]
    values = [frame.loc[frame["label"] == label, "word_count"] for label in labels]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.boxplot(values, tick_labels=["Likely legitimate", "Suspicious"], showmeans=True)
    ax.set_title("Message word-count distribution by class")
    ax.set_ylabel("Words per message")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_message_types(frame: pd.DataFrame, path: Path) -> None:
    counts = frame.groupby(["message_type", "label"]).size().unstack(fill_value=0)
    counts = counts.reindex(["sms", "email", "other"], fill_value=0)
    colors = {"likely_legitimate": "#2b7a78", "suspicious": "#bd4b4b"}
    ax = counts[["likely_legitimate", "suspicious"]].plot(
        kind="bar", figsize=(7, 4.2), color=[colors["likely_legitimate"], colors["suspicious"]]
    )
    ax.set_title("Synthetic message types by class")
    ax.set_xlabel("Message type")
    ax.set_ylabel("Messages")
    ax.legend(["Likely legitimate", "Suspicious"])
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=160)
    plt.close(ax.figure)


def _save_common_terms(frame: pd.DataFrame, path: Path, top_n: int = 15) -> list[tuple[str, int]]:
    vectorizer = CountVectorizer(stop_words="english", max_features=top_n)
    matrix = vectorizer.fit_transform(frame["message_text"])
    counts = matrix.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, counts.tolist()), key=lambda item: (-item[1], item[0]))
    ranked.reverse()
    ranked = sorted(ranked, key=lambda item: item[1])
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.barh([term for term, _ in ranked], [count for _, count in ranked], color="#476a8a")
    ax.set_title("Common synthetic message terms (English stop words removed)")
    ax.set_xlabel("Term frequency")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return list(reversed(ranked))


def _normalized_duplicate_count(texts: pd.Series) -> int:
    """Count repeated message bodies after case and punctuation normalization."""
    normalized = texts.map(lambda text: re.sub(r"[^a-z0-9]+", " ", text.lower()).strip())
    return int(normalized.duplicated().sum())


def run_eda(data_path: Path = DEFAULT_DATA_PATH, figures_dir: Path = FIGURES_DIR) -> dict[str, Any]:
    """Load the synthetic CSV, save exploratory plots, and return key findings."""
    frame = pd.read_csv(data_path)
    prepared = prepare_frame(frame)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _save_class_distribution(prepared, figures_dir / "class_distribution.png")
    _save_length_distribution(prepared, figures_dir / "message_length_by_class.png")
    _save_message_types(prepared, figures_dir / "message_type_by_class.png")
    common_terms = _save_common_terms(prepared, figures_dir / "common_terms.png")

    label_counts = prepared["label"].value_counts().sort_index().to_dict()
    type_counts = prepared["message_type"].value_counts().sort_index().to_dict()
    split_counts = prepared["split"].value_counts().sort_index().to_dict()
    scenario_split_counts = prepared.groupby("split")["scenario"].nunique().sort_index().to_dict()
    length_stats = prepared.groupby("label")["word_count"].agg(["mean", "median", "min", "max"]).round(2)
    url_by_class = prepared.assign(has_url=prepared["message_text"].str.contains(r"https?://", case=False, regex=True)).groupby("label")["has_url"].sum().to_dict()
    findings = {
        "rows": int(len(prepared)),
        "class_counts": {str(key): int(value) for key, value in label_counts.items()},
        "message_type_counts": {str(key): int(value) for key, value in type_counts.items()},
        "split_counts": {str(key): int(value) for key, value in split_counts.items()},
        "scenario_families_per_split": {str(key): int(value) for key, value in scenario_split_counts.items()},
        "exact_normalized_duplicates": _normalized_duplicate_count(prepared["message_text"]),
        "word_count_stats": length_stats.to_dict(orient="index"),
        "url_message_counts_by_class": {str(key): int(value) for key, value in url_by_class.items()},
        "common_terms": common_terms,
        "figure_paths": [
            "reports/figures/class_distribution.png",
            "reports/figures/message_length_by_class.png",
            "reports/figures/message_type_by_class.png",
            "reports/figures/common_terms.png",
        ],
    }
    return findings


def write_findings(findings: dict[str, Any], path: Path = FINDINGS_PATH) -> None:
    """Save actual EDA values and cautious interpretation to Markdown."""
    stats_lines = [
        f"- **{label}:** mean {stats['mean']} words, median {stats['median']}, range {stats['min']}–{stats['max']}"
        for label, stats in findings["word_count_stats"].items()
    ]
    terms = ", ".join(f"`{term}` ({count})" for term, count in findings["common_terms"][:10])
    content = f"""# TextShield Exploratory Data Analysis

Generated from the fixed **synthetic-only** CSV at `data/processed/synthetic_messages.csv` using `python -m src.preprocess`. Counts and summaries below are computed from the file; they are not model performance metrics.

## Dataset profile

- Rows: **{findings['rows']}**
- Class counts: `{findings['class_counts']}`
- Message types: `{findings['message_type_counts']}`
- Split counts: `{findings['split_counts']}`
- Scenario families per split: `{findings['scenario_families_per_split']}`
- Duplicate normalized messages: **{findings['exact_normalized_duplicates']}**
- Messages containing a URL by class: `{findings['url_message_counts_by_class']}`

## Message length

Word-count statistics:
{chr(10).join(stats_lines)}

## Common terms

Top terms by raw frequency after scikit-learn's English stop-word list: {terms}.

These counts describe the hand-authored synthetic templates; frequent terms may reflect template construction and should not be interpreted as real-world indicators.

## Figures

- `reports/figures/class_distribution.png`
- `reports/figures/message_length_by_class.png`
- `reports/figures/message_type_by_class.png`
- `reports/figures/common_terms.png`

## Preprocessing decision

The reusable preprocessing step only trims and collapses whitespace. It preserves punctuation and URL text before TF-IDF vectorization. `build_text_pipeline()` places normalization, `TfidfVectorizer`, and the estimator in one scikit-learn `Pipeline`; fit the complete pipeline on training data only so the vocabulary is not learned from validation or test data.

## Limitations

All observations come from synthetic templates. EDA does not show that these patterns distinguish real malicious and legitimate messages. No model has been trained or evaluated in this phase.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    """Run EDA and write actual findings and figures under reports/."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="CSV file to analyze")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR, help="figure output directory")
    args = parser.parse_args()
    findings = run_eda(args.data, args.figures_dir)
    write_findings(findings)
    print(f"Analyzed {findings['rows']} rows from {args.data}")
    print(f"Class counts: {findings['class_counts']}")
    print(f"Message-type counts: {findings['message_type_counts']}")
    print(f"Split counts: {findings['split_counts']}")
    print(f"Duplicate normalized messages: {findings['exact_normalized_duplicates']}")
    print(f"Findings saved to: {FINDINGS_PATH.relative_to(PROJECT_ROOT)}")
    for figure in findings["figure_paths"]:
        print(f"Figure saved to: {figure}")


if __name__ == "__main__":
    main()
