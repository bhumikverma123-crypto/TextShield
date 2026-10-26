"""Data generation and preprocessing checks for synthetic-only inputs."""
from __future__ import annotations

import pandas as pd
import pytest

from src.generate_synthetic_data import generate_dataset, validate_dataset
from src.preprocess import REQUIRED_COLUMNS, normalize_text, prepare_frame, validate_input_frame


def test_generator_is_deterministic_and_schema_valid() -> None:
    first = generate_dataset(seed=17, examples_per_scenario=2)
    second = generate_dataset(seed=17, examples_per_scenario=2)

    assert first == second
    assert len(first) == 32
    assert validate_dataset(first)["scenario_split_consistent"] is True
    assert all(set(row) == REQUIRED_COLUMNS for row in first)


def test_generated_messages_only_use_reserved_example_urls() -> None:
    rows = generate_dataset(seed=21, examples_per_scenario=2)
    urls = [word for row in rows for word in row["message_text"].split() if "http" in word]
    assert urls
    assert all("example.invalid" in url for url in urls)


def test_preprocessing_normalizes_whitespace_without_changing_url() -> None:
    message = "  Hello\tthere!  https://info.example.invalid/item  "
    assert normalize_text(message) == "Hello there! https://info.example.invalid/item"


def test_frame_validation_and_preparation() -> None:
    frame = pd.DataFrame(
        [{"message_id": "SYN-1", "message_text": "  fictional\tmessage ", "label": "likely_legitimate",
          "message_type": "other", "scenario": "demo", "split": "train"}]
    )
    validate_input_frame(frame)
    prepared = prepare_frame(frame)
    assert prepared.loc[0, "message_text"] == "fictional message"
    assert prepared.loc[0, "char_length"] == len("fictional message")
    assert prepared.loc[0, "word_count"] == 2

    with pytest.raises(ValueError, match="missing required columns"):
        validate_input_frame(frame.drop(columns=["split"]))


def test_preprocessing_rejects_non_string_messages() -> None:
    with pytest.raises(TypeError, match="Expected a string"):
        normalize_text(None)  # type: ignore[arg-type]
