"""Streamlit integration smoke checks using its in-process AppTest runner."""
from __future__ import annotations

from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

from src.prototype_analysis import SAMPLES

ROOT = Path(__file__).resolve().parents[1]


def _run_message(message: str):
    app = AppTest.from_file(ROOT / "app.py").run()
    assert not app.exception
    app.text_area(key="message_input").set_value(message)
    app.button[0].click().run()
    assert not app.exception
    return app


def test_ui_analyzes_representative_benign_and_suspicious_synthetic_messages() -> None:
    benign = _run_message("The fictional reading group meets tomorrow in the west hall.")
    assert any("Model output" in item.value for item in benign.subheader)
    assert any("Rule-based warning signals" in item.value for item in benign.subheader)

    suspicious = _run_message(
        "Fictional Northstar support: send your one-time verification code to confirm your account."
    )
    assert not suspicious.exception
    assert any("Potential warning sign" in item.value for item in suspicious.markdown)


def test_ui_sample_choices_are_fictional_examples() -> None:
    app = AppTest.from_file(ROOT / "app.py").run()
    assert not app.exception
    assert SAMPLES
    assert all("example.invalid" in text or "http" not in text for text in SAMPLES.values())


def test_ui_empty_and_over_limit_messages_are_rejected() -> None:
    empty = _run_message("   ")
    assert any("Enter a message" in item.value for item in empty.warning)

    long_input = _run_message("x" * 20_001)
    assert any("prototype limit" in item.value for item in long_input.warning)
