"""Saved model loading and inference smoke tests."""
from __future__ import annotations

from pathlib import Path

import joblib
import pytest

from src.explain import explain_with_model

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("artifact", ["textshield_baseline.joblib", "textshield_tuned.joblib"])
def test_saved_model_pipeline_loads_and_predicts(artifact: str) -> None:
    model_path = ROOT / "models" / artifact
    if not model_path.is_file():
        pytest.skip(f"Optional saved model artifact is absent: {model_path.name}")

    model = joblib.load(model_path)
    result = explain_with_model(
        "Fictional library group: the reading-circle notes are ready for tomorrow.", model
    )

    assert result["predicted_class"] in {"likely_legitimate", "suspicious"}
    assert 0.0 <= result["estimated_model_score"] <= 100.0
    assert result["risk_band"] in {"Low", "Medium", "High"}
    assert result["urls_opened"] is False
