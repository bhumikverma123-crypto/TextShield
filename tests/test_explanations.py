"""Unit checks for transparent TextShield explanation rules."""
from __future__ import annotations

import pytest

from src.explain import analyze_rules, risk_band_from_score


def names(message: str) -> set[str]:
    """Return matched indicator names for concise assertions."""
    return {signal["name"] for signal in analyze_rules(message)["signals"]}


def test_benign_link_is_context_only() -> None:
    """A reserved fictional newsletter link alone must not raise rule review."""
    result = analyze_rules("The reading list is here: https://books.example.invalid/list")
    assert "link_present" in {signal["name"] for signal in result["signals"]}
    assert "suspicious_url_structure" not in {signal["name"] for signal in result["signals"]}
    assert result["rule_review"] == "No high-concern rule pattern found"
    assert result["urls_opened"] is False


def test_harmless_urgent_coordination_is_not_rule_labeled_as_scam() -> None:
    """Harmless urgency should be visible as context without high-concern review."""
    result = analyze_rules("Urgent for rehearsal: the room moved to the west hall; bring the printed schedule.")
    assert "urgency_language" in {signal["name"] for signal in result["signals"]}
    assert result["high_concern_signal_count"] == 0
    assert result["rule_review"] == "No high-concern rule pattern found"


def test_sensitive_request_combination_triggers_potential_indicators() -> None:
    message = (
        "This is Mira with the Northstar Parcel help desk. Confirm your account and send your "
        "one-time verification code immediately to keep access active."
    )
    matched = names(message)
    assert "credential_request" in matched
    assert "account_verification_request" in matched
    assert "support_impersonation_request" in matched
    assert "urgency_language" in matched
    assert analyze_rules(message)["rule_review"] == "Potential warning signs need review"


def test_payment_amount_without_request_is_not_enough() -> None:
    """A money amount alone does not trigger the payment-request rule."""
    result = analyze_rules("The fictional club's annual budget is 20 credits in the planning note.")
    assert "payment_request" not in {signal["name"] for signal in result["signals"]}
    assert result["rule_review"] == "No high-concern rule pattern found"


def test_url_format_warning_never_fetches_url() -> None:
    """A structurally unusual reserved example URL is only flagged, never fetched."""
    result = analyze_rules("Review this sample: http://go.example.invalid/x")
    assert "suspicious_url_structure" in {signal["name"] for signal in result["signals"]}
    assert result["urls_opened"] is False


def test_risk_band_thresholds_are_explicit() -> None:
    assert risk_band_from_score(34.9) == "Low"
    assert risk_band_from_score(35.0) == "Medium"
    assert risk_band_from_score(69.9) == "Medium"
    assert risk_band_from_score(70.0) == "High"
    with pytest.raises(ValueError):
        risk_band_from_score(100.1)
