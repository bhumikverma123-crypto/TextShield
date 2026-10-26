"""Transparent rule indicators and model-assisted explanations for TextShield.

This module never opens or fetches URLs. Its rules report potential context,
not definitive scam labels. The model score is an uncalibrated demo score.
"""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit

URL_PATTERN = re.compile(r"https?://[^\s<>()\[\]{}\"']+", re.IGNORECASE)
URGENCY_PATTERN = re.compile(
    r"\b(?:urgent|immediately|right away|expires? today|last chance|before\s+(?:\w+|\d{1,2}:\d{2})|within\s+\d+\s+hours?|act now)\b",
    re.IGNORECASE,
)
CREDENTIAL_OBJECT_PATTERN = re.compile(
    r"\b(?:password|passcode|one[ -]?time code|verification code|access code|sign[ -]?in details|login credentials|otp)\b",
    re.IGNORECASE,
)
REQUEST_VERB_PATTERN = re.compile(
    r"\b(?:send|share|provide|reply with|enter|type|text|submit|confirm|verify)\b",
    re.IGNORECASE,
)
ACCOUNT_PATTERN = re.compile(r"\b(?:account|profile|identity|sign[ -]?in)\b", re.IGNORECASE)
PAYMENT_PATTERN = re.compile(
    r"\b(?:pay|payment|fee|processing charge|transfer|wire|billing details|payment details|release charge)\b",
    re.IGNORECASE,
)
PAYMENT_ACTION_PATTERN = re.compile(
    r"\b(?:pay|send|transfer|settle|submit|provide|reply|due|charge|required)\b",
    re.IGNORECASE,
)
PRIZE_PATTERN = re.compile(r"\b(?:prize|winner|reward|lottery|giveaway|selected for)\b", re.IGNORECASE)
CLAIM_PATTERN = re.compile(r"\b(?:claim|collect|release|processing charge|assigned|chosen|selected)\b", re.IGNORECASE)
SUPPORT_IDENTITY_PATTERN = re.compile(
    r"\b(?:this is|i am|i'm)\b.{0,50}\b(?:support|help desk|service desk|security team|account desk|protection team)\b",
    re.IGNORECASE,
)

HIGH_CONCERN_SIGNALS = {
    "credential_request",
    "account_verification_request",
    "payment_request",
    "prize_or_reward_claim",
    "support_impersonation_request",
    "suspicious_url_structure",
}

SIGNAL_DESCRIPTIONS = {
    "urgency_language": "Time-pressure wording is present. Urgency can be harmless and is not a scam verdict.",
    "link_present": "A link appears in the text. A link by itself is not evidence that the message is unsafe.",
    "credential_request": "The message appears to ask the recipient to share or enter a password, passcode, or verification code.",
    "account_verification_request": "The message asks the recipient to verify or confirm an account or identity.",
    "payment_request": "The message appears to request a payment, fee, transfer, or billing action.",
    "prize_or_reward_claim": "The message combines prize or reward language with a claim or release action.",
    "support_impersonation_request": "The sender claims to represent a support or security desk while requesting sensitive action.",
    "suspicious_url_structure": "The URL has an unusual structure (for example, insecure HTTP, embedded user information, or a punycode hostname). The URL was not opened or checked online.",
}


def _extract_urls(message: str) -> list[str]:
    """Return URL strings found in text without making any network request."""
    return [match.rstrip(".,!?;:") for match in URL_PATTERN.findall(message)]


def _nearby_match(first: re.Pattern[str], second: re.Pattern[str], message: str, window: int = 80) -> bool:
    """Check whether two classes of terms appear close enough to describe one request."""
    for first_match in first.finditer(message):
        start = max(0, first_match.start() - window)
        end = min(len(message), first_match.end() + window)
        if second.search(message[start:end]):
            return True
    return False


def _url_has_suspicious_structure(url: str) -> bool:
    """Flag URL structure only; never resolve, fetch, or follow a URL."""
    try:
        parsed = urlsplit(url)
        hostname = (parsed.hostname or "").lower().rstrip(".")
    except ValueError:
        return True
    if not hostname:
        return True
    if parsed.scheme.lower() == "http":
        return True
    if "@" in parsed.netloc or "xn--" in hostname:
        return True
    # Reserved synthetic domains are intentionally treated as neutral examples.
    if hostname == "example.invalid" or hostname.endswith(".example.invalid"):
        return False
    # Treat explicit generic redirect-style host labels as a review prompt only.
    if any(label in {"go", "redirect", "short", "link"} for label in hostname.split(".")):
        return True
    return False


def analyze_rules(message: str) -> dict[str, Any]:
    """Identify potential warning indicators without assigning a scam label."""
    if not isinstance(message, str):
        raise TypeError("message must be a string")
    urls = _extract_urls(message)
    signals: list[dict[str, str]] = []

    def add_signal(name: str, kind: str = "potential") -> None:
        signals.append({
            "name": name,
            "kind": kind,
            "description": SIGNAL_DESCRIPTIONS[name],
        })

    if URGENCY_PATTERN.search(message):
        add_signal("urgency_language", "context")
    if urls:
        add_signal("link_present", "context")
    if _nearby_match(CREDENTIAL_OBJECT_PATTERN, REQUEST_VERB_PATTERN, message):
        add_signal("credential_request")
    if _nearby_match(ACCOUNT_PATTERN, re.compile(r"\b(?:verify|confirm|restore|unlock|check)\b", re.I), message):
        add_signal("account_verification_request")
    if _nearby_match(PAYMENT_PATTERN, PAYMENT_ACTION_PATTERN, message):
        add_signal("payment_request")
    if _nearby_match(PRIZE_PATTERN, CLAIM_PATTERN, message):
        add_signal("prize_or_reward_claim")
    if SUPPORT_IDENTITY_PATTERN.search(message) and (
        "credential_request" in {signal["name"] for signal in signals}
        or "payment_request" in {signal["name"] for signal in signals}
        or "account_verification_request" in {signal["name"] for signal in signals}
    ):
        add_signal("support_impersonation_request")
    if any(_url_has_suspicious_structure(url) for url in urls):
        add_signal("suspicious_url_structure")

    high_concern = [signal["name"] for signal in signals if signal["name"] in HIGH_CONCERN_SIGNALS]
    return {
        "signals": signals,
        "high_concern_signal_count": len(high_concern),
        "rule_review": "Potential warning signs need review" if high_concern else "No high-concern rule pattern found",
        "rule_review_note": (
            "This is not a scam verdict. Rule review is raised by credential, account-verification, payment, "
            "prize-claim, support-impersonation, or unusual URL-structure patterns. An isolated link or urgency wording does not raise it."
        ),
        "urls_detected_count": len(urls),
        "urls_opened": False,
    }


def risk_band_from_score(score: float) -> str:
    """Map an uncalibrated 0–100 demo score to documented display bands."""
    if not 0.0 <= score <= 100.0:
        raise ValueError("score must be between 0 and 100")
    if score < 35.0:
        return "Low"
    if score < 70.0:
        return "Medium"
    return "High"


def explain_with_model(message: str, pipeline: Any) -> dict[str, Any]:
    """Combine classifier output with independent transparent rule indicators."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must contain non-whitespace text")
    if not hasattr(pipeline, "predict_proba"):
        raise TypeError("pipeline must support predict_proba to produce the demo model score")
    predicted_class = str(pipeline.predict([message])[0])
    classes = list(pipeline.named_steps["classifier"].classes_)
    if "suspicious" not in classes:
        raise ValueError("pipeline does not include the 'suspicious' class")
    probability = float(pipeline.predict_proba([message])[0][classes.index("suspicious")])
    score = round(probability * 100.0, 1)
    rules = analyze_rules(message)
    return {
        "predicted_class": predicted_class,
        "estimated_model_score": score,
        "score_description": "Estimated model score from Logistic Regression; not a calibrated probability.",
        "risk_band": risk_band_from_score(score),
        "rule_signals": rules["signals"],
        "rule_review": rules["rule_review"],
        "rule_review_note": rules["rule_review_note"],
        "urls_detected_count": rules["urls_detected_count"],
        "urls_opened": rules["urls_opened"],
        "combination_note": (
            "The risk band is based on the model score only. Rule indicators are shown independently as supporting review context; "
            "they do not change the score or band. Neither output is definitive."
        ),
    }
