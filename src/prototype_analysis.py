"""Reusable mock analysis for the Phase 1 TextShield prototype."""
from __future__ import annotations
import re

SAMPLES = {
    "Fictional delivery notice (mock suspicious example)": "Parcel notice from Northstar Parcel: confirm your account at https://tracking.example.invalid before the delivery window closes.",
    "Fictional everyday coordination (mock benign example)": "Hi Mira, are we still meeting at the library at 4? I can bring the notes.",
    "Fictional promotion with a safe example link (mock benign example)": "Maple Books newsletter: our spring reading list is ready at https://maple-books.example.invalid. Unsubscribe any time.",
}

SIGNAL_PATTERNS = {
    "Urgency or time pressure": re.compile(r"\b(urgent|immediately|now|expires?|within\s+\d+\s+hours?)\b", re.I),
    "Account or credential request": re.compile(r"\b(verify|confirm|password|sign[ -]?in|account)\b", re.I),
    "Payment or fee wording": re.compile(r"\b(pay|payment|fee|invoice)\b", re.I),
    "Prize or reward wording": re.compile(r"\b(prize|winner|reward|lottery)\b", re.I),
    "Link present": re.compile(r"https?://\S+", re.I),
}


def mock_analysis(message: str) -> tuple[str, list[str], str]:
    """Return transparent placeholder output for the UI prototype only."""
    found = [name for name, pattern in SIGNAL_PATTERNS.items() if pattern.search(message)]
    if len(found) >= 3:
        return ("Prototype label: suspicious", "Pause and verify through a contact method you already trust. Do not share credentials or payment details.", found)
    if found:
        return ("Prototype label: needs a closer look", "Treat these as prompts to check the context. A link or isolated keyword does not establish that a message is unsafe.", found)
    return ("Prototype label: no listed demo signals found", "This mock result cannot confirm that a message is legitimate. Verify unexpected requests independently.", found)
