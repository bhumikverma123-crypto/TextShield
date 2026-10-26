"""TextShield local Streamlit app backed by the saved synthetic-data model."""
from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any

import joblib
import streamlit as st

from src.explain import explain_with_model
from src.prototype_analysis import SAMPLES

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "textshield_tuned.joblib"
MAX_MESSAGE_CHARS = 20_000
EMPTY_SAMPLE = "Choose a sample…"


@st.cache_resource(show_spinner=False)
def load_model() -> Any:
    """Load the saved pipeline once per local Streamlit process."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "The tuned model file is missing. From the project root run "
            "`.\\.venv\\Scripts\\python.exe -m src.tune` to create it."
        )
    return joblib.load(MODEL_PATH)


def sanitize_message(message: str) -> str:
    """Normalize Unicode and remove NUL bytes without stripping useful punctuation."""
    return unicodedata.normalize("NFC", message.replace("\x00", "")).strip()


def validate_message(message: str) -> str:
    """Validate one submitted message and return its normalized text."""
    cleaned = sanitize_message(message)
    if not cleaned:
        raise ValueError("Enter a message or choose a sample before analyzing.")
    if len(cleaned) > MAX_MESSAGE_CHARS:
        raise ValueError(
            f"This message has {len(cleaned):,} characters. The prototype limit is "
            f"{MAX_MESSAGE_CHARS:,}; please analyze a shorter excerpt."
        )
    return cleaned


def _load_sample() -> None:
    """Copy the selected fictional example into the message field."""
    selected = st.session_state.get("sample_choice", EMPTY_SAMPLE)
    if selected in SAMPLES:
        st.session_state.message_input = SAMPLES[selected]
        st.session_state.pop("analysis_result", None)


def _clear_message() -> None:
    """Clear the message and all displayed analysis."""
    st.session_state.message_input = ""
    st.session_state.pop("analysis_result", None)
    st.session_state.sample_choice = EMPTY_SAMPLE


def _safety_advice(result: dict[str, Any]) -> str:
    """Return cautious, practical guidance based on displayed model/rule outputs."""
    if result["predicted_class"] == "suspicious" or result["rule_review"] == "Potential warning signs need review":
        return (
            "Pause before responding. Verify unexpected requests using contact details you already know. "
            "Do not share passwords, one-time codes, or payment details, and do not follow a link just because the message asks you to."
        )
    return (
        "No strong warning pattern was identified by this demo. That does not confirm the message is safe. "
        "Verify unexpected requests independently and avoid sharing sensitive information."
    )


def main() -> None:
    """Render the message-analysis flow using the saved tuned model."""
    st.set_page_config(page_title="TextShield", page_icon="🛡️", layout="centered")
    st.title("🛡️ TextShield")
    st.caption("Explainable NLP-Based Social Engineering Message Risk Analyzer")
    st.info(
        "This local demonstration uses a model trained on synthetic messages. "
        "Its score is an estimate, not a calibrated probability or security verdict.",
        icon="ℹ️",
    )

    st.subheader("Analyze a message")
    st.selectbox(
        "Try a fictional sample",
        [EMPTY_SAMPLE, *SAMPLES.keys()],
        key="sample_choice",
        on_change=_load_sample,
        help="Samples use fictional names and reserved example.invalid domains.",
    )
    st.selectbox("Message type", ["SMS", "Email", "Other text"], key="message_type")
    st.text_area(
        "Message text",
        key="message_input",
        height=180,
        max_chars=MAX_MESSAGE_CHARS + 1,
        placeholder="Paste a message to analyze. Avoid sensitive information.",
        help=(
            f"Maximum {MAX_MESSAGE_CHARS:,} characters. Text is processed in this local app session. "
            "Do not enter passwords, OTPs, account numbers, or other sensitive details."
        ),
    )
    left, right = st.columns(2)
    with left:
        analyze = st.button("Analyze message", type="primary", use_container_width=True)
    with right:
        st.button("Clear", on_click=_clear_message, use_container_width=True)

    if analyze:
        try:
            message = validate_message(st.session_state.get("message_input", ""))
        except (ValueError, TypeError, UnicodeError) as exc:
            st.warning(str(exc))
            st.session_state.pop("analysis_result", None)
        else:
            try:
                with st.spinner("Analyzing locally…"):
                    pipeline = load_model()
                    result = explain_with_model(message, pipeline)
                    result["message_type"] = st.session_state.get("message_type", "Other text")
                    st.session_state.analysis_result = result
            except FileNotFoundError as exc:
                st.error(str(exc))
                st.session_state.pop("analysis_result", None)
            except (OSError, ValueError, TypeError, UnicodeError, EOFError) as exc:
                st.error(f"Analysis could not be completed: {exc}")
                st.session_state.pop("analysis_result", None)

    result = st.session_state.get("analysis_result")
    if result:
        st.divider()
        st.subheader("Model output")
        friendly_class = {
            "suspicious": "Suspicious pattern predicted",
            "likely_legitimate": "Likely legitimate pattern predicted",
        }.get(result["predicted_class"], result["predicted_class"])
        score_col, band_col = st.columns(2)
        with score_col:
            st.metric(
                "Estimated model score",
                f"{result['estimated_model_score']:.1f} / 100",
                help=result["score_description"],
            )
        with band_col:
            st.metric("Model risk band", result["risk_band"])
        st.markdown(f"**Predicted class:** {friendly_class}")
        st.caption(
            f"Message type selected: {result['message_type']} (context only; the current model analyzes message text). "
            f"{result['score_description']} "
            "Risk bands use the documented demo thresholds: Low <35, Medium 35–<70, High ≥70."
        )

        st.subheader("Rule-based warning signals")
        if result["rule_signals"]:
            for signal in result["rule_signals"]:
                label = "Context only" if signal["kind"] == "context" else "Potential warning sign"
                st.markdown(f"- **{label} — {signal['name'].replace('_', ' ').title()}:** {signal['description']}")
        else:
            st.write("No listed rule indicators matched. This does not prove the message is safe.")
        st.markdown(f"**Rule review:** {result['rule_review']}")
        st.caption(result["rule_review_note"])
        st.caption(result["combination_note"])

        st.subheader("Safety guidance")
        st.write(_safety_advice(result))
        st.warning(
            "TextShield can be wrong or miss warning signs. It does not replace security software or professional security advice.",
            icon="⚠️",
        )

    with st.expander("Privacy and how this demo works"):
        st.write(
            "The saved classifier was trained and evaluated using fictional synthetic messages. "
            "The model predicts a class and an uncalibrated score; readable rules identify potential contextual signals. "
            "Rules do not change the model score or band. Message text is processed in the local running app. "
            "The app does not transmit text, call external APIs, track usage, or open URLs. Avoid entering sensitive content."
        )


if __name__ == "__main__":
    main()
