"""Console fallback for local model-assisted TextShield analysis in Spyder."""
from __future__ import annotations

from pathlib import Path

import joblib

from src.explain import explain_with_model
from src.prototype_analysis import SAMPLES

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "textshield_tuned.joblib"
MAX_MESSAGE_CHARS = 20_000


def main() -> None:
    """Load the local tuned model, analyze one message, and print the explanation."""
    print("TextShield — local synthetic-data demonstration")
    print("Avoid entering passwords, OTPs, account numbers, or other sensitive information.\n")
    print("Fictional sample messages:")
    for index, (name, sample) in enumerate(SAMPLES.items(), start=1):
        print(f"  {index}. {name}: {sample}")
    if not MODEL_PATH.is_file():
        print("\nSaved tuned model not found. From the project root run:")
        print("  .\\.venv\\Scripts\\python.exe -m src.tune")
        return
    message = input("\nMessage to inspect (blank to exit): ").strip()
    if not message:
        print("No message entered; exiting.")
        return
    if len(message) > MAX_MESSAGE_CHARS:
        print(f"Message is too long. The limit is {MAX_MESSAGE_CHARS:,} characters.")
        return
    try:
        pipeline = joblib.load(MODEL_PATH)
        result = explain_with_model(message, pipeline)
    except (OSError, ValueError, TypeError, UnicodeError, EOFError) as exc:
        print(f"Analysis could not be completed: {exc}")
        return

    print("\nMODEL OUTPUT")
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Estimated model score: {result['estimated_model_score']:.1f}/100 (not calibrated)")
    print(f"Model risk band: {result['risk_band']}")
    print("\nRULE-BASED SIGNALS")
    if result["rule_signals"]:
        for signal in result["rule_signals"]:
            print(f"- {signal['kind']}: {signal['name']} — {signal['description']}")
    else:
        print("- No listed indicators matched.")
    print(f"Rule review: {result['rule_review']}")
    print(result["combination_note"])
    print("\nSafety guidance: verify unexpected requests through contact details you already know. Avoid sharing sensitive information.")
    print("This estimate may be wrong and is not a definitive security assessment.")


if __name__ == "__main__":
    main()
