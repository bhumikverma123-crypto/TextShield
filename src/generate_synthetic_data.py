"""Generate a reproducible synthetic-only TextShield message dataset.

No external data, network access, or real personal information is used.
"""
from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "synthetic_messages.csv"
DATA_CARD_PATH = PROJECT_ROOT / "data" / "data_card.md"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "dataset_summary.md"

# Each scenario family belongs to exactly one split. All messages produced from
# its templates stay together, reducing near-duplicate template leakage.
SCENARIOS: tuple[dict[str, Any], ...] = (
    {
        "scenario": "account_review", "label": "suspicious", "split": "train",
        "templates": (
            "{org} account desk: check the review notice at {url}; access is paused until details are confirmed.",
            "A request is waiting for your {org} profile. Use {url} to confirm the account before the review closes.",
            "We could not finish the {org} account check. Open {url} and provide the requested sign-in details.",
        ),
    },
    {
        "scenario": "delivery_fee", "label": "suspicious", "split": "train",
        "templates": (
            "{org} delivery update: a small release fee of {amount} is due before {day}. Continue at {url}.",
            "Your parcel is waiting at the local desk. Settle the handling charge of {amount} at {url} to arrange another attempt.",
            "The courier could not complete the drop-off. Pay the stated {amount} through {url} to keep the item moving.",
        ),
    },
    {
        "scenario": "prize_claim", "label": "suspicious", "split": "train",
        "templates": (
            "Good news from {org}: your name was selected for a reward. Reply today with the claim details to reserve it.",
            "A {item} prize has been assigned to your entry. Contact the prize desk before {time} to arrange collection.",
            "You have been chosen for this week's {org} giveaway. A processing charge is required to release the reward.",
        ),
    },
    {
        "scenario": "support_impersonation", "label": "suspicious", "split": "train",
        "templates": (
            "This is {person} with the {org} help desk. Send the one-time sign-in code so I can keep your access active.",
            "{org} support here: we need your password to complete a security review. Please reply in this chat.",
            "I am handling your {org} case. Confirm the account recovery code now or the request will be closed.",
        ),
    },
    {
        "scenario": "invoice_change", "label": "suspicious", "split": "validation",
        "templates": (
            "{org} billing note: use the revised payment details for invoice {ref}; the earlier instructions are no longer valid.",
            "Please send the open invoice amount of {amount} to the updated account today. Reply when the transfer is complete.",
            "A change was made to the payment destination for reference {ref}. Confirm the transfer with this desk before processing.",
        ),
    },
    {
        "scenario": "document_deadline", "label": "suspicious", "split": "validation",
        "templates": (
            "The {org} document needs your signature by {time}. Open the review copy and enter your account password to proceed.",
            "Final reminder: the shared file for {org} will be removed today unless you verify your sign-in details.",
            "A pending file requires immediate review. Send the access code to {person} so the document can be released.",
        ),
    },
    {
        "scenario": "account_lock_warning", "label": "suspicious", "split": "test",
        "templates": (
            "We noticed a problem with the {org} profile. Restore access now by replying with the requested verification code.",
            "Your account may be locked after {time}. Confirm your password with the service desk to avoid interruption.",
            "The {org} protection team needs a quick sign-in check. Provide the requested details before the review window ends.",
        ),
    },
    {
        "scenario": "unexpected_refund", "label": "suspicious", "split": "test",
        "templates": (
            "A refund of {amount} is waiting for you. Pay the return-processing fee first, then reply with the receipt number.",
            "{org} refunds: your request is approved. Share your account details so we can deposit the credit today.",
            "The refund desk cannot complete reference {ref} until you verify the payment card information by message.",
        ),
    },
    {
        "scenario": "team_coordination", "label": "likely_legitimate", "split": "train",
        "templates": (
            "Hi {person}, are we still meeting with the {org} group at {time}? I can bring the notes.",
            "Quick check for the team: should we meet in {place} or use the usual room this afternoon?",
            "I shared the draft agenda with {org}. Let me know if you want any changes before our next check-in.",
        ),
    },
    {
        "scenario": "school_update", "label": "likely_legitimate", "split": "train",
        "templates": (
            "{org} class note: the reading discussion is in {place} on {day}. Bring your notebook if you can.",
            "Hi {person}, the group project outline is ready. We can review it together during study time.",
            "A reminder for the class: the library room is reserved for our regular session at {time}.",
        ),
    },
    {
        "scenario": "appointment_reminder", "label": "likely_legitimate", "split": "train",
        "templates": (
            "Reminder from {org}: your appointment is scheduled for {day} at {time}. Call the office if you need to reschedule.",
            "Your visit with the {org} desk is on the calendar for {time}. Please bring any notes you would like to discuss.",
            "Hello {person}, confirming our meeting at {place} on {day}. Let us know if the time no longer works.",
        ),
    },
    {
        "scenario": "parcel_tracking", "label": "likely_legitimate", "split": "train",
        "templates": (
            "{org} parcel update: your package is expected on {day}. View the delivery window at {url}.",
            "The parcel was checked in at the neighborhood depot. Follow the status at {url}; no reply is needed.",
            "A delivery for {person} is on its way. You can review the estimated arrival at {url}.",
        ),
    },
    {
        "scenario": "newsletter_promotion", "label": "likely_legitimate", "split": "validation",
        "templates": (
            "{org} monthly note: this week's reading list is available at {url}. Unsubscribe any time.",
            "Our seasonal {item} collection is ready to browse at {url}. This is an optional newsletter update.",
            "A new set of community recommendations is posted at {url}. Visit when convenient; no action is required.",
        ),
    },
    {
        "scenario": "harmless_urgent_coordination", "label": "likely_legitimate", "split": "validation",
        "templates": (
            "Urgent for our rehearsal: the room changed to {place}. Please bring the printed schedule if you have it.",
            "Time-sensitive team note: could you send the meeting notes before {time}? No action is needed outside our group.",
            "The {org} club needs a quick headcount for {day}. Reply when you get a chance so we can reserve enough seats.",
        ),
    },
    {
        "scenario": "community_event", "label": "likely_legitimate", "split": "test",
        "templates": (
            "{org} community board: the open workshop starts at {time} in {place}. Everyone is welcome.",
            "Hi {person}, the neighborhood book exchange is on {day}. Bring a title if you would like to trade.",
            "The library posted its upcoming event calendar at {url}. It is fine to just browse the dates.",
        ),
    },
    {
        "scenario": "routine_service_notice", "label": "likely_legitimate", "split": "test",
        "templates": (
            "{org} service note: planned maintenance is scheduled for {day}. Details are posted at {url}.",
            "Your membership renewal date is {day}. You can review the standard plan information at {url} whenever convenient.",
            "The community service desk will close early at {time}. Please use the usual contact channel if you need help.",
        ),
    },
)

NAMES = ("Mira", "Tavi", "Niko", "Luma", "Ravi", "Sena", "Oren", "Kavi", "Noor", "Ira")
ORGS = ("Maple Learning Circle", "Northstar Parcel", "Cedar Lane Library", "Blue Pebble Studio", "Willow Community Desk", "Amber Field Club", "Cloudberry Books", "Juniper Workshop")
PLACES = ("the west hall", "Room 12", "the reading room", "the garden pavilion", "the usual studio", "the north meeting space")
ITEMS = ("book bundle", "garden kit", "art set", "reading pack", "craft box", "music pass")
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")
TIMES = ("10:30", "noon", "2:15 pm", "4:00 pm", "6:30 pm")
AMOUNTS = ("8 credits", "12 tokens", "15 units", "20 credits")
PREFIXES = ("", "Note: ", "Hello {person} — ", "Update: ")
CLOSINGS = ("", " Thanks, {person}.", " Please review when convenient.", " — {org}")


def make_context(rng: random.Random, index: int) -> dict[str, str]:
    """Create fictional values for one generated message."""
    return {
        "person": rng.choice(NAMES),
        "org": rng.choice(ORGS),
        "place": rng.choice(PLACES),
        "item": rng.choice(ITEMS),
        "day": rng.choice(DAYS),
        "time": rng.choice(TIMES),
        "amount": rng.choice(AMOUNTS),
        "ref": f"TS-{rng.randrange(1000, 9999)}-{index % 97:02d}",
        "url": f"https://{rng.choice(('info', 'status', 'notice', 'events'))}.{rng.choice(('maple-books', 'northstar-parcel', 'cedar-library', 'willow-desk'))}.example.invalid/{rng.randrange(100, 999)}",
    }


def generate_dataset(seed: int = 42, examples_per_scenario: int = 75) -> list[dict[str, str]]:
    """Generate deterministic fictional messages with scenario-group splits."""
    if examples_per_scenario < 1:
        raise ValueError("examples_per_scenario must be at least 1")
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    seen_texts: set[str] = set()
    row_number = 1
    for scenario in SCENARIOS:
        for index in range(examples_per_scenario):
            for _attempt in range(100):
                context = make_context(rng, row_number)
                template = rng.choice(scenario["templates"])
                prefix = rng.choice(PREFIXES).format(**context)
                closing = rng.choice(CLOSINGS).format(**context)
                message = f"{prefix}{template.format(**context)}{closing}"
                normalized = normalized_text(message)
                if normalized not in seen_texts:
                    seen_texts.add(normalized)
                    break
            else:
                raise ValueError(
                    f"Could not create a unique message for scenario {scenario['scenario']}"
                )
            rows.append({
                "message_id": f"SYN-{row_number:06d}",
                "message_text": message,
                "label": scenario["label"],
                "message_type": rng.choice(("sms", "email", "other")),
                "scenario": scenario["scenario"],
                "split": scenario["split"],
            })
            row_number += 1
    rng.shuffle(rows)
    return rows


def normalized_text(value: str) -> str:
    """Normalize whitespace and punctuation for duplicate auditing."""
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def validate_dataset(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Validate schema, labels, distributions, duplicates, and group splits."""
    required = ("message_id", "message_text", "label", "message_type", "scenario", "split")
    if not rows:
        raise ValueError("Dataset is empty")
    if any(set(row) != set(required) for row in rows):
        raise ValueError(f"Rows must have exactly these columns: {', '.join(required)}")
    if any(not row["message_text"].strip() for row in rows):
        raise ValueError("Dataset contains missing or empty messages")
    if len({row["message_id"] for row in rows}) != len(rows):
        raise ValueError("message_id values must be unique")
    expected_labels = {scenario["scenario"]: scenario["label"] for scenario in SCENARIOS}
    expected_splits = {scenario["scenario"]: scenario["split"] for scenario in SCENARIOS}
    if any(row["label"] != expected_labels.get(row["scenario"]) for row in rows):
        raise ValueError("A scenario contains an inconsistent label")
    if any(row["split"] != expected_splits.get(row["scenario"]) for row in rows):
        raise ValueError("A scenario family crosses or mismatches its assigned split")
    if any(row["message_type"] not in {"sms", "email", "other"} for row in rows):
        raise ValueError("Unsupported message_type value")
    texts = [normalized_text(row["message_text"]) for row in rows]
    exact_duplicates = len(texts) - len(set(texts))
    if exact_duplicates:
        raise ValueError(f"Found {exact_duplicates} duplicate message texts")

    # Audit cross-split near duplicates only. Scenario-grouped splitting is the
    # primary leakage guard; this additional check catches unusually similar text.
    candidates: list[tuple[str, str, str]] = [
        (text, row["split"], row["scenario"]) for text, row in zip(texts, rows)
    ]
    near_pairs: list[tuple[str, str, float]] = []
    for i, (left, left_split, left_scenario) in enumerate(candidates):
        for j in range(i + 1, len(candidates)):
            right, right_split, right_scenario = candidates[j]
            if left_split == right_split or left_scenario == right_scenario:
                continue
            if abs(len(left) - len(right)) > 12:
                continue
            matcher = SequenceMatcher(None, left, right, autojunk=False)
            # quick_ratio is an upper bound on ratio; skip full matching when
            # the pair cannot meet the near-duplicate threshold.
            if matcher.quick_ratio() < 0.92:
                continue
            similarity = matcher.ratio()
            if similarity >= 0.92:
                near_pairs.append((left_scenario, right_scenario, similarity))
    if near_pairs:
        examples = near_pairs[:3]
        raise ValueError(f"Found {len(near_pairs)} cross-split near-duplicate pairs; examples: {examples}")

    counts = {
        "rows": len(rows),
        "class_counts": dict(Counter(row["label"] for row in rows)),
        "message_type_counts": dict(Counter(row["message_type"] for row in rows)),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "scenario_counts": dict(Counter(row["scenario"] for row in rows)),
        "exact_duplicate_count": exact_duplicates,
        "cross_split_near_duplicate_count": len(near_pairs),
        "scenario_split_consistent": True,
        "seed_reproducible": True,
    }
    return counts


def write_csv(rows: list[dict[str, str]], path: Path = OUTPUT_PATH) -> None:
    """Write generated messages as UTF-8 CSV with a stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("message_id", "message_text", "label", "message_type", "scenario", "split")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_data_card(summary: dict[str, Any], seed: int, examples_per_scenario: int) -> None:
    """Write provenance, generation details, checks, and limitations."""
    card = f"""# TextShield Synthetic Data Card

## Status and provenance

This dataset is **fully synthetic** and was generated locally by `src/generate_synthetic_data.py`. It contains no downloaded corpus, scraped messages, real personal information, or functioning non-reserved links. All organizations, names, references, and scenarios are fictional. URL examples use reserved `.invalid` hostnames and are never opened or fetched by the generator or analyzer.

## Generation method

- **Generator:** `src/generate_synthetic_data.py`
- **Default random seed:** `{seed}`
- **Examples per scenario family:** `{examples_per_scenario}`
- **Scenario families:** {len(SCENARIOS)} predefined fictional template families
- **Template variation:** Each row selects a scenario-specific template and varies fictional names, organizations, dates, times, places, references, safe placeholder links, and message types with Python's seeded `random.Random`.
- **Splitting:** Families are assigned wholly to train, validation, or test before examples are generated, preventing a single scenario family's templates from crossing splits.
- **Output:** `data/processed/synthetic_messages.csv`

## Dataset summary

- **Total rows:** {summary['rows']}
- **Labels:** `{summary['class_counts']}`
- **Message types:** `{summary['message_type_counts']}`
- **Splits:** `{summary['split_counts']}`
- **Exact duplicate normalized texts:** {summary['exact_duplicate_count']}
- **Cross-split near-duplicate pairs at normalized SequenceMatcher ratio ≥ 0.92:** {summary['cross_split_near_duplicate_count']}

## Fields

`message_id` (synthetic unique identifier), `message_text`, `label` (`suspicious` or `likely_legitimate`), `message_type` (`sms`, `email`, or `other`), `scenario` (template family), and `split` (`train`, `validation`, or `test`).

## Intended use

Safe classroom demonstration, development, and reproducible software exercises only. This dataset is not suitable for operational filtering, security decisions, or estimating performance on real-world messages.

## Limitations

The examples are authored from a limited set of templates and fictional tokens. Their language, class balance, and scenario splits do not represent real message populations. Models may learn artifacts of these templates, and even clean held-out synthetic metrics would not establish real-world effectiveness. The rule-based labels are assigned by scenario, not by human review of real cases.

## Reproducibility and validation

Re-run from the project root with `python -m src.generate_synthetic_data --seed {seed} --examples-per-scenario {examples_per_scenario}` (use `.venv\\Scripts\\python.exe` on Windows). The generator validates schema, non-empty text, unique IDs, scenario/label/split consistency, message types, exact duplicates, and cross-split near duplicates.
"""
    DATA_CARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_CARD_PATH.write_text(card, encoding="utf-8")


def write_summary(summary: dict[str, Any], seed: int, examples_per_scenario: int) -> None:
    """Write a concise human-readable dataset summary."""
    lines = [
        "# Synthetic Dataset Summary",
        "",
        "Generated locally from fictional templates; this is not real-world message data.",
        "",
        f"- Seed: `{seed}`",
        f"- Examples per scenario: `{examples_per_scenario}`",
        f"- Rows: **{summary['rows']}**",
        f"- Class counts: `{summary['class_counts']}`",
        f"- Message type counts: `{summary['message_type_counts']}`",
        f"- Split counts: `{summary['split_counts']}`",
        f"- Exact duplicates: `{summary['exact_duplicate_count']}`",
        f"- Cross-split near duplicates (threshold 0.92): `{summary['cross_split_near_duplicate_count']}`",
        "- Split policy: scenario families are kept entirely within one split.",
        "- Scope limitation: synthetic-only results do not establish real-world detection performance.",
        "",
    ]
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Generate, validate, and save the reproducible synthetic dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="deterministic generation seed (default: 42)")
    parser.add_argument("--examples-per-scenario", type=int, default=75, help="rows per family (default: 75; 16 families produce 1,200 rows)")
    args = parser.parse_args()
    rows = generate_dataset(seed=args.seed, examples_per_scenario=args.examples_per_scenario)
    if len(rows) < 1200:
        raise SystemExit(f"Generated {len(rows)} rows; at least 1,200 are required. Increase --examples-per-scenario.")
    summary = validate_dataset(rows)
    write_csv(rows)
    write_data_card(summary, args.seed, args.examples_per_scenario)
    write_summary(summary, args.seed, args.examples_per_scenario)
    print(f"Synthetic dataset saved to: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Data card saved to: {DATA_CARD_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Summary saved to: {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    for name, value in summary.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
