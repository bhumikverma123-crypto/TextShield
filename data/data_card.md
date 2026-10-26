# TextShield Synthetic Data Card

## Status and provenance

This dataset is **fully synthetic** and was generated locally by `src/generate_synthetic_data.py`. It contains no downloaded corpus, scraped messages, real personal information, or functioning non-reserved links. All organizations, names, references, and scenarios are fictional. URL examples use reserved `.invalid` hostnames and are never opened or fetched by the generator or analyzer.

## Generation method

- **Generator:** `src/generate_synthetic_data.py`
- **Default random seed:** `42`
- **Examples per scenario family:** `75`
- **Scenario families:** 16 predefined fictional template families
- **Template variation:** Each row selects a scenario-specific template and varies fictional names, organizations, dates, times, places, references, safe placeholder links, and message types with Python's seeded `random.Random`.
- **Splitting:** Families are assigned wholly to train, validation, or test before examples are generated, preventing a single scenario family's templates from crossing splits.
- **Output:** `data/processed/synthetic_messages.csv`

## Dataset summary

- **Total rows:** 1200
- **Labels:** `{'likely_legitimate': 600, 'suspicious': 600}`
- **Message types:** `{'email': 419, 'other': 415, 'sms': 366}`
- **Splits:** `{'train': 600, 'test': 300, 'validation': 300}`
- **Exact duplicate normalized texts:** 0
- **Cross-split near-duplicate pairs at normalized SequenceMatcher ratio ≥ 0.92:** 0

## Fields

`message_id` (synthetic unique identifier), `message_text`, `label` (`suspicious` or `likely_legitimate`), `message_type` (`sms`, `email`, or `other`), `scenario` (template family), and `split` (`train`, `validation`, or `test`).

## Intended use

Safe classroom demonstration, development, and reproducible software exercises only. This dataset is not suitable for operational filtering, security decisions, or estimating performance on real-world messages.

## Limitations

The examples are authored from a limited set of templates and fictional tokens. Their language, class balance, and scenario splits do not represent real message populations. Models may learn artifacts of these templates, and even clean held-out synthetic metrics would not establish real-world effectiveness. The rule-based labels are assigned by scenario, not by human review of real cases.

## Reproducibility and validation

Re-run from the project root with `python -m src.generate_synthetic_data --seed 42 --examples-per-scenario 75` (use `.venv\Scripts\python.exe` on Windows). The generator validates schema, non-empty text, unique IDs, scenario/label/split consistency, message types, exact duplicates, and cross-split near duplicates.
