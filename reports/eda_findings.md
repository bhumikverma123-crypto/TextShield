# TextShield Exploratory Data Analysis

Generated from the fixed **synthetic-only** CSV at `data/processed/synthetic_messages.csv` using `python -m src.preprocess`. Counts and summaries below are computed from the file; they are not model performance metrics.

## Dataset profile

- Rows: **1200**
- Class counts: `{'likely_legitimate': 600, 'suspicious': 600}`
- Message types: `{'email': 419, 'other': 415, 'sms': 366}`
- Split counts: `{'test': 300, 'train': 600, 'validation': 300}`
- Scenario families per split: `{'test': 4, 'train': 8, 'validation': 4}`
- Duplicate normalized messages: **0**
- Messages containing a URL by class: `{'likely_legitimate': 229, 'suspicious': 150}`

## Message length

Word-count statistics:
- **likely_legitimate:** mean 22.02 words, median 22.0, range 15–30
- **suspicious:** mean 22.98 words, median 23.0, range 17–29

## Common terms

Top terms by raw frequency after scikit-learn's English stop-word list: `review` (503), `desk` (430), `note` (427), `update` (388), `example` (379), `https` (379), `invalid` (379), `convenient` (341), `hello` (335), `thanks` (308).

These counts describe the hand-authored synthetic templates; frequent terms may reflect template construction and should not be interpreted as real-world indicators.

## Figures

- `reports/figures/class_distribution.png`
- `reports/figures/message_length_by_class.png`
- `reports/figures/message_type_by_class.png`
- `reports/figures/common_terms.png`

## Preprocessing decision

The reusable preprocessing step only trims and collapses whitespace. It preserves punctuation and URL text before TF-IDF vectorization. `build_text_pipeline()` places normalization, `TfidfVectorizer`, and the estimator in one scikit-learn `Pipeline`; fit the complete pipeline on training data only so the vocabulary is not learned from validation or test data.

## Limitations

All observations come from synthetic templates. EDA does not show that these patterns distinguish real malicious and legitimate messages. No model has been trained or evaluated in this phase.
