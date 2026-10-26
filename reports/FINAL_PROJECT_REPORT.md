# TextShield: Explainable NLP-Based Social Engineering Message Risk Analyzer

## 1. Title page

- **Project purpose:** Local educational demonstration of NLP-based message risk analysis using synthetic messages only.
- **Report date:** 2026-10-25
- **Author:** [Student name pending]
- **Project status:** Implementation and handoff documentation complete through Phase 10. Replace the author placeholder before submission.

## 2. Abstract

TextShield is a local classroom project that analyzes fictional social-engineering-style messages using a TF-IDF and Logistic Regression classifier together with separate, readable warning-signal rules. The project includes a deterministic 1,200-row synthetic dataset, scenario-family splits, exploratory analysis, a baseline model, grouped cross-validation and validation-based model selection, held-out synthetic evaluation, a Streamlit interface, a console fallback, and automated checks. The selected classifier retained the baseline hyperparameters because the candidate did not improve validation macro F1; the final pipeline was then refit on the train and validation rows. It scored 1.0000 accuracy on the 300-row synthetic test split. That result reflects a small authored synthetic benchmark, not real-world detection ability. The project does not open message URLs or call a remote analyzer.

## 3. Problem statement and motivation

Social-engineering messages can pressure people to reveal credentials, verify accounts, send payments, or claim rewards. A classroom tool can demonstrate how text features and explicit indicators might support a cautious review. TextShield provides that demonstration on fictional examples so students can inspect a complete data-to-inference workflow without collecting real messages or personal information.

## 4. Objectives

- Generate a reproducible, balanced, synthetic message dataset with scenario-family-separated train, validation, and test splits.
- Demonstrate conservative text preprocessing, TF-IDF features, Logistic Regression training, and held-out evaluation.
- Present readable potential warning signals independently from model predictions and scores.
- Provide a Windows- and Spyder-friendly local interface, with a terminal-launched Streamlit UI and console fallback.
- Document privacy boundaries, limitations, actual checks, and measured synthetic-only results.

## 5. Scope, ethics, and privacy

All examples are generated locally from fictional names, organizations, and scenarios. Any demonstration links use reserved `.invalid` domains such as `example.invalid`; they are analyzed as strings only and are never opened. No external datasets, message scraping, paid APIs, or real user messages are part of the project. The local analyzer does not make network requests to classify a message. Users are instructed not to enter credentials or sensitive content.

The model and rules can produce incorrect results and are not a security product or professional advice. The risk score is not calibrated. An apparently low-risk output does not establish that a message is safe.

## 6. Related concepts and background

TextShield illustrates a standard supervised text-classification workflow. TF-IDF represents a document using term frequency weighted by inverse document frequency; Logistic Regression learns a linear decision function from those features. A scikit-learn pipeline keeps normalization, vectorization, and classification together so the vectorizer is fitted with the training data. The project also demonstrates held-out evaluation and group-aware splitting: examples from one authored scenario family stay within a single split to reduce direct template leakage.

The explanation component combines two distinct kinds of output. The classifier supplies a predicted class and an uncalibrated score; transparent text rules identify possible indicators such as credential requests, payment requests, urgency, and unusual URL structure. Rule indicators are not added to the score. These concepts are demonstrated on authored examples and are not validated as a real-world threat model.

## 7. System architecture and data flow

```text
Seeded synthetic generator
  -> schema / duplicate / scenario-split validation
  -> CSV dataset and data card
  -> preprocessing and exploratory analysis
  -> baseline train / held-out evaluation
  -> grouped cross-validation and validation-based selection
  -> final tuned pipeline artifact

Message entered locally
  -> Unicode normalization, NUL removal, and input-length validation
  -> TF-IDF + Logistic Regression prediction
  -> independent rule analysis (URLs are never fetched)
  -> separate model, rule, and safety-guidance display
```

The Streamlit app and console fallback load `models/textshield_tuned.joblib` and call the explanation module in `src/explain.py`. Training, tuning, evaluation, and EDA are separate commands and can run without Streamlit. Files are organized under `src/`, `data/`, `models/`, `reports/`, and `tests/`.

## 8. Tools and technologies

- Python 3.14.3 was used for the Phase 9 clean-environment verification; the project code targets Python 3.10 or newer.
- Observed project-environment versions: pandas 3.1.0, NumPy 2.5.4, scikit-learn 1.9.1, Matplotlib 3.11.2, joblib 1.6.0, Streamlit 1.66.1, and pytest 9.1.1.
- `requirements.txt` declares compatible version ranges rather than an exact transitive lockfile.
- Windows PowerShell was used for setup and commands. Core Python modules are intended to run with the project interpreter from the project root; Spyder can run the console and module scripts after selecting that interpreter and setting the working directory. Streamlit is launched from a terminal.

## 9. Synthetic dataset design and data card summary

The fixed dataset contains 1,200 synthetic messages generated with seed 42: 600 `suspicious` and 600 `likely_legitimate`. Sixteen fictional scenario families contribute 75 rows each. Families are assigned wholly to one split: 8 families / 600 messages for training, 4 / 300 for validation, and 4 / 300 for testing. Message types are 419 email, 415 other, and 366 SMS. The required columns are `message_id`, `message_text`, `label`, `message_type`, `scenario`, and `split`.

The generator checks labels, schema, message types, scenario assignments, exact duplicates, and cross-split near-duplicate text. The recorded run found zero exact normalized duplicates and zero cross-split near-duplicate pairs at the configured similarity threshold. Links appear in both classes, and harmless urgency examples are included. All domain examples use reserved `.invalid` hosts. Provenance, generation method, intended use, and limitations are described in [the data card](../data/data_card.md); dataset counts are summarized in [the dataset summary](dataset_summary.md).

## 10. Preprocessing and exploratory analysis

Preprocessing trims leading/trailing whitespace and collapses repeated whitespace while preserving punctuation and URL strings. `build_text_pipeline()` combines this normalization, a TF-IDF vectorizer, and a classifier; the entire pipeline is fitted on training data for the baseline workflow.

EDA found 600 examples per class and the message-type counts described above. URL-containing messages numbered 150 in the suspicious class and 229 in the likely-legitimate class. Mean lengths were 22.98 words for suspicious messages and 22.02 for likely-legitimate messages. Common terms included `review`, `desk`, `note`, `update`, and `example`/`https`/`invalid`. These observations reflect the authored templates. Findings and four plots (class counts, length, message type, and common terms) are in `eda_findings.md` and `figures/`.

## 11. Baseline model and rationale

The baseline is a scikit-learn pipeline containing whitespace normalization, word-level TF-IDF with unigrams and bigrams, and Logistic Regression using the `liblinear` solver. It uses seed 42 and `max_iter=1000`. It was fitted on the 600 training rows only (300 per class), with zero held-out rows used for fitting. The pipeline and metadata are saved as `models/textshield_baseline.joblib` and `models/textshield_baseline_metadata.json`; the training record is `baseline_training.md`.

This compact pipeline was selected as a reproducible, interpretable classroom baseline with a modest dependency footprint. A separate fictional smoke inference verified artifact loading during training; that smoke prediction is not a performance result.

## 12. Evaluation methodology and actual results

The baseline artifact was evaluated on the preassigned 300-row test split only. Test rows belong to four scenario families that do not overlap train or validation families. No fitting or model selection was performed on this test split. `suspicious` is the positive class for ROC-AUC and average precision.

| Baseline test metric | Result |
|---|---:|
| Accuracy | 0.9000 |
| Suspicious precision / recall / F1 | 0.8409 / 0.9867 / 0.9080 |
| Likely-legitimate precision / recall / F1 | 0.9839 / 0.8133 / 0.8905 |
| Macro F1 | 0.8992 |
| ROC-AUC | 0.9876 |
| Average precision | 0.9886 |

The confusion matrix in `[likely_legitimate, suspicious]` order is `[[122, 28], [2, 148]]`. The baseline made 28 false-positive predictions (22 `routine_service_notice`, 6 `community_event`) and 2 false negatives (both `unexpected_refund`). Metrics, predictions, error examples, and plots are saved in `evaluation_metrics.json`, `evaluation_predictions.csv`, `error_analysis.csv`, `evaluation_report.md`, and `figures/`.

## 13. Fine-tuning method and actual before/after results

The search evaluated 24 TF-IDF and Logistic Regression parameter combinations using four-fold `StratifiedGroupKFold` on the 600 training rows, with scenario families kept intact. Macro F1 was the selection metric. The highest grouped-CV macro F1 was 0.7285. On the validation split, the best candidate and baseline both scored 0.6700 accuracy and 0.6602 macro F1, so the baseline configuration was retained: unigrams and bigrams, `min_df=1`, `C=1.0`, and no class weighting.

After selection, the final pipeline was refit on the 900 train-plus-validation examples. The test split was evaluated only after this model-selection and fitting sequence. The results on the same 300-row held-out synthetic test split were:

| Test metric | Phase 4 baseline (600 fit rows) | Selected final model (900 fit rows) |
|---|---:|---:|
| Accuracy | 0.9000 | 1.0000 |
| Suspicious precision / recall / F1 | 0.8409 / 0.9867 / 0.9080 | 1.0000 / 1.0000 / 1.0000 |
| Macro F1 | 0.8992 | 1.0000 |
| ROC-AUC | 0.9876 | 1.0000 |
| Average precision | 0.9886 | 1.0000 |

The selected hyperparameters did not change. The final fit used 300 additional validation examples, so the score difference cannot be attributed to hyperparameter tuning. Perfect test scores on this small synthetic benchmark do not establish real-world performance. Search records and detailed comparisons are in `tuning_cv_results.csv`, `tuning_report.md`, and `tuned_evaluation_metrics.json`.

## 14. Explainability and risk scoring

`src/explain.py` reports potential indicators for urgency, links, credential/account verification requests, payment requests, prize/reward claims, support impersonation combined with sensitive requests, and unusual URL structure. A link alone and harmless urgency are context-only. URL parsing is string-only; no URL is resolved or fetched. Reserved `.example.invalid` links are neutral.

The model score is the suspicious-class `predict_proba` value scaled to 0–100 and is explicitly **not calibrated**. Display bands are Low `<35`, Medium `35–<70`, and High `>=70`. Rule evidence is presented independently and does not change the model score or band. Six focused explanation tests and integration examples were run; details are in `reports/PROGRESS.md`.

## 15. UI features and screenshots

The Streamlit app provides fictional sample selection, message-type context, a message field, analyze and clear controls, model class/score/band, separate rule indicators, safety guidance, and a privacy explanation. It normalizes Unicode, removes NUL characters, rejects empty content, and enforces a 20,000-character limit. If the tuned model file is missing, the app shows the project command needed to create it. The console fallback uses the same model and explanation logic.

**Screenshots were not captured.** To capture one if required by the course, start the app from PowerShell in the project root with `\.venv\Scripts\python.exe -m streamlit run app.py`, select a fictional sample or paste one of the fictional demo messages from `README.md`, click **Analyze message**, and capture the browser window with the model and rule sections visible. Do not use a real message or include personal information in a screenshot.

## 16. Testing and reproducibility

Phase 9 added tests for synthetic generation/schema, preprocessing, saved model inference, explanations, and the UI smoke flow. The complete suite passed with **16 tests** in the project environment and **16 tests** in a fresh Python 3.14.3 virtual environment. In the fresh environment, `pip install -r requirements.txt` completed and `pip check` reported no broken requirements. The temporary verification environment was removed. Streamlit AppTest emitted its standard bare-mode `ScriptRunContext` warning but reported no app exceptions. A local Streamlit server returned HTTP 200 during Phase 1.

For a fresh Windows setup, open PowerShell in this project root and run:

```powershell
python -m venv .venv
\.venv\Scripts\python.exe -m pip install -r requirements.txt
\.venv\Scripts\python.exe -m pytest -q
```

For Spyder, select `PROJECT_ROOT\.venv\Scripts\python.exe` as the interpreter, restart the IPython console if prompted, and set the working directory to the project root (the folder containing `app.py`). Run `console_demo.py` or a core module with Spyder's Run button. Launch Streamlit from a terminal, or from the Spyder IPython console use `!python -m streamlit run app.py`. The generator and training/evaluation/tuning commands are documented in `README.md`. The fixed seed and saved artifacts support reproducibility; because requirements are version ranges rather than a lockfile, regenerate model artifacts if library versions change.

## 17. Limitations and threats to validity

- Every message and scenario is synthetic and authored from a limited set of templates. Vocabulary, labels, and results may reflect template design more than general language patterns.
- Scenario-family separation reduces direct template overlap but does not provide independent evidence from real message sources. There are only four test scenario families and 300 test rows.
- The final test score of 1.0000 is unusually strong but must be interpreted within the small, authored synthetic benchmark. It cannot be used as a claim about real-world detection accuracy.
- The final model was trained on more rows than the baseline; unchanged hyperparameters mean the difference is not an isolated tuning effect.
- Logistic Regression scores are not calibrated probabilities. Threshold bands are demonstration labels, not validated risk thresholds.
- Rules are simple patterns and can miss paraphrases, multilingual messages, context, or new attack strategies; they can also surface harmless wording.
- No adversarial, demographic, language-coverage, or real-world prospective evaluation was performed. No human-security outcome was measured.
- The tool should not be used as the sole basis for deciding whether a message is safe or malicious.

## 18. Future improvements

Potential future work, requiring suitable authorization and ethical review, includes broader synthetic scenario coverage, more robust linguistic tests, calibration analysis on an appropriate dataset, accessibility improvements, and a documented model-version/environment lock. Any real-world dataset work would require separate consent, privacy safeguards, and governance; it is outside this project's synthetic-only scope.

## 19. Conclusion

TextShield completes an end-to-end educational workflow from reproducible synthetic data generation through modeling, evaluation, explanations, and a local interface. It demonstrates how model predictions and simple rules can be presented separately with explicit caveats. The implemented checks and test results are recorded in the progress log. Its measured performance applies only to the authored synthetic benchmark, and its score and rules are not a dependable security verdict.

## 20. References

No external papers, web pages, or library-documentation pages were consulted as project sources for this implementation. No external references are claimed here. The software dependencies and their observed versions are listed in Section 7 and `requirements.txt`.

## Appendix A. Handoff checklist

- [x] Project specification, README, data card, progress log, and final report are present.
- [x] Synthetic-only data generation and reserved fake domains are documented.
- [x] Baseline and final model artifacts, evaluation outputs, and figures are retained in the project.
- [x] Automated suite passed in the project and a fresh environment (16 tests in each).
- [x] Windows PowerShell and Spyder setup/run instructions are included.
- [ ] Replace `[Student name pending]` on the title page before submission.
- [ ] Capture a UI screenshot only if the course submission requires one; none is currently included.
