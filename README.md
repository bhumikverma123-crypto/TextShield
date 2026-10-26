# TextShield

TextShield is an educational local demonstration of an explainable NLP-based social engineering message risk analyzer. Development uses **synthetic fictional messages only**; dataset results must not be interpreted as evidence of real-world scam-detection performance.

## Current status

**Through Phase 10:** the Streamlit UI and console fallback use the saved tuned model and show model output separately from rule-based warning signals. The test suite, fresh virtual-environment install, and final report have been completed. The project uses synthetic examples only; reported metrics apply only to the generated scenario-held-out data.

## Architecture

```text
Synthetic generator -> processed CSV -> preprocessing / EDA
                                      -> baseline training -> baseline evaluation
                                      -> grouped tuning -> saved tuned pipeline
User message -> local normalization -> model prediction + independent rules -> explanation
```

The model and rules run locally. The analyzer does not fetch or open message URLs. Training, evaluation, and tuning are separate Python commands and do not depend on Streamlit. Python sources are in `src/`, tests in `tests/`, model artifacts in `models/`, and experiment outputs in `reports/`.

## Phase status

Phases 0–10 are complete: planning, UI prototype, synthetic generation, preprocessing/EDA, baseline training, evaluation, tuning, explanations, UI/model integration, testing/reproducibility, and final report/handoff. See [the final report](reports/FINAL_PROJECT_REPORT.md) and [progress log](reports/PROGRESS.md) for implementation, results, and checks. The report's student-name placeholder must be replaced before submission.

## Windows setup

Open PowerShell in the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use the project environment executable for commands; the system Python may be a different interpreter. Select `.venv\Scripts\python.exe` in Spyder. If PowerShell activation is desired, run `.\.venv\Scripts\Activate.ps1`. If script execution is restricted, use the explicit executable commands above; changing the machine execution policy is not required.

## Run the analyzer

Streamlit is launched from a terminal (not by pressing Spyder's Run button on `app.py`):

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The console fallback uses the same saved model and explanation module, and can be run from PowerShell or Spyder's Run button:

```powershell
.\.venv\Scripts\python.exe console_demo.py
```

## Fictional demo messages

These examples are fabricated for classroom demonstrations. They do not represent real people or services:

- Likely ordinary: `The fictional reading group meets tomorrow in the west hall.`
- Potentially concerning: `Fictional Northstar support: send your one-time verification code to confirm your account.`
- Reserved demonstration link: `The reading list is here: https://books.example.invalid/list`

Model scores are uncalibrated and can be wrong. Do not enter real credentials or sensitive messages.

## Spyder interpreter and working directory

1. Configure Spyder to use the project's `.venv` interpreter: `PROJECT_ROOT\.venv\Scripts\python.exe` (replace `PROJECT_ROOT` with the actual project folder path). Restart the IPython console if Spyder asks.
2. Set Spyder's working directory to the project root, the folder containing `app.py` and `PROJECT_SPECIFICATION.md`.
3. Run `console_demo.py` with Spyder's Run button. Launch the Streamlit UI from PowerShell. From Spyder's IPython console, the equivalent is `!python -m streamlit run app.py` after setting the working directory and selecting the project interpreter.

## Generate synthetic data

From the project root, generate the fixed default dataset and update the data card and summary:

```powershell
.\.venv\Scripts\python.exe -m src.generate_synthetic_data
```

The generator defaults to seed 42 and 75 examples per each of 16 scenario families (1,200 rows). Both values can be configured:

```powershell
.\.venv\Scripts\python.exe -m src.generate_synthetic_data --seed 42 --examples-per-scenario 75
```

The output is `data/processed/synthetic_messages.csv`. Every message is fictional; any included links use reserved `.invalid` domains. See [the data card](data/data_card.md) and [dataset summary](reports/dataset_summary.md).

## Preprocessing and exploratory analysis

Run EDA from the project root to normalize whitespace for analysis, validate the dataset schema, and regenerate the findings and plots:

```powershell
.\.venv\Scripts\python.exe -m src.preprocess
```

See [reports/eda_findings.md](reports/eda_findings.md) and `reports/figures/`. `build_text_pipeline()` in `src/preprocess.py` combines whitespace normalization, TF-IDF, and a classifier; fit the full pipeline on training data only.

## Evaluate the baseline

Evaluate the saved baseline on the preassigned test split and regenerate JSON metrics, predictions, error examples, and plots:

```powershell
.\.venv\Scripts\python.exe -m src.evaluate
```

See [reports/evaluation_report.md](reports/evaluation_report.md), [reports/evaluation_metrics.json](reports/evaluation_metrics.json), and `reports/figures/`. The reported results describe the synthetic scenario-held-out test set only; they do not establish real-world performance.

## Explainability module

`src/explain.py` exposes `analyze_rules()` and `explain_with_model()`. The model score is shown on a 0–100 scale as an uncalibrated score. Risk bands use documented thresholds: Low below 35, Medium from 35 to below 70, and High from 70 upward. Rule indicators are displayed separately and do not change the model score or band. A link or urgency phrase alone does not raise the rule-only review status. URL strings are inspected locally and never opened.

Run the focused explanation checks with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_explanations.py
```

## Analyzer behavior and limitations

- The UI includes fictional sample selection, SMS/email/other message types, analyze and clear controls, model output, separately listed rule signals, and privacy/safety guidance.
- The predicted class and estimated score come from the saved tuned TF-IDF + Logistic Regression pipeline. The 0–100 score is uncalibrated; Low/Medium/High bands are display thresholds, not validated security levels. Rule indicators do not change the score or band.
- Rule patterns are readable illustrative signals. Neither model output nor rules provide a definitive security assessment.
- The prototype does not fetch links, call external APIs, track users, or transmit messages to a project server. Avoid entering sensitive content.
- TextShield does not replace security software or professional security advice.
- Synthetic examples are for safe classroom development only. Metrics from them cannot establish real-world performance.

## Train the baseline

Train and save the fitted pipeline (uses only `split=train` rows):

```powershell
.\.venv\Scripts\python.exe -m src.train
```

The model and metadata are saved under `models/`; the configuration and training-only fitting details are in `reports/baseline_training.md`. This command does not compute held-out performance metrics.

## Fine-tune and compare

Run scenario-grouped cross-validation on training rows, select settings using validation macro F1, refit the selected pipeline on train + validation rows, then evaluate the saved final model on the test split:

```powershell
.\.venv\Scripts\python.exe -m src.tune
```

The test split is used only after model selection. Results and trade-offs are in [reports/tuning_report.md](reports/tuning_report.md); the selected artifact and parameters are in `models/`. Perfect synthetic test scores do not demonstrate real-world performance.

## Tests and later-phase commands

Run the complete automated test suite from the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

See [PROJECT_SPECIFICATION.md](PROJECT_SPECIFICATION.md) for the phase plan and [reports/PROGRESS.md](reports/PROGRESS.md) for actual work and check results.

## Troubleshooting

- **`No module named streamlit` or another dependency:** confirm Spyder and the terminal use `PROJECT_ROOT\.venv\Scripts\python.exe`, then run `\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
- **Streamlit does not start from Spyder's Run button:** launch it from PowerShell with `\.venv\Scripts\python.exe -m streamlit run app.py`. From Spyder's IPython console use `!python -m streamlit run app.py` after selecting the project interpreter and project-root working directory.
- **The model artifact is missing:** from the project root run `\.venv\Scripts\python.exe -m src.tune` to regenerate the saved tuned pipeline. Tuning also writes evaluation artifacts and can take longer than a simple demo run.
- **The analyzer rejects a long or blank message:** enter non-empty text no longer than 20,000 characters, or analyze a shorter excerpt.
- **A saved model cannot be loaded after changing library versions:** recreate it in the selected environment with `\.venv\Scripts\python.exe -m src.tune`; do not use an untrusted joblib artifact.
