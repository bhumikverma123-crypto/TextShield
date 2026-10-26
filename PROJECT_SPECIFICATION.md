# TextShield --- Master Project Specification and Execution Prompt

## 1. Role and Mission

You are Codex acting as a senior Python/NLP engineer, UI developer, ML
practitioner, QA engineer, and technical writer.

Build a complete, beginner-friendly college project named **TextShield:
Explainable NLP-Based Social Engineering Message Risk Analyzer**.

The application accepts a text message (SMS or email snippet), analyzes
it for potentially suspicious patterns, estimates a risk level, explains
the warning signs, and gives practical safety guidance. The project must
be built in explicit phases, with validation and documentation at every
phase.

**Use synthetic data only. Do not download or use external datasets.**
Clearly label the dataset as synthetic in the UI, reports, code
comments, and README. Do not imply that results trained on synthetic
data prove real-world scam-detection performance.

## 2. Core Goals

1.  Build a polished, responsive Streamlit UI.
2.  Generate a reproducible synthetic dataset for safe development and
    demonstration.
3.  Implement text preprocessing and exploratory data analysis.
4.  Train a baseline text-classification model.
5.  Evaluate the model with appropriate metrics and visualizations.
6.  Fine-tune model parameters using a reproducible validation strategy.
7.  Explain predictions using interpretable signals and transparent
    rule-based indicators.
8.  Keep a continuously updated progress file and a final project
    report.
9.  Include tests, setup instructions, limitations, and reproducibility
    information.
10. Make the project easy for a beginner to run locally on Windows,
    including from a PowerShell terminal. The user may use Spyder for
    editing Python files.

## 3. Product Scope

### User workflow

1.  User opens the TextShield app.
2.  User pastes a message into a text area.
3.  User chooses the message type: SMS, email, or other text.
4.  User clicks **Analyze message**.
5.  The app displays:
    -   predicted class (for example, `suspicious` or
        `likely_legitimate`);
    -   risk band (`Low`, `Medium`, or `High`);
    -   a numeric score from 0--100, clearly described as a
        heuristic/demo score rather than a calibrated probability unless
        calibration is actually implemented and validated;
    -   warning signals found in the message;
    -   brief, actionable safety advice;
    -   a disclaimer that the output is an estimate and may be wrong.
6.  User can clear the message and analyze another one.
7.  The app includes a sample-message selector for quick demonstrations.
8.  Optional: include a small evaluation/dashboard page showing
    synthetic dataset summaries and model evaluation results.

### Explainability requirements

Use a combination of: - transparent rule-based indicators (for example,
urgency language, prize/lottery claims, credential requests, requests
for payment, shortened or suspicious-looking URLs, impersonation
wording); - model-based prediction from the trained classifier.

Do not claim that a message is dangerous solely because it contains a
URL, money amount, or urgent phrase. Use cautious wording such as
"potential warning sign." Explain which indicators were triggered and
avoid claiming that the model's features are a complete or definitive
explanation of its prediction.

### Safety and privacy

-   No real message is transmitted to a server controlled by the
    project.
-   Do not add tracking, telemetry, external API calls, or automatic
    link opening.
-   Never open or fetch URLs contained in submitted messages.
-   Do not ask users to paste passwords, OTPs, account numbers, or other
    sensitive information.
-   Include a short privacy notice explaining that the demo processes
    text locally in the running app and that users should avoid entering
    sensitive content.
-   Do not claim that TextShield replaces security software or
    professional security advice.

## 4. Synthetic Data Requirements

Generate all training, validation, and test examples locally with a
deterministic seed. Do not copy real messages from the internet or use
downloaded corpora.

Create a balanced or near-balanced dataset with at least **1,200
synthetic examples** and these fields: - `message_id`: synthetic unique
ID; - `message_text`: generated message text; - `label`: `suspicious` or
`likely_legitimate`; - `message_type`: `sms`, `email`, or `other`; -
`scenario`: template/category used to generate the example; - `split`:
assigned train/validation/test split, or store splits separately without
leakage.

Include a broad variety of synthetic patterns, including: - urgency and
time pressure; - prize, reward, or lottery claims; - credential or
account-verification requests; - payment or fee requests; - delivery and
parcel notices; - impersonation claims; - benign everyday
coordination; - legitimate appointment or delivery updates; - ordinary
work or school messages; - legitimate promotional messages; - messages
containing links in both classes, so a URL alone does not determine the
label; - negative examples with words such as "urgent" used in harmless
contexts where practical.

Avoid generating a dataset where labels can be predicted by a single
obvious keyword or where suspicious examples all have the same wording.
Vary templates, names, punctuation, capitalization, message length, and
combinations of signals. Keep the examples fictional and do not use real
personal data, real phone numbers, real domains, or functioning
malicious links. Use reserved example domains such as `example.com` and
clearly fake placeholders.

### Dataset quality checks

-   Check duplicate and near-duplicate template outputs.
-   Check class counts and message-type distribution.
-   Check missing/empty messages.
-   Check label consistency with scenario rules.
-   Avoid leakage: split by template family/scenario where feasible so
    nearly identical template variants do not appear across train and
    test.
-   Save a fixed synthetic dataset CSV under `data/processed/` or a
    clearly documented data directory.
-   Include a data card documenting generation method, seed, class
    balance, limitations, and synthetic-only status.
-   Never report the dataset as real-world data.

## 5. Required Technology

Prefer a small, maintainable stack: - Python 3.10+ (choose and document
a version that works with the dependencies); - Streamlit for UI; -
pandas and NumPy for data handling; - scikit-learn for vectorization,
modeling, splitting, metrics, and tuning; - matplotlib and/or seaborn
for evaluation charts; - joblib for saving the trained pipeline; -
pytest for tests.

Use NLTK only if it provides a clear benefit. Do not introduce heavy
dependencies or deep-learning frameworks without a compelling reason.
The baseline should use `TfidfVectorizer` and a scikit-learn classifier
such as Logistic Regression. Compare at least one sensible alternative
(such as Multinomial Naive Bayes or Linear SVM) if time and
synthetic-data quality allow.

## 6. Project Structure

Create a clean structure similar to this, adapting only when there is a
clear reason:

``` text
textshield/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── config.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── DATA_CARD.md
├── models/
│   └── .gitkeep
├── reports/
│   ├── FINAL_PROJECT_REPORT.md
│   ├── PROGRESS.md
│   └── figures/
├── src/
│   ├── __init__.py
│   ├── generate_synthetic_data.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── tune.py
│   ├── explain.py
│   └── utils.py
├── tests/
│   ├── test_data.py
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_explanations.py
└── notebooks/
    └── (optional exploration notebook)
```

Do not create empty or decorative files that serve no purpose. Keep code
modular and explainable.

## 7. Mandatory Phases and Acceptance Criteria

Work in order. Do not silently skip phases. After each phase, run
relevant checks, update `reports/PROGRESS.md`, and report what was
completed, what was tested, any results, and what remains. If a phase
fails, fix it before moving forward.

### Phase 0 --- Inspect and Plan

-   Inspect the existing workspace before changing files.
-   If files already exist, preserve useful work and avoid destructive
    overwrites.
-   Create a brief implementation plan and initial
    `reports/PROGRESS.md`.
-   Record assumptions, environment, planned commands, and phase
    checklist.
-   Confirm the project is scoped to synthetic data only.

**Acceptance:** plan and progress file exist before major
implementation.

### Phase 1 --- UI Prototype

-   Build the Streamlit interface with sample inputs, message-type
    selector, analyze/clear controls, explanatory sections, and
    privacy/safety notice.
-   During this phase, use clearly marked placeholder results or a
    simple mock analysis only; do not misrepresent mock output as
    trained model output.
-   Use accessible contrast, clear hierarchy, helpful empty states, and
    readable layouts.
-   Avoid excessive animations or visual clutter.

**Acceptance:** app starts locally; empty input is handled gracefully;
sample selection works; UI clearly marks results as prototype until the
model phase is integrated.

### Phase 2 --- Synthetic Data Generation

-   Implement deterministic data generation with a configurable random
    seed.
-   Generate at least 1,200 examples with the required schema and
    diverse categories.
-   Add validation checks and save the dataset plus `data/DATA_CARD.md`.
-   Produce a short dataset summary.

**Acceptance:** running the generator twice with the same seed produces
the same dataset; schema and class checks pass; dataset is synthetic and
documented.

### Phase 3 --- Preprocessing and EDA

-   Implement reusable preprocessing functions.
-   Use a scikit-learn `Pipeline` where possible to prevent data
    leakage.
-   Inspect class distribution, message lengths, common terms,
    duplicates, and message types.
-   Generate clear plots under `reports/figures/`.
-   Keep preprocessing appropriate for TF-IDF; do not over-clean
    punctuation or URLs if those features may be useful.

**Acceptance:** preprocessing is tested; plots are saved; EDA findings
are recorded in progress notes.

### Phase 4 --- Baseline ML Training

-   Create a reproducible split, ideally split by scenario/template
    family to reduce leakage.
-   Fit a TF-IDF + Logistic Regression baseline using training data
    only.
-   Save the complete fitted pipeline with joblib under `models/`.
-   Record seed, features, parameters, and split strategy.
-   Never fit vectorizers on the full dataset before splitting.

**Acceptance:** training script runs end-to-end; saved pipeline can be
loaded and used for inference; no test-set data is used during fitting.

### Phase 5 --- ML Evaluation

Evaluate on held-out data and report: - accuracy; - precision, recall,
and F1 for each class; - macro F1; - confusion matrix; - classification
report; - ROC-AUC only if the model and test labels support it; - PR-AUC
or average precision if appropriate; - inference examples and error
analysis.

Save evaluation metrics to a machine-readable JSON or CSV and plots
under `reports/figures/`. Interpret results cautiously because the data
is synthetic. Do not invent scores or manually edit metrics to look
better.

**Acceptance:** evaluation is reproducible and uses held-out test data;
plots and metrics are saved; results and limitations are recorded.

### Phase 6 --- Fine-Tuning and Parameter Adjustment

-   Use training/validation data only for model selection and tuning.
-   Use a suitable method such as `GridSearchCV` or
    `RandomizedSearchCV`, with a documented scoring metric such as macro
    F1.
-   Tune reasonable parameters for TF-IDF and Logistic Regression, for
    example `ngram_range`, `min_df`, `max_df`, `C`, and `class_weight`,
    adjusting the search space to the dataset size.
-   Optionally compare against Multinomial Naive Bayes or Linear SVM.
-   Avoid overfitting to the synthetic templates.
-   Evaluate the selected final model on the held-out test set only
    after tuning is complete.
-   Record baseline-versus-tuned metrics, chosen parameters, and
    trade-offs.

**Acceptance:** tuning script runs reproducibly; selected parameters are
saved; comparison is documented; test data is not used to choose
hyperparameters.

### Phase 7 --- Explanation and Risk Scoring

-   Implement rule-based warning indicators and explanations in a
    separate module.
-   Keep rules readable, testable, and easy to adjust.
-   Implement risk bands with documented thresholds.
-   If converting model probabilities to a 0--100 score, label it as an
    estimated model score and do not call it a calibrated probability
    unless calibration has been performed and evaluated.
-   Combine model output and rule-based indicators transparently. Avoid
    unexplained arbitrary weighting; document any heuristic formula.
-   Include counterexamples and benign messages that contain isolated
    warning words.

**Acceptance:** explanations identify triggered signals; rules are
unit-tested; benign links and harmless urgency are not automatically
labeled as scams by rules alone.

### Phase 8 --- Integrate UI and Final Model

-   Replace mock output with the saved pipeline and explanation module.
-   Handle missing model files with a clear user-facing instruction.
-   Handle empty, very long, malformed, and unusual input safely.
-   Make clear which outputs come from the ML model and which come from
    rules.
-   Test representative benign and suspicious synthetic examples.

**Acceptance:** full user flow works from input to prediction and
explanations.

### Phase 9 --- Testing, Polish, and Reproducibility

-   Add and run pytest tests for data schema, deterministic generation,
    preprocessing, model loading/inference, and explanations.
-   Check clean setup from a fresh virtual environment.
-   Check `requirements.txt` includes only needed dependencies.
-   Add a comprehensive README with installation, commands,
    architecture, phases, demo examples, limitations, and
    troubleshooting.
-   Add `.gitignore` for virtual environments, caches, local secrets,
    and large/unnecessary artifacts. Do not add secrets.
-   If possible, add a simple automated smoke test for the app's main
    components.

**Acceptance:** all automated tests pass, or any failures are clearly
reported with reasons and fixes attempted; setup steps are verified.

### Phase 10 --- Final Report and Handoff

Create `reports/FINAL_PROJECT_REPORT.md` with: 1. Title page details
(project name, purpose, date, author placeholder); 2. Abstract; 3.
Problem statement and motivation; 4. Objectives; 5. Scope and
ethical/privacy considerations; 6. Related concepts/background; 7.
System architecture and data flow; 8. Tools and technologies; 9.
Synthetic dataset design and data card summary; 10. Preprocessing and
EDA; 11. Baseline model and rationale; 12. Evaluation methodology and
actual results; 13. Fine-tuning method and actual before/after results;
14. Explainability and risk scoring; 15. UI features and screenshots or
screenshot instructions; 16. Testing and reproducibility; 17.
Limitations and threats to validity; 18. Future improvements; 19.
Conclusion; 20. References to library documentation only where actually
used.

Do not fabricate experiments, screenshots, test outcomes, references, or
performance numbers. Use clearly labeled placeholders for anything that
could not be verified, and explain how to fill them in.

**Acceptance:** final report matches the implemented code and actual
measured outputs; progress log is complete.

## 8. The Two Required Reports

Maintain both files throughout development:

### `reports/PROGRESS.md`

-   Update after every phase, not just at the end.
-   Include timestamp/date, phase, files created/modified, commands run,
    checks/results, blockers, decisions, and next steps.
-   Mark phases as `Not started`, `In progress`, `Blocked`, or
    `Complete`.
-   Record actual outcomes, not intended outcomes.
-   Preserve earlier entries; append a dated log entry rather than
    replacing history.
-   Keep a current status table near the top.

### `reports/FINAL_PROJECT_REPORT.md`

-   Create a skeleton early, but fill it with verified findings as the
    project progresses.
-   Finalize only after implementation and evaluation.
-   Do not let the final report contradict the progress log or source
    code.
-   Clearly separate measured results from expected behavior and
    limitations.

## 9. Code Quality Standards

-   Use descriptive names, type hints where useful, and focused
    functions.
-   Add docstrings to public functions.
-   Keep UI code separate from data generation, training, evaluation,
    and explanations.
-   Use `pathlib` for paths and ensure scripts work from the project
    root.
-   Set random seeds and document them.
-   Use logging or informative console output instead of silent
    failures.
-   Handle errors with actionable messages.
-   Avoid hard-coded machine-specific absolute paths.
-   Never swallow exceptions without explanation.
-   Avoid unnecessary abstractions and overengineering.
-   Do not claim the app detects every scam or guarantees security.
-   Use the current installed Python environment's capabilities; do not
    assume GPU access.

## 10. Suggested Commands

Document and verify commands equivalent to:

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.generate_synthetic_data
python -m src.preprocess
python -m src.train
python -m src.evaluate
python -m src.tune
pytest -q
streamlit run app.py
```

If the project uses a different entry point, adjust the commands
consistently in the README and report. Explain the PowerShell
execution-policy issue only if it actually occurs; provide a safe
alternative such as invoking the environment's Python executable
directly.

## 11. Required Final Handoff

At completion, provide: - concise summary of the implementation; - exact
commands to install, generate data, train/tune, test, and run the app; -
project tree; - actual evaluation results; - location of
`reports/FINAL_PROJECT_REPORT.md` and `reports/PROGRESS.md`; - tests
passed and any known failures; - limitations and recommended next steps.

Do not stop after writing a plan. Implement the project in phases in the
workspace available to you. If the environment cannot execute a command,
state that clearly and still provide the code plus the exact command the
user should run locally. Never claim a command or test ran unless it
actually did.

## 12. Operating Instructions for Codex

Start by inspecting the repository and available runtime. Then: 1.
Explain the short plan. 2. Create/update the progress log and
final-report skeleton. 3. Execute Phase 1 and validate it. 4. Continue
one phase at a time through Phase 10. 5. After each phase, update
`reports/PROGRESS.md`. 6. Before making destructive changes, inspect the
affected files and preserve existing work. 7. Do not ask the user to
approve every routine step. Ask only when a genuinely blocking decision
is required. 8. At the end of each phase, provide a brief summary of
changes and tests, then proceed to the next phase unless blocked. 9.
Ensure the final report is based on real outputs from the completed
project.

**Primary rule:** a smaller working project with verified tests and
honest reporting is better than a larger project with unverified claims.
