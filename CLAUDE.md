# CLAUDE.md

Guidance for Claude Code or any coding assistant working on the O-Beast obesity-risk ML application.

## Project Summary

O-Beast is an educational FastAPI web app for estimating obesity-risk probability from lifestyle, body, and family-history inputs. It is built for student research and professor presentation, not medical diagnosis.

The app should stay understandable for high-school students: explain machine-learning algorithms as "guess, check, improve" workflows, while keeping the implementation scientifically careful.

## Important Safety Rule

Never describe the app as a diagnosis or medical authority. Use language such as:

- "educational probability estimate"
- "research prototype"
- "risk band for the current training data"
- "not a replacement for a doctor, nutrition professional, or clinical screening"

Do not claim universal model accuracy from the included sample dataset. The sample data is tiny and exists only to prove that the pipeline runs.

## Project Structure

```text
src/obesity_ml/app.py          FastAPI app, UI HTML/CSS, routes, result page, method visuals
src/obesity_ml/train.py        ML training pipeline, algorithms, SMOTENC, metrics, model selection
src/obesity_ml/predict.py      Loads trained artifact and predicts probability
src/obesity_ml/features.py     Input validation and engineered features
src/obesity_ml/risk_tiers.py   Probability risk-band classification
src/obesity_ml/advice.py       Rule-based wellness advice logic and advice sources
src/obesity_ml/form_import.py  Normalizes the two Google Form CSV formats
src/obesity_ml/config.py       Project paths and model path
data/sample_obesity_training.csv
models/obesity_probability_model.joblib
tests/test_app_regressions.py
```

## Run Locally

From the project root:

```bash
cd "/Users/phawichpilathong/Documents/Obesity machine learning"
source .venv/bin/activate
PYTHONPATH=src .venv/bin/python -m uvicorn obesity_ml.app:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/
```

If port 8000 is already in use, inspect it before killing anything:

```bash
lsof -iTCP:8000 -sTCP:LISTEN -n -P
```

## Training

Prototype training command:

```bash
PYTHONPATH=src .venv/bin/python -m obesity_ml.train --data data/sample_obesity_training.csv
```

Real Google Form CSVs should be normalized before training:

```bash
PYTHONPATH=src .venv/bin/python -m obesity_ml.form_import \
  --input form1.csv form2.csv \
  --output data/raw/normalized_obesity_training.csv
```

## ML Protocol

Keep the model workflow research-safe:

- Use stratified cross-validation for model comparison when enough data exists.
- Keep final test data separate for reporting.
- Use SMOTENC for class imbalance when categorical columns are present.
- Do not use plain SMOTE after one-hot encoding, because it can create unrealistic fractional category values.
- Do not calibrate on the final test set.
- Say "best for the current training data," not "best algorithm forever."
- Include prototype warnings when the dataset is too small.

Current candidate algorithms:

- Logistic regression
- Support vector machine
- Random forest
- Gaussian Naive Bayes
- Neural network MLP
- XGBoost, when installed and available

Useful model metrics:

- ROC-AUC for ranking probability discrimination
- F1 score for class-balance behavior
- Brier score for probability calibration/error
- Accuracy, Cohen's Kappa, sensitivity, and specificity for research-style reporting
- Confusion matrix when explaining classification performance

## Risk Bands

Probability risk bands are implemented in `src/obesity_ml/risk_tiers.py`. Keep bands transparent and avoid pretending that the cutoffs are clinical diagnosis thresholds.

When changing risk-band logic, update:

- Result page wording in `src/obesity_ml/app.py`
- Advice priority in `src/obesity_ml/advice.py`, if needed
- Tests or smoke checks that assert result text

## Advice Feature

Advice is rule-based and intentionally transparent. It should use user answers plus prediction probability to give practical wellness suggestions.

Advice should be:

- Specific to the user's answers
- Educational, not clinical
- Based on trustworthy public-health sources such as WHO, CDC, or Thai dietary guidance
- Clear that family history means "be more aware," not "you are guaranteed to develop obesity"

## UI Guidance

The UI style is O-Beast: bold, polished, Instagram-like, and student-friendly.

Preserve these expectations:

- Keep the logo and O-Beast brand visible.
- Use clear visual explanations for algorithms.
- Write visible app copy for students and general users, not for developers or the assistant/user conversation.
- Do not put phrases like "your command," "your REF papers," or internal build notes into the app UI.
- Keep technical terms in deeper method/result sections only when they help explain the research; the home page should feel simple and welcoming.
- Avoid hidden or overlapping text.
- Make result pages focus on the result only.
- Use animations carefully so clicks feel alive but do not distract.
- For algorithm visuals, explain intuitively what the algorithm does.

## Verification Before Completion

After any code or project change, run relevant checks before saying the task is complete.

Minimum checks for most Python/app changes:

```bash
PYTHONPATH=src .venv/bin/python -m compileall src
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
git diff --check
```

If the server is running, also smoke-check the app:

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/
```

Use honest wording in final responses:

- "The checks I ran passed."
- "I did not find errors in these tests."
- "I cannot guarantee zero bugs, but this passed the relevant verification."

Avoid:

- "No bugs guaranteed"
- "100% correct"
- "clinically accurate"

## Code References In Final Replies

After every code change, include useful code references with absolute file links and line numbers.

Example:

```markdown
Code references:
- [train.py](/Users/phawichpilathong/Documents/Obesity%20machine%20learning/src/obesity_ml/train.py:108): candidate model definitions
- [risk_tiers.py](/Users/phawichpilathong/Documents/Obesity%20machine%20learning/src/obesity_ml/risk_tiers.py:52): probability tier classification
```

Reference the important changed areas, not every small style line.

## Git Rules

The user wants updates pushed to GitHub when requested. Before committing:

```bash
git status --short --branch
git diff --check
```

Do not revert user changes. If the working tree has unrelated changes, leave them alone and only commit the files you intentionally changed.

After pushing:

```bash
git status --short --branch
git log --oneline --decorate -3
git ls-remote origin refs/heads/main
```

Report whether local `main` and `origin/main` match.

## Current Repository

GitHub remote:

```text
https://github.com/choruslover0/Machine-Learning-within-Application-for-analysis-a-probability-of-Obesity-in-Thai-People.git
```

Main local path:

```text
/Users/phawichpilathong/Documents/Obesity machine learning
```
