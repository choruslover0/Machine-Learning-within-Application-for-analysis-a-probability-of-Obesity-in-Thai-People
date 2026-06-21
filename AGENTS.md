# Agent Instructions

Use this file before changing O-Beast.

## Project

O-Beast is an educational FastAPI app for Thai obesity-risk research. The app estimates probability from lifestyle ML plus BMI screening. It is not a diagnosis.

## Required Behavior

- Read [README.md](./README.md), [CONTEXT.md](./CONTEXT.md), and [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) before large changes.
- Keep user-facing web copy clean. Do not include chat history, user commands, or agent conversation text in app UI.
- Use project language from [CONTEXT.md](./CONTEXT.md).
- Preserve medical safety wording: educational estimate, not diagnosis.
- Do not treat SMOTENC as a prediction algorithm. It is data balancing only.
- Do not put BMI, height, or weight into the lifestyle ML feature list.
- Keep final probability as `50% lifestyle_probability + 50% bmi_screen_score` unless research protocol changes.

## Code Rules

- Use `rg` for search.
- Use `apply_patch` for manual edits.
- Do not revert unrelated user changes.
- Keep edits scoped.
- After code changes, run relevant tests.
- After ML logic changes, retrain model artifact and inspect stored feature columns.

## Verification

Preferred checks:

```bash
PYTHONPATH=src python3 -m compileall src
PYTHONPATH=src python3 -m unittest discover -s tests
git diff --check
```

For app smoke:

```bash
PYTHONPATH=src uvicorn obesity_ml.app:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

## Current ML Contract

- Target column: `obesity`
- Lifestyle ML features: habits, missingness flags, sex, family history
- Body screen features: `height_cm`, `weight_kg`, `bmi`
- Candidate models: Logistic Regression, SVM RBF, Random Forest, Gaussian Naive Bayes, MLP, XGBoost
- Model choice: cross-validated ROC-AUC first, then F1, then fewer extreme probabilities, then lower Brier score
- Result tiers: very low, low, moderate, high, very high

## Docs Contract

- `README.md`: first reader entry
- `AGENTS.md`: agent instructions
- `CLAUDE.md`: Claude/Codex CLI instructions
- `CONTEXT.md`: glossary only
- `docs/ARCHITECTURE.md`: system design
- `docs/API.md`: routes and data contracts
- `docs/TASKS.md`: task queue
- `CHANGELOG.md`: history
