# Claude Instructions

Follow [AGENTS.md](./AGENTS.md). This file exists for Claude-style tools that look for `CLAUDE.md`.

## Operating Mode

- Be concise.
- Prefer direct code edits when request is implementation work.
- Do not add web UI text that talks about the conversation, prompts, or agents.
- Cite changed files and line numbers after edits.
- Verify before saying work is complete.

## O-Beast Research Rules

- App is educational, not diagnostic.
- BMI is a separate screening signal.
- Lifestyle ML must exclude `height_cm`, `weight_kg`, and `bmi`.
- Final probability is the 50/50 blend of lifestyle ML and BMI screening.
- SMOTENC is balancing, not an algorithm.
- Current dataset is still prototype-grade until doctor-diagnosis labels arrive.

## Common Commands

```bash
PYTHONPATH=src python3 -m obesity_ml.form_import \
  --input data/raw/google_form_1_current.csv data/raw/google_form_2_current.csv \
  --output data/processed/current_google_forms_training.csv
```

```bash
PYTHONPATH=src python3 -m obesity_ml.train \
  --data data/processed/current_google_forms_training.csv \
  --model models/obesity_probability_model.joblib
```

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

```bash
PYTHONPATH=src uvicorn obesity_ml.app:app --reload
```
