# O-Beast

Educational obesity-risk probability app for Thai student research.

O-Beast collects body, lifestyle, and family-history answers, estimates an educational obesity-risk probability, explains model methods, and gives transparent wellness advice. It is not a medical diagnosis tool.

## Start Here

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python3 -m obesity_ml.train --data data/processed/current_google_forms_training.csv
PYTHONPATH=src uvicorn obesity_ml.app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Current Workflow

1. Import Google Form CSV data.
2. Clean and normalize survey columns.
3. Create BMI for screening, not for lifestyle ML features.
4. Train lifestyle ML model from habits and family-history signals.
5. Compare candidate algorithms with stratified cross-validation.
6. Keep hold-out test data for final reporting only.
7. Calculate final probability with a 50/50 blend:

```text
final probability = 0.5 * lifestyle_probability + 0.5 * bmi_screen_score
```

8. Convert final probability into five risk tiers and show advice.

## Main Files

```text
src/obesity_ml/app.py          FastAPI routes and UI
src/obesity_ml/train.py        ML training, model comparison, metrics
src/obesity_ml/predict.py      Prediction, BMI score, final probability blend
src/obesity_ml/form_import.py  Google Form normalization
src/obesity_ml/advice.py       Wellness advice logic
src/obesity_ml/risk_tiers.py   Five probability tiers and BMI screening tiers
data/                          Sample, raw, and processed data
models/                        Trained joblib artifacts
docs/                          Architecture, API, tasks, research docs
memory/                        Project memory snapshots and older notes
research-assets/               Research workflow images and PDFs
```

## Core Docs

- [AGENTS.md](./AGENTS.md) - instructions for Codex and AI coding agents
- [CLAUDE.md](./CLAUDE.md) - instructions for Claude/Codex CLI-style agents
- [CONTEXT.md](./CONTEXT.md) - project language glossary
- [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) - system design and data flow
- [docs/API.md](./docs/API.md) - app routes and request shapes
- [docs/TASKS.md](./docs/TASKS.md) - current work queue
- [CHANGELOG.md](./CHANGELOG.md) - project history

## Data Notes

Current processed training data:

```text
data/processed/current_google_forms_training.csv
```

Raw Google Form exports:

```text
data/raw/google_form_1_current.csv
data/raw/google_form_2_current.csv
```

The current prototype target still comes from BMI thresholding. Future doctor-diagnosis labels should replace this target when available.

## Safety

O-Beast gives educational probability estimates only. It should not replace doctors, nutrition professionals, clinical diagnosis, or treatment plans.
