# Changelog

## Unreleased

### Documentation

- Added core documentation set:
  - `README.md`
  - `AGENTS.md`
  - `CLAUDE.md`
  - `CONTEXT.md`
  - `docs/ARCHITECTURE.md`
  - `docs/API.md`
  - `docs/TASKS.md`
  - `CHANGELOG.md`
- Cleaned `CONTEXT.md` into glossary-only project language.
- Added docs map so new readers can find architecture, API, tasks, and project history.

### ML Workflow

- Separated BMI, height, and weight from lifestyle ML features.
- Added final probability blend:

```text
0.5 * lifestyle_probability + 0.5 * bmi_screen_score
```

- Retrained model artifact with lifestyle-only ML features.
- Stored excluded body screen features and blend strategy in model artifact.
- Made the BMI screen a config-driven graded logistic curve (`BMI_SCREEN_MIDPOINT = 25`, `BMI_SCREEN_STEEPNESS = 3`), replacing the hardcoded `slope = 0.8`. Crosses 0.5 at BMI 25 but rises gradually so BMI informs — not dominates — the 50/50 blend, matching the research report.

### Testing

- Added regression tests for BMI exclusion and final probability blend.
- Added `httpx2` dependency to remove Starlette TestClient deprecation warning.
- Added regression test proving TestClient import emits no Starlette deprecation warning.

## Earlier Project Work

- Built FastAPI web app with home, predictor, result, advice, methods, and chatbot pages.
- Added Google Form normalization for two survey schemas.
- Added SMOTENC balancing before preprocessing.
- Added model tournament across Logistic Regression, SVM, Random Forest, Gaussian Naive Bayes, MLP, and XGBoost.
- Added five probability risk tiers.
- Added wellness advice cards based on user answers and public-health references.
- Added research workflow PNG/PDF assets.
