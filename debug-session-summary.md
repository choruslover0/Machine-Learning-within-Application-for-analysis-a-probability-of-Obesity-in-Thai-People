# Debug Session Summary

Date: 2026-05-27

Project: `/Users/phawichpilathong/Documents/Obesity machine learning`

## Original Request

Inspect the web app, identify bugs first, explain them for review, and only fix them after approval.

## Bugs Found

1. The local `.venv` still pointed to an old project path:
   `/Users/phawichpilathong/Documents/New project 2/.venv`

2. The app could not reliably import or run from normal commands because the package install was stale.

3. Static files used a cwd-relative path:
   `src/obesity_ml/static`

4. The JSON API rejected invalid values, but HTML form routes accepted impossible values and returned predictions.

5. Shared prediction validation only checked missing columns, not invalid ranges or invalid categories.

6. FastAPI `TestClient` could not run because `httpx` was missing.

7. Installed-package execution could not find the project model path correctly.

8. The SVG logo was not included in the installed package.

## Fixes Made

- Repaired `.venv/bin` script paths so they point to this project.
- Reinstalled the project into the current `.venv`.
- Added `httpx` to dependencies.
- Changed static file loading to use the package directory.
- Added package-data config for `static/*.svg`.
- Added shared prediction value validation.
- Made `/predict-form` and `/advice` return a clean `400` input-error page for invalid values.
- Restricted API `sex` values to `M` or `F`.
- Updated model path resolution so installed app runs from the project directory.
- Added regression tests in `tests/test_app_regressions.py`.

## Important Files Changed

- `src/obesity_ml/app.py`
- `src/obesity_ml/features.py`
- `src/obesity_ml/config.py`
- `pyproject.toml`
- `requirements.txt`
- `tests/test_app_regressions.py`

## Verification Passed

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall src tests
.venv/bin/python -m obesity_ml.train --data data/sample_obesity_training.csv --model /tmp/obeast_tmp_model.joblib
git diff --check
```

Live server smoke test with `.venv/bin/uvicorn obesity_ml.app:app --host 127.0.0.1 --port 8001`:

- `GET /` returned `200`
- `GET /static/obeast-logo.svg` returned `200`
- Invalid `POST /predict-form` returned `400`
- Valid `POST /predict-form` returned `200`

## Current Git Status

There are uncommitted changes. Review with:

```bash
git status --short
git diff
```

## Notes

This app is still an educational research prototype, not a medical diagnosis tool. The sample dataset is intentionally tiny, so model metrics are only useful for checking that the pipeline runs.
