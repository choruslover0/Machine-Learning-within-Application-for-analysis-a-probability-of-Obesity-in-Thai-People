# Architecture

O-Beast is a single Python FastAPI app with a scikit-learn training pipeline, joblib model artifacts, static assets, and markdown research docs.

## System Map

```text
UCI-style obesity survey CSV
  -> uci_import.py
  -> data/processed/uci_obesity_training.csv
  -> train.py
  -> models/obesity_probability_model.joblib
  -> predict.py
  -> app.py + routes/api.py
  -> Jinja2 templates + chart fragments
  -> result page + advice page
```

## Runtime Components

```text
src/obesity_ml/app.py
```

FastAPI routes and HTML/CSS UI. It owns page rendering for home, predictor, result, advice, methods, and chatbot.

```text
src/obesity_ml/routes/api.py
```

APIRouter module for JSON endpoints: `/predict` and `/chat`.

```text
src/obesity_ml/templating.py
src/obesity_ml/templates/
```

Jinja2 rendering layer. `base.html` owns the page shell and `partials/styles.html` keeps CSS in a template partial so tests can still inspect inline CSS in HTML responses.

```text
src/obesity_ml/charts.py
```

Builds Methods-page metric bars, metric tables, selected-model banner, and ROC visual fragments from stored model metrics.

```text
src/obesity_ml/predict.py
```

Loads the model artifact, computes lifestyle ML probability, computes BMI screen score, blends both into final probability, and returns risk-tier data.

```text
src/obesity_ml/advice.py
```

Creates educational wellness advice cards from user input, probability tier, BMI tier, and public-health reference logic.

```text
src/obesity_ml/chatbot.py
```

Trains and serves the small Beast 1.0 question-answer intent model.

## Training Components

```text
src/obesity_ml/uci_import.py
```

Normalizes the UCI ObesityDataSet-style source into the O-Beast training schema. The UCI source is the active prototype training source for testing the app pipeline and future survey format.

```text
src/obesity_ml/features.py
```

Validates user input and computes engineered fields such as BMI and optional missingness flags.

```text
src/obesity_ml/train.py
```

Builds preprocessing pipelines, applies SMOTENC when possible, compares candidate algorithms, stores metrics, stores ROC curves, and writes `models/obesity_probability_model.joblib`.

```text
src/obesity_ml/config.py
```

Defines paths and feature groups:

- `NUMERIC_FEATURES`: lifestyle ML numeric features only
- `CATEGORICAL_FEATURES`: lifestyle ML categorical features
- `BODY_SCREEN_FEATURES`: `height_cm`, `weight_kg`, `bmi`

## Probability Design

Lifestyle ML and BMI screening are intentionally separate.

```text
lifestyle_probability = model.predict_proba(lifestyle_features)
bmi_screen_score = logistic BMI screening score
final_probability = 0.5 * lifestyle_probability + 0.5 * bmi_screen_score
```

Reason: the behavior model should learn lifestyle and survey patterns, while body measurements stay visible as a separate BMI screening signal. This keeps the app easier to explain and prevents body measurements from dominating a model that is meant to test habit-format questions.

## ML Pipeline

1. Validate training columns and target values.
2. Compute BMI and optional defaults.
3. Select only lifestyle ML features for model training.
4. Split data with stratification when possible.
5. Apply SMOTENC to training data when minority class count is enough.
6. Preprocess numeric and categorical features.
7. Compare candidate models with stratified cross-validation.
8. Select best model by ROC-AUC, F1, extreme-probability rate, then Brier score.
9. Fit selected pipeline.
10. Store model, metrics, feature list, and blend strategy.

## Candidate Models

- Logistic Regression
- Support Vector Machine RBF
- Random Forest
- Gaussian Naive Bayes
- MLP Neural Network
- XGBoost

## Data Directories

```text
data/raw/          Original UCI-style source and legacy raw files
data/processed/    Normalized active UCI-style training CSV
data/sample_*      Tiny local smoke-test data
models/            Trained joblib artifacts
research-assets/   Research workflow diagrams and PDFs
memory/            Older memory notes and project snapshots
```

## Safety Boundary

The app must describe outputs as educational estimates, not diagnosis. Advice output must stay wellness-oriented and transparent.

## Module layout (updated 2026-06-24)

Data source switched from the 205-row self-collected Google Forms prototype to the
2111-row UCI ObesityDataSet-style prototype source (Mendoza Palechor & de la Hoz
Manotas, 2019). App rendering moved to a hybrid Jinja2 architecture.

```
src/obesity_ml/
├── app.py            # FastAPI factory + HTML page routes
├── routes/
│   └── api.py        # JSON endpoints: /predict, /chat (APIRouter)
├── charts.py         # SVG/metric HTML fragments (kept in Python; injected via |safe)
├── templating.py     # Jinja2 Environment singleton + render()
├── templates/
│   ├── base.html             # page layout (nav, head, widget slot)
│   └── partials/styles.html  # CSS, inlined into <head> (kept inline for test contract)
├── uci_import.py     # active UCI-style CSV -> training schema loader
├── config.py         # pure UCI feature lists (FAVC/FCVC/NCP/CAEC/CH2O/SCC/FAF/TUE/CALC/MTRANS)
├── features.py       # feature engineering + validation
├── predict.py        # lifestyle ML + BMI screen blend
├── train.py          # model tournament + calibration
├── advice.py, risk_tiers.py, chatbot.py
```

CSS note: tests assert CSS rule substrings in `response.text`, so styles are
extracted out of Python but served inline via a Jinja `{% include %}`, not an
external `<link>` stylesheet.

## Pure UCI feature set (updated 2026-06-24)

The Thai Google-Form schema and loader were removed. The model, the predictor
form, the `/predict` API, and the advice engine all use the UCI ObesityDataSet
question set directly (no rescaling, no defaulted lifestyle inputs): FAVC, FCVC,
NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS plus age/sex/height/weight/
family-history. height/weight/bmi remain excluded from the lifestyle model and
feed only the BMI screen. A site-wide footer states the data is verified
open-source data (Colombia/Peru/Mexico), not Thai-population data.
