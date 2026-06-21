# Architecture

O-Beast is a single Python FastAPI app with a scikit-learn training pipeline, joblib model artifacts, static assets, and markdown research docs.

## System Map

```text
Google Forms CSV
  -> form_import.py
  -> data/processed/current_google_forms_training.csv
  -> train.py
  -> models/obesity_probability_model.joblib
  -> predict.py
  -> app.py routes/UI
  -> result page + advice page
```

## Runtime Components

```text
src/obesity_ml/app.py
```

FastAPI routes and HTML/CSS UI. It owns page rendering for home, predictor, result, advice, methods, and chatbot.

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
src/obesity_ml/form_import.py
```

Normalizes current Google Form exports into one training CSV. Current prototype target is BMI-derived until doctor diagnosis labels are available.

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

Reason: current prototype target is BMI-derived, so putting BMI directly into the ML model creates target leakage. Separating BMI keeps the research workflow honest and easier to explain.

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
data/raw/          Original Google Form exports
data/processed/    Normalized training CSV
data/sample_*      Tiny local smoke-test data
models/            Trained joblib artifacts
research-assets/   Research workflow diagrams and PDFs
memory/            Older memory notes and project snapshots
```

## Safety Boundary

The app must describe outputs as educational estimates, not diagnosis. Advice output must stay wellness-oriented and transparent.
