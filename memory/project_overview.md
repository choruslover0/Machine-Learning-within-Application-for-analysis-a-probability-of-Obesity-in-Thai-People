---
name: project-overview
description: O-Beast FastAPI obesity-risk ML app — purpose, tech stack, module responsibilities, ML rules, UI constraints
metadata:
  type: project
---

O-Beast is an educational FastAPI web application that estimates obesity-risk probability for Thai people using lifestyle, body, and family-history inputs. Built for student research and professor presentations — explicitly NOT medical diagnosis.

**Why:** Research prototype for high-school students to learn ML concepts in a real application.

**Stack:** Python, FastAPI, scikit-learn, imbalanced-learn (SMOTENC), XGBoost (optional), joblib, Pandas, Uvicorn.

**Module map:**
- `app.py` — FastAPI routes, full UI HTML/CSS (Instagram-style), result page, algorithm visual explanations
- `train.py` — ML pipeline: SMOTENC balancing, ColumnTransformer, 6 candidate algorithms, stratified CV, model selection, metrics
- `predict.py` — loads joblib artifact, returns probability
- `features.py` — input validation, BMI engineered feature, missing-value flags
- `risk_tiers.py` — 5-tier probability bands (0–20%, 20–40%, 40–60%, 60–80%, 80–100%), coarse 3-band, Asian BMI screening tiers
- `advice.py` — rule-based wellness advice keyed on user answers + probability, sourced from WHO/CDC/FAO/Thai guidelines
- `config.py` — paths, feature lists (numeric + categorical), required/optional input columns
- `form_import.py` — normalizes Google Form CSV exports for training

**ML rules (hard constraints):**
- SMOTENC before one-hot encoding (not plain SMOTE after encoding)
- Stratified cross-validation for model comparison
- Separate final test set; do NOT calibrate on it
- Language: "best for the current training data," never absolute claims

**Candidate algorithms:** Logistic Regression, SVM, Random Forest, Gaussian Naive Bayes, MLP Neural Network, XGBoost.

**Key metrics:** ROC-AUC, F1, Brier score, Accuracy, Cohen's Kappa, Sensitivity, Specificity, Confusion Matrix.

**How to apply:** Respect the educational framing — no clinical claims, always mention it's a prototype on small sample data. The Asian BMI thresholds (23/25/30) differ from Western thresholds and matter for this Thai-context app.
