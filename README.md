# Obesity Probability ML App

This project is a starter machine-learning application for estimating the probability that a person belongs to an obesity-risk group.

Important: this is an educational research tool, not a medical diagnosis tool.

## Data Format

Put your real training CSV at:

```text
data/raw/obesity_training.csv
```

Required columns:

```text
age
sex
height_cm
weight_kg
physical_activity_hours_per_week
screen_time_hours_per_day
sleep_hours
fast_food_meals_per_week
sugary_drinks_per_day
family_history_obesity
obesity
```

`obesity` should be `0` for no obesity and `1` for obesity.

Optional columns supported by the newer importer:

```text
high_calorie_food_frequency
vegetable_frequency
main_meals_per_day
food_between_meals_frequency
smoke
physical_activity_missing
screen_time_missing
fast_food_missing
sugary_drinks_missing
```

## Google Form Import

The project can now normalize both Google Form styles used in the research:

- Form 1: lifestyle, screen time, sleep, fast food, sugary drinks, family history, and self-reported obesity.
- Form 2: UCI-style habits such as high-calorie food, vegetables, meals per day, food between meals, smoking, sleep, family history, height, and weight.

Form 2 does not ask every question from Form 1. The importer fills neutral values for missing cross-form questions and adds missingness flags, so the model can learn that those answers were not actually collected.

Example:

```bash
PYTHONPATH=src python -m obesity_ml.form_import \
  --input form1.csv form2.csv \
  --output data/raw/normalized_obesity_training.csv
```

The prototype target is created from BMI using a default threshold of `25.0`, which is a high-risk screening threshold in many Asian adult BMI references. This should be reviewed before the final research analysis.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m obesity_ml.train --data data/sample_obesity_training.csv
uvicorn obesity_ml.app:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```
# Machine-Learning-within-Application-for-analysis-a-probability-of-Obesity-in-Thai-People

## Current ML Protocol

The training pipeline now uses a safer research prototype workflow:

- Model selection is based on stratified cross-validation on the training split.
- The final test split is kept separate for reporting only.
- Class imbalance is handled with SMOTENC before preprocessing, so categorical survey answers are not turned into unrealistic fractional one-hot values.
- Calibration is only performed when there is enough training data for a separate calibration split; the final test set is not used for calibration.
- The UI describes the selected model as the best model for the current training data, not as a universal medical truth.
- Candidate algorithms include logistic regression, support vector machine, random forest, Gaussian Naive Bayes, neural network, and XGBoost when installed.
- Evaluation now records Accuracy, Cohen's Kappa, Sensitivity, and Specificity, matching the style of comparative obesity ML papers, while still using ROC-AUC, F1, and Brier score for probability-model selection.
- Prediction output now uses five probability tiers: very low, low, moderate, high, and very high.

The included sample CSV is intentionally tiny and exists only to prove the app and algorithms run. Real conclusions should wait for the completed experimental dataset.

## Wellness Advice Page

The app includes an educational rule-based advice page. It uses the user's answers plus the model probability to suggest practical wellness habits for activity, sedentary time, sleep, sugary drinks, fast food, BMI awareness, and family-history awareness.

The advice rules are intentionally transparent and are based on public-health references from WHO, CDC, and Thai food-based dietary guidance. They are not a diagnosis or treatment plan.
