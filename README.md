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

The included sample CSV is intentionally tiny and exists only to prove the app and algorithms run. Real conclusions should wait for the completed experimental dataset.

## Wellness Advice Page

The app includes an educational rule-based advice page. It uses the user's answers plus the model probability to suggest practical wellness habits for activity, sedentary time, sleep, sugary drinks, fast food, BMI awareness, and family-history awareness.

The advice rules are intentionally transparent and are based on public-health references from WHO, CDC, and Thai food-based dietary guidance. They are not a diagnosis or treatment plan.
