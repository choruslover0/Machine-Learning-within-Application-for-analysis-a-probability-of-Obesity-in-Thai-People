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
