# API

O-Beast exposes HTML routes for users and JSON routes for programmatic use.

## Base URL

```text
http://127.0.0.1:8000
```

## HTML Routes

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Home page |
| `GET` | `/predictor` | Multi-step prediction form |
| `POST` | `/predict-form` | Submit form and render result page |
| `GET` | `/advice` | Advice intro/example page |
| `POST` | `/advice` | Render advice from submitted form data |
| `GET` | `/methods` | ML methods, metrics, and evaluation page |

## JSON Routes

### `POST /predict`

Returns probability estimate and model metadata.

Request:

```json
{
  "age": 16,
  "sex": "M",
  "height_cm": 170,
  "weight_kg": 65,
  "physical_activity_hours_per_week": 3,
  "screen_time_hours_per_day": 5,
  "sleep_hours": 7,
  "fast_food_meals_per_week": 2,
  "sugary_drinks_per_day": 1,
  "family_history_obesity": 0
}
```

Response shape (example values are the real output for the sample request above, BMI 22.49):

```json
{
  "obesity_probability": 0.1925,
  "lifestyle_probability": 0.0826,
  "bmi_screen_score": 0.3023,
  "probability_blend_strategy": "Final probability = 50% lifestyle_probability + 50% bmi_screen_score...",
  "risk_tier": "very_low",
  "risk_tier_label": "Very low risk tier",
  "risk_tier_range": "0-19.9%",
  "risk_band": "low",
  "base_model_name": "logistic_regression_balanced",
  "candidate_methods": ["logistic_regression_balanced"],
  "used_smote": true,
  "selection_rule": "...",
  "validation_strategy": "...",
  "resampling_strategy": "...",
  "dataset_warning": "",
  "metrics": {},
  "test_metrics": {},
  "disclaimer": "Educational risk-estimation model only; not a medical diagnosis."
}
```

Validation:

- `age`: 5-100
- `sex`: `M` or `F`
- `height_cm`: 80-230
- `weight_kg`: 20-250
- `physical_activity_hours_per_week`: 0-40
- `screen_time_hours_per_day`: 0-24
- `sleep_hours`: 0-16
- `fast_food_meals_per_week`: 0-30
- `sugary_drinks_per_day`: 0-20
- `family_history_obesity`: 0 or 1

### `POST /chat`

Returns Beast 1.0 chatbot response.

Request:

```json
{
  "message": "What is BMI?",
  "lang": "auto",
  "context": {
    "risk_tier": "Very low risk tier",
    "probability": 0.12
  }
}
```

Response shape:

```json
{
  "intent": "bmi",
  "answer": "...",
  "confidence": 0.91,
  "lang": "en"
}
```

## Error Behavior

- Invalid JSON/form values return validation errors.
- Missing model artifact raises clear model-not-found error.
- HTML form errors render an input error page instead of a result page.

## Related Files

- `src/obesity_ml/app.py`
- `src/obesity_ml/predict.py`
- `src/obesity_ml/advice.py`
- `src/obesity_ml/chatbot.py`
