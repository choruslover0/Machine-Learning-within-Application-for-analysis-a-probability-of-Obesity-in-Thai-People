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

The API uses the full UCI ObesityDataSet question set. All sixteen fields are required.

Request:

```json
{
  "age": 21,
  "sex": "F",
  "height_cm": 162,
  "weight_kg": 64,
  "family_history_obesity": 1,
  "high_calorie_food_frequency": 0,
  "vegetable_frequency": 2,
  "main_meals_per_day": 3,
  "food_between_meals_frequency": 1,
  "smoke": 0,
  "water_daily": 2,
  "calorie_monitoring": 0,
  "physical_activity_freq": 2,
  "screen_time_band": 1,
  "alcohol_frequency": 0,
  "transportation": "Public_Transportation"
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

- `age`: 14-100
- `sex`: `M` or `F`
- `height_cm`: 80-230
- `weight_kg`: 20-250
- `family_history_obesity`: 0 or 1
- `high_calorie_food_frequency` (FAVC): 0 or 1
- `vegetable_frequency` (FCVC): 1-3
- `main_meals_per_day` (NCP): 1-4
- `food_between_meals_frequency` (CAEC): 0-3
- `smoke` (SMOKE): 0 or 1 (user's own smoking, not second-hand)
- `water_daily` (CH2O): 1-3
- `calorie_monitoring` (SCC): 0 or 1
- `physical_activity_freq` (FAF): 0-3
- `screen_time_band` (TUE): 0-2
- `alcohol_frequency` (CALC): integer 0-3 (No/Sometimes/Frequently/Always)
- `transportation` (MTRANS): `Automobile`, `Bike`, `Motorbike`, `Public_Transportation`, or `Walking`

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
- `src/obesity_ml/routes/api.py`
- `src/obesity_ml/templates/base.html`
- `src/obesity_ml/templates/partials/styles.html`
- `src/obesity_ml/predict.py`
- `src/obesity_ml/advice.py`
- `src/obesity_ml/chatbot.py`
