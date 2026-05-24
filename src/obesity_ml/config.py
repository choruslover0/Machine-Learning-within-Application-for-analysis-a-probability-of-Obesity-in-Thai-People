from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "obesity_probability_model.joblib"

TARGET_COLUMN = "obesity"

NUMERIC_FEATURES = [
    "age",
    "height_cm",
    "weight_kg",
    "bmi",
    "physical_activity_hours_per_week",
    "screen_time_hours_per_day",
    "sleep_hours",
    "fast_food_meals_per_week",
    "sugary_drinks_per_day",
]

CATEGORICAL_FEATURES = [
    "sex",
    "family_history_obesity",
]

REQUIRED_INPUT_COLUMNS = [
    "age",
    "sex",
    "height_cm",
    "weight_kg",
    "physical_activity_hours_per_week",
    "screen_time_hours_per_day",
    "sleep_hours",
    "fast_food_meals_per_week",
    "sugary_drinks_per_day",
    "family_history_obesity",
]

