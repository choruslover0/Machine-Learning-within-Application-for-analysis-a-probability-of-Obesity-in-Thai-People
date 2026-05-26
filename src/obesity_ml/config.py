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
    "high_calorie_food_frequency",
    "vegetable_frequency",
    "main_meals_per_day",
    "food_between_meals_frequency",
    "smoke",
    "physical_activity_missing",
    "screen_time_missing",
    "fast_food_missing",
    "sugary_drinks_missing",
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

OPTIONAL_INPUT_DEFAULTS = {
    "high_calorie_food_frequency": 0.0,
    "vegetable_frequency": 2.0,
    "main_meals_per_day": 3.0,
    "food_between_meals_frequency": 1.0,
    "smoke": 0.0,
    "physical_activity_missing": 0.0,
    "screen_time_missing": 0.0,
    "fast_food_missing": 0.0,
    "sugary_drinks_missing": 0.0,
}
