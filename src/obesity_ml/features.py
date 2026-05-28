import pandas as pd

from obesity_ml.config import OPTIONAL_INPUT_DEFAULTS, REQUIRED_INPUT_COLUMNS, TARGET_COLUMN


NUMERIC_PREDICTION_RANGES = {
    "age": (5, 100),
    "height_cm": (80, 230),
    "weight_kg": (20, 250),
    "physical_activity_hours_per_week": (0, 40),
    "screen_time_hours_per_day": (0, 24),
    "sleep_hours": (0, 16),
    "fast_food_meals_per_week": (0, 30),
    "sugary_drinks_per_day": (0, 20),
}


def validate_training_frame(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_INPUT_COLUMNS + [TARGET_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")

    invalid_targets = set(df[TARGET_COLUMN].dropna().unique()) - {0, 1}
    if invalid_targets:
        raise ValueError("Target column 'obesity' must contain only 0 and 1.")


def validate_prediction_frame(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_INPUT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Prediction input is missing required columns: {missing}")

    invalid = []
    for column, (minimum, maximum) in NUMERIC_PREDICTION_RANGES.items():
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any() or (~values.between(minimum, maximum)).any():
            invalid.append(f"{column} must be between {minimum} and {maximum}")

    sex_values = df["sex"].astype(str).str.upper().str.strip()
    if (~sex_values.isin({"M", "F"})).any():
        invalid.append("sex must be M or F")

    family_values = pd.to_numeric(df["family_history_obesity"], errors="coerce")
    if family_values.isna().any() or (~family_values.isin({0, 1})).any():
        invalid.append("family_history_obesity must be 0 or 1")

    if invalid:
        raise ValueError("Prediction input has invalid values: " + "; ".join(invalid))


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Use a small epsilon to avoid division by zero
    height_m = out["height_cm"].astype(float).clip(lower=1.0) / 100
    out["bmi"] = out["weight_kg"].astype(float) / (height_m**2)
    for column, default in OPTIONAL_INPUT_DEFAULTS.items():
        if column not in out.columns:
            out[column] = default
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(default)
    out["family_history_obesity"] = out["family_history_obesity"].astype(str)
    out["sex"] = out["sex"].astype(str).str.upper().str.strip()
    return out
