import pandas as pd

from obesity_ml.config import OPTIONAL_INPUT_DEFAULTS, REQUIRED_INPUT_COLUMNS, TARGET_COLUMN


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


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    height_m = out["height_cm"].astype(float) / 100
    out["bmi"] = out["weight_kg"].astype(float) / (height_m**2)
    for column, default in OPTIONAL_INPUT_DEFAULTS.items():
        if column not in out.columns:
            out[column] = default
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(default)
    out["family_history_obesity"] = out["family_history_obesity"].astype(str)
    out["sex"] = out["sex"].astype(str).str.upper().str.strip()
    return out
