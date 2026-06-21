from pathlib import Path
from math import exp

import joblib
import pandas as pd

from obesity_ml.config import BMI_SCREEN_MIDPOINT, BMI_SCREEN_STEEPNESS, MODEL_PATH
from obesity_ml.features import add_engineered_features, validate_prediction_frame
from obesity_ml.risk_tiers import classify_probability


def load_artifact(model_path: Path = MODEL_PATH) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train it first with: "
            "python -m obesity_ml.train --data data/sample_obesity_training.csv"
        )
    return joblib.load(model_path)


def bmi_screen_score(
    bmi: float,
    midpoint: float = BMI_SCREEN_MIDPOINT,
    steepness: float = BMI_SCREEN_STEEPNESS,
) -> float:
    # Graded logistic screen: crosses 0.5 at `midpoint`, rises gradually over
    # `steepness` BMI units. Dividing by steepness (vs a large slope multiplier)
    # keeps the curve soft so BMI informs — not dominates — the final blend.
    return 1 / (1 + exp(-(float(bmi) - midpoint) / steepness))


def blend_lifestyle_and_bmi_probabilities(lifestyle_probability: float, bmi_score: float) -> float:
    lifestyle = max(0.0, min(1.0, float(lifestyle_probability)))
    bmi_signal = max(0.0, min(1.0, float(bmi_score)))
    return (0.5 * lifestyle) + (0.5 * bmi_signal)


def predict_probability(input_data: dict, model_path: Path = MODEL_PATH) -> dict:
    df = pd.DataFrame([input_data])
    validate_prediction_frame(df)
    df = add_engineered_features(df)

    artifact = load_artifact(model_path)
    model = artifact["model"]
    lifestyle_probability = float(model.predict_proba(df)[0, 1])
    bmi_score = bmi_screen_score(float(df.iloc[0]["bmi"]))
    probability = blend_lifestyle_and_bmi_probabilities(lifestyle_probability, bmi_score)
    risk = classify_probability(probability)

    return {
        "obesity_probability": round(probability, 4),
        "lifestyle_probability": round(lifestyle_probability, 4),
        "bmi_screen_score": round(bmi_score, 4),
        "probability_blend_strategy": artifact.get(
            "probability_blend_strategy",
            "Final probability = 50% lifestyle ML probability + 50% BMI screen score.",
        ),
        **risk,
        "base_model_name": artifact["base_model_name"],
        "candidate_methods": artifact.get("candidate_methods", []),
        "used_smote": bool(artifact.get("used_smote", False)),
        "selection_rule": artifact.get("selection_rule", ""),
        "validation_strategy": artifact.get("validation_strategy", ""),
        "resampling_strategy": artifact.get("resampling_strategy", ""),
        "dataset_warning": artifact.get("dataset_warning", ""),
        "metrics": artifact.get("metrics", {}),
        "test_metrics": artifact.get("test_metrics", {}),
        "disclaimer": artifact["disclaimer"],
    }
