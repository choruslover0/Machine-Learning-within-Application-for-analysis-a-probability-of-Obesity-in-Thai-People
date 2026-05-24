from pathlib import Path

import joblib
import pandas as pd

from obesity_ml.config import MODEL_PATH
from obesity_ml.features import add_engineered_features, validate_prediction_frame


def load_artifact(model_path: Path = MODEL_PATH) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train it first with: "
            "python -m obesity_ml.train --data data/sample_obesity_training.csv"
        )
    return joblib.load(model_path)


def predict_probability(input_data: dict, model_path: Path = MODEL_PATH) -> dict:
    df = pd.DataFrame([input_data])
    validate_prediction_frame(df)
    df = add_engineered_features(df)

    artifact = load_artifact(model_path)
    model = artifact["model"]
    probability = float(model.predict_proba(df)[0, 1])

    if probability < 0.35:
        risk_band = "low"
    elif probability < 0.65:
        risk_band = "medium"
    else:
        risk_band = "high"

    return {
        "obesity_probability": round(probability, 4),
        "risk_band": risk_band,
        "base_model_name": artifact["base_model_name"],
        "candidate_methods": artifact.get("candidate_methods", []),
        "used_smote": bool(artifact.get("used_smote", False)),
        "selection_rule": artifact.get("selection_rule", ""),
        "metrics": artifact.get("metrics", {}),
        "disclaimer": artifact["disclaimer"],
    }
