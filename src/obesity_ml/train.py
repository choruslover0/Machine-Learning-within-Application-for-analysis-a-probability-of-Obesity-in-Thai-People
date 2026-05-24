import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from obesity_ml.config import (
    CATEGORICAL_FEATURES,
    MODEL_DIR,
    MODEL_PATH,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from obesity_ml.features import add_engineered_features, validate_training_frame

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:  # pragma: no cover - optional dependency fallback
    SMOTE = None
    ImbPipeline = None

XGBOOST_IMPORT_ERROR = None
try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover - optional dependency fallback
    XGBClassifier = None
    XGBOOST_IMPORT_ERROR = str(exc)


REFERENCE_NOTES = [
    "Notion ref: systematic literature review on obesity causes, consequences, and ML prediction approaches.",
    "Notion ref: systematic review/meta-analysis comparing logistic regression and machine learning for obesity risk prediction.",
    "Notion ref: Thai AI health-risk paper using model comparison and SHAP-style interpretability.",
]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def can_use_smote(y_train: pd.Series) -> bool:
    return SMOTE is not None and y_train.nunique() == 2 and y_train.value_counts().min() >= 3


def make_pipeline(model, y_train: pd.Series) -> Pipeline:
    steps = [("preprocess", build_preprocessor())]
    if can_use_smote(y_train):
        minority_count = int(y_train.value_counts().min())
        steps.append(("smote", SMOTE(k_neighbors=max(1, min(5, minority_count - 1)), random_state=42)))
    steps.append(("model", model))
    if len(steps) > 2 and ImbPipeline is not None:
        return ImbPipeline(steps=steps)
    return Pipeline(steps=steps)


def candidate_models(y_train: pd.Series) -> dict[str, Pipeline]:
    candidates = {
        "logistic_regression_balanced": make_pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
            y_train,
        ),
        "support_vector_machine_rbf": make_pipeline(
            SVC(kernel="rbf", C=2.0, probability=True, class_weight="balanced", random_state=42),
            y_train,
        ),
        "random_forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
            ),
            y_train,
        ),
        "neural_network_mlp": make_pipeline(
            MLPClassifier(
                hidden_layer_sizes=(32, 16),
                alpha=0.01,
                learning_rate_init=0.003,
                max_iter=1200,
                random_state=42,
            ),
            y_train,
        ),
    }

    if XGBClassifier is not None:
        candidates["xgboost_gradient_boosting"] = make_pipeline(
            XGBClassifier(
                n_estimators=250,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
            ),
            y_train,
        )

    return candidates


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    probability = model.predict_proba(x_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
        "precision": precision_score(y_test, prediction, zero_division=0),
        "recall": recall_score(y_test, prediction, zero_division=0),
        "f1": f1_score(y_test, prediction, zero_division=0),
        "brier_score": brier_score_loss(y_test, probability),
    }
    metrics["roc_auc"] = roc_auc_score(y_test, probability) if len(set(y_test)) > 1 else float("nan")
    return metrics


def score_for_selection(metrics: dict[str, float]) -> tuple[float, float, float]:
    roc_auc = metrics["roc_auc"]
    if roc_auc != roc_auc:  # NaN check
        roc_auc = 0.0
    return (roc_auc, metrics["f1"], -metrics["brier_score"])


def train(data_path: Path, model_path: Path = MODEL_PATH) -> dict[str, object]:
    df = pd.read_csv(data_path)
    validate_training_frame(df)
    df = add_engineered_features(df)

    x = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    results = {}
    trained_models = {}
    for name, model in candidate_models(y_train).items():
        model.fit(x_train, y_train)
        results[name] = evaluate_model(model, x_test, y_test)
        trained_models[name] = model

    best_name = max(results, key=lambda name: score_for_selection(results[name]))
    best_model = trained_models[best_name]

    if y_test.nunique() == 2 and len(y_test) >= 4:
        calibrated_model = CalibratedClassifierCV(FrozenEstimator(best_model), method="sigmoid")
        calibrated_model.fit(x_test, y_test)
    else:
        calibrated_model = best_model

    artifact = {
        "model": calibrated_model,
        "base_model_name": best_name,
        "metrics": results,
        "feature_columns": list(x.columns),
        "target_column": TARGET_COLUMN,
        "used_smote": can_use_smote(y_train),
        "candidate_methods": list(results.keys()),
        "xgboost_status": "enabled" if XGBClassifier is not None else f"disabled: {XGBOOST_IMPORT_ERROR}",
        "reference_notes": REFERENCE_NOTES,
        "selection_rule": "Choose highest ROC-AUC, then F1, then lowest Brier score.",
        "disclaimer": "Educational risk-estimation model only; not a medical diagnosis.",
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the obesity probability model.")
    parser.add_argument("--data", type=Path, required=True, help="Path to training CSV.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Output model path.")
    args = parser.parse_args()

    artifact = train(args.data, args.model)
    print(f"Saved model to: {args.model}")
    print(f"Best base model: {artifact['base_model_name']}")
    print(f"SMOTE used: {artifact['used_smote']}")
    print(f"Selection rule: {artifact['selection_rule']}")
    print("Metrics:")
    for name, metrics in artifact["metrics"].items():
        print(f"  {name}: {metrics}")


if __name__ == "__main__":
    main()
