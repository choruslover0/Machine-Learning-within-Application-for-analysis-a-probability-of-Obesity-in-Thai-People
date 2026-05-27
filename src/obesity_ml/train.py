import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
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
    from imblearn.over_sampling import SMOTENC
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:  # pragma: no cover - optional dependency fallback
    SMOTENC = None
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
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0.0,
    )


def can_use_smote(y_train: pd.Series) -> bool:
    return SMOTENC is not None and y_train.nunique() == 2 and y_train.value_counts().min() >= 3


def make_pipeline(model, y_train: pd.Series) -> Pipeline:
    steps = []
    if can_use_smote(y_train):
        minority_count = int(y_train.value_counts().min())
        steps.append(
            (
                "smotenc",
                SMOTENC(
                    categorical_features=CATEGORICAL_FEATURES,
                    k_neighbors=max(1, min(5, minority_count - 2)),
                    random_state=42,
                ),
            )
        )
    steps.append(("preprocess", build_preprocessor()))
    steps.append(("model", model))
    if any(step_name == "smotenc" for step_name, _ in steps) and ImbPipeline is not None:
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
        "naive_bayes_gaussian": make_pipeline(
            GaussianNB(var_smoothing=1e-2),
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


def specificity_score(y_true: pd.Series, prediction) -> float:
    labels = sorted(set(y_true) | set(prediction))
    if len(labels) == 2:
        return recall_score(y_true, prediction, pos_label=0, zero_division=0)

    matrix = confusion_matrix(y_true, prediction, labels=labels)
    specificities = []
    total = matrix.sum()
    for index in range(len(labels)):
        true_negative = total - (matrix[index, :].sum() + matrix[:, index].sum() - matrix[index, index])
        false_positive = matrix[:, index].sum() - matrix[index, index]
        denominator = true_negative + false_positive
        specificities.append(true_negative / denominator if denominator else 0.0)
    return float(sum(specificities) / len(specificities)) if specificities else 0.0


def evaluate_probabilities(y_true: pd.Series, probability) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    probability_series = pd.Series(probability)
    metrics = {
        "accuracy": accuracy_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "kappa": cohen_kappa_score(y_true, prediction),
        "sensitivity": recall_score(y_true, prediction, zero_division=0),
        "specificity": specificity_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "brier_score": brier_score_loss(y_true, probability),
        "extreme_probability_rate": float(((probability_series <= 0.01) | (probability_series >= 0.99)).mean()),
    }
    metrics["roc_auc"] = roc_auc_score(y_true, probability) if len(set(y_true)) > 1 else float("nan")
    return metrics


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    probability = model.predict_proba(x_test)[:, 1]
    return evaluate_probabilities(y_test, probability)


def score_for_selection(metrics: dict[str, float]) -> tuple[float, float, float, float]:
    roc_auc = metrics["roc_auc"]
    if roc_auc != roc_auc:  # NaN check
        roc_auc = 0.0
    return (roc_auc, metrics["f1"], -metrics.get("extreme_probability_rate", 0.0), -metrics["brier_score"])


def cross_validation_strategy(y_train: pd.Series) -> StratifiedKFold | None:
    if y_train.nunique() != 2:
        return None
    minority_count = int(y_train.value_counts().min())
    if minority_count < 2:
        return None
    return StratifiedKFold(n_splits=min(5, minority_count), shuffle=True, random_state=42)


def cross_validated_metrics(model: Pipeline, x_train: pd.DataFrame, y_train: pd.Series) -> dict[str, float]:
    cv = cross_validation_strategy(y_train)
    if cv is None:
        model.fit(x_train, y_train)
        return evaluate_model(model, x_train, y_train)
    probability = cross_val_predict(model, x_train, y_train, cv=cv, method="predict_proba")[:, 1]
    return evaluate_probabilities(y_train, probability)


def split_training_and_calibration(x_train: pd.DataFrame, y_train: pd.Series):
    can_calibrate = y_train.nunique() == 2 and y_train.value_counts().min() >= 3 and len(y_train) >= 12
    if not can_calibrate:
        return x_train, None, y_train, None
    return train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )


def dataset_warning(sample_count: int) -> str:
    if sample_count < 50:
        return (
            "Prototype warning: this training file is very small. Metrics are useful for checking the "
            "pipeline, but not for research claims until the real dataset is collected."
        )
    return ""


def train(data_path: Path, model_path: Path = MODEL_PATH) -> dict[str, object]:
    df = pd.read_csv(data_path)
    validate_training_frame(df)
    df = add_engineered_features(df)

    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    x = df[all_features]
    y = df[TARGET_COLUMN].astype(int)

    stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    validation_results = {}
    trained_models = {}
    for name, model in candidate_models(y_train).items():
        validation_results[name] = cross_validated_metrics(model, x_train, y_train)
        x_fit, x_calibration, y_fit, y_calibration = split_training_and_calibration(x_train, y_train)
        model.fit(x_fit, y_fit)
        if x_calibration is not None and y_calibration is not None:
            model = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
            model.fit(x_calibration, y_calibration)
        trained_models[name] = model

    best_name = max(validation_results, key=lambda name: score_for_selection(validation_results[name]))
    best_model = trained_models[best_name]
    test_results = {name: evaluate_model(model, x_test, y_test) for name, model in trained_models.items()}

    artifact = {
        "model": best_model,
        "base_model_name": best_name,
        "metrics": validation_results,
        "test_metrics": test_results,
        "feature_columns": list(x.columns),
        "target_column": TARGET_COLUMN,
        "used_smote": can_use_smote(y_train),
        "candidate_methods": list(validation_results.keys()),
        "xgboost_status": "enabled" if XGBClassifier is not None else f"disabled: {XGBOOST_IMPORT_ERROR}",
        "reference_notes": REFERENCE_NOTES,
        "selection_rule": "Choose highest cross-validated ROC-AUC, then F1, then fewer extreme probabilities, then lowest Brier score; report Accuracy, Kappa, Sensitivity, and Specificity like obesity ML comparison papers.",
        "validation_strategy": "Stratified cross-validation on the training split; final test split is kept for reporting only.",
        "resampling_strategy": "SMOTENC before preprocessing for categorical-safe balancing." if can_use_smote(y_train) else "Class weights/fallback only; SMOTENC skipped because data is too small or unavailable.",
        "dataset_warning": dataset_warning(len(df)),
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
    print(f"SMOTENC used: {artifact['used_smote']}")
    print(f"Selection rule: {artifact['selection_rule']}")
    print(f"Validation strategy: {artifact['validation_strategy']}")
    print(f"Resampling strategy: {artifact['resampling_strategy']}")
    if artifact["dataset_warning"]:
        print(artifact["dataset_warning"])
    print("Cross-validation metrics used for selection:")
    for name, metrics in artifact["metrics"].items():
        print(f"  {name}: {metrics}")
    print("Final hold-out test metrics for reporting only:")
    for name, metrics in artifact["test_metrics"].items():
        print(f"  {name}: {metrics}")


if __name__ == "__main__":
    main()
