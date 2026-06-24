# Scikit-Learn Model Performance Analysis Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or `superpowers:subagent-driven-development` to execute this plan task-by-task. Keep this file updated by changing each checkbox from `[ ]` to `[x]` immediately after the step is finished.

**Goal:** Build a repeatable analysis workflow that measures every obesity-risk model with fair validation metrics, exports research-ready result tables, and clearly separates lifestyle-model performance from the final BMI-plus-lifestyle app output.

**Current App Idea To Preserve:** The ML model learns from lifestyle and survey features only. BMI, height, and weight are kept outside the ML model and blended afterward as a separate screening signal:

```text
final_probability = 0.5 * lifestyle_model_probability + 0.5 * bmi_screen_score
```

**Main Output:** A reproducible evaluation report saved under:

```text
/Users/phawichpilathong/Documents/Obesity machine learning/reports/model_performance/
```

The report should contain CSV tables for presentation/research and a Markdown summary explaining which model performs best on the current dataset.

## Current Reference Code Already Executing This Idea

Use these code refs when explaining or implementing the evaluation workflow.

### 1. Lifestyle features exclude body screen features

Reference:

```text
/Users/phawichpilathong/Documents/Obesity machine learning/src/obesity_ml/config.py:23
/Users/phawichpilathong/Documents/Obesity machine learning/src/obesity_ml/config.py:37
```

Key code:

```python
BODY_SCREEN_FEATURES = [
    "height_cm",
    "weight_kg",
    "bmi",
]

NUMERIC_FEATURES = [
    "age",
    "physical_activity_hours_per_week",
    "screen_time_hours_per_day",
    "sleep_hours",
    "fast_food_meals_per_week",
    "sugary_drinks_per_day",
    ...
]
```

Meaning: `height_cm`, `weight_kg`, and `bmi` are not part of `NUMERIC_FEATURES`, so they are not direct ML inputs.

### 2. BMI gets engineered, but used outside model

Reference:

```text
/Users/phawichpilathong/Documents/Obesity machine learning/src/obesity_ml/features.py:51
```

Key code:

```python
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    height_m = out["height_cm"].astype(float).clip(lower=1.0) / 100
    out["bmi"] = out["weight_kg"].astype(float) / (height_m**2)
    ...
    return out
```

Meaning: BMI is calculated consistently from user body data, then kept as separate screening signal.

### 3. SMOTENC is balancing method, not algorithm

Reference:

```text
/Users/phawichpilathong/Documents/Obesity machine learning/src/obesity_ml/train.py:108
```

Key code:

```python
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
```

Meaning: SMOTENC happens before preprocessing and before each model. It balances training data; it is not listed as model candidate.

### 4. Candidate algorithms compared in same tournament

Reference:

```text
/Users/phawichpilathong/Documents/Obesity machine learning/src/obesity_ml/train.py:129
```

Key code:

```python
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
```

Meaning: each algorithm receives same prepared feature set and same balancing/preprocessing pipeline.

### 5. Metrics already calculated from probabilities

Reference:

```text
/Users/phawichpilathong/Documents/Obesity machine learning/src/obesity_ml/train.py:198
```

Key code:

```python
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
```

Meaning: plan extends existing metrics with `average_precision`, confusion counts, CSV export, and clearer report output.

### 6. Fair validation and selection rule

Reference:

```text
/Users/phawichpilathong/Documents/Obesity machine learning/src/obesity_ml/train.py:222
/Users/phawichpilathong/Documents/Obesity machine learning/src/obesity_ml/train.py:229
/Users/phawichpilathong/Documents/Obesity machine learning/src/obesity_ml/train.py:238
```

Key code:

```python
def score_for_selection(metrics: dict[str, float]) -> tuple[float, float, float, float]:
    roc_auc = metrics["roc_auc"]
    if roc_auc != roc_auc:
        roc_auc = 0.0
    return (roc_auc, metrics["f1"], -metrics.get("extreme_probability_rate", 0.0), -metrics["brier_score"])

def cross_validation_strategy(y_train: pd.Series) -> StratifiedKFold | None:
    ...
    return StratifiedKFold(n_splits=min(5, minority_count), shuffle=True, random_state=42)

def cross_validated_evaluation(
    model: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[dict[str, float], dict[str, list[float]] | None]:
    ...
    probability = cross_val_predict(model, x_train, y_train, cv=cv, method="predict_proba")[:, 1]
    metrics = evaluate_probabilities(y_train, probability)
```

Meaning: best model is chosen from cross-validation, not from one direct test guess.

### 7. Final app output blends lifestyle ML with BMI screen

Reference:

```text
/Users/phawichpilathong/Documents/Obesity machine learning/src/obesity_ml/predict.py:21
/Users/phawichpilathong/Documents/Obesity machine learning/src/obesity_ml/predict.py:32
/Users/phawichpilathong/Documents/Obesity machine learning/src/obesity_ml/predict.py:38
```

Key code:

```python
def bmi_screen_score(
    bmi: float,
    midpoint: float = BMI_SCREEN_MIDPOINT,
    steepness: float = BMI_SCREEN_STEEPNESS,
) -> float:
    return 1 / (1 + exp(-(float(bmi) - midpoint) / steepness))

def blend_lifestyle_and_bmi_probabilities(lifestyle_probability: float, bmi_score: float) -> float:
    lifestyle = max(0.0, min(1.0, float(lifestyle_probability)))
    bmi_signal = max(0.0, min(1.0, float(bmi_score)))
    return (0.5 * lifestyle) + (0.5 * bmi_signal)

def predict_probability(input_data: dict, model_path: Path = MODEL_PATH) -> dict:
    ...
    lifestyle_probability = float(model.predict_proba(df)[0, 1])
    bmi_score = bmi_screen_score(float(df.iloc[0]["bmi"]))
    probability = blend_lifestyle_and_bmi_probabilities(lifestyle_probability, bmi_score)
```

Meaning: evaluation report must measure model-only performance and final blended app-output performance separately.

### 8. Final probability becomes five user-facing tiers

Reference:

```text
/Users/phawichpilathong/Documents/Obesity machine learning/src/obesity_ml/risk_tiers.py:13
/Users/phawichpilathong/Documents/Obesity machine learning/src/obesity_ml/risk_tiers.py:52
```

Key code:

```python
PROBABILITY_TIERS = [
    ProbabilityTier(key="very_low", label="Very low risk tier", lower=0.0, upper=0.20, ...),
    ProbabilityTier(key="low", label="Low risk tier", lower=0.20, upper=0.40, ...),
    ProbabilityTier(key="moderate", label="Moderate risk tier", lower=0.40, upper=0.60, ...),
    ProbabilityTier(key="high", label="High risk tier", lower=0.60, upper=0.80, ...),
    ProbabilityTier(key="very_high", label="Very high risk tier", lower=0.80, upper=1.01, ...),
]

def classify_probability(probability: float) -> dict[str, object]:
    bounded = max(0.0, min(1.0, float(probability)))
    for tier in PROBABILITY_TIERS:
        if tier.lower <= bounded < tier.upper:
            return {
                "risk_tier": tier.key,
                "risk_tier_label": tier.label,
                "risk_tier_range": _format_range(tier.lower, tier.upper),
                "risk_tier_explanation": tier.explanation,
                "risk_band": coarse_band_from_probability(bounded),
            }
```

Meaning: final evaluation can also report tier distribution, not only raw probability metrics.

## Files To Create Or Edit

- Create `/Users/phawichpilathong/Documents/Obesity machine learning/src/obesity_ml/evaluate.py`
  - Dedicated model-performance analysis module.
  - Runs all model candidates through cross-validation and holdout evaluation.
  - Exports metrics, confusion matrices, ROC curve points, and a Markdown summary.

- Edit `/Users/phawichpilathong/Documents/Obesity machine learning/src/obesity_ml/train.py`
  - Reuse current candidate model definitions.
  - Add a KNN candidate only if we decide the project should compare an instance-based model too.
  - Keep SMOTENC as a balancing method, not an algorithm.

- Create `/Users/phawichpilathong/Documents/Obesity machine learning/tests/test_model_evaluation.py`
  - Tests for metric calculation, report export, and BMI-plus-lifestyle blended evaluation.

- Edit `/Users/phawichpilathong/Documents/Obesity machine learning/docs/ARCHITECTURE.md`
  - Add the performance-analysis workflow and report output path.

- Edit `/Users/phawichpilathong/Documents/Obesity machine learning/docs/TASKS.md`
  - Add completed/remaining research-evaluation tasks.

## Metrics We Should Use

Do not judge the models by accuracy alone because the current dataset is imbalanced. The positive obesity-risk group is much smaller than the non-risk group, so a model can look accurate by mostly predicting the majority class.

Use these metrics:

- `ROC-AUC`: How well the model separates risk and non-risk cases across thresholds.
- `Average Precision / PR-AUC`: More informative when the positive class is rare.
- `F1-score`: Balance between precision and recall.
- `Sensitivity / Recall`: How many true risk cases the model catches.
- `Specificity`: How many non-risk cases the model correctly avoids flagging.
- `Balanced Accuracy`: Average of sensitivity and specificity.
- `Cohen's Kappa`: Agreement beyond random guessing.
- `Brier Score`: Probability calibration error; lower is better.
- `Confusion Matrix`: Counts of TP, FP, TN, FN for presentation.

## Model Candidates

Use the current tournament models:

- Logistic Regression
- Support Vector Machine with RBF kernel
- Random Forest
- Gaussian Naive Bayes
- Neural Network MLP
- XGBoost Gradient Boosting, if installed

Recommended additional scikit-learn model:

- K-Nearest Neighbors

KNN should be tuned by validation because it is instance-based learning. It does not learn coefficients like logistic regression. Its key parameter is `n_neighbors`, meaning how many nearby training examples vote for the prediction. Test several values such as `3`, `5`, `7`, `11`, and `15`, then select the value that gives the best validation performance without overfitting.

## Implementation Steps

- [ ] Step 1: Add metric helper tests first.

  Create `/Users/phawichpilathong/Documents/Obesity machine learning/tests/test_model_evaluation.py` with tests that verify metric calculation includes both classification metrics and probability metrics.

  Expected helper API:

  ```python
  from obesity_ml.evaluate import evaluate_probability_predictions

  row = evaluate_probability_predictions(
      model_name="demo",
      split_name="unit",
      y_true=[0, 0, 1, 1],
      y_probability=[0.05, 0.40, 0.70, 0.90],
      threshold=0.5,
  )
  ```

  Required output keys:

  ```python
  {
      "model",
      "split",
      "roc_auc",
      "average_precision",
      "accuracy",
      "balanced_accuracy",
      "kappa",
      "sensitivity",
      "specificity",
      "precision",
      "f1",
      "brier_score",
      "tn",
      "fp",
      "fn",
      "tp",
  }
  ```

- [ ] Step 2: Implement `evaluate_probability_predictions`.

  In `/Users/phawichpilathong/Documents/Obesity machine learning/src/obesity_ml/evaluate.py`, use scikit-learn metrics:

  ```python
  from sklearn.metrics import (
      accuracy_score,
      average_precision_score,
      balanced_accuracy_score,
      brier_score_loss,
      cohen_kappa_score,
      confusion_matrix,
      f1_score,
      precision_score,
      recall_score,
      roc_auc_score,
  )
  ```

  Implementation rules:

  - Convert probabilities to labels with threshold `0.5`.
  - Sensitivity is `recall_score(y_true, y_pred, pos_label=1)`.
  - Specificity is recall for class `0`.
  - Use `zero_division=0` for precision/recall/F1.
  - If ROC-AUC or average precision cannot be calculated because only one class exists, return `None` for that metric instead of crashing.

- [ ] Step 3: Add cross-validation evaluation for every candidate model.

  Add a function:

  ```python
  def evaluate_candidate_models(data_path: Path, threshold: float = 0.5) -> EvaluationResult:
      ...
  ```

  It should:

  - Load the current training CSV.
  - Validate required columns.
  - Keep BMI, height, and weight outside the ML feature matrix.
  - Use lifestyle/survey features for model prediction.
  - Split train/test as `75% / 25%` with stratification when possible.
  - Use Stratified K-Fold cross-validation on the training set.
  - Apply SMOTENC only inside training folds, never on the test set.
  - Train every candidate model with the same preprocessing logic.
  - Return cross-validation metrics and holdout-test metrics for each model.

  Expected result object:

  ```python
  @dataclass
  class EvaluationResult:
      metrics: list[dict[str, Any]]
      confusion_matrices: list[dict[str, Any]]
      roc_curves: list[dict[str, Any]]
      best_model_name: str
      dataset_rows: int
      positive_count: int
      negative_count: int
  ```

- [ ] Step 4: Evaluate the final app probability separately.

  Add a function:

  ```python
  def evaluate_final_blend(
      lifestyle_probability: Sequence[float],
      bmi_screen_score: Sequence[float],
      y_true: Sequence[int],
  ) -> dict[str, Any]:
      ...
  ```

  Logic:

  ```python
  final_probability = 0.5 * lifestyle_probability + 0.5 * bmi_screen_score
  ```

  This produces a separate row named `final_bmi_lifestyle_blend`. This matters because the app output is not only the ML model; it is the ML model plus BMI screening.

- [ ] Step 5: Export research-ready files.

  Add:

  ```python
  def write_evaluation_report(result: EvaluationResult, output_dir: Path) -> None:
      ...
  ```

  Required outputs:

  ```text
  reports/model_performance/model_metrics.csv
  reports/model_performance/confusion_matrices.csv
  reports/model_performance/roc_curves.csv
  reports/model_performance/model_performance_summary.md
  ```

  The Markdown summary must state:

  - Dataset row count.
  - Positive and negative class counts.
  - Best model for current training data only.
  - Top metrics for the best model.
  - Warning that current results are prototype results if the target label is BMI-derived.
  - Note that doctor diagnosis labels are needed for final medical research conclusions.

- [ ] Step 6: Add a command-line entry point.

  Support:

  ```bash
  python -m obesity_ml.evaluate \
    --data "/Users/phawichpilathong/Documents/Obesity machine learning/data/processed/uci_obesity_training.csv" \
    --out-dir "/Users/phawichpilathong/Documents/Obesity machine learning/reports/model_performance"
  ```

  It should print:

  ```text
  Evaluation complete.
  Best current model: <model_name>
  Metrics saved to: reports/model_performance/model_metrics.csv
  ```

- [ ] Step 7: Decide whether to add KNN to the tournament.

  If added, edit `/Users/phawichpilathong/Documents/Obesity machine learning/src/obesity_ml/train.py`.

  Candidate configuration:

  ```python
  from sklearn.neighbors import KNeighborsClassifier

  "knn_distance": KNeighborsClassifier(
      n_neighbors=7,
      weights="distance",
      metric="minkowski",
      p=2,
  )
  ```

  Better version for research:

  - Evaluate multiple K values: `3`, `5`, `7`, `11`, `15`.
  - Use cross-validation to choose K.
  - Do not claim one K is best for all datasets.

- [ ] Step 8: Update docs.

  In `/Users/phawichpilathong/Documents/Obesity machine learning/docs/ARCHITECTURE.md`, add:

  ```text
  Model evaluation is produced by src/obesity_ml/evaluate.py. It compares candidate models with stratified cross-validation and holdout testing. The report separates lifestyle-model metrics from final BMI-plus-lifestyle app-output metrics.
  ```

  In `/Users/phawichpilathong/Documents/Obesity machine learning/docs/TASKS.md`, add:

  ```text
  Next: rerun model evaluation after doctor diagnosis labels are collected.
  ```

- [ ] Step 9: Run verification.

  Run:

  ```bash
  cd "/Users/phawichpilathong/Documents/Obesity machine learning"
  python -m pytest
  python -m obesity_ml.evaluate --data data/processed/uci_obesity_training.csv --out-dir reports/model_performance
  ```

  Check the exported CSV:

  ```bash
  python - <<'PY'
  import pandas as pd
  df = pd.read_csv("reports/model_performance/model_metrics.csv")
  print(df[["model", "split", "roc_auc", "average_precision", "f1", "sensitivity", "specificity", "brier_score"]])
  PY
  ```

## How To Explain This In Presentation

Use this short explanation:

```text
We compare several machine-learning models using the same dataset split and the same validation method. Because obesity-risk data can be imbalanced, we do not rely on accuracy alone. We also measure ROC-AUC, F1-score, sensitivity, specificity, balanced accuracy, Cohen's Kappa, and Brier score. The best model is selected only for the current training data. The final app probability combines two signals: 50% from the lifestyle machine-learning model and 50% from BMI screening.
```

## Acceptance Criteria

- All candidate models have rows in `model_metrics.csv`.
- Cross-validation and holdout-test results are clearly separated.
- SMOTENC is described as a data-balancing method, not as an algorithm.
- BMI, height, and weight are not used as ML input features.
- The final blended app probability has its own metric row.
- The report warns when labels are prototype BMI-derived labels.
- `python -m pytest` passes.
- The evaluation command runs without errors.
