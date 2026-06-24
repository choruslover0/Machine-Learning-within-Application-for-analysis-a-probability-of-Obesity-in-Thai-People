import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


class AppRegressionTests(unittest.TestCase):
    def test_testclient_import_does_not_emit_starlette_deprecation_warning(self):
        import importlib
        import sys
        import warnings
        try:
            from starlette.exceptions import StarletteDeprecationWarning
        except ImportError:
            return

        sys.modules.pop("fastapi.testclient", None)
        sys.modules.pop("starlette.testclient", None)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", StarletteDeprecationWarning)
            importlib.import_module("fastapi.testclient")

        messages = [
            str(warning.message)
            for warning in caught
            if issubclass(warning.category, StarletteDeprecationWarning)
        ]
        self.assertEqual(messages, [])

    def test_app_imports_from_non_project_working_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC_DIR)
            result = subprocess.run(
                [sys.executable, "-c", "import obesity_ml.app; print('ok')"],
                cwd=tmpdir,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ok", result.stdout)

    def test_installed_config_uses_project_working_directory_for_model_path(self):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from obesity_ml.config import MODEL_PATH; print(MODEL_PATH)",
            ],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            str(PROJECT_ROOT / "models" / "obesity_probability_model.joblib"),
        )

    def test_static_logo_route_works(self):
        from fastapi.testclient import TestClient
        from obesity_ml.app import app

        client = TestClient(app)
        response = client.get("/static/obeast-logo.png")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")

    def test_predict_api_rejects_invalid_json_values(self):
        from fastapi.testclient import TestClient
        from obesity_ml.app import app

        client = TestClient(app)
        response = client.post(
            "/predict",
            json={
                "age": 16,
                "sex": "X",
                "height_cm": 0,
                "weight_kg": 65,
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
                "transportation": "Public_Transportation",
                "family_history_obesity": 0,
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_predict_form_rejects_invalid_values(self):
        from fastapi.testclient import TestClient
        from obesity_ml.app import app

        client = TestClient(app)
        response = client.post(
            "/predict-form",
            data={
                "age": "16",
                "sex": "X",
                "height_cm": "0",
                "weight_kg": "65",
                "high_calorie_food_frequency": "1",
                "vegetable_frequency": "2",
                "main_meals_per_day": "3",
                "food_between_meals_frequency": "1",
                "smoke": "0",
                "water_daily": "2",
                "calorie_monitoring": "0",
                "physical_activity_freq": "2",
                "screen_time_band": "1",
                "alcohol_frequency": "0",
                "transportation": "Public_Transportation",
                "family_history_obesity": "9",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("height_cm", response.text)
        self.assertNotIn("Estimated Probability", response.text)

    def test_predict_form_accepts_valid_values(self):
        from fastapi.testclient import TestClient
        from obesity_ml.app import app

        client = TestClient(app)
        response = client.post(
            "/predict-form",
            data={
                "age": "16",
                "sex": "M",
                "height_cm": "170",
                "weight_kg": "65",
                "high_calorie_food_frequency": "1",
                "vegetable_frequency": "2",
                "main_meals_per_day": "3",
                "food_between_meals_frequency": "1",
                "smoke": "0",
                "water_daily": "2",
                "calorie_monitoring": "0",
                "physical_activity_freq": "2",
                "screen_time_band": "1",
                "alcohol_frequency": "0",
                "transportation": "Public_Transportation",
                "family_history_obesity": "0",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Estimated Probability", response.text)

    def test_predict_api_accepts_uci_style_values(self):
        from fastapi.testclient import TestClient
        from obesity_ml.app import app

        client = TestClient(app)
        response = client.post(
            "/predict",
            json={
                "age": 21,
                "sex": "F",
                "height_cm": 162,
                "weight_kg": 64,
                "physical_activity_freq": 2,
                "screen_time_band": 1,
                "family_history_obesity": 1,
                "high_calorie_food_frequency": 0,
                "vegetable_frequency": 2,
                "main_meals_per_day": 3,
                "food_between_meals_frequency": 1,
                "smoke": 0,
                "water_daily": 2,
                "calorie_monitoring": 0,
                "alcohol_frequency": 0,
                "transportation": "Public_Transportation",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("obesity_probability", response.json())


class TrainingPipelineTests(unittest.TestCase):
    def test_training_artifact_excludes_body_features_from_lifestyle_model(self):
        import tempfile
        import pandas as pd
        from pathlib import Path
        from obesity_ml.train import train

        rows = []
        for i in range(60):
            rows.append({
                "age": 16 + (i % 4), "sex": "M" if i % 2 else "F",
                "height_cm": 160.0 + (i % 12), "weight_kg": 50.0 + i,
                "physical_activity_freq": float(i % 4),
                "screen_time_band": i % 3,
                "high_calorie_food_frequency": i % 2,
                "vegetable_frequency": 1 + (i % 3),
                "main_meals_per_day": 1 + (i % 4),
                "food_between_meals_frequency": i % 4,
                "smoke": 0,
                "water_daily": 1 + (i % 3),
                "calorie_monitoring": i % 2,
                "alcohol_frequency": i % 4,
                "transportation": "Public_Transportation",
                "family_history_obesity": i % 2,
                "obesity": 0,
            })
        for i in range(40):
            rows.append({
                "age": 17 + (i % 3), "sex": "M" if i % 2 else "F",
                "height_cm": 155.0 + (i % 10), "weight_kg": 85.0 + i,
                "physical_activity_freq": float(i % 2),
                "screen_time_band": 2,
                "high_calorie_food_frequency": 1,
                "vegetable_frequency": 1 + (i % 2),
                "main_meals_per_day": 3,
                "food_between_meals_frequency": 2 + (i % 2),
                "smoke": 0,
                "water_daily": 1 + (i % 2),
                "calorie_monitoring": 0,
                "alcohol_frequency": 1 + (i % 2),
                "transportation": "Automobile",
                "family_history_obesity": 1,
                "obesity": 1,
            })

        df = pd.DataFrame(rows)
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            csv_path = Path(f.name)
            df.to_csv(csv_path, index=False)
        model_path = csv_path.with_suffix(".joblib")
        try:
            artifact = train(csv_path, model_path=model_path)
            self.assertNotIn("height_cm", artifact["feature_columns"])
            self.assertNotIn("weight_kg", artifact["feature_columns"])
            self.assertNotIn("bmi", artifact["feature_columns"])
            self.assertIn("physical_activity_freq", artifact["feature_columns"])
            self.assertIn("bmi_screen_score", artifact["probability_blend_strategy"])
        finally:
            csv_path.unlink(missing_ok=True)
            model_path.unlink(missing_ok=True)

    def test_predict_probability_blends_lifestyle_model_with_bmi_screen_score(self):
        import numpy as np
        from unittest.mock import patch
        from obesity_ml.predict import predict_probability

        class FixedLifestyleModel:
            def predict_proba(self, _frame):
                return np.array([[0.8, 0.2]])

        artifact = {
            "model": FixedLifestyleModel(),
            "base_model_name": "fixed_lifestyle_model",
            "candidate_methods": [],
            "used_smote": False,
            "selection_rule": "",
            "validation_strategy": "",
            "resampling_strategy": "",
            "dataset_warning": "",
            "metrics": {},
            "test_metrics": {},
            "disclaimer": "Educational risk-estimation model only; not a medical diagnosis.",
        }
        input_data = {
            "age": 16,
            "sex": "M",
            "height_cm": 200,
            "weight_kg": 100,
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
            "transportation": "Public_Transportation",
            "family_history_obesity": 0,
        }

        with patch("obesity_ml.predict.load_artifact", return_value=artifact):
            result = predict_probability(input_data)

        self.assertEqual(result["lifestyle_probability"], 0.2)
        self.assertEqual(result["bmi_screen_score"], 0.5)
        self.assertEqual(result["obesity_probability"], 0.35)
        self.assertIn("50% lifestyle", result["probability_blend_strategy"])

    def test_cross_validated_evaluation_returns_real_roc_points(self):
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from obesity_ml.train import cross_validated_evaluation

        x = pd.DataFrame({"signal": [-3, -2, -1, -0.5, 0.5, 1, 2, 3]})
        y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

        metrics, curve = cross_validated_evaluation(LogisticRegression(), x, y)

        self.assertIn("roc_auc", metrics)
        self.assertIsNotNone(curve)
        self.assertEqual(curve["false_positive_rate"][0], 0.0)
        self.assertEqual(curve["true_positive_rate"][0], 0.0)
        self.assertEqual(curve["false_positive_rate"][-1], 1.0)
        self.assertEqual(curve["true_positive_rate"][-1], 1.0)
        for point in curve["false_positive_rate"] + curve["true_positive_rate"]:
            self.assertGreaterEqual(point, 0.0)
            self.assertLessEqual(point, 1.0)

    def test_clamp_smotenc_k_neighbors_adjusts_to_fit_data_size(self):
        """RED for Bug 4: k_neighbors is computed from y_train but the model is
        fitted on y_fit (80% after calibration split). _clamp_smotenc_k_neighbors
        must tighten k_neighbors to y_fit's minority count minus 1 so SMOTENC
        never requests more neighbors than samples available at fit time."""
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from obesity_ml.train import _clamp_smotenc_k_neighbors, make_pipeline

        # y_train has 5 minority → k_neighbors = max(1, min(5, 5-2)) = 3
        y_train = pd.Series([1] * 5 + [0] * 10)
        pipeline = make_pipeline(LogisticRegression(), y_train)

        # y_fit after 80% calibration split would have only 4 minority
        # but let's test a tighter case: only 3 minority in y_fit
        y_fit = pd.Series([1] * 3 + [0] * 8)
        _clamp_smotenc_k_neighbors(pipeline, y_fit)

        smotenc_step = dict(pipeline.steps).get("smotenc")
        self.assertIsNotNone(smotenc_step, "Pipeline should have a smotenc step")
        # minority=3, so max safe k_neighbors = 3-1 = 2
        self.assertEqual(
            smotenc_step.k_neighbors, 2,
            f"k_neighbors should be clamped to 2 (minority-1), got {smotenc_step.k_neighbors}"
        )

    def test_training_with_minimum_viable_smote_data_completes(self):
        """Regression: 12 total samples (3 minority) is the exact boundary where
        SMOTENC, calibration split, and cross-validation all activate simultaneously.
        Training must succeed without ValueError."""
        import tempfile
        import pandas as pd
        from pathlib import Path
        from obesity_ml.train import train

        rows = []
        for i in range(9):
            rows.append({
                "age": 18 + i, "sex": "M", "height_cm": 170.0,
                "weight_kg": 60.0 + i,
                "physical_activity_freq": 2.0,
                "screen_time_band": 1,
                "high_calorie_food_frequency": 0, "vegetable_frequency": 2,
                "main_meals_per_day": 3, "food_between_meals_frequency": 1,
                "smoke": 0, "water_daily": 2, "calorie_monitoring": 0,
                "alcohol_frequency": 0, "transportation": "Public_Transportation",
                "family_history_obesity": 0, "obesity": 0,
            })
        for i in range(3):
            rows.append({
                "age": 30 + i, "sex": "M", "height_cm": 160.0,
                "weight_kg": 100.0 + i,
                "physical_activity_freq": 0.0,
                "screen_time_band": 2,
                "high_calorie_food_frequency": 1, "vegetable_frequency": 1,
                "main_meals_per_day": 3, "food_between_meals_frequency": 3,
                "smoke": 0, "water_daily": 1, "calorie_monitoring": 0,
                "alcohol_frequency": 2, "transportation": "Automobile",
                "family_history_obesity": 1, "obesity": 1,
            })

        df = pd.DataFrame(rows)
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            csv_path = Path(f.name)
            df.to_csv(csv_path, index=False)
        model_path = csv_path.with_suffix(".joblib")
        try:
            artifact = train(csv_path, model_path=model_path)
            self.assertIsNotNone(artifact)
            self.assertIn("model", artifact)
            self.assertIn("roc_curves", artifact)
            self.assertTrue(artifact["roc_curves"])
        finally:
            csv_path.unlink(missing_ok=True)
            model_path.unlink(missing_ok=True)


class BestModelReasonTests(unittest.TestCase):
    def test_nan_brier_score_does_not_render_nan_in_reason_text(self):
        """Bug 2: isinstance(float('nan'), float) is True, so the old guard
        lets NaN reach :.4f and prints 'Brier score of nan' in the UI."""
        from obesity_ml.app import best_model_reason

        result = {
            "base_model_name": "logistic_regression_balanced",
            "metrics": {
                "logistic_regression_balanced": {
                    "roc_auc": 0.85,
                    "f1": 0.80,
                    "brier_score": float("nan"),
                    "extreme_probability_rate": 0.0,
                },
                "naive_bayes_gaussian": {
                    "roc_auc": 0.70,
                    "f1": 0.65,
                    "brier_score": 0.20,
                    "extreme_probability_rate": 0.0,
                },
            },
        }
        reason = best_model_reason(result)
        self.assertNotIn("nan", reason.lower(), f"Reason must not contain 'nan': {reason}")


class EvaluationDashboardTests(unittest.TestCase):
    def setUp(self):
        self.metrics = {
            "random_forest": {
                "roc_auc": 0.91,
                "f1": 0.84,
                "accuracy": 0.88,
                "kappa": 0.75,
                "sensitivity": 0.82,
                "specificity": 0.90,
            },
            "naive_bayes_gaussian": {
                "roc_auc": float("nan"),
                "f1": 0.71,
                "accuracy": 0.74,
                "kappa": 0.48,
                "sensitivity": 0.76,
                "specificity": 0.72,
            },
        }

    def test_selected_model_banner_uses_stored_metrics(self):
        from obesity_ml.app import selected_model_banner_html

        html = selected_model_banner_html("random_forest", self.metrics)

        self.assertIn("Random Forest", html)
        self.assertIn("ROC-AUC 0.91", html)
        self.assertIn("F1 score 0.84", html)
        self.assertIn("Selected for the current training data", html)

    def test_metric_comparison_chart_includes_six_metrics_and_unavailable_values(self):
        from obesity_ml.app import metric_comparison_chart_html

        html = metric_comparison_chart_html(self.metrics)

        for label in ("ROC-AUC", "F1 score", "Accuracy", "Kappa", "Sensitivity", "Specificity"):
            self.assertIn(label, html)
        self.assertIn("Not available", html)
        self.assertNotIn("nan", html.lower())

    def test_roc_curves_requires_real_stored_points(self):
        from obesity_ml.app import roc_curves_html

        html = roc_curves_html({}, self.metrics)

        self.assertIn("ROC curves are available after retraining the models.", html)
        self.assertNotIn('class="roc-model-curve"', html)

    def test_roc_curves_renders_real_stored_points(self):
        from obesity_ml.app import roc_curves_html

        curves = {
            "random_forest": {
                "false_positive_rate": [0.0, 0.2, 1.0],
                "true_positive_rate": [0.0, 0.8, 1.0],
            }
        }
        html = roc_curves_html(curves, self.metrics)

        self.assertIn('role="img"', html)
        self.assertIn('class="roc-model-curve"', html)
        self.assertIn("Random Forest", html)
        self.assertIn("AUC 0.91", html)
        self.assertNotIn("available after retraining", html)


class RouteRegressionTests(unittest.TestCase):
    def _client(self):
        from fastapi.testclient import TestClient
        from obesity_ml.app import app
        return TestClient(app)

    def test_home_page_renders(self):
        """GET / must return 200 and contain the O-Beast brand."""
        response = self._client().get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("O-Beast", response.text)

    def test_premium_health_ui_contract_renders_on_home_and_predictor(self):
        """Primary user pages should expose the approved premium-health journey."""
        client = self._client()
        home_response = client.get("/")
        predictor_response = client.get("/predictor")

        self.assertEqual(home_response.status_code, 200)
        self.assertIn("Know your risk", home_response.text)
        self.assertIn("premium-health-pattern.png", home_response.text)
        self.assertIn("prefers-reduced-motion", home_response.text)

        self.assertEqual(predictor_response.status_code, 200)
        self.assertIn('class="form-step active"', predictor_response.text)
        self.assertIn('data-step="2"', predictor_response.text)
        self.assertIn('id="formProgress"', predictor_response.text)
        self.assertIn('name="vegetable_frequency"', predictor_response.text)
        self.assertIn('name="transportation"', predictor_response.text)

    def test_result_prioritizes_clear_actions(self):
        """Result should give users practical next steps before research detail."""
        response = self._client().post(
            "/predict-form",
            data={
                "age": "16",
                "sex": "M",
                "height_cm": "170",
                "weight_kg": "65",
                "high_calorie_food_frequency": "1",
                "vegetable_frequency": "2",
                "main_meals_per_day": "3",
                "food_between_meals_frequency": "1",
                "smoke": "0",
                "water_daily": "2",
                "calorie_monitoring": "0",
                "physical_activity_freq": "2",
                "screen_time_band": "1",
                "alcohol_frequency": "0",
                "transportation": "Public_Transportation",
                "family_history_obesity": "0",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Three useful next steps", response.text)
        self.assertIn("What this result means", response.text)

    def test_methods_page_renders(self):
        """GET /methods must return 200 and mention model-selection content."""
        response = self._client().get("/methods")
        self.assertEqual(response.status_code, 200)
        self.assertIn("How O-Beast Checks Results", response.text)
        self.assertIn(".section-grid > * { min-width: 0; }", response.text)
        self.assertIn("overflow-wrap: anywhere", response.text)
        self.assertIn("Selected for the current training data", response.text)
        self.assertNotIn("Library not loaded", response.text)

    def test_methods_page_separates_smotenc_from_prediction_algorithms(self):
        """SMOTENC prepares training data; it must not appear as a prediction algorithm."""
        from obesity_ml.app import methods_html

        response = self._client().get("/methods")
        algorithm_html = methods_html()

        self.assertNotIn("SMOTENC", algorithm_html)
        self.assertIn("SMOTENC is not a prediction algorithm", response.text)
        self.assertIn("Uneven groups", response.text)
        self.assertIn("Balanced training data", response.text)
        self.assertIn("Prediction algorithms learn", response.text)

    def test_model_tournament_details_live_on_methods_page_not_result_page(self):
        """Normal users should see a clean result page; detailed metrics belong on Methods."""
        client = self._client()
        result_response = client.post(
            "/predict-form",
            data={
                "age": "16",
                "sex": "M",
                "height_cm": "170",
                "weight_kg": "65",
                "high_calorie_food_frequency": "1",
                "vegetable_frequency": "2",
                "main_meals_per_day": "3",
                "food_between_meals_frequency": "1",
                "smoke": "0",
                "water_daily": "2",
                "calorie_monitoring": "0",
                "physical_activity_freq": "2",
                "screen_time_band": "1",
                "alcohol_frequency": "0",
                "transportation": "Public_Transportation",
                "family_history_obesity": "0",
            },
        )
        methods_response = client.get("/methods")

        self.assertEqual(result_response.status_code, 200)
        self.assertNotIn("Model tournament scores", result_response.text)
        self.assertNotIn('<table class="metric-table"', result_response.text)
        self.assertEqual(methods_response.status_code, 200)
        self.assertIn("Model tournament scores", methods_response.text)
        self.assertIn('<table class="metric-table"', methods_response.text)
        self.assertIn('class="evaluation-dashboard"', methods_response.text)
        self.assertLess(
            methods_response.text.index('class="metric-table"'),
            methods_response.text.index('class="evaluation-dashboard"'),
        )
        self.assertIn("Selected for the current training data", methods_response.text)

    def test_methods_page_explains_when_roc_curves_require_retraining(self):
        from unittest.mock import patch

        artifact = {
            "base_model_name": "random_forest",
            "metrics": {
                "random_forest": {
                    "roc_auc": 0.91,
                    "f1": 0.84,
                    "accuracy": 0.88,
                    "kappa": 0.75,
                    "sensitivity": 0.82,
                    "specificity": 0.90,
                }
            },
        }

        with patch("obesity_ml.app.load_artifact", return_value=artifact):
            response = self._client().get("/methods")

        self.assertEqual(response.status_code, 200)
        self.assertIn("ROC curves are available after retraining the models.", response.text)
        self.assertNotIn('class="roc-model-curve"', response.text)

    def test_methods_page_renders_real_stored_roc_curves(self):
        from unittest.mock import patch

        artifact = {
            "base_model_name": "random_forest",
            "metrics": {
                "random_forest": {
                    "roc_auc": 0.91,
                    "f1": 0.84,
                    "accuracy": 0.88,
                    "kappa": 0.75,
                    "sensitivity": 0.82,
                    "specificity": 0.90,
                }
            },
            "roc_curves": {
                "random_forest": {
                    "false_positive_rate": [0.0, 0.2, 1.0],
                    "true_positive_rate": [0.0, 0.8, 1.0],
                }
            },
        }

        with patch("obesity_ml.app.load_artifact", return_value=artifact):
            response = self._client().get("/methods")

        self.assertEqual(response.status_code, 200)
        self.assertIn('class="roc-model-curve"', response.text)
        self.assertIn("Random Forest (AUC 0.91)", response.text)
        self.assertNotIn("ROC curves are available after retraining the models.", response.text)

    def test_advice_get_renders_example_cards(self):
        """GET /advice must return 200 with at least one advice card."""
        response = self._client().get("/advice")
        self.assertEqual(response.status_code, 200)
        self.assertIn("advice-card", response.text)

    def test_advice_post_returns_wellness_page(self):
        """POST /advice from result page must return 200 with advice cards."""
        response = self._client().post(
            "/advice",
            data={
                "age": "17",
                "sex": "F",
                "height_cm": "162",
                "weight_kg": "58",
                "high_calorie_food_frequency": "1",
                "vegetable_frequency": "2",
                "main_meals_per_day": "3",
                "food_between_meals_frequency": "1",
                "smoke": "0",
                "water_daily": "2",
                "calorie_monitoring": "0",
                "physical_activity_freq": "2",
                "screen_time_band": "1",
                "alcohol_frequency": "0",
                "transportation": "Public_Transportation",
                "family_history_obesity": "1",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Your Wellness Advice", response.text)
        self.assertIn("advice-card", response.text)


class RiskTierTests(unittest.TestCase):
    def test_classify_probability_covers_all_seven_tiers(self):
        """Every UCI-style tier boundary must resolve to the right key."""
        from obesity_ml.risk_tiers import classify_probability

        cases = [
            (0.00, "insufficient_weight"),
            (0.13, "insufficient_weight"),
            (0.14, "normal_weight"),
            (0.27, "normal_weight"),
            (0.28, "overweight_level_i"),
            (0.42, "overweight_level_i"),
            (0.43, "overweight_level_ii"),
            (0.56, "overweight_level_ii"),
            (0.57, "obesity_type_i"),
            (0.70, "obesity_type_i"),
            (0.71, "obesity_type_ii"),
            (0.85, "obesity_type_ii"),
            (0.86, "obesity_type_iii"),
            (1.00, "obesity_type_iii"),
        ]
        for prob, expected_key in cases:
            with self.subTest(prob=prob):
                result = classify_probability(prob)
                self.assertEqual(
                    result["risk_tier"], expected_key,
                    f"P={prob} → expected '{expected_key}', got '{result['risk_tier']}'"
                )

    def test_classify_probability_never_returns_nan_fields(self):
        """No field in the returned dict should be NaN for any probability in [0,1]."""
        import math
        from obesity_ml.risk_tiers import classify_probability

        for p in [0.0, 0.19, 0.20, 0.399, 0.40, 0.599, 0.60, 0.799, 0.80, 1.0]:
            result = classify_probability(p)
            for key, val in result.items():
                if isinstance(val, float):
                    self.assertFalse(math.isnan(val), f"NaN in field '{key}' at P={p}")


class GenerateAdviceTests(unittest.TestCase):
    def test_generate_advice_always_returns_nonempty_cards(self):
        """Even for an ideal user, cards must never be empty.
        This also proves the 'if not cards' dead-code block in advice.py is unreachable."""
        from obesity_ml.advice import generate_advice

        ideal_input = {
            "age": 20,
            "sex": "M",
            "height_cm": 175,
            "weight_kg": 68,
            "high_calorie_food_frequency": 0,
            "vegetable_frequency": 3,
            "main_meals_per_day": 3,
            "food_between_meals_frequency": 1,
            "smoke": 0,
            "water_daily": 3,
            "calorie_monitoring": 1,
            "physical_activity_freq": 3,
            "screen_time_band": 0,
            "alcohol_frequency": 0,
            "transportation": "Walking",
            "family_history_obesity": 0,
        }
        advice = generate_advice(ideal_input, {"obesity_probability": 0.08})
        self.assertGreater(len(advice["cards"]), 0)
        self.assertIn("bmi", advice)
        self.assertIn("sources", advice)

    def test_generate_advice_disclaimer_says_not_a_diagnosis(self):
        """Disclaimer must contain the safety phrase 'not a diagnosis'."""
        from obesity_ml.advice import generate_advice

        advice = generate_advice(
            {"age": 16, "sex": "F", "height_cm": 160, "weight_kg": 80,
             "physical_activity_freq": 0, "screen_time_band": 2,
             "high_calorie_food_frequency": 1, "vegetable_frequency": 1,
             "main_meals_per_day": 3, "food_between_meals_frequency": 3,
             "smoke": 0, "water_daily": 1, "calorie_monitoring": 0, "alcohol_frequency": 2,
             "transportation": "Automobile",
             "family_history_obesity": 1},
            {"obesity_probability": 0.92},
        )
        self.assertIn("disclaimer", advice)
        self.assertIn("not a diagnosis", advice["disclaimer"].lower())


if __name__ == "__main__":
    unittest.main()
