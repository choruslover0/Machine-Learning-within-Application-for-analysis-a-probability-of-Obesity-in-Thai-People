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
        response = client.get("/static/obeast-logo.svg")

        self.assertEqual(response.status_code, 200)
        self.assertIn("svg", response.headers["content-type"])

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
                "physical_activity_hours_per_week": 3,
                "screen_time_hours_per_day": 5,
                "sleep_hours": 7,
                "fast_food_meals_per_week": 2,
                "sugary_drinks_per_day": 1,
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
                "physical_activity_hours_per_week": "-10",
                "screen_time_hours_per_day": "100",
                "sleep_hours": "-2",
                "fast_food_meals_per_week": "-1",
                "sugary_drinks_per_day": "-5",
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
                "physical_activity_hours_per_week": "3",
                "screen_time_hours_per_day": "5",
                "sleep_hours": "7",
                "fast_food_meals_per_week": "2",
                "sugary_drinks_per_day": "1",
                "family_history_obesity": "0",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("Estimated Probability", response.text)


class TrainingPipelineTests(unittest.TestCase):
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
                "physical_activity_hours_per_week": 3.0,
                "screen_time_hours_per_day": 5.0, "sleep_hours": 7.0,
                "fast_food_meals_per_week": 2, "sugary_drinks_per_day": 1.0,
                "family_history_obesity": 0, "obesity": 0,
            })
        for i in range(3):
            rows.append({
                "age": 30 + i, "sex": "M", "height_cm": 160.0,
                "weight_kg": 100.0 + i,
                "physical_activity_hours_per_week": 0.5,
                "screen_time_hours_per_day": 10.0, "sleep_hours": 5.0,
                "fast_food_meals_per_week": 7, "sugary_drinks_per_day": 3.0,
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

    def test_methods_page_renders(self):
        """GET /methods must return 200 and mention model-selection content."""
        response = self._client().get("/methods")
        self.assertEqual(response.status_code, 200)
        self.assertIn("How O-Beast Checks Results", response.text)

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
                "physical_activity_hours_per_week": "4",
                "screen_time_hours_per_day": "6",
                "sleep_hours": "7",
                "fast_food_meals_per_week": "3",
                "sugary_drinks_per_day": "2",
                "family_history_obesity": "1",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Your Wellness Advice", response.text)
        self.assertIn("advice-card", response.text)


class RiskTierTests(unittest.TestCase):
    def test_classify_probability_covers_all_five_tiers(self):
        """Every tier boundary must resolve to the right key."""
        from obesity_ml.risk_tiers import classify_probability

        cases = [
            (0.00, "very_low"),
            (0.10, "very_low"),
            (0.20, "low"),
            (0.39, "low"),
            (0.40, "moderate"),
            (0.59, "moderate"),
            (0.60, "high"),
            (0.79, "high"),
            (0.80, "very_high"),
            (1.00, "very_high"),
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
            "physical_activity_hours_per_week": 6,
            "screen_time_hours_per_day": 2,
            "sleep_hours": 8,
            "fast_food_meals_per_week": 1,
            "sugary_drinks_per_day": 0,
            "family_history_obesity": 0,
        }
        advice = generate_advice(ideal_input, {"obesity_probability": 0.08})
        self.assertGreater(len(advice["cards"]), 0)
        self.assertIn("bmi", advice)
        self.assertIn("sources", advice)

    def test_generate_advice_disclaimer_says_not_a_diagnosis(self):
        """Disclaimer must contain the safety phrase 'not a diagnosis' per CLAUDE.md rules."""
        from obesity_ml.advice import generate_advice

        advice = generate_advice(
            {"age": 16, "sex": "F", "height_cm": 160, "weight_kg": 80,
             "physical_activity_hours_per_week": 0.5, "screen_time_hours_per_day": 10,
             "sleep_hours": 5, "fast_food_meals_per_week": 7, "sugary_drinks_per_day": 3,
             "family_history_obesity": 1},
            {"obesity_probability": 0.92},
        )
        self.assertIn("disclaimer", advice)
        self.assertIn("not a diagnosis", advice["disclaimer"].lower())


if __name__ == "__main__":
    unittest.main()
