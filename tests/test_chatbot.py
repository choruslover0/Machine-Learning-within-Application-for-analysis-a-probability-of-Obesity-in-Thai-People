import sys
import unittest
import json
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from fastapi.testclient import TestClient
from obesity_ml.app import app as fastapi_app
from obesity_ml.config import CHATBOT_MODEL_PATH
from obesity_ml.chatbot import get_answer, train_chatbot, chat


class ChatbotConfigTests(unittest.TestCase):
    def test_chatbot_model_path_is_in_models_dir(self):
        self.assertEqual(CHATBOT_MODEL_PATH.parent.name, "models")
        self.assertEqual(CHATBOT_MODEL_PATH.name, "chatbot_model.joblib")


class ChatbotTrainingDataTests(unittest.TestCase):
    def setUp(self):
        self.data_path = PROJECT_ROOT / "data" / "chatbot_training.json"
        self.rows = json.loads(self.data_path.read_text(encoding="utf-8"))

    def test_training_data_structure(self):
        self.assertTrue(self.data_path.exists())
        self.assertIsInstance(self.rows, list)
        self.assertGreaterEqual(len(self.rows), 100)
        for row in self.rows:
            self.assertIn("text", row)
            self.assertIn("intent", row)
            self.assertIn("lang", row)
            self.assertIn(row["lang"], ("en", "th"))

    def test_training_data_has_all_intents(self):
        expected = {
            "greeting", "causes", "prevention", "treatment",
            "diet", "exercise", "sleep", "bmi", "genetics", "risk_factors",
        }
        found = {r["intent"] for r in self.rows}
        self.assertEqual(found, expected)


class ChatbotGetAnswerTests(unittest.TestCase):
    def test_get_answer_returns_required_keys(self):
        result = get_answer("causes", "en")
        self.assertIn("answer", result)
        self.assertIn("source", result)

    def test_get_answer_en_causes_has_content(self):
        result = get_answer("causes", "en")
        self.assertGreater(len(result["answer"]), 30)
        self.assertTrue("WHO" in result["source"] or "CDC" in result["source"])

    def test_get_answer_th_diet_has_content(self):
        result = get_answer("diet", "th")
        self.assertGreater(len(result["answer"]), 10)

    def test_get_answer_treatment_with_very_high_context(self):
        ctx = {"risk_tier": "Very High Risk", "probability": 0.72}
        result = get_answer("treatment", "en", context=ctx)
        self.assertTrue("72" in result["answer"] or "Very High" in result["answer"])

    def test_get_answer_prevention_with_low_context(self):
        ctx = {"risk_tier": "Low Risk", "probability": 0.20}
        result = get_answer("prevention", "th", context=ctx)
        self.assertGreater(len(result["answer"]), 10)

    def test_get_answer_unknown_lang_defaults_to_en(self):
        result = get_answer("bmi", "fr")
        self.assertGreater(len(result["answer"]), 10)


class ChatbotTrainTests(unittest.TestCase):
    def test_train_chatbot_creates_model_file(self):
        data_path = PROJECT_ROOT / "data" / "chatbot_training.json"
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "chatbot_model.joblib"
            result = train_chatbot(data_path, out_path)
            self.assertTrue(out_path.exists())
            self.assertGreater(result["cv_mean"], 0.5)
            self.assertEqual(
                set(result["classes"]),
                {"greeting", "causes", "prevention", "treatment",
                 "diet", "exercise", "sleep", "bmi", "genetics", "risk_factors"},
            )


class ChatbotChatTests(unittest.TestCase):
    def test_chat_causes_en(self):
        result = chat("What causes obesity?", lang="en")
        self.assertEqual(result["intent"], "causes")
        self.assertEqual(result["detected_lang"], "en")
        self.assertGreater(len(result["answer"]), 20)

    def test_chat_diet_th(self):
        result = chat("ควรกินอะไร", lang="th")
        self.assertEqual(result["intent"], "diet")
        self.assertEqual(result["detected_lang"], "th")

    def test_chat_bmi_auto_detect_en(self):
        result = chat("What is BMI?", lang="auto")
        self.assertEqual(result["intent"], "bmi")
        self.assertEqual(result["detected_lang"], "en")

    def test_chat_low_confidence_returns_unknown(self):
        result = chat("xyzqqqabcdef999", lang="en")
        self.assertEqual(result["intent"], "unknown")

    def test_chat_with_context_treatment(self):
        ctx = {"risk_tier": "Very High Risk", "probability": 0.78}
        result = chat("How to treat obesity?", lang="en", context=ctx)
        self.assertEqual(result["intent"], "treatment")
        self.assertTrue("78" in result["answer"] or "Very High" in result["answer"])

    def test_chat_response_has_all_keys(self):
        result = chat("hello", lang="en")
        for key in ("answer", "intent", "detected_lang", "source"):
            self.assertIn(key, result)


class ChatbotEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(fastapi_app)

    def test_chat_endpoint_shape(self):
        resp = self.client.post("/chat", json={"message": "What causes obesity?", "lang": "en"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        for key in ("answer", "intent", "detected_lang", "source"):
            self.assertIn(key, data)

    def test_chat_endpoint_unknown_returns_200(self):
        resp = self.client.post("/chat", json={"message": "xyzqqqabc999", "lang": "en"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["intent"], "unknown")

    def test_chat_endpoint_with_context(self):
        resp = self.client.post("/chat", json={
            "message": "how to treat obesity",
            "lang": "en",
            "context": {"risk_tier": "High Risk", "probability": 0.55},
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["intent"], "treatment")


class ChatbotWidgetTests(unittest.TestCase):
    def setUp(self):
        # reuse the TestClient already imported
        self.client = TestClient(fastapi_app)

    def test_page_shell_contains_beast_fab(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn('id="beast-fab"', resp.text)

    def test_page_shell_contains_chat_window(self):
        resp = self.client.get("/")
        self.assertIn('id="beast-chat"', resp.text)

    def test_widget_shows_on_all_pages(self):
        for path in ("/", "/predictor", "/advice", "/methods"):
            resp = self.client.get(path)
            self.assertIn('id="beast-fab"', resp.text, msg=f"Widget missing on {path}")

    def test_widget_javascript_uses_parseable_quotes(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn("‘", resp.text)
        self.assertNotIn("’", resp.text)
        self.assertIn("document.getElementById('beast-close-btn')", resp.text)


class ChatbotResultContextTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(fastapi_app)

    def test_result_page_has_risk_tier_data_attribute(self):
        resp = self.client.post("/predict-form", data={
            "age": "16", "sex": "M", "height_cm": "170", "weight_kg": "90",
            "family_history_obesity": "1",
            "high_calorie_food_frequency": "1", "vegetable_frequency": "2",
            "main_meals_per_day": "3", "food_between_meals_frequency": "2",
            "smoke": "0", "water_daily": "1", "calorie_monitoring": "0",
            "physical_activity_freq": "1", "screen_time_band": "2",
            "alcohol_frequency": "1", "transportation": "Automobile",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertIn('data-risk-tier=', resp.text)
        self.assertIn('data-probability=', resp.text)
