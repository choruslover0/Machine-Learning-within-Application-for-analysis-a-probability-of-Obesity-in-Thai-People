import sys
import unittest
import json
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from obesity_ml.config import CHATBOT_MODEL_PATH
from obesity_ml.chatbot import get_answer, train_chatbot


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
