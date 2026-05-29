import sys
import unittest
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from obesity_ml.config import CHATBOT_MODEL_PATH


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
