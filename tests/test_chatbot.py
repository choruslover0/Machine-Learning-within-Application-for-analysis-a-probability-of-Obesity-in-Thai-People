import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from obesity_ml.config import CHATBOT_MODEL_PATH


class ChatbotConfigTests(unittest.TestCase):
    def test_chatbot_model_path_is_in_models_dir(self):
        self.assertEqual(CHATBOT_MODEL_PATH.parent.name, "models")
        self.assertEqual(CHATBOT_MODEL_PATH.name, "chatbot_model.joblib")
