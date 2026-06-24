import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


class UciImportTests(unittest.TestCase):
    def _raw_frame(self):
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "Gender": "Female", "Age": 21, "Height": 1.62, "Weight": 64,
                    "family_history_with_overweight": "yes", "FAVC": "no",
                    "FCVC": 2, "NCP": 3, "CAEC": "Sometimes", "SMOKE": "no",
                    "CH2O": 2, "SCC": "no", "FAF": 0, "TUE": 1, "CALC": "no",
                    "MTRANS": "Public_Transportation", "NObeyesdad": "Normal_Weight",
                },
                {
                    "Gender": "Male", "Age": 27, "Height": 1.80, "Weight": 120,
                    "family_history_with_overweight": "yes", "FAVC": "yes",
                    "FCVC": 3, "NCP": 3, "CAEC": "Frequently", "SMOKE": "no",
                    "CH2O": 3, "SCC": "no", "FAF": 2, "TUE": 0, "CALC": "Sometimes",
                    "MTRANS": "Automobile", "NObeyesdad": "Obesity_Type_II",
                },
            ]
        )

    def test_maps_columns_units_and_binary_target(self):
        from obesity_ml.uci_import import normalize_uci_frame

        out = normalize_uci_frame(self._raw_frame())
        self.assertEqual(len(out), 2)

        normal = out.iloc[0]
        self.assertEqual(normal["sex"], "F")
        self.assertEqual(normal["height_cm"], 162.0)
        self.assertEqual(normal["family_history_obesity"], 1)
        self.assertEqual(normal["food_between_meals_frequency"], 1.0)
        self.assertEqual(normal["water_daily"], 2.0)
        self.assertEqual(normal["obesity"], 0)
        self.assertEqual(normal["obesity_class"], "Normal_Weight")

        obese = out.iloc[1]
        self.assertEqual(obese["sex"], "M")
        self.assertEqual(obese["high_calorie_food_frequency"], 1.0)
        self.assertEqual(obese["alcohol_frequency"], 1.0)
        self.assertEqual(obese["transportation"], "Automobile")
        self.assertEqual(obese["obesity"], 1)

    def test_trains_through_config_feature_set(self):
        from obesity_ml.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
        from obesity_ml.features import add_engineered_features
        from obesity_ml.uci_import import normalize_uci_frame

        engineered = add_engineered_features(normalize_uci_frame(self._raw_frame()))
        for column in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
            self.assertIn(column, engineered.columns, f"missing model feature: {column}")


if __name__ == "__main__":
    unittest.main()
