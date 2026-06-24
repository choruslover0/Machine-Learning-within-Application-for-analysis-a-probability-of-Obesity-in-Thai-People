from pathlib import Path


def _project_root() -> Path:
    working_directory = Path.cwd()
    if (working_directory / "models").exists() or (working_directory / "pyproject.toml").exists():
        return working_directory

    source_project_root = Path(__file__).resolve().parents[2]
    if (source_project_root / "pyproject.toml").exists():
        return source_project_root

    return working_directory


PROJECT_ROOT = _project_root()
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "obesity_probability_model.joblib"
CHATBOT_MODEL_PATH = MODEL_DIR / "chatbot_model.joblib"

TARGET_COLUMN = "obesity"

# Body-screen features feed the separate BMI screen, never the lifestyle ML model.
BODY_SCREEN_FEATURES = [
    "height_cm",
    "weight_kg",
    "bmi",
]

# --- BMI screening signal (independent of the lifestyle ML model) ---
# bmi_screen_score crosses 0.5 at the midpoint and rises GRADUALLY (graded, not a
# hard cutoff at BMI 25). Kept separate from the model so BMI informs the final
# blend WITHOUT leaking into a model whose label is itself BMI-derived. A larger
# steepness = gentler curve, so lifestyle keeps a real say in the 50/50 blend.
BMI_SCREEN_MIDPOINT = 25.0   # Asian-adult obesity threshold; screen score = 0.5 here
BMI_SCREEN_STEEPNESS = 3.0   # logistic scale in BMI units (smaller = sharper step)

# --- Pure UCI ObesityDataSet feature schema (Mendoza Palechor & de la Hoz, 2019) ---
# Internal name (UCI column): semantics
#   high_calorie_food_frequency (FAVC): 0/1 eats high-calorie food often
#   vegetable_frequency         (FCVC): 1-3 vegetable intake
#   main_meals_per_day          (NCP):  1-4 number of main meals
#   food_between_meals_frequency(CAEC): 0-3 never..always snacking
#   smoke                       (SMOKE):0/1
#   water_daily                 (CH2O): 1-3 daily water (<1L / 1-2L / >2L)
#   calorie_monitoring          (SCC):  0/1 monitors calories
#   physical_activity_freq      (FAF):  0-3 physical-activity frequency
#   screen_time_band            (TUE):  0-2 device-use band (0-2h / 3-5h / >5h)
#   alcohol_frequency           (CALC): 0-3 never..always
#   transportation              (MTRANS): categorical travel mode
NUMERIC_FEATURES = [
    "age",
    "physical_activity_freq",
    "screen_time_band",
    "high_calorie_food_frequency",
    "vegetable_frequency",
    "main_meals_per_day",
    "food_between_meals_frequency",
    "smoke",
    "water_daily",
    "calorie_monitoring",
    "alcohol_frequency",
]

CATEGORICAL_FEATURES = [
    "sex",
    "family_history_obesity",
    "transportation",
]

# Every field the predictor form and /predict API must supply. The form now
# collects the full UCI question set, so there are no defaulted lifestyle inputs.
REQUIRED_INPUT_COLUMNS = [
    "age",
    "sex",
    "height_cm",
    "weight_kg",
    "family_history_obesity",
    "high_calorie_food_frequency",
    "vegetable_frequency",
    "main_meals_per_day",
    "food_between_meals_frequency",
    "smoke",
    "water_daily",
    "calorie_monitoring",
    "physical_activity_freq",
    "screen_time_band",
    "alcohol_frequency",
    "transportation",
]

# The form supplies every lifestyle input, so there are no numeric optionals.
OPTIONAL_INPUT_DEFAULTS: dict[str, float] = {}

# Safety default for the one categorical lifestyle feature, used only if a caller
# omits it (the live form always sends it). OneHotEncoder ignores unknown levels.
CATEGORICAL_OPTIONAL_DEFAULTS = {
    "transportation": "Public_Transportation",
}
