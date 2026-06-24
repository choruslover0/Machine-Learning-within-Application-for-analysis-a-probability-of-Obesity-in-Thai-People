"""Load the UCI ObesityDataSet into the O-Beast training schema.

Source: Mendoza Palechor, F. & de la Hoz Manotas, A. (2019). "Dataset for
estimation of obesity levels based on eating habits and physical condition in
individuals from Colombia, Peru and Mexico." Data in Brief 25, 104344.
UCI Machine Learning Repository id 544. CC BY 4.0.

The UCI file has 2111 records and 17 attributes. This module maps those columns
onto the O-Beast training schema consumed by ``train``, keeping the UCI-native
ordinal codes (FAF 0-3, TUE 0-2, FCVC 1-3, ...) rather than rescaling them.
Body-screen features (height_cm, weight_kg, bmi) are kept for the BMI screen but
excluded from the lifestyle ML model by ``config.NUMERIC_FEATURES`` /
``config.CATEGORICAL_FEATURES``.
"""

import argparse
from pathlib import Path

import pandas as pd

from obesity_ml.config import TARGET_COLUMN
from obesity_ml.features import TRANSPORTATION_VALUES

UCI_SOURCE = "uci_obesity_2019"

# NObeyesdad categories that count as "not obese" for the binary target. The
# remaining categories (overweight I/II, obesity I/II/III) map to 1.
NON_OBESE_CLASSES = {"Insufficient_Weight", "Normal_Weight"}

# Ordinal frequency encodings shared by CAEC (food between meals) and CALC
# (alcohol): never -> 0 ... always -> 3.
FREQUENCY_MAP = {"no": 0.0, "Sometimes": 1.0, "Frequently": 2.0, "Always": 3.0}

DEFAULT_TRANSPORTATION = "Public_Transportation"

OUTPUT_COLUMNS = [
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
    "bmi",
    TARGET_COLUMN,
    "obesity_class",
    "form_source",
]


def _yes_no(value: object) -> int:
    return 1 if str(value).strip().lower() in {"yes", "true", "1"} else 0


def _number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_uci_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Map a raw UCI ObesityDataSet frame to the O-Beast training schema."""
    rows = []
    for _, row in df.iterrows():
        height_m = _number(row.get("Height"), default=0.0)
        weight_kg = _number(row.get("Weight"), default=0.0)
        bmi = round(weight_kg / (height_m**2), 2) if height_m > 0 else pd.NA
        obesity_class = str(row.get("NObeyesdad")).strip()
        transportation = str(row.get("MTRANS")).strip()
        if transportation not in TRANSPORTATION_VALUES:
            transportation = DEFAULT_TRANSPORTATION
        rows.append(
            {
                "age": round(_number(row.get("Age"))),
                "sex": "M" if str(row.get("Gender")).strip().lower() == "male" else "F",
                "height_cm": round(height_m * 100, 1),
                "weight_kg": round(weight_kg, 1),
                "family_history_obesity": _yes_no(row.get("family_history_with_overweight")),
                "high_calorie_food_frequency": float(_yes_no(row.get("FAVC"))),
                "vegetable_frequency": round(_number(row.get("FCVC"), default=2.0), 2),
                "main_meals_per_day": round(_number(row.get("NCP"), default=3.0), 2),
                "food_between_meals_frequency": FREQUENCY_MAP.get(str(row.get("CAEC")).strip(), 1.0),
                "smoke": float(_yes_no(row.get("SMOKE"))),
                "water_daily": round(_number(row.get("CH2O"), default=2.0), 2),
                "calorie_monitoring": float(_yes_no(row.get("SCC"))),
                "physical_activity_freq": round(_number(row.get("FAF")), 2),   # FAF 0-3, native
                "screen_time_band": round(_number(row.get("TUE")), 2),         # TUE 0-2, native
                "alcohol_frequency": FREQUENCY_MAP.get(str(row.get("CALC")).strip(), 0.0),
                "transportation": transportation,
                "bmi": bmi,
                TARGET_COLUMN: 0 if obesity_class in NON_OBESE_CLASSES else 1,
                "obesity_class": obesity_class,
                "form_source": UCI_SOURCE,
            }
        )

    normalized = pd.DataFrame(rows)
    normalized = normalized.dropna(subset=["age", "height_cm", "weight_kg", "bmi"])
    normalized = normalized[normalized["bmi"].between(10, 60)]
    return normalized[OUTPUT_COLUMNS]


def normalize_uci_csv(path: Path) -> pd.DataFrame:
    return normalize_uci_frame(pd.read_csv(path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert the UCI ObesityDataSet CSV into the O-Beast training schema."
    )
    parser.add_argument("--input", type=Path, required=True, help="Raw UCI ObesityDataSet CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Where to save the normalized training CSV.")
    args = parser.parse_args()

    normalized = normalize_uci_csv(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(args.output, index=False)
    obese = int(normalized[TARGET_COLUMN].sum())
    print(f"Saved {len(normalized)} normalized rows to {args.output}")
    print(f"  obese (1): {obese}  |  not obese (0): {len(normalized) - obese}")


if __name__ == "__main__":
    main()
