import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from obesity_ml.config import OPTIONAL_INPUT_DEFAULTS, REQUIRED_INPUT_COLUMNS, TARGET_COLUMN
from obesity_ml.features import add_engineered_features


FORM1_SOURCE = "google_form_lifestyle_screen"
FORM2_SOURCE = "google_form_uci_style_habits"

NEUTRAL_MISSING_DEFAULTS = {
    "physical_activity_hours_per_week": 2.5,
    "screen_time_hours_per_day": 4.0,
    "fast_food_meals_per_week": 2.0,
    "sugary_drinks_per_day": 0.5,
}


def _text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def parse_number(value, default=pd.NA, *, minimum: float | None = None, maximum: float | None = None):
    text = _text(value)
    if not text:
        return default
    if "ไม่กิน" in text or text.lower() in {"none", "no", "never"}:
        number = 0.0
    elif "กินตลอด" in text or "ตลอด" in text:
        number = 3.0
    else:
        numbers = [float(match) for match in re.findall(r"\d+(?:\.\d+)?", text.replace(",", "."))]
        if not numbers:
            return default
        number = sum(numbers[:2]) / min(2, len(numbers))
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


def normalize_sex(value: object) -> str:
    text = _text(value).lower()
    if "หญิง" in text or text in {"f", "female", "woman"}:
        return "F"
    if "ชาย" in text or text in {"m", "male", "man"}:
        return "M"
    return "U"


def normalize_yes_no(value: object, default: int = 0) -> int:
    text = _text(value).lower()
    if not text:
        return default
    negative_markers = ("ไม่ใช่", "ไม่มี", "ไม่เป็น", "ไม่สูบ", "ไม่กิน", "no", "false", "0")
    positive_markers = ("ใช่", "มี", "เป็น", "สูบ", "yes", "true", "1")
    if any(marker in text for marker in negative_markers):
        return 0
    if any(marker in text for marker in positive_markers):
        return 1
    return default


def consent_allows(value: object) -> bool:
    text = _text(value).lower()
    if "ไม่อนุญาต" in text:
        return False
    return "อนุญาต" in text or text in {"yes", "y", "true", "1"}


def parse_vegetable_frequency(value: object) -> float:
    text = _text(value)
    if "ระดับ3" in text or "ชอบกินผัก" in text:
        return 3.0
    if "ระดับ2" in text:
        return 2.0
    if "ระดับ1" in text or "ไม่กิน" in text:
        return 1.0
    return parse_number(value, default=OPTIONAL_INPUT_DEFAULTS["vegetable_frequency"], minimum=1, maximum=3)


def parse_between_meals_frequency(value: object) -> float:
    text = _text(value)
    if "ตลอด" in text or "บ่อย" in text:
        return 3.0
    if "บาง" in text:
        return 2.0
    if "น้อย" in text:
        return 1.0
    if "ไม่" in text:
        return 0.0
    return parse_number(value, default=OPTIONAL_INPUT_DEFAULTS["food_between_meals_frequency"], minimum=0, maximum=3)


def add_bmi_target(frame: pd.DataFrame, *, threshold: float = 25.0) -> pd.DataFrame:
    out = frame.copy()
    engineered = add_engineered_features(out)
    out["bmi"] = engineered["bmi"].round(2)
    out[TARGET_COLUMN] = (engineered["bmi"] >= threshold).astype(int)
    return out


def normalize_form1_frame(df: pd.DataFrame, *, bmi_target_threshold: float = 25.0) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        if not consent_allows(row.get("ขออนุญาตนำข้อมูลไปใช้นะครับ")):
            continue
        fast_food = parse_number(row.get("กินอาหารไร้ประโยชน์กี่มื้อต่อสัปดาห์"), default=2, minimum=0, maximum=30)
        sugary_drinks = parse_number(row.get("กินน้ำหวานหรือน้ำอัดลมกี่ครั้งต่อวัน"), default=0.5, minimum=0, maximum=20)
        rows.append(
            {
                "age": parse_number(row.get("อายุ"), minimum=5, maximum=100),
                "sex": normalize_sex(row.get("เพศต้นกำเนิด")),
                "height_cm": parse_number(row.get("ส่วนสูง"), minimum=80, maximum=230),
                "weight_kg": parse_number(row.get("น้ำหนัก"), minimum=20, maximum=250),
                "physical_activity_hours_per_week": parse_number(row.get("ออกกำลังกายกี่ชั่วโมงต่อสัปดาห์"), default=2.5, minimum=0, maximum=40),
                "screen_time_hours_per_day": parse_number(row.get("ใช้หน้าจอวันละกี่ชั่วโมงต่อวัน"), default=4, minimum=0, maximum=24),
                "sleep_hours": parse_number(row.get("นอนกี่ชั่วโมงต่อวัน"), default=7, minimum=0, maximum=16),
                "fast_food_meals_per_week": fast_food,
                "sugary_drinks_per_day": sugary_drinks,
                "family_history_obesity": normalize_yes_no(row.get("มีคนในครอบครัวเป็นโรคอ้วนหรือไม่")),
                "high_calorie_food_frequency": 1.0 if fast_food >= 3 or sugary_drinks >= 1 else 0.0,
                "physical_activity_missing": 0.0,
                "screen_time_missing": 0.0,
                "fast_food_missing": 0.0,
                "sugary_drinks_missing": 0.0,
                "self_reported_obesity": normalize_yes_no(row.get("คุณคิดว่าคุณเป็นโรคอ้วนมั้ย"), default=pd.NA),
                "form_source": FORM1_SOURCE,
            }
        )
    normalized = pd.DataFrame(rows)
    return _finalize_normalized_frame(normalized, bmi_target_threshold=bmi_target_threshold)


def normalize_form2_frame(df: pd.DataFrame, *, bmi_target_threshold: float = 25.0) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        if not consent_allows(row.get("อนุญาตนำข้อมูลนี้ไปใส่ในงานวิจัยนะครับ")):
            continue
        rows.append(
            {
                "age": parse_number(row.get("อายุ"), minimum=5, maximum=100),
                "sex": normalize_sex(row.get("เพศ")),
                "height_cm": parse_number(row.get("ส่วนสูง"), minimum=80, maximum=230),
                "weight_kg": parse_number(row.get("น้ำหนัก"), minimum=20, maximum=250),
                "physical_activity_hours_per_week": NEUTRAL_MISSING_DEFAULTS["physical_activity_hours_per_week"],
                "screen_time_hours_per_day": NEUTRAL_MISSING_DEFAULTS["screen_time_hours_per_day"],
                "sleep_hours": parse_number(row.get("คุณนอนวันละกี่ชั่วโมง"), default=7, minimum=0, maximum=16),
                "fast_food_meals_per_week": NEUTRAL_MISSING_DEFAULTS["fast_food_meals_per_week"],
                "sugary_drinks_per_day": NEUTRAL_MISSING_DEFAULTS["sugary_drinks_per_day"],
                "family_history_obesity": normalize_yes_no(row.get("คุณมีสมาชิกในครอบครัวเป็นโรคอ้วนหรือน้ำหนักเกินหรือไม่")),
                "high_calorie_food_frequency": normalize_yes_no(row.get("คุณกินสิ่งที่มีแคลอรี่สูงบ่อยหรือไม่")),
                "vegetable_frequency": parse_vegetable_frequency(row.get("ส่วนใหญ่คุณกินผักมั้ย")),
                "main_meals_per_day": parse_number(row.get("คุณกินข้าววันละกี่มื้อ"), default=3, minimum=1, maximum=6),
                "food_between_meals_frequency": parse_between_meals_frequency(row.get("คุณกินอะไรระหว่างมื้อรึป่าว")),
                "smoke": normalize_yes_no(row.get("คุณสูบบุหรี่มั้ย")),
                "physical_activity_missing": 1.0,
                "screen_time_missing": 1.0,
                "fast_food_missing": 1.0,
                "sugary_drinks_missing": 1.0,
                "form_source": FORM2_SOURCE,
            }
        )
    normalized = pd.DataFrame(rows)
    return _finalize_normalized_frame(normalized, bmi_target_threshold=bmi_target_threshold)


def _finalize_normalized_frame(frame: pd.DataFrame, *, bmi_target_threshold: float) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for column, default in OPTIONAL_INPUT_DEFAULTS.items():
        if column not in out.columns:
            out[column] = default
    for column in REQUIRED_INPUT_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    out = out.dropna(subset=["age", "height_cm", "weight_kg", "sleep_hours"])
    out = add_bmi_target(out, threshold=bmi_target_threshold)
    out = out[out["bmi"].between(10, 60)]
    ordered = REQUIRED_INPUT_COLUMNS + list(OPTIONAL_INPUT_DEFAULTS) + [
        "bmi",
        TARGET_COLUMN,
        "self_reported_obesity",
        "form_source",
    ]
    for column in ordered:
        if column not in out.columns:
            out[column] = pd.NA
    return out[ordered]


def normalize_google_form_frame(df: pd.DataFrame, *, bmi_target_threshold: float = 25.0) -> pd.DataFrame:
    columns = set(df.columns)
    if "เพศต้นกำเนิด" in columns and "คุณคิดว่าคุณเป็นโรคอ้วนมั้ย" in columns:
        return normalize_form1_frame(df, bmi_target_threshold=bmi_target_threshold)
    if "คุณกินสิ่งที่มีแคลอรี่สูงบ่อยหรือไม่" in columns and "คุณกินอะไรระหว่างมื้อรึป่าว" in columns:
        return normalize_form2_frame(df, bmi_target_threshold=bmi_target_threshold)
    raise ValueError("Could not recognize this Google Form schema. Check the header row.")


def normalize_google_form_csv(path: Path, *, bmi_target_threshold: float = 25.0) -> pd.DataFrame:
    return normalize_google_form_frame(pd.read_csv(path), bmi_target_threshold=bmi_target_threshold)


def combine_google_form_csvs(paths: Iterable[Path], *, bmi_target_threshold: float = 25.0) -> pd.DataFrame:
    frames = [normalize_google_form_csv(path, bmi_target_threshold=bmi_target_threshold) for path in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize O-Beast Google Form CSV exports for later ML training.")
    parser.add_argument("--input", nargs="+", type=Path, required=True, help="One or more exported Google Form CSV files.")
    parser.add_argument("--output", type=Path, required=True, help="Where to save the normalized training CSV.")
    parser.add_argument(
        "--bmi-target-threshold",
        type=float,
        default=25.0,
        help="BMI threshold used to create the prototype obesity-risk target. Default is 25.0 for Asian adult high-risk screening.",
    )
    args = parser.parse_args()

    normalized = combine_google_form_csvs(args.input, bmi_target_threshold=args.bmi_target_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(args.output, index=False)
    print(f"Saved {len(normalized)} normalized rows to {args.output}")


if __name__ == "__main__":
    main()
