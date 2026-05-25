import pandas as pd

from obesity_ml.features import add_engineered_features


SOURCE_NOTES = [
    {
        "name": "WHO physical activity and sedentary behaviour guidelines",
        "url": "https://www.who.int/publications/i/item/9789240015128",
        "supports": "Physical activity and reducing sedentary time.",
    },
    {
        "name": "WHO guideline on sugars intake",
        "url": "https://www.who.int/publications/i/item/9789241549028",
        "supports": "Reducing free-sugar and sugary-drink intake.",
    },
    {
        "name": "CDC BMI and healthy weight",
        "url": "https://www.cdc.gov/bmi/",
        "supports": "BMI as a screening signal, not a diagnosis.",
    },
    {
        "name": "CDC sleep and sleep health",
        "url": "https://www.cdc.gov/sleep/",
        "supports": "Sleep routine and sleep-duration guidance.",
    },
    {
        "name": "FAO summary of Thailand food-based dietary guidelines",
        "url": "https://www.fao.org/nutrition/education/food-dietary-guidelines/regions/countries/Thailand/en/",
        "supports": "Thai-context food variety, vegetables, fruits, and moderation.",
    },
]


def _bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "below the usual adult screening range"
    if bmi < 23:
        return "in the lower-risk Asian adult screening range"
    if bmi < 25:
        return "in the increased-risk Asian adult screening range"
    if bmi < 30:
        return "in the high-risk Asian adult screening range"
    return "in the very-high-risk Asian adult screening range"


def _priority_for_probability(probability: float) -> str:
    if probability >= 0.65:
        return "High priority"
    if probability >= 0.35:
        return "Medium priority"
    return "Build good habits"


def generate_advice(input_data: dict, prediction: dict | None = None) -> dict:
    framed = add_engineered_features(pd.DataFrame([input_data]))
    row = framed.iloc[0]
    bmi = float(row["bmi"])
    probability = float(prediction.get("obesity_probability", 0)) if prediction else 0

    cards = []
    if float(row["physical_activity_hours_per_week"]) < 2.5:
        cards.append(
            {
                "title": "Move more during the week",
                "priority": "High impact habit",
                "why": "Your activity answer is below a common public-health target for adults and older teens.",
                "action": "Start with 10-15 minute walks or active breaks, then build toward at least 150 minutes of moderate movement per week when appropriate.",
                "source": "WHO physical activity guidelines",
            }
        )
    else:
        cards.append(
            {
                "title": "Keep your activity habit",
                "priority": "Protective habit",
                "why": "Your weekly activity answer is already moving in a helpful direction.",
                "action": "Keep a regular routine and add variety: walking, sports, cycling, dance, or strength exercises.",
                "source": "WHO physical activity guidelines",
            }
        )

    if float(row["screen_time_hours_per_day"]) >= 6:
        cards.append(
            {
                "title": "Break up long sitting time",
                "priority": "Daily routine",
                "why": "High screen time usually means long sedentary periods, which public-health guidance recommends reducing.",
                "action": "Use a simple rule: after 45-60 minutes of sitting, stand, stretch, walk, or do light movement for 3-5 minutes.",
                "source": "WHO sedentary behaviour guidance",
            }
        )

    if float(row["sleep_hours"]) < 7:
        cards.append(
            {
                "title": "Improve sleep consistency",
                "priority": "Recovery",
                "why": "Short sleep can make appetite, energy, and exercise habits harder to manage.",
                "action": "Try a fixed sleep/wake time, reduce bright screens before bed, and avoid heavy late-night snacks or sugary drinks.",
                "source": "CDC sleep health",
            }
        )

    if float(row["sugary_drinks_per_day"]) >= 1:
        cards.append(
            {
                "title": "Reduce sugary drinks step by step",
                "priority": "Nutrition",
                "why": "Sugary drinks are an easy source of free sugars because they do not make people feel full for long.",
                "action": "Replace one sweet drink with water, unsweetened tea, or low-sugar options first; gradual change is easier to keep.",
                "source": "WHO sugars guideline",
            }
        )

    if int(row["fast_food_meals_per_week"]) >= 3:
        cards.append(
            {
                "title": "Make fast food less frequent",
                "priority": "Food pattern",
                "why": "Frequent fast-food meals can increase energy intake and make vegetables, fruits, and balanced meals harder to fit in.",
                "action": "Plan a few Thai-style balanced meals around rice or grains, vegetables, lean protein, fruit, and less fried or sweet food.",
                "source": "Thailand food-based dietary guidance",
            }
        )

    if str(row["family_history_obesity"]) == "1":
        cards.append(
            {
                "title": "Use family history as an early-warning signal",
                "priority": "Awareness",
                "why": "Family history can reflect shared genes, food habits, activity patterns, and home environment.",
                "action": "Track habits gently and consider discussing weight, blood pressure, or nutrition questions with a health professional.",
                "source": "CDC healthy weight context",
            }
        )

    cards.append(
        {
            "title": "Understand your BMI signal",
            "priority": _priority_for_probability(probability),
            "why": f"Your calculated BMI is {bmi:.1f}, which is {_bmi_category(bmi)}. BMI is a screening clue, not a diagnosis.",
            "action": "Use BMI together with lifestyle answers, waist/health measurements if available, and professional advice when needed.",
            "source": "CDC BMI guidance",
        }
    )

    if not cards:
        cards.append(
            {
                "title": "Maintain your current healthy pattern",
                "priority": "Maintenance",
                "why": "Your answers do not show a major lifestyle warning in this prototype rule set.",
                "action": "Keep regular activity, enough sleep, balanced meals, and low sugary-drink intake.",
                "source": "WHO, CDC, and Thai dietary guidance",
            }
        )

    return {
        "bmi": round(bmi, 1),
        "focus": _priority_for_probability(probability),
        "cards": cards,
        "sources": SOURCE_NOTES,
        "disclaimer": "Educational wellness advice only; it is not a diagnosis or treatment plan.",
    }
