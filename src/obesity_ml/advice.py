import pandas as pd

from obesity_ml.features import add_engineered_features
from obesity_ml.risk_tiers import asian_bmi_risk_tier, classify_probability


SOURCE_NOTES = [
    {
        "name": "WHO physical activity and sedentary behaviour guidelines",
        "url": "https://www.who.int/publications/i/item/9789240015128",
        "supports": "Physical activity and reducing sedentary time.",
    },
    {
        "name": "WHO guideline on sugars intake",
        "url": "https://www.who.int/publications/i/item/9789241549028",
        "supports": "Reducing free-sugar and high-calorie food intake.",
    },
    {
        "name": "WHO healthy diet fact sheet",
        "url": "https://www.who.int/news-room/fact-sheets/detail/healthy-diet",
        "supports": "Vegetables, water, meal patterns, and moderation.",
    },
    {
        "name": "CDC BMI and healthy weight",
        "url": "https://www.cdc.gov/bmi/",
        "supports": "BMI as a screening signal, not a diagnosis.",
    },
    {
        "name": "FAO summary of Thailand food-based dietary guidelines",
        "url": "https://www.fao.org/nutrition/education/food-dietary-guidelines/regions/countries/Thailand/en/",
        "supports": "Thai-context food variety, vegetables, fruits, and moderation.",
    },
]


def _priority_for_probability(probability: float) -> str:
    if probability >= 0.80:
        return "Very high priority"
    if probability >= 0.60:
        return "High priority"
    if probability >= 0.40:
        return "Moderate priority"
    if probability >= 0.20:
        return "Build good habits"
    return "Maintenance"


def generate_advice(input_data: dict, prediction: dict | None = None) -> dict:
    framed = add_engineered_features(pd.DataFrame([input_data]))
    row = framed.iloc[0]
    bmi = float(row["bmi"])
    probability = float(prediction.get("obesity_probability", 0)) if prediction else 0
    probability_tier = classify_probability(probability)
    bmi_tier = asian_bmi_risk_tier(bmi)

    activity = float(row.get("physical_activity_freq", 1))       # FAF 0-3
    screen_band = float(row.get("screen_time_band", 1))          # TUE 0-2
    water = float(row.get("water_daily", 2))                     # CH2O 1-3
    vegetables = float(row.get("vegetable_frequency", 2))        # FCVC 1-3
    high_calorie = float(row.get("high_calorie_food_frequency", 0))  # FAVC 0/1
    between_meals = float(row.get("food_between_meals_frequency", 1))  # CAEC 0-3
    main_meals = float(row.get("main_meals_per_day", 3))         # NCP 1-4
    calorie_monitoring = float(row.get("calorie_monitoring", 0))  # SCC 0/1
    alcohol = float(row.get("alcohol_frequency", 0))             # CALC 0-3
    smoke = float(row.get("smoke", 0))                           # SMOKE 0/1

    cards = []
    cards.append(
        {
            "title": "Your fine risk tier",
            "priority": probability_tier["risk_tier_label"],
            "why": f"The model probability falls in the {probability_tier['risk_tier_range']} range. {probability_tier['risk_tier_explanation']}",
            "action": "Use this tier to choose how strongly to focus on habits. It is a research estimate, not a diagnosis.",
            "source": "Model probability tiering",
        }
    )

    if activity < 1:
        cards.append(
            {
                "title": "Start with small movement goals",
                "priority": "High impact habit",
                "why": "Your physical-activity answer is very low, so even small increases can improve the weekly pattern.",
                "action": "Try 10 minutes of walking or sport on 3-4 days this week, then build up slowly.",
                "source": "WHO physical activity guidelines",
            }
        )
    elif activity < 2:
        cards.append(
            {
                "title": "Move a little more each week",
                "priority": "High impact habit",
                "why": "Your activity answer is below a helpful weekly target for teens and adults.",
                "action": "Add 15-20 minutes of moderate movement on a few extra days, aiming toward regular activity.",
                "source": "WHO physical activity guidelines",
            }
        )
    else:
        cards.append(
            {
                "title": "Keep your activity habit",
                "priority": "Protective habit",
                "why": "Your activity answer is already moving in a helpful direction.",
                "action": "Keep a regular routine with variety: walking, sport, cycling, dance, or strength work.",
                "source": "WHO physical activity guidelines",
            }
        )

    if screen_band >= 2:
        cards.append(
            {
                "title": "Reduce very long device time",
                "priority": "Daily routine",
                "why": "Your device-use answer is in the highest band, which usually means long sedentary time.",
                "action": "Protect one screen-free block each day and take a short movement break after each long session.",
                "source": "WHO sedentary behaviour guidance",
            }
        )
    elif screen_band >= 1:
        cards.append(
            {
                "title": "Break up sitting time",
                "priority": "Daily routine",
                "why": "Moderate device time still means long sitting periods that guidance recommends breaking up.",
                "action": "After 45-60 minutes of sitting, stand, stretch, or walk for a few minutes.",
                "source": "WHO sedentary behaviour guidance",
            }
        )

    if water < 2:
        cards.append(
            {
                "title": "Drink a little more water",
                "priority": "Daily routine",
                "why": "Your water answer is on the low side. Water is a calorie-free way to stay hydrated through the day.",
                "action": "Aim for at least 1-2 litres daily and keep water nearby during study, sport, and meals.",
                "source": "WHO healthy diet guidance",
            }
        )

    if high_calorie >= 1:
        cards.append(
            {
                "title": "Watch frequent high-calorie food",
                "priority": "Food choice",
                "why": "You reported frequent high-calorie food, an important variable in obesity survey research.",
                "action": "Keep the food you enjoy, but reduce portion size or frequency and pair meals with vegetables or fruit.",
                "source": "WHO sugars guideline",
            }
        )

    if vegetables < 2:
        cards.append(
            {
                "title": "Add vegetables to one meal first",
                "priority": "Food balance",
                "why": "Vegetable frequency is low. Vegetable intake is one of the useful lifestyle variables in the reference data.",
                "action": "Add one vegetable serving to lunch or dinner before changing the whole diet.",
                "source": "Thailand food-based dietary guidance",
            }
        )

    if between_meals >= 3:
        cards.append(
            {
                "title": "Plan snacks between meals",
                "priority": "Eating pattern",
                "why": "Frequent food between meals was one of the strongest variables in obesity ML research.",
                "action": "Prepare planned snacks such as fruit, yogurt, or nuts instead of turning every break into a high-calorie snack.",
                "source": "WHO healthy diet guidance",
            }
        )

    if main_meals > 3.5:
        cards.append(
            {
                "title": "Check meal frequency",
                "priority": "Eating pattern",
                "why": "Your main-meal answer is high, so the app flags it for review rather than treating it as automatically bad.",
                "action": "Check whether this means true meals or snacks, and keep meals balanced around grains, vegetables, and protein.",
                "source": "WHO healthy diet guidance",
            }
        )

    if alcohol >= 2:
        cards.append(
            {
                "title": "Reduce frequent alcohol",
                "priority": "Nutrition",
                "why": "Frequent alcohol adds calories and can affect sleep, appetite, and overall health.",
                "action": "Reduce frequency step by step and replace some occasions with water or unsweetened drinks.",
                "source": "WHO healthy diet guidance",
            }
        )

    if calorie_monitoring < 1:
        cards.append(
            {
                "title": "Try light awareness of intake",
                "priority": "Awareness",
                "why": "You do not currently monitor calories. Light awareness, not strict counting, can help notice patterns.",
                "action": "Notice portion sizes and sugary or fried items for a week, without aiming for a perfect diet.",
                "source": "CDC healthy weight context",
            }
        )

    if smoke >= 1:
        cards.append(
            {
                "title": "Do not use smoking for weight control",
                "priority": "Health safety",
                "why": "Smoking is a major health risk and should not be treated as a weight-management strategy.",
                "action": "If smoking is real in the answer, talk with a trusted adult or health professional about support.",
                "source": "Public-health safety guidance",
            }
        )

    if str(row["family_history_obesity"]) == "1":
        cards.append(
            {
                "title": "Use family history as an early-warning signal",
                "priority": "Awareness",
                "why": "Family history can reflect shared genes, food habits, activity patterns, and home environment.",
                "action": "Track habits gently and consider discussing weight, blood pressure, or nutrition with a health professional.",
                "source": "CDC healthy weight context",
            }
        )

    cards.append(
        {
            "title": "Understand your BMI signal",
            "priority": _priority_for_probability(probability),
            "why": f"Your calculated BMI is {bmi:.1f}. {bmi_tier['detail']} BMI is a screening clue, not a diagnosis.",
            "action": "Use BMI together with lifestyle answers, waist/health measurements if available, and professional advice when needed.",
            "source": "CDC BMI guidance",
        }
    )

    return {
        "bmi": round(bmi, 1),
        "focus": _priority_for_probability(probability),
        "risk_tier": probability_tier["risk_tier_label"],
        "cards": cards,
        "sources": SOURCE_NOTES,
        "disclaimer": "Educational wellness advice only; it is not a diagnosis or treatment plan.",
    }
