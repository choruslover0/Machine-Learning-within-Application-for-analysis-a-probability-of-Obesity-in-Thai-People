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
    activity = float(row["physical_activity_hours_per_week"])
    screen_time = float(row["screen_time_hours_per_day"])
    sleep = float(row["sleep_hours"])
    sugary_drinks = float(row["sugary_drinks_per_day"])
    fast_food = float(row["fast_food_meals_per_week"])
    vegetables = float(row.get("vegetable_frequency", 2))
    high_calorie = float(row.get("high_calorie_food_frequency", 0))
    between_meals = float(row.get("food_between_meals_frequency", 1))
    main_meals = float(row.get("main_meals_per_day", 3))
    smoke = float(row.get("smoke", 0))

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
                "title": "Start with very small movement goals",
                "priority": "High impact habit",
                "why": "Your activity answer is under 1 hour per week, so even small increases can improve the weekly pattern.",
                "action": "Try 10 minutes of walking or sports practice on 4-5 days this week, then increase slowly.",
                "source": "WHO physical activity guidelines",
            }
        )
    elif activity < 2.5:
        cards.append(
            {
                "title": "Move more during the week",
                "priority": "High impact habit",
                "why": "Your activity answer is below a common public-health target for adults and older teens.",
                "action": "Add 15-20 minutes of moderate movement on 3 extra days, aiming toward 150 minutes per week when appropriate.",
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

    if screen_time >= 8:
        cards.append(
            {
                "title": "Reduce very long screen sessions",
                "priority": "Daily routine",
                "why": "Your answer suggests very long daily sedentary time, which can crowd out sleep and activity.",
                "action": "Protect one screen-free block each day and take a 3-5 minute movement break after each study or gaming session.",
                "source": "WHO sedentary behaviour guidance",
            }
        )
    elif screen_time >= 6:
        cards.append(
            {
                "title": "Break up long sitting time",
                "priority": "Daily routine",
                "why": "High screen time usually means long sedentary periods, which public-health guidance recommends reducing.",
                "action": "Use a simple rule: after 45-60 minutes of sitting, stand, stretch, walk, or do light movement for 3-5 minutes.",
                "source": "WHO sedentary behaviour guidance",
            }
        )

    if sleep < 6:
        cards.append(
            {
                "title": "Protect sleep first",
                "priority": "Recovery",
                "why": "Your sleep answer is very short. Short sleep can make appetite, mood, school focus, and movement habits harder.",
                "action": "Move bedtime earlier by 15-30 minutes for a week and reduce bright screen use close to bedtime.",
                "source": "CDC sleep health",
            }
        )
    elif sleep < 7:
        cards.append(
            {
                "title": "Improve sleep consistency",
                "priority": "Recovery",
                "why": "Short sleep can make appetite, energy, and exercise habits harder to manage.",
                "action": "Try a fixed sleep/wake time, reduce bright screens before bed, and avoid heavy late-night snacks or sugary drinks.",
                "source": "CDC sleep health",
            }
        )

    if sugary_drinks >= 2:
        cards.append(
            {
                "title": "Cut sugary drinks in half first",
                "priority": "Nutrition",
                "why": "Two or more sweet drinks per day can add many extra calories and free sugars without strong fullness.",
                "action": "For the first target, reduce by half and replace the removed drinks with water or unsweetened tea.",
                "source": "WHO sugars guideline",
            }
        )
    elif sugary_drinks >= 1:
        cards.append(
            {
                "title": "Reduce sugary drinks step by step",
                "priority": "Nutrition",
                "why": "Sugary drinks are an easy source of free sugars because they do not make people feel full for long.",
                "action": "Replace one sweet drink with water, unsweetened tea, or low-sugar options first; gradual change is easier to keep.",
                "source": "WHO sugars guideline",
            }
        )

    if fast_food >= 6:
        cards.append(
            {
                "title": "Replace some frequent fast-food meals",
                "priority": "Food pattern",
                "why": "Fast-food frequency is high, so the first goal is reducing frequency rather than trying to be perfect.",
                "action": "Swap 2 meals per week for a simple balanced meal: rice or grains, vegetables, lean protein, fruit, and water.",
                "source": "Thailand food-based dietary guidance",
            }
        )
    elif fast_food >= 3:
        cards.append(
            {
                "title": "Make fast food less frequent",
                "priority": "Food pattern",
                "why": "Frequent fast-food meals can increase energy intake and make vegetables, fruits, and balanced meals harder to fit in.",
                "action": "Plan a few Thai-style balanced meals around rice or grains, vegetables, lean protein, fruit, and less fried or sweet food.",
                "source": "Thailand food-based dietary guidance",
            }
        )

    if high_calorie >= 1:
        cards.append(
            {
                "title": "Watch frequent high-calorie food",
                "priority": "Food choice",
                "why": "Your second-form style answer says high-calorie food is frequent, which matches an important variable in obesity survey research.",
                "action": "Keep the food you enjoy, but reduce portion size or frequency and pair meals with vegetables or fruit.",
                "source": "E3S obesity ML lifestyle-variable study",
            }
        )

    if vegetables < 2:
        cards.append(
            {
                "title": "Add vegetables to one meal first",
                "priority": "Food balance",
                "why": "Vegetable frequency is low. In the reference paper, vegetable intake was one of the useful lifestyle variables.",
                "action": "Add one vegetable serving to lunch or dinner before changing the whole diet.",
                "source": "Thailand food-based dietary guidance",
            }
        )

    if between_meals >= 3:
        cards.append(
            {
                "title": "Plan snacks between meals",
                "priority": "Eating pattern",
                "why": "Frequent food between meals was one of the strongest variables in the reference obesity ML paper.",
                "action": "Prepare planned snacks such as fruit, yogurt, or nuts, and avoid turning every break into a high-calorie snack.",
                "source": "E3S obesity ML lifestyle-variable study",
            }
        )

    if main_meals > 4:
        cards.append(
            {
                "title": "Check meal frequency",
                "priority": "Eating pattern",
                "why": "Your main-meal answer is unusually high, so the app flags it for review rather than treating it as automatically bad.",
                "action": "Check whether this answer means true meals or snacks. For research data, this may need cleaning before training.",
                "source": "Survey data quality check",
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
                "action": "Track habits gently and consider discussing weight, blood pressure, or nutrition questions with a health professional.",
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
