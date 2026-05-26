from dataclasses import dataclass


@dataclass(frozen=True)
class ProbabilityTier:
    key: str
    label: str
    lower: float
    upper: float
    explanation: str


PROBABILITY_TIERS = [
    ProbabilityTier(
        key="very_low",
        label="Very low risk tier",
        lower=0.0,
        upper=0.20,
        explanation="The model sees only a small pattern match with the current obesity-risk training examples.",
    ),
    ProbabilityTier(
        key="low",
        label="Low risk tier",
        lower=0.20,
        upper=0.40,
        explanation="The model sees some risk signals, but the overall probability is still below the middle range.",
    ),
    ProbabilityTier(
        key="moderate",
        label="Moderate risk tier",
        lower=0.40,
        upper=0.60,
        explanation="The model sees a mixed profile, so the result deserves closer habit review.",
    ),
    ProbabilityTier(
        key="high",
        label="High risk tier",
        lower=0.60,
        upper=0.80,
        explanation="The model sees several answers that look similar to higher-risk training examples.",
    ),
    ProbabilityTier(
        key="very_high",
        label="Very high risk tier",
        lower=0.80,
        upper=1.01,
        explanation="The model sees a strong pattern match with higher-risk training examples.",
    ),
]


def classify_probability(probability: float) -> dict[str, object]:
    bounded = max(0.0, min(1.0, float(probability)))
    for tier in PROBABILITY_TIERS:
        if tier.lower <= bounded < tier.upper:
            return {
                "risk_tier": tier.key,
                "risk_tier_label": tier.label,
                "risk_tier_range": _format_range(tier.lower, tier.upper),
                "risk_tier_explanation": tier.explanation,
                "risk_band": coarse_band_from_probability(bounded),
            }
    return classify_probability(1.0)


def coarse_band_from_probability(probability: float) -> str:
    if probability < 0.35:
        return "low"
    if probability < 0.65:
        return "medium"
    return "high"


def _format_range(lower: float, upper: float) -> str:
    top = min(100.0, (upper * 100) - 0.1)
    return f"{lower * 100:.0f}-{top:.1f}%"


def asian_bmi_risk_tier(bmi: float) -> dict[str, str]:
    value = float(bmi)
    if value < 18.5:
        return {
            "key": "below_range",
            "label": "Below usual adult BMI screening range",
            "detail": "BMI is below 18.5. Weight status needs a different wellness conversation, not weight-loss advice.",
        }
    if value < 23:
        return {
            "key": "lower_risk",
            "label": "Lower-risk Asian adult BMI screening range",
            "detail": "BMI is between 18.5 and 22.9, a lower-risk screening range for many Asian adult references.",
        }
    if value < 25:
        return {
            "key": "increased_risk",
            "label": "Increased-risk Asian adult BMI screening range",
            "detail": "BMI is between 23.0 and 24.9, where Asian adult screening references start flagging higher metabolic risk.",
        }
    if value < 30:
        return {
            "key": "high_risk",
            "label": "High-risk Asian adult BMI screening range",
            "detail": "BMI is between 25.0 and 29.9, a high-risk screening range in Asian adult references.",
        }
    return {
        "key": "very_high_risk",
        "label": "Very-high-risk adult BMI screening range",
        "detail": "BMI is 30.0 or higher, which is also an obesity class threshold in many international references.",
    }
