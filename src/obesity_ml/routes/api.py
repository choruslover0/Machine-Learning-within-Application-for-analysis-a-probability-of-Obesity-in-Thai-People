"""JSON API endpoints: probability prediction and chatbot replies."""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from obesity_ml.predict import predict_probability

router = APIRouter()

Transportation = Literal[
    "Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"
]


class ObesityInput(BaseModel):
    """Full UCI ObesityDataSet question set (Mendoza Palechor & de la Hoz, 2019)."""

    age: int = Field(..., ge=14, le=100)
    sex: Literal["M", "F"] = Field(..., examples=["M", "F"])
    height_cm: float = Field(..., ge=80, le=230)
    weight_kg: float = Field(..., ge=20, le=250)
    family_history_obesity: int = Field(..., ge=0, le=1)
    high_calorie_food_frequency: int = Field(..., ge=0, le=1)   # FAVC
    vegetable_frequency: float = Field(..., ge=1, le=3)         # FCVC
    main_meals_per_day: float = Field(..., ge=1, le=4)          # NCP
    food_between_meals_frequency: int = Field(..., ge=0, le=3)  # CAEC
    smoke: int = Field(..., ge=0, le=1)                         # SMOKE
    water_daily: float = Field(..., ge=1, le=3)                 # CH2O
    calorie_monitoring: int = Field(..., ge=0, le=1)            # SCC
    physical_activity_freq: float = Field(..., ge=0, le=3)      # FAF
    screen_time_band: int = Field(..., ge=0, le=2)              # TUE
    alcohol_frequency: int = Field(..., ge=0, le=3)             # CALC
    transportation: Transportation = Field(..., examples=["Public_Transportation"])


class ChatRequest(BaseModel):
    message: str
    lang: str = "auto"
    context: dict | None = None


@router.post("/predict")
def predict_api(payload: ObesityInput) -> dict:
    return predict_probability(payload.model_dump())


@router.post("/chat")
def chat_endpoint(payload: ChatRequest) -> dict:
    from obesity_ml.chatbot import chat

    return chat(payload.message, payload.lang, payload.context)
