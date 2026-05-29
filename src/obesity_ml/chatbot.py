from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from obesity_ml.config import CHATBOT_MODEL_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DATA_PATH = PROJECT_ROOT / "data" / "chatbot_training.json"

KNOWLEDGE_BASE: dict[str, dict] = {
    "greeting": {
        "en": {
            "answer": "Hi! I'm Beast 1.0 — your O-Beast wellness assistant. I can answer questions about obesity causes, prevention, treatment, diet, exercise, sleep, BMI, and genetics. Tap a chip below or type your question!",
            "source": "",
        },
        "th": {
            "answer": "สวัสดี! ฉัน Beast 1.0 — ผู้ช่วยด้านสุขภาพของ O-Beast ฉันสามารถตอบคำถามเกี่ยวกับสาเหตุ การป้องกัน การรักษาโรคอ้วน อาหาร การออกกำลังกาย การนอนหลับ ค่า BMI และพันธุกรรม กดปุ่มด้านล่างหรือพิมพ์คำถามได้เลย!",
            "source": "",
        },
    },
    "causes": {
        "en": {
            "answer": "Obesity develops from a combination of factors: consuming more calories than you burn (caloric surplus), low physical activity, poor diet quality (fast food and sugary drinks), insufficient sleep, chronic stress, genetics, and certain medications. According to the WHO and CDC, no single cause is responsible — it is the interaction of lifestyle and biology.",
            "source": "WHO, CDC",
        },
        "th": {
            "answer": "โรคอ้วนเกิดจากหลายปัจจัยร่วมกัน ได้แก่ การบริโภคแคลอรีมากกว่าที่ร่างกายเผาผลาญ การขาดการออกกำลังกาย อาหารที่มีคุณภาพต่ำ การนอนหลับไม่เพียงพอ ความเครียด พันธุกรรม และยาบางชนิด องค์การอนามัยโลกระบุว่าไม่มีสาเหตุเดียว แต่เป็นปฏิสัมพันธ์ระหว่างพฤติกรรมและชีววิทยา",
            "source": "WHO, CDC",
        },
    },
    "prevention": {
        "en": {
            "answer": "The WHO recommends: maintain caloric balance, do at least 150 minutes of moderate aerobic activity per week, limit screen time, get 7–9 hours of sleep per night, eat fruits and vegetables daily, and avoid sugary drinks and ultra-processed food. Small consistent habits over time prevent weight gain effectively.",
            "source": "WHO, CDC",
        },
        "th": {
            "answer": "องค์การอนามัยโลกแนะนำ: รักษาสมดุลแคลอรี ออกกำลังกายระดับปานกลางอย่างน้อย 150 นาทีต่อสัปดาห์ จำกัดเวลาหน้าจอ นอนหลับ 7–9 ชั่วโมงต่อคืน กินผักผลไม้ทุกวัน และหลีกเลี่ยงเครื่องดื่มน้ำตาลสูงและอาหารแปรรูป นิสัยเล็กๆ ที่สม่ำเสมอสามารถป้องกันน้ำหนักที่เพิ่มขึ้นได้",
            "source": "WHO, CDC",
        },
    },
    "treatment": {
        "en": {
            "answer": "WHO clinical guidelines recommend: reduce daily calorie intake by 500 kcal, increase physical activity to 150–300 minutes per week, join a behavioural support programme, and consult a doctor if BMI > 30. Surgical options (bariatric surgery) may be considered for BMI > 40 with related health conditions.",
            "source": "WHO Clinical Guidelines",
        },
        "th": {
            "answer": "แนวทางทางคลินิกขององค์การอนามัยโลกแนะนำ: ลดแคลอรีวันละ 500 กิโลแคลอรี เพิ่มการออกกำลังกาย 150–300 นาทีต่อสัปดาห์ เข้าร่วมโปรแกรมปรับพฤติกรรม และปรึกษาแพทย์หาก BMI > 30 การผ่าตัด (bariatric surgery) อาจพิจารณาสำหรับ BMI > 40 ร่วมกับปัญหาสุขภาพที่เกี่ยวข้อง",
            "source": "WHO Clinical Guidelines",
        },
    },
    "diet": {
        "en": {
            "answer": "A healthy diet includes: whole grains, lean protein (fish, chicken, legumes), vegetables (half your plate), limited added sugar and saturated fat. Thai MOPH guidelines suggest eating 3 meals a day without skipping breakfast, reducing fried food, and choosing steamed or grilled dishes. Limit fast food to no more than once a week.",
            "source": "Thai MOPH, WHO",
        },
        "th": {
            "answer": "อาหารเพื่อสุขภาพ: ธัญพืชไม่ขัดสี โปรตีนไขมันต่ำ (ปลา ไก่ ถั่ว) ผักครึ่งจาน จำกัดน้ำตาลและไขมันอิ่มตัว กรมอนามัยแนะนำ 3 มื้อต่อวันไม่ข้ามอาหารเช้า ลดอาหารทอด เลือกอาหารนึ่งหรือย่าง และจำกัดอาหารจานด่วนไม่เกินสัปดาห์ละครั้ง",
            "source": "Thai MOPH, WHO",
        },
    },
    "exercise": {
        "en": {
            "answer": "WHO recommends adults get 150–300 minutes of moderate-intensity aerobic activity per week (brisk walking, swimming, cycling) or 75–150 minutes of vigorous activity. Muscle-strengthening exercises on 2+ days per week are also beneficial. For students, team sports, running, or dancing count as moderate activity.",
            "source": "WHO, ACSM",
        },
        "th": {
            "answer": "องค์การอนามัยโลกแนะนำผู้ใหญ่ควรออกกำลังกายระดับปานกลาง 150–300 นาทีต่อสัปดาห์ (เดินเร็ว ว่ายน้ำ ปั่นจักรยาน) หรือระดับหนัก 75–150 นาที ควรฝึกเสริมสร้างกล้ามเนื้ออย่างน้อย 2 วันต่อสัปดาห์ สำหรับนักเรียน กีฬาทีม วิ่ง หรือเต้นรำนับเป็นการออกกำลังกายระดับปานกลาง",
            "source": "WHO, ACSM",
        },
    },
    "sleep": {
        "en": {
            "answer": "Poor sleep raises ghrelin (hunger hormone) and lowers leptin (fullness hormone), increasing appetite and cravings for high-calorie food. The CDC recommends 8–10 hours for teenagers and 7–9 hours for adults. Good sleep hygiene (no screens 1 hour before bed, consistent schedule) supports healthy weight.",
            "source": "CDC, Sleep Foundation",
        },
        "th": {
            "answer": "การนอนหลับไม่เพียงพอทำให้เกรลิน (ฮอร์โมนหิว) เพิ่มขึ้นและเลปติน (ฮอร์โมนอิ่ม) ลดลง ส่งผลให้อยากกินอาหารแคลอรีสูง CDC แนะนำวัยรุ่นควรนอน 8–10 ชั่วโมง และผู้ใหญ่ 7–9 ชั่วโมง การปรับสุขลักษณะการนอน (ปิดหน้าจอก่อนนอน 1 ชั่วโมง นอนตื่นเวลาเดิม) ช่วยควบคุมน้ำหนักได้",
            "source": "CDC, Sleep Foundation",
        },
    },
    "bmi": {
        "en": {
            "answer": "BMI = weight (kg) ÷ height² (m²). WHO: < 18.5 Underweight, 18.5–24.9 Normal, 25–29.9 Overweight, ≥ 30 Obese. For Asian populations including Thailand, the Thai MOPH uses lower cut-offs: ≥ 23 at-risk, ≥ 25 overweight, ≥ 30 obese. BMI is a screening tool only — it does not directly measure body fat.",
            "source": "WHO, Thai MOPH",
        },
        "th": {
            "answer": "ค่า BMI = น้ำหนัก (กก.) ÷ ส่วนสูง² (ม.) เกณฑ์ WHO: < 18.5 น้ำหนักต่ำ, 18.5–24.9 ปกติ, 25–29.9 น้ำหนักเกิน, ≥ 30 อ้วน สำหรับคนเอเชียรวมถึงไทย กรมอนามัยใช้เกณฑ์: ≥ 23 เสี่ยง, ≥ 25 น้ำหนักเกิน, ≥ 30 อ้วน BMI เป็นเครื่องมือคัดกรองเท่านั้น",
            "source": "WHO, Thai MOPH",
        },
    },
    "genetics": {
        "en": {
            "answer": "Genetics account for 40–70% of BMI variation (NIH). If one parent has obesity, a child has ~40% risk; if both parents do, the risk rises to 70–80%. However, genetics set a predisposition — not a destiny. Healthy lifestyle choices can significantly reduce genetic risk through gene-environment interaction.",
            "source": "NIH, CDC",
        },
        "th": {
            "answer": "พันธุกรรมมีผลต่อค่า BMI ถึง 40–70% (NIH) หากพ่อหรือแม่คนหนึ่งเป็นโรคอ้วน บุตรมีความเสี่ยงประมาณ 40% หากทั้งสองคน ความเสี่ยงสูงขึ้นเป็น 70–80% แต่พันธุกรรมเป็นแค่แนวโน้ม ไม่ใช่ชะตากรรม การเลือกวิถีชีวิตที่ดีสามารถลดความเสี่ยงทางพันธุกรรมได้",
            "source": "NIH, CDC",
        },
    },
    "risk_factors": {
        "en": {
            "answer": "Major modifiable risk factors: excessive screen time (> 4 hrs/day linked to adolescent obesity), frequent fast food, sugary drinks, insufficient physical activity, chronic stress, skipping breakfast, and poor sleep. Non-modifiable factors include genetics, age, and sex. Socioeconomic status also plays a role (WHO, CDC).",
            "source": "WHO, CDC",
        },
        "th": {
            "answer": "ปัจจัยเสี่ยงที่ปรับเปลี่ยนได้: เวลาหน้าจอมากเกินไป (> 4 ชั่วโมง/วัน) บริโภคอาหารจานด่วนบ่อย ดื่มเครื่องดื่มน้ำตาลสูง ขาดการออกกำลังกาย ความเครียดเรื้อรัง ข้ามมื้ออาหาร และนอนหลับไม่เพียงพอ ปัจจัยที่เปลี่ยนไม่ได้ ได้แก่ พันธุกรรม อายุ และเพศ",
            "source": "WHO, CDC",
        },
    },
}

_CONTEXT_PREFIXES: dict[str, dict[str, dict[str, str]]] = {
    "treatment": {
        "en": {
            "Very High Risk": "With your Very High risk score ({prob}%), WHO clinical guidelines recommend immediate action: consult a doctor or dietitian, reduce calorie intake by 500 kcal/day, start with 30 min of brisk walking 5 days/week, and join a structured weight-management programme. ",
            "High Risk": "With your High risk score ({prob}%), WHO recommends: create a 300–500 kcal daily deficit, increase to 150–250 min of moderate exercise per week, track your food intake, and consider talking to a school nurse or doctor. ",
            "Moderate Risk": "With your Moderate risk score ({prob}%), WHO suggests acting now: cut one high-calorie snack per day, add 30 min of activity 3 days/week, and replace sugary drinks with water. ",
            "Low Risk": "With your Low risk score ({prob}%), keep up your current habits. If you have concerns, talking to a school nurse is a good first step. ",
            "Very Low Risk": "With your Very Low risk score ({prob}%), you are doing great! WHO recommends annual weight monitoring to stay on track. ",
        },
        "th": {
            "Very High Risk": "จากผลความเสี่ยงสูงมากของคุณ ({prob}%) แนวทาง WHO แนะนำ: ปรึกษาแพทย์ทันที ลดแคลอรีวันละ 500 กิโลแคลอรี เริ่มเดินเร็ว 30 นาที 5 วัน/สัปดาห์ และเข้าร่วมโปรแกรมจัดการน้ำหนัก ",
            "High Risk": "จากผลความเสี่ยงสูงของคุณ ({prob}%) WHO แนะนำ: สร้างการขาดดุลแคลอรี 300–500 กิโลแคลอรีต่อวัน เพิ่มการออกกำลังกาย 150–250 นาที/สัปดาห์ และปรึกษาพยาบาลโรงเรียน ",
            "Moderate Risk": "จากผลความเสี่ยงปานกลางของคุณ ({prob}%) WHO แนะนำ: ลดของขบเคี้ยวแคลอรีสูง เพิ่มกิจกรรม 30 นาที 3 วัน/สัปดาห์ และเปลี่ยนเครื่องดื่มน้ำตาลเป็นน้ำเปล่า ",
            "Low Risk": "จากผลความเสี่ยงต่ำของคุณ ({prob}%) รักษาพฤติกรรมที่ดีในปัจจุบัน หากมีข้อกังวล ปรึกษาพยาบาลโรงเรียนเป็นขั้นตอนแรกที่ดี ",
            "Very Low Risk": "จากผลความเสี่ยงต่ำมากของคุณ ({prob}%) ยอดเยี่ยม! WHO แนะนำการตรวจสอบน้ำหนักประจำปีเพื่อติดตามผล ",
        },
    },
    "prevention": {
        "en": {
            "Very High Risk": "Based on your Very High risk result ({prob}%), WHO recommends: reduce calorie intake, limit fast food to once per week, cut all sugary drinks, get 7–9 hours sleep, and aim for 150 min/week of exercise. ",
            "High Risk": "Based on your High risk result ({prob}%), WHO recommends: 150 min exercise per week, 5 servings of vegetables and fruit daily, limit fast food, and a regular sleep schedule. ",
            "Moderate Risk": "Based on your Moderate risk result ({prob}%), WHO recommends: replace one processed snack per day with fruit, walk at least 30 min/day, reduce screen time before bed, and eat breakfast daily. ",
            "Low Risk": "Your Low risk result ({prob}%) shows your habits are largely healthy. Keep up regular physical activity and a balanced diet to maintain this. ",
            "Very Low Risk": "Your Very Low risk result ({prob}%) is excellent — keep doing what you are doing! Continue physical activity and a varied diet for long-term health. ",
        },
        "th": {
            "Very High Risk": "จากผลความเสี่ยงสูงมากของคุณ ({prob}%) WHO แนะนำ: ลดแคลอรี จำกัดอาหารจานด่วนสัปดาห์ละครั้ง หลีกเลี่ยงเครื่องดื่มน้ำตาล และนอนหลับ 7–9 ชั่วโมง ",
            "High Risk": "จากผลความเสี่ยงสูงของคุณ ({prob}%) WHO แนะนำ: ออกกำลังกาย 150 นาทีต่อสัปดาห์ กินผักผลไม้ 5 มื้อต่อวัน และนอนเป็นเวลา ",
            "Moderate Risk": "จากผลความเสี่ยงปานกลางของคุณ ({prob}%) WHO แนะนำ: เปลี่ยนของขบเคี้ยวแปรรูปเป็นผลไม้ เดิน 30 นาทีต่อวัน ลดเวลาหน้าจอก่อนนอน ",
            "Low Risk": "ผลความเสี่ยงต่ำของคุณ ({prob}%) บ่งชี้ว่าพฤติกรรมดีอยู่แล้ว รักษาการออกกำลังกายและอาหารสมดุลต่อไป ",
            "Very Low Risk": "ผลความเสี่ยงต่ำมากของคุณ ({prob}%) ดีเยี่ยม — ทำต่อไปเลย! รักษากิจกรรมทางกายและอาหารหลากหลายเพื่อสุขภาพระยะยาว ",
        },
    },
}

_cached_model: Optional[Pipeline] = None


def load_chatbot() -> Pipeline:
    global _cached_model
    if _cached_model is None:
        _cached_model = joblib.load(CHATBOT_MODEL_PATH)
    return _cached_model


def get_answer(intent: str, lang: str, context: Optional[dict] = None) -> dict:
    lang = lang if lang in ("en", "th") else "en"
    kb = KNOWLEDGE_BASE.get(intent, KNOWLEDGE_BASE["greeting"])
    entry = kb.get(lang, kb.get("en", {"answer": "", "source": ""}))
    answer = entry["answer"]
    source = entry["source"]

    if context and intent in _CONTEXT_PREFIXES:
        risk_tier = context.get("risk_tier", "")
        probability = context.get("probability", 0.0)
        prob_pct = round(float(probability) * 100, 1)
        prefixes = _CONTEXT_PREFIXES[intent][lang]
        prefix = prefixes.get(risk_tier, "")
        if prefix:
            answer = prefix.format(prob=prob_pct) + answer

    return {"answer": answer, "source": source}


def train_chatbot(data_path: Path, model_path: Path) -> dict:
    rows = json.loads(Path(data_path).read_text(encoding="utf-8"))
    texts = [r["text"] for r in rows]
    labels = [r["intent"] for r in rows]

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                sublinear_tf=True,
                min_df=1,
            ),
        ),
        ("clf", LogisticRegression(C=1.0, max_iter=1000)),
    ])

    scores = cross_val_score(pipeline, texts, labels, cv=5, scoring="accuracy")
    pipeline.fit(texts, labels)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    return {
        "cv_mean": float(scores.mean()),
        "cv_std": float(scores.std()),
        "classes": list(pipeline.classes_),
    }


def train_chatbot_cli() -> None:
    result = train_chatbot(TRAINING_DATA_PATH, CHATBOT_MODEL_PATH)
    print(f"Trained. CV accuracy: {result['cv_mean']:.3f} ± {result['cv_std']:.3f}")
    print(f"Classes: {result['classes']}")
    print(f"Model saved to {CHATBOT_MODEL_PATH}")
