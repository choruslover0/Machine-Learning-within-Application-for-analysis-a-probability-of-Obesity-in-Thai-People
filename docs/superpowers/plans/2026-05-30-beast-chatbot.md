# Beast 1.0 Chatbot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bilingual (EN/TH) floating Q&A chatbot named Beast 1.0 to all O-Beast pages, backed by a trained TF-IDF + Logistic Regression intent classifier and a static obesity knowledge base, with personalised answers on the Result page.

**Architecture:** A new `chatbot.py` module handles classifier training, language detection, intent classification, and knowledge-base answer lookup. A `POST /chat` endpoint in `app.py` calls these functions. The chat widget (HTML + CSS + vanilla JS) is injected into every page via `page_shell()`. On the Result page, `predict_form()` passes `risk_tier` and `probability` as `data-` attributes so the JS includes them in every `/chat` request.

**Tech Stack:** scikit-learn (TF-IDF + LogisticRegression), joblib, langdetect, FastAPI (existing), vanilla JS fetch API.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `src/obesity_ml/chatbot.py` | Knowledge base, train, detect, answer lookup |
| Create | `data/chatbot_training.json` | ~200 labelled bilingual training sentences |
| Create | `tests/test_chatbot.py` | Unit tests for chatbot functions and `/chat` endpoint |
| Modify | `src/obesity_ml/config.py` | Add `CHATBOT_MODEL_PATH` |
| Modify | `src/obesity_ml/app.py` | Add `/chat` endpoint + inject widget into `page_shell()` |
| Modify | `pyproject.toml` | Add `obesity_ml-train-chat` CLI entry point |
| Modify | `requirements.txt` | Add `langdetect` |

---

## Task 1: Add `langdetect` dependency

**Files:**
- Modify: `requirements.txt`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add to requirements.txt**

```
langdetect>=1.0.9
```

Add after the `httpx` line.

- [ ] **Step 2: Add to pyproject.toml dependencies**

In the `dependencies` list under `[project]`, add:
```toml
"langdetect>=1.0.9",
```

- [ ] **Step 3: Install**

```bash
pip install langdetect>=1.0.9
```

Expected: installs without error.

- [ ] **Step 4: Verify**

```bash
python -c "from langdetect import detect; print(detect('hello world'))"
```

Expected output: `en`

- [ ] **Step 5: Commit**

```bash
git add requirements.txt pyproject.toml
git commit -m "feat: add langdetect dependency for chatbot language detection"
```

---

## Task 2: Add `CHATBOT_MODEL_PATH` to config

**Files:**
- Modify: `src/obesity_ml/config.py`
- Test: `tests/test_chatbot.py` (create file)

- [ ] **Step 1: Write failing test**

Create `tests/test_chatbot.py`:

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from obesity_ml.config import CHATBOT_MODEL_PATH


def test_chatbot_model_path_is_in_models_dir():
    assert CHATBOT_MODEL_PATH.parent.name == "models"
    assert CHATBOT_MODEL_PATH.name == "chatbot_model.joblib"
```

- [ ] **Step 2: Run — expect FAIL**

```bash
python -m pytest tests/test_chatbot.py::test_chatbot_model_path_is_in_models_dir -v
```

Expected: `ImportError: cannot import name 'CHATBOT_MODEL_PATH'`

- [ ] **Step 3: Add to config.py**

In `src/obesity_ml/config.py`, after the `MODEL_PATH` line add:

```python
CHATBOT_MODEL_PATH = MODEL_DIR / "chatbot_model.joblib"
```

- [ ] **Step 4: Run — expect PASS**

```bash
python -m pytest tests/test_chatbot.py::test_chatbot_model_path_is_in_models_dir -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/obesity_ml/config.py tests/test_chatbot.py
git commit -m "feat: add CHATBOT_MODEL_PATH to config"
```

---

## Task 3: Create training data

**Files:**
- Create: `data/chatbot_training.json`

- [ ] **Step 1: Write failing test** (add to `tests/test_chatbot.py`)

```python
import json

def test_training_data_structure():
    data_path = PROJECT_ROOT / "data" / "chatbot_training.json"
    assert data_path.exists(), "data/chatbot_training.json not found"
    rows = json.loads(data_path.read_text(encoding="utf-8"))
    assert isinstance(rows, list)
    assert len(rows) >= 100
    for row in rows:
        assert "text" in row
        assert "intent" in row
        assert "lang" in row
        assert row["lang"] in ("en", "th")

EXPECTED_INTENTS = {
    "greeting", "causes", "prevention", "treatment",
    "diet", "exercise", "sleep", "bmi", "genetics", "risk_factors",
}

def test_training_data_has_all_intents():
    data_path = PROJECT_ROOT / "data" / "chatbot_training.json"
    rows = json.loads(data_path.read_text(encoding="utf-8"))
    found = {r["intent"] for r in rows}
    assert found == EXPECTED_INTENTS
```

- [ ] **Step 2: Run — expect FAIL**

```bash
python -m pytest tests/test_chatbot.py::test_training_data_structure tests/test_chatbot.py::test_training_data_has_all_intents -v
```

Expected: `AssertionError: data/chatbot_training.json not found`

- [ ] **Step 3: Create `data/chatbot_training.json`**

```json
[
  {"text": "hi", "intent": "greeting", "lang": "en"},
  {"text": "hello", "intent": "greeting", "lang": "en"},
  {"text": "hey there", "intent": "greeting", "lang": "en"},
  {"text": "what can you do", "intent": "greeting", "lang": "en"},
  {"text": "how can you help me", "intent": "greeting", "lang": "en"},
  {"text": "who are you", "intent": "greeting", "lang": "en"},
  {"text": "what is beast 1.0", "intent": "greeting", "lang": "en"},
  {"text": "help", "intent": "greeting", "lang": "en"},
  {"text": "start", "intent": "greeting", "lang": "en"},
  {"text": "what topics do you cover", "intent": "greeting", "lang": "en"},
  {"text": "สวัสดี", "intent": "greeting", "lang": "th"},
  {"text": "หวัดดี", "intent": "greeting", "lang": "th"},
  {"text": "คุณทำอะไรได้บ้าง", "intent": "greeting", "lang": "th"},
  {"text": "ช่วยอะไรได้บ้าง", "intent": "greeting", "lang": "th"},
  {"text": "คุณคือใคร", "intent": "greeting", "lang": "th"},
  {"text": "beast 1.0 คืออะไร", "intent": "greeting", "lang": "th"},
  {"text": "ขอความช่วยเหลือ", "intent": "greeting", "lang": "th"},
  {"text": "เริ่มต้น", "intent": "greeting", "lang": "th"},
  {"text": "หัวข้ออะไรบ้าง", "intent": "greeting", "lang": "th"},
  {"text": "มีอะไรให้ถามได้บ้าง", "intent": "greeting", "lang": "th"},

  {"text": "What causes obesity?", "intent": "causes", "lang": "en"},
  {"text": "Why do people become obese?", "intent": "causes", "lang": "en"},
  {"text": "What makes people fat?", "intent": "causes", "lang": "en"},
  {"text": "Why am I overweight?", "intent": "causes", "lang": "en"},
  {"text": "causes of obesity", "intent": "causes", "lang": "en"},
  {"text": "what leads to weight gain", "intent": "causes", "lang": "en"},
  {"text": "why is obesity increasing", "intent": "causes", "lang": "en"},
  {"text": "what factors cause obesity", "intent": "causes", "lang": "en"},
  {"text": "what are the reasons for obesity", "intent": "causes", "lang": "en"},
  {"text": "how do people get obese", "intent": "causes", "lang": "en"},
  {"text": "สาเหตุของโรคอ้วนคืออะไร", "intent": "causes", "lang": "th"},
  {"text": "ทำไมคนถึงอ้วน", "intent": "causes", "lang": "th"},
  {"text": "ทำไมถึงน้ำหนักเกิน", "intent": "causes", "lang": "th"},
  {"text": "อะไรทำให้อ้วน", "intent": "causes", "lang": "th"},
  {"text": "สาเหตุน้ำหนักเพิ่ม", "intent": "causes", "lang": "th"},
  {"text": "เหตุใดโรคอ้วนจึงเพิ่มขึ้น", "intent": "causes", "lang": "th"},
  {"text": "ปัจจัยที่ทำให้เป็นโรคอ้วน", "intent": "causes", "lang": "th"},
  {"text": "ทำไมฉันถึงอ้วน", "intent": "causes", "lang": "th"},
  {"text": "สาเหตุที่ทำให้น้ำหนักขึ้น", "intent": "causes", "lang": "th"},
  {"text": "โรคอ้วนเกิดจากอะไร", "intent": "causes", "lang": "th"},

  {"text": "How to prevent obesity?", "intent": "prevention", "lang": "en"},
  {"text": "How can I avoid getting fat?", "intent": "prevention", "lang": "en"},
  {"text": "ways to prevent weight gain", "intent": "prevention", "lang": "en"},
  {"text": "how to stay healthy and not obese", "intent": "prevention", "lang": "en"},
  {"text": "tips to prevent obesity", "intent": "prevention", "lang": "en"},
  {"text": "what can I do to avoid obesity", "intent": "prevention", "lang": "en"},
  {"text": "how to maintain healthy weight", "intent": "prevention", "lang": "en"},
  {"text": "how to not become overweight", "intent": "prevention", "lang": "en"},
  {"text": "obesity prevention tips", "intent": "prevention", "lang": "en"},
  {"text": "what habits prevent obesity", "intent": "prevention", "lang": "en"},
  {"text": "ป้องกันโรคอ้วนอย่างไร", "intent": "prevention", "lang": "th"},
  {"text": "วิธีไม่ให้อ้วน", "intent": "prevention", "lang": "th"},
  {"text": "จะป้องกันน้ำหนักเกินได้อย่างไร", "intent": "prevention", "lang": "th"},
  {"text": "เคล็ดลับป้องกันโรคอ้วน", "intent": "prevention", "lang": "th"},
  {"text": "ทำอย่างไรถึงไม่อ้วน", "intent": "prevention", "lang": "th"},
  {"text": "วิธีรักษาน้ำหนักให้ปกติ", "intent": "prevention", "lang": "th"},
  {"text": "นิสัยอะไรป้องกันโรคอ้วน", "intent": "prevention", "lang": "th"},
  {"text": "จะหลีกเลี่ยงโรคอ้วนได้อย่างไร", "intent": "prevention", "lang": "th"},
  {"text": "การป้องกันน้ำหนักเกิน", "intent": "prevention", "lang": "th"},
  {"text": "วิธีมีสุขภาพดีไม่อ้วน", "intent": "prevention", "lang": "th"},

  {"text": "How to treat obesity?", "intent": "treatment", "lang": "en"},
  {"text": "How to cure obesity?", "intent": "treatment", "lang": "en"},
  {"text": "how to lose weight", "intent": "treatment", "lang": "en"},
  {"text": "how to reduce obesity", "intent": "treatment", "lang": "en"},
  {"text": "treatment for obesity", "intent": "treatment", "lang": "en"},
  {"text": "how to get rid of obesity", "intent": "treatment", "lang": "en"},
  {"text": "what is the treatment for being overweight", "intent": "treatment", "lang": "en"},
  {"text": "how to fight obesity", "intent": "treatment", "lang": "en"},
  {"text": "how do I reduce my weight", "intent": "treatment", "lang": "en"},
  {"text": "obesity treatment options", "intent": "treatment", "lang": "en"},
  {"text": "รักษาโรคอ้วนอย่างไร", "intent": "treatment", "lang": "th"},
  {"text": "วิธีรักษาโรคอ้วน", "intent": "treatment", "lang": "th"},
  {"text": "วิธีลดน้ำหนัก", "intent": "treatment", "lang": "th"},
  {"text": "ลดความอ้วนอย่างไร", "intent": "treatment", "lang": "th"},
  {"text": "การรักษาโรคอ้วน", "intent": "treatment", "lang": "th"},
  {"text": "จะแก้ไขโรคอ้วนได้อย่างไร", "intent": "treatment", "lang": "th"},
  {"text": "ทำอย่างไรให้น้ำหนักลด", "intent": "treatment", "lang": "th"},
  {"text": "วิธีต่อสู้กับโรคอ้วน", "intent": "treatment", "lang": "th"},
  {"text": "ตัวเลือกการรักษาน้ำหนักเกิน", "intent": "treatment", "lang": "th"},
  {"text": "จะลดน้ำหนักได้อย่างไร", "intent": "treatment", "lang": "th"},

  {"text": "What should I eat to avoid obesity?", "intent": "diet", "lang": "en"},
  {"text": "diet tips for weight loss", "intent": "diet", "lang": "en"},
  {"text": "what food is healthy", "intent": "diet", "lang": "en"},
  {"text": "what should I avoid eating", "intent": "diet", "lang": "en"},
  {"text": "healthy eating habits", "intent": "diet", "lang": "en"},
  {"text": "best foods to eat to lose weight", "intent": "diet", "lang": "en"},
  {"text": "foods that cause weight gain", "intent": "diet", "lang": "en"},
  {"text": "how many calories should I eat", "intent": "diet", "lang": "en"},
  {"text": "what is a healthy diet", "intent": "diet", "lang": "en"},
  {"text": "Thai food and obesity", "intent": "diet", "lang": "en"},
  {"text": "ควรกินอะไร", "intent": "diet", "lang": "th"},
  {"text": "อาหารที่ดีต่อสุขภาพ", "intent": "diet", "lang": "th"},
  {"text": "อาหารลดน้ำหนัก", "intent": "diet", "lang": "th"},
  {"text": "ควรหลีกเลี่ยงอาหารอะไร", "intent": "diet", "lang": "th"},
  {"text": "อาหารไทยเพื่อสุขภาพ", "intent": "diet", "lang": "th"},
  {"text": "กินอาหารอย่างไรไม่อ้วน", "intent": "diet", "lang": "th"},
  {"text": "อาหารที่ทำให้น้ำหนักขึ้น", "intent": "diet", "lang": "th"},
  {"text": "แคลอรีที่ควรกินต่อวัน", "intent": "diet", "lang": "th"},
  {"text": "โภชนาการที่ดีคืออะไร", "intent": "diet", "lang": "th"},
  {"text": "เคล็ดลับการกินที่ดี", "intent": "diet", "lang": "th"},

  {"text": "How much should I exercise?", "intent": "exercise", "lang": "en"},
  {"text": "what exercise is best for weight loss", "intent": "exercise", "lang": "en"},
  {"text": "how to start exercising", "intent": "exercise", "lang": "en"},
  {"text": "best workout for obesity", "intent": "exercise", "lang": "en"},
  {"text": "exercise tips for overweight people", "intent": "exercise", "lang": "en"},
  {"text": "how many minutes should I work out", "intent": "exercise", "lang": "en"},
  {"text": "what sports help with weight loss", "intent": "exercise", "lang": "en"},
  {"text": "physical activity and obesity", "intent": "exercise", "lang": "en"},
  {"text": "how often should I exercise", "intent": "exercise", "lang": "en"},
  {"text": "what is moderate exercise", "intent": "exercise", "lang": "en"},
  {"text": "ควรออกกำลังกายเท่าไหร่", "intent": "exercise", "lang": "th"},
  {"text": "กีฬาอะไรช่วยลดน้ำหนัก", "intent": "exercise", "lang": "th"},
  {"text": "วิธีเริ่มออกกำลังกาย", "intent": "exercise", "lang": "th"},
  {"text": "การออกกำลังกายสำหรับคนอ้วน", "intent": "exercise", "lang": "th"},
  {"text": "ออกกำลังกายกี่นาทีต่อวัน", "intent": "exercise", "lang": "th"},
  {"text": "กีฬาอะไรเผาผลาญแคลอรีมาก", "intent": "exercise", "lang": "th"},
  {"text": "การออกกำลังกายและโรคอ้วน", "intent": "exercise", "lang": "th"},
  {"text": "ควรออกกำลังกายบ่อยแค่ไหน", "intent": "exercise", "lang": "th"},
  {"text": "การออกกำลังกายระดับปานกลางคืออะไร", "intent": "exercise", "lang": "th"},
  {"text": "กิจกรรมทางกายที่แนะนำ", "intent": "exercise", "lang": "th"},

  {"text": "Does sleep affect weight?", "intent": "sleep", "lang": "en"},
  {"text": "how does sleep affect obesity", "intent": "sleep", "lang": "en"},
  {"text": "how many hours of sleep do I need", "intent": "sleep", "lang": "en"},
  {"text": "sleep and obesity relationship", "intent": "sleep", "lang": "en"},
  {"text": "can lack of sleep make you fat", "intent": "sleep", "lang": "en"},
  {"text": "how does poor sleep cause weight gain", "intent": "sleep", "lang": "en"},
  {"text": "sleep recommendations for teenagers", "intent": "sleep", "lang": "en"},
  {"text": "does staying up late cause obesity", "intent": "sleep", "lang": "en"},
  {"text": "sleep deprivation and obesity", "intent": "sleep", "lang": "en"},
  {"text": "how to sleep better", "intent": "sleep", "lang": "en"},
  {"text": "การนอนหลับส่งผลต่อน้ำหนักอย่างไร", "intent": "sleep", "lang": "th"},
  {"text": "นอนน้อยทำให้อ้วนไหม", "intent": "sleep", "lang": "th"},
  {"text": "ควรนอนกี่ชั่วโมง", "intent": "sleep", "lang": "th"},
  {"text": "ความสัมพันธ์ระหว่างการนอนและโรคอ้วน", "intent": "sleep", "lang": "th"},
  {"text": "นอนดึกทำให้อ้วนไหม", "intent": "sleep", "lang": "th"},
  {"text": "การนอนหลับไม่เพียงพอทำให้น้ำหนักขึ้น", "intent": "sleep", "lang": "th"},
  {"text": "เวลานอนที่แนะนำสำหรับวัยรุ่น", "intent": "sleep", "lang": "th"},
  {"text": "การพักผ่อนไม่เพียงพอและโรคอ้วน", "intent": "sleep", "lang": "th"},
  {"text": "วิธีนอนหลับให้ดีขึ้น", "intent": "sleep", "lang": "th"},
  {"text": "การนอนหลับกับการควบคุมน้ำหนัก", "intent": "sleep", "lang": "th"},

  {"text": "What is BMI?", "intent": "bmi", "lang": "en"},
  {"text": "how do I calculate my BMI", "intent": "bmi", "lang": "en"},
  {"text": "what is normal BMI", "intent": "bmi", "lang": "en"},
  {"text": "BMI and obesity", "intent": "bmi", "lang": "en"},
  {"text": "is my BMI healthy", "intent": "bmi", "lang": "en"},
  {"text": "what does a high BMI mean", "intent": "bmi", "lang": "en"},
  {"text": "BMI for Asian people", "intent": "bmi", "lang": "en"},
  {"text": "how to interpret BMI", "intent": "bmi", "lang": "en"},
  {"text": "what is the BMI formula", "intent": "bmi", "lang": "en"},
  {"text": "BMI categories", "intent": "bmi", "lang": "en"},
  {"text": "BMI คืออะไร", "intent": "bmi", "lang": "th"},
  {"text": "คำนวณค่า BMI อย่างไร", "intent": "bmi", "lang": "th"},
  {"text": "ค่า BMI ปกติคือเท่าไหร่", "intent": "bmi", "lang": "th"},
  {"text": "BMI กับโรคอ้วน", "intent": "bmi", "lang": "th"},
  {"text": "ค่า BMI ของฉันปกติไหม", "intent": "bmi", "lang": "th"},
  {"text": "ค่า BMI สูงหมายถึงอะไร", "intent": "bmi", "lang": "th"},
  {"text": "ค่า BMI สำหรับคนเอเชีย", "intent": "bmi", "lang": "th"},
  {"text": "สูตรคำนวณ BMI", "intent": "bmi", "lang": "th"},
  {"text": "ระดับค่า BMI", "intent": "bmi", "lang": "th"},
  {"text": "BMI บอกอะไรได้บ้าง", "intent": "bmi", "lang": "th"},

  {"text": "Does family history affect obesity?", "intent": "genetics", "lang": "en"},
  {"text": "is obesity genetic", "intent": "genetics", "lang": "en"},
  {"text": "can obesity be inherited", "intent": "genetics", "lang": "en"},
  {"text": "does genetics cause obesity", "intent": "genetics", "lang": "en"},
  {"text": "family history and weight", "intent": "genetics", "lang": "en"},
  {"text": "if my parents are obese will I be obese", "intent": "genetics", "lang": "en"},
  {"text": "how much does genetics influence weight", "intent": "genetics", "lang": "en"},
  {"text": "can I overcome genetic obesity", "intent": "genetics", "lang": "en"},
  {"text": "hereditary obesity", "intent": "genetics", "lang": "en"},
  {"text": "is being fat in my genes", "intent": "genetics", "lang": "en"},
  {"text": "ประวัติครอบครัวส่งผลต่อโรคอ้วนไหม", "intent": "genetics", "lang": "th"},
  {"text": "โรคอ้วนถ่ายทอดทางพันธุกรรมได้ไหม", "intent": "genetics", "lang": "th"},
  {"text": "พันธุกรรมทำให้อ้วนไหม", "intent": "genetics", "lang": "th"},
  {"text": "ถ้าพ่อแม่อ้วนลูกจะอ้วนไหม", "intent": "genetics", "lang": "th"},
  {"text": "พันธุกรรมกับน้ำหนักตัว", "intent": "genetics", "lang": "th"},
  {"text": "ยีนส่งผลต่อน้ำหนักอย่างไร", "intent": "genetics", "lang": "th"},
  {"text": "โรคอ้วนทางพันธุกรรม", "intent": "genetics", "lang": "th"},
  {"text": "จะเอาชนะพันธุกรรมที่ทำให้อ้วนได้ไหม", "intent": "genetics", "lang": "th"},
  {"text": "ความสัมพันธ์ระหว่างพันธุกรรมกับโรคอ้วน", "intent": "genetics", "lang": "th"},
  {"text": "โรคอ้วนในครอบครัว", "intent": "genetics", "lang": "th"},

  {"text": "What are risk factors for obesity?", "intent": "risk_factors", "lang": "en"},
  {"text": "what increases risk of obesity", "intent": "risk_factors", "lang": "en"},
  {"text": "does screen time cause obesity", "intent": "risk_factors", "lang": "en"},
  {"text": "does fast food cause obesity", "intent": "risk_factors", "lang": "en"},
  {"text": "what habits lead to obesity", "intent": "risk_factors", "lang": "en"},
  {"text": "am I at risk of obesity", "intent": "risk_factors", "lang": "en"},
  {"text": "what behaviours increase obesity risk", "intent": "risk_factors", "lang": "en"},
  {"text": "does stress cause weight gain", "intent": "risk_factors", "lang": "en"},
  {"text": "risk factors for being overweight", "intent": "risk_factors", "lang": "en"},
  {"text": "what makes someone more likely to be obese", "intent": "risk_factors", "lang": "en"},
  {"text": "ปัจจัยเสี่ยงของโรคอ้วนคืออะไร", "intent": "risk_factors", "lang": "th"},
  {"text": "อะไรเพิ่มความเสี่ยงโรคอ้วน", "intent": "risk_factors", "lang": "th"},
  {"text": "เล่นมือถือนานทำให้อ้วนไหม", "intent": "risk_factors", "lang": "th"},
  {"text": "กินอาหารจานด่วนทำให้อ้วนไหม", "intent": "risk_factors", "lang": "th"},
  {"text": "นิสัยอะไรทำให้เสี่ยงโรคอ้วน", "intent": "risk_factors", "lang": "th"},
  {"text": "ฉันเสี่ยงโรคอ้วนไหม", "intent": "risk_factors", "lang": "th"},
  {"text": "พฤติกรรมอะไรเพิ่มความเสี่ยงโรคอ้วน", "intent": "risk_factors", "lang": "th"},
  {"text": "ความเครียดทำให้น้ำหนักขึ้นไหม", "intent": "risk_factors", "lang": "th"},
  {"text": "ปัจจัยเสี่ยงน้ำหนักเกิน", "intent": "risk_factors", "lang": "th"},
  {"text": "อะไรทำให้มีแนวโน้มเป็นโรคอ้วนมากขึ้น", "intent": "risk_factors", "lang": "th"}
]
```

- [ ] **Step 4: Run — expect PASS**

```bash
python -m pytest tests/test_chatbot.py::test_training_data_structure tests/test_chatbot.py::test_training_data_has_all_intents -v
```

Expected: both `PASSED`

- [ ] **Step 5: Commit**

```bash
git add data/chatbot_training.json tests/test_chatbot.py
git commit -m "feat: add bilingual chatbot training data (200 sentences, 10 intents)"
```

---

## Task 4: Create `chatbot.py` — knowledge base + `get_answer`

**Files:**
- Create: `src/obesity_ml/chatbot.py`

- [ ] **Step 1: Write failing tests** (add to `tests/test_chatbot.py`)

```python
from obesity_ml.chatbot import get_answer

def test_get_answer_returns_required_keys():
    result = get_answer("causes", "en")
    assert "answer" in result
    assert "source" in result

def test_get_answer_en_causes_has_content():
    result = get_answer("causes", "en")
    assert len(result["answer"]) > 30
    assert "WHO" in result["source"] or "CDC" in result["source"]

def test_get_answer_th_diet_has_content():
    result = get_answer("diet", "th")
    assert len(result["answer"]) > 10

def test_get_answer_treatment_with_very_high_context():
    ctx = {"risk_tier": "Very High Risk", "probability": 0.72}
    result = get_answer("treatment", "en", context=ctx)
    assert "72" in result["answer"] or "Very High" in result["answer"]

def test_get_answer_prevention_with_low_context():
    ctx = {"risk_tier": "Low Risk", "probability": 0.20}
    result = get_answer("prevention", "th", context=ctx)
    assert len(result["answer"]) > 10

def test_get_answer_unknown_lang_defaults_to_en():
    result = get_answer("bmi", "fr")
    assert len(result["answer"]) > 10
```

- [ ] **Step 2: Run — expect FAIL**

```bash
python -m pytest tests/test_chatbot.py -k "get_answer" -v
```

Expected: `ImportError: cannot import name 'get_answer' from 'obesity_ml.chatbot'`

- [ ] **Step 3: Create `src/obesity_ml/chatbot.py`**

```python
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

# Context-personalised prefixes for treatment and prevention intents
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
```

- [ ] **Step 4: Run — expect PASS**

```bash
python -m pytest tests/test_chatbot.py -k "get_answer" -v
```

Expected: all `get_answer` tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/obesity_ml/chatbot.py
git commit -m "feat: add chatbot knowledge base and get_answer function"
```

---

## Task 5: Add `train_chatbot` and CLI entry point; train the model

**Files:**
- Modify: `src/obesity_ml/chatbot.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing test** (add to `tests/test_chatbot.py`)

```python
import tempfile
from pathlib import Path
from obesity_ml.chatbot import train_chatbot

def test_train_chatbot_creates_model_file():
    data_path = PROJECT_ROOT / "data" / "chatbot_training.json"
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "chatbot_model.joblib"
        result = train_chatbot(data_path, out_path)
        assert out_path.exists()
        assert result["cv_mean"] > 0.5
        assert set(result["classes"]) == {
            "greeting", "causes", "prevention", "treatment",
            "diet", "exercise", "sleep", "bmi", "genetics", "risk_factors",
        }
```

- [ ] **Step 2: Run — expect FAIL**

```bash
python -m pytest tests/test_chatbot.py::test_train_chatbot_creates_model_file -v
```

Expected: `ImportError: cannot import name 'train_chatbot'`

- [ ] **Step 3: Add `train_chatbot` to `chatbot.py`**

Add this function at the bottom of `src/obesity_ml/chatbot.py` (before any `if __name__` block):

```python
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
        ("clf", LogisticRegression(C=1.0, max_iter=1000, multi_class="multinomial")),
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
```

- [ ] **Step 4: Add CLI entry to `pyproject.toml`**

Add a `[project.scripts]` section after the `[tool.setuptools.package-data]` block:

```toml
[project.scripts]
obesity_ml-train-chat = "obesity_ml.chatbot:train_chatbot_cli"
```

- [ ] **Step 5: Reinstall package**

```bash
pip install -e .
```

- [ ] **Step 6: Run test — expect PASS**

```bash
python -m pytest tests/test_chatbot.py::test_train_chatbot_creates_model_file -v
```

Expected: `PASSED`

- [ ] **Step 7: Train the real model**

```bash
obesity_ml-train-chat
```

Expected output (values will vary):
```
Trained. CV accuracy: 0.850 ± 0.045
Classes: ['bmi', 'causes', 'diet', 'exercise', 'genetics', 'greeting', 'prevention', 'risk_factors', 'sleep', 'treatment']
Model saved to .../models/chatbot_model.joblib
```

- [ ] **Step 8: Commit**

```bash
git add src/obesity_ml/chatbot.py pyproject.toml models/chatbot_model.joblib
git commit -m "feat: add train_chatbot function and CLI entry point; include trained model"
```

---

## Task 6: Add `_detect_language` and `chat` function; test intent classification

**Files:**
- Modify: `src/obesity_ml/chatbot.py`

- [ ] **Step 1: Write failing tests** (add to `tests/test_chatbot.py`)

```python
from obesity_ml.chatbot import chat

def test_chat_causes_en():
    result = chat("What causes obesity?", lang="en")
    assert result["intent"] == "causes"
    assert result["detected_lang"] == "en"
    assert len(result["answer"]) > 20

def test_chat_diet_th():
    result = chat("ควรกินอะไร", lang="th")
    assert result["intent"] == "diet"
    assert result["detected_lang"] == "th"

def test_chat_bmi_auto_detect_en():
    result = chat("What is BMI?", lang="auto")
    assert result["intent"] == "bmi"
    assert result["detected_lang"] == "en"

def test_chat_low_confidence_returns_unknown():
    result = chat("xyzqqqabcdef999", lang="en")
    assert result["intent"] == "unknown"

def test_chat_with_context_treatment():
    ctx = {"risk_tier": "Very High Risk", "probability": 0.78}
    result = chat("How to treat obesity?", lang="en", context=ctx)
    assert result["intent"] == "treatment"
    assert "78" in result["answer"] or "Very High" in result["answer"]

def test_chat_response_has_all_keys():
    result = chat("hello", lang="en")
    for key in ("answer", "intent", "detected_lang", "source"):
        assert key in result
```

- [ ] **Step 2: Run — expect FAIL**

```bash
python -m pytest tests/test_chatbot.py -k "test_chat" -v
```

Expected: `ImportError: cannot import name 'chat'`

- [ ] **Step 3: Add `_detect_language` and `chat` to `chatbot.py`**

Add these functions in `src/obesity_ml/chatbot.py` (after `get_answer`, before `train_chatbot`):

```python
def _detect_language(text: str) -> str:
    try:
        from langdetect import detect
        lang = detect(text)
        return "th" if lang == "th" else "en"
    except Exception:
        return "en"


def _classify_intent(text: str, model: Pipeline) -> tuple[str, float]:
    proba = model.predict_proba([text])[0]
    idx = int(np.argmax(proba))
    return str(model.classes_[idx]), float(proba[idx])


def chat(
    message: str,
    lang: str = "auto",
    context: Optional[dict] = None,
) -> dict:
    detected_lang = _detect_language(message) if lang == "auto" else lang
    if detected_lang not in ("en", "th"):
        detected_lang = "en"

    try:
        model = load_chatbot()
    except FileNotFoundError:
        fallback = {
            "en": "Beast 1.0 is not trained yet. Run `obesity_ml-train-chat` first.",
            "th": "Beast 1.0 ยังไม่ได้รับการฝึก กรุณารัน `obesity_ml-train-chat` ก่อน",
        }
        return {"answer": fallback[detected_lang], "intent": "error", "detected_lang": detected_lang, "source": ""}

    intent, confidence = _classify_intent(message, model)

    if confidence < 0.45:
        fallback = {
            "en": "I'm not sure I understood that — try rephrasing, or tap a chip below.",
            "th": "ฉันไม่แน่ใจว่าเข้าใจถูกต้อง — ลองพิมพ์ใหม่ หรือกดปุ่มด้านล่าง",
        }
        return {"answer": fallback[detected_lang], "intent": "unknown", "detected_lang": detected_lang, "source": ""}

    result = get_answer(intent, detected_lang, context)
    return {
        "answer": result["answer"],
        "intent": intent,
        "detected_lang": detected_lang,
        "source": result["source"],
    }
```

- [ ] **Step 4: Run — expect PASS**

```bash
python -m pytest tests/test_chatbot.py -k "test_chat" -v
```

Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/obesity_ml/chatbot.py tests/test_chatbot.py
git commit -m "feat: add chat function with language detection and intent classification"
```

---

## Task 7: Add `POST /chat` endpoint to `app.py`

**Files:**
- Modify: `src/obesity_ml/app.py`

- [ ] **Step 1: Write failing tests** (add to `tests/test_chatbot.py`)

```python
from fastapi.testclient import TestClient
from obesity_ml.app import app

_client = TestClient(app)

def test_chat_endpoint_shape():
    resp = _client.post("/chat", json={"message": "What causes obesity?", "lang": "en"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "intent" in data
    assert "detected_lang" in data
    assert "source" in data

def test_chat_endpoint_unknown_returns_200():
    resp = _client.post("/chat", json={"message": "xyzqqqabc999", "lang": "en"})
    assert resp.status_code == 200
    assert resp.json()["intent"] == "unknown"

def test_chat_endpoint_with_context():
    resp = _client.post("/chat", json={
        "message": "how to treat obesity",
        "lang": "en",
        "context": {"risk_tier": "High Risk", "probability": 0.55},
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "treatment"
```

- [ ] **Step 2: Run — expect FAIL**

```bash
python -m pytest tests/test_chatbot.py -k "endpoint" -v
```

Expected: `404 Not Found` (endpoint missing)

- [ ] **Step 3: Add Pydantic model and endpoint to `app.py`**

At the top of `src/obesity_ml/app.py`, the `BaseModel` and `Field` imports already exist. Add `Optional` to the typing import:

```python
from typing import Literal, Optional
```

After the `ObesityInput` class, add:

```python
class ChatRequest(BaseModel):
    message: str
    lang: str = "auto"
    context: Optional[dict] = None
```

After the existing `@app.post("/predict")` endpoint, add:

```python
@app.post("/chat")
def chat_endpoint(payload: ChatRequest) -> dict:
    from obesity_ml.chatbot import chat
    return chat(payload.message, payload.lang, payload.context)
```

- [ ] **Step 4: Run — expect PASS**

```bash
python -m pytest tests/test_chatbot.py -k "endpoint" -v
```

Expected: all `PASSED`

- [ ] **Step 5: Commit**

```bash
git add src/obesity_ml/app.py tests/test_chatbot.py
git commit -m "feat: add POST /chat endpoint to FastAPI app"
```

---

## Task 8: Inject chat widget into `page_shell()`

**Files:**
- Modify: `src/obesity_ml/app.py`

- [ ] **Step 1: Write failing test** (add to `tests/test_chatbot.py`)

```python
def test_page_shell_contains_beast_fab():
    resp = _client.get("/")
    assert resp.status_code == 200
    assert 'id="beast-fab"' in resp.text

def test_page_shell_contains_chat_window():
    resp = _client.get("/")
    assert 'id="beast-chat"' in resp.text

def test_widget_shows_on_all_pages():
    for path in ("/", "/predictor", "/advice", "/methods"):
        resp = _client.get(path)
        assert 'id="beast-fab"' in resp.text, f"Widget missing on {path}"
```

- [ ] **Step 2: Run — expect FAIL**

```bash
python -m pytest tests/test_chatbot.py -k "beast_fab or chat_window or widget" -v
```

Expected: `AssertionError` (no `beast-fab` in HTML)

- [ ] **Step 3: Add `CHAT_WIDGET_STYLE` constant to `app.py`**

Add this constant after the existing `STYLE` string (before `class ObesityInput`):

```python
CHAT_WIDGET_STYLE = """
<style>
  .beast-fab {
    position: fixed; bottom: 20px; right: 20px; z-index: 1000;
    width: 72px; height: 72px; border-radius: 22px;
    background: linear-gradient(135deg, var(--violet), var(--hot), var(--sun));
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 3px; padding: 6px 6px 4px;
    box-shadow: 0 16px 36px rgba(225,48,108,.34);
    border: 0; cursor: pointer; color: white;
    transition: transform 180ms ease, box-shadow 180ms ease;
  }
  .beast-fab:hover { transform: translateY(-2px) scale(1.04); }
  .beast-fab img { width: 46px; height: 46px; object-fit: contain; border-radius: 8px; }
  .beast-fab-label { font-size: 9px; font-weight: 900; letter-spacing: .04em; font-family: inherit; }
  .beast-fab-badge {
    position: absolute; top: -4px; right: -4px;
    width: 16px; height: 16px; border-radius: 50%;
    background: #22c55e; border: 2px solid white; display: none;
  }
  .beast-chat {
    position: fixed; bottom: 104px; right: 16px; z-index: 1000;
    width: 310px; border-radius: 26px;
    background: rgba(255,255,255,0.97);
    border: 1px solid var(--line);
    box-shadow: 0 28px 72px rgba(21,21,26,.20);
    overflow: hidden; font-size: 13px;
    font-family: Inter, ui-sans-serif, system-ui, sans-serif;
  }
  .beast-head {
    padding: 12px 14px;
    background: linear-gradient(135deg, var(--violet), var(--hot), var(--sun));
    color: white; display: flex; align-items: center; gap: 9px;
  }
  .beast-head-img { width: 36px; height: 36px; object-fit: contain; border-radius: 10px; background: rgba(255,255,255,.18); flex-shrink: 0; }
  .beast-head-info strong { display: block; font-size: 14px; }
  .beast-head-info small { opacity: .82; font-size: 11px; }
  .beast-lang-toggle { margin-left: auto; display: flex; gap: 4px; background: rgba(255,255,255,.20); border-radius: 999px; padding: 3px; }
  .beast-lang-btn { padding: 3px 8px; border-radius: 999px; font-size: 10px; font-weight: 900; color: rgba(255,255,255,.75); border: 0; background: transparent; cursor: pointer; font-family: inherit; }
  .beast-lang-btn.active { background: rgba(255,255,255,.30); color: white; }
  .beast-ctx { margin: 8px 12px 0; border-radius: 14px; padding: 8px 12px; background: rgba(225,48,108,.08); border: 1px solid rgba(225,48,108,.22); font-size: 11px; color: var(--hot); font-weight: 800; line-height: 1.4; }
  .beast-msgs { padding: 12px; display: flex; flex-direction: column; gap: 8px; max-height: 220px; overflow-y: auto; }
  .beast-msg { line-height: 1.4; max-width: 88%; }
  .beast-bot { background: #f4f4f5; border-radius: 16px 16px 16px 4px; padding: 9px 12px; }
  .beast-user { background: linear-gradient(135deg, var(--violet), var(--hot)); color: white; border-radius: 16px 16px 4px 16px; padding: 9px 12px; align-self: flex-end; }
  .beast-src { display: block; margin-top: 4px; font-size: 10px; color: var(--muted); font-weight: 800; }
  .beast-chips { display: flex; flex-wrap: wrap; gap: 6px; padding: 4px 12px 10px; }
  .beast-chip { border: 1px solid rgba(225,48,108,.28); border-radius: 999px; padding: 5px 11px; font-size: 11px; font-weight: 900; color: var(--hot); background: rgba(225,48,108,.06); cursor: pointer; font-family: inherit; }
  .beast-form { display: flex; gap: 8px; padding: 10px 12px; border-top: 1px solid var(--line); }
  .beast-input { flex: 1; border: 1px solid var(--line); border-radius: 999px; padding: 8px 13px; font-size: 12px; outline: none; background: #fafafa; font-family: inherit; }
  .beast-input:focus { box-shadow: 0 0 0 3px rgba(225,48,108,.16); background: #fff; }
  .beast-send { width: 34px; height: 34px; border-radius: 50%; flex-shrink: 0; background: linear-gradient(135deg, var(--hot), var(--sun)); border: 0; color: white; font-size: 13px; cursor: pointer; display: grid; place-items: center; }
</style>
"""
```

- [ ] **Step 4: Add `chat_widget_html()` function to `app.py`**

Add this function after `producer_section_html()`:

```python
def chat_widget_html(risk_tier: str = "", probability: str = "") -> str:
    tier_attr = f'data-risk-tier="{escape(risk_tier, quote=True)}"'
    prob_attr = f'data-probability="{escape(probability, quote=True)}"'
    badge_style = 'style="display:block"' if risk_tier else ""
    return f"""
{CHAT_WIDGET_STYLE}
<button id="beast-fab" class="beast-fab" aria-label="Open Beast 1.0 chat" {tier_attr} {prob_attr}>
  <img src="/static/beast1-logo.png" alt="Beast 1.0 logo">
  <span class="beast-fab-label">Beast 1.0</span>
  <span class="beast-fab-badge" {badge_style}></span>
</button>
<div id="beast-chat" class="beast-chat" hidden>
  <div class="beast-head">
    <img class="beast-head-img" src="/static/beast1-logo.png" alt="Beast 1.0">
    <div class="beast-head-info"><strong>Beast 1.0</strong><small>Obesity Q&amp;A · Online</small></div>
    <div class="beast-lang-toggle">
      <button class="beast-lang-btn active" data-lang="en">EN</button>
      <button class="beast-lang-btn" data-lang="th">TH</button>
    </div>
  </div>
  <div id="beast-ctx" class="beast-ctx" hidden></div>
  <div id="beast-msgs" class="beast-msgs">
    <div class="beast-msg beast-bot">Hi! I&apos;m Beast 1.0 &#x1F981; Ask me about obesity — or tap a chip below.</div>
  </div>
  <div class="beast-chips">
    <button class="beast-chip" data-query="What causes obesity?">Causes</button>
    <button class="beast-chip" data-query="How to prevent obesity?">Prevention</button>
    <button class="beast-chip" data-query="What should I eat?">Diet</button>
    <button class="beast-chip" data-query="How much should I exercise?">Exercise</button>
  </div>
  <form id="beast-form" class="beast-form" onsubmit="return false">
    <input id="beast-input" class="beast-input" placeholder="Ask Beast 1.0…" autocomplete="off">
    <button type="submit" class="beast-send" aria-label="Send">&#x27A4;</button>
  </form>
</div>
<script>
(function(){{
  var fab=document.getElementById('beast-fab');
  var chat=document.getElementById('beast-chat');
  var msgs=document.getElementById('beast-msgs');
  var form=document.getElementById('beast-form');
  var inp=document.getElementById('beast-input');
  var ctx=document.getElementById('beast-ctx');
  var lang='auto';
  var tier=fab.dataset.riskTier||'';
  var prob=fab.dataset.probability||'';
  if(tier){{ctx.hidden=false;ctx.textContent='📊 Your result: '+tier+' ('+Math.round(parseFloat(prob)*100)+'%) — I’ll tailor my answers to your score.';}}
  fab.addEventListener('click',function(){{chat.hidden=!chat.hidden;if(!chat.hidden)inp.focus();}});
  document.querySelectorAll('.beast-lang-btn').forEach(function(b){{
    b.addEventListener('click',function(){{
      document.querySelectorAll('.beast-lang-btn').forEach(function(x){{x.classList.remove('active');}});
      b.classList.add('active');lang=b.dataset.lang;updateChips(lang);
      inp.placeholder=lang==='th'?'พิมพ์คำถาม…':'Ask Beast 1.0…';
    }});
  }});
  document.querySelectorAll('.beast-chip').forEach(function(c){{c.addEventListener('click',function(){{send(c.dataset.query);}});}});
  form.addEventListener('submit',function(){{var t=inp.value.trim();if(!t)return;inp.value='';send(t);}});
  function send(text){{
    bubble(text,'user');
    var body={{message:text,lang:lang}};
    if(tier)body.context={{risk_tier:tier,probability:parseFloat(prob)}};
    fetch('/chat',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(body)}})
      .then(function(r){{return r.json();}})
      .then(function(d){{if(lang==='auto'&&d.detected_lang)lang=d.detected_lang;bubble(d.answer,'bot',d.source);}})
      .catch(function(){{bubble('Sorry, something went wrong.','bot');}});
  }}
  function bubble(text,role,src){{
    var d=document.createElement('div');
    d.className='beast-msg beast-'+role;d.textContent=text;
    if(src){{var s=document.createElement('span');s.className='beast-src';s.textContent='Source: '+src;d.appendChild(s);}}
    msgs.appendChild(d);msgs.scrollTop=msgs.scrollHeight;
  }}
  function updateChips(l){{
    var labels={{en:['Causes','Prevention','Diet','Exercise'],th:['สาเหตุ','การป้องกัน','อาหาร','ออกกำลังกาย']}};
    var queries={{en:['What causes obesity?','How to prevent obesity?','What should I eat?','How much should I exercise?'],th:['สาเหตุของโรคอ้วนคืออะไร','ป้องกันโรคอ้วนอย่างไร','ควรกินอะไร','ควรออกกำลังกายเท่าไหร่']}};
    var lk=(l==='auto')?'en':l;
    document.querySelectorAll('.beast-chip').forEach(function(c,i){{c.textContent=labels[lk][i];c.dataset.query=queries[lk][i];}});
  }}
}})();
</script>
"""
```

- [ ] **Step 5: Modify `page_shell()` to inject the widget**

Change the `page_shell` signature and body in `app.py`:

```python
def page_shell(title: str, body: str, chat_context: dict | None = None) -> str:
    risk_tier = chat_context.get("risk_tier", "") if chat_context else ""
    probability = str(chat_context.get("probability", "")) if chat_context else ""
    widget = chat_widget_html(risk_tier, probability)
    return f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>{title}</title>
      {STYLE}
    </head>
    <body>
      <main>
        <nav class="nav">
          <div class="brand"><img class="brand-logo" src="/static/obeast-logo.svg" alt="O-Beast logo">O-Beast</div>
          <div class="nav-links">
            <a href="/">Home</a>
            <a href="/predictor">Predictor</a>
            <a href="/advice">Advice</a>
            <a href="/methods">Methods</a>
          </div>
        </nav>
        {body}
      </main>
      {widget}
    </body>
    </html>
    """
```

- [ ] **Step 6: Run — expect PASS**

```bash
python -m pytest tests/test_chatbot.py -k "beast_fab or chat_window or widget" -v
```

Expected: all `PASSED`

- [ ] **Step 7: Commit**

```bash
git add src/obesity_ml/app.py
git commit -m "feat: inject Beast 1.0 chat widget into all pages via page_shell()"
```

---

## Task 9: Add context awareness to Result page

**Files:**
- Modify: `src/obesity_ml/app.py`

- [ ] **Step 1: Write failing test** (add to `tests/test_chatbot.py`)

```python
def test_result_page_has_risk_tier_data_attribute():
    resp = _client.post("/predict-form", data={
        "age": "16", "sex": "M", "height_cm": "170", "weight_kg": "90",
        "physical_activity_hours_per_week": "1", "screen_time_hours_per_day": "8",
        "sleep_hours": "5", "fast_food_meals_per_week": "5",
        "sugary_drinks_per_day": "3", "family_history_obesity": "1",
    })
    assert resp.status_code == 200
    assert 'data-risk-tier=' in resp.text
    assert 'data-probability=' in resp.text
```

- [ ] **Step 2: Run — expect FAIL**

```bash
python -m pytest tests/test_chatbot.py::test_result_page_has_risk_tier_data_attribute -v
```

Expected: `AssertionError` (`data-risk-tier` not in HTML because `predict_form` doesn't pass `chat_context`)

- [ ] **Step 3: Update `predict_form` in `app.py`**

Find the `return page_shell("Result - SK Obesity ML", body)` line at the end of `predict_form()` and change it to:

```python
    return page_shell(
        "Result - SK Obesity ML",
        body,
        chat_context={
            "risk_tier": result["risk_tier_label"],
            "probability": result["obesity_probability"],
        },
    )
```

- [ ] **Step 4: Run — expect PASS**

```bash
python -m pytest tests/test_chatbot.py::test_result_page_has_risk_tier_data_attribute -v
```

Expected: `PASSED`

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all tests pass (including existing regression tests).

- [ ] **Step 6: Commit**

```bash
git add src/obesity_ml/app.py tests/test_chatbot.py
git commit -m "feat: pass risk tier context to Beast 1.0 widget on Result page"
```

---

## Task 10: Manual verification checklist

- [ ] Start the app: `uvicorn obesity_ml.app:app --reload`
- [ ] Open `http://localhost:8000` — confirm Beast 1.0 rounded-square button appears bottom-right
- [ ] Click the button — confirm chat window opens with logo in header and EN/TH toggle
- [ ] Type "What causes obesity?" — confirm answer appears with "Source: WHO, CDC"
- [ ] Click the "TH" toggle — confirm chips switch to Thai labels
- [ ] Click the "สาเหตุ" chip — confirm Thai answer returned
- [ ] Navigate to `/predictor`, submit the form — confirm widget appears on Result page
- [ ] On Result page, confirm pink context banner shows risk tier and probability
- [ ] Ask "how to treat obesity" on Result page — confirm answer is personalised with risk tier
- [ ] Navigate to `/advice` and `/methods` — confirm widget appears on both pages

---

## Self-Review

**Spec coverage:**
- ✅ Bilingual EN/TH answers — Task 4 (knowledge base), Task 6 (language detection)
- ✅ All 10 intent categories — Task 3 (training data), Task 4 (knowledge base)
- ✅ TF-IDF + LR classifier — Task 5 (train_chatbot)
- ✅ `POST /chat` endpoint — Task 7
- ✅ Widget on all pages — Task 8 (page_shell injection)
- ✅ Context-aware on Result page — Task 9
- ✅ Quick-reply chips — Task 8 (widget HTML)
- ✅ EN/TH language toggle — Task 8 (widget JS)
- ✅ Source citations in bot messages — Task 8 (bubble() function)
- ✅ Green badge on Result page FAB — Task 8 (badge_style logic)
- ✅ `/static/beast1-logo.png` used in widget — Task 8
- ✅ Fallback for low-confidence — Task 6 (confidence < 0.45)
- ✅ `obesity_ml-train-chat` CLI — Task 5

**Type consistency:** `get_answer` returns `{"answer": str, "source": str}`. `chat` returns `{"answer", "intent", "detected_lang", "source"}`. `/chat` endpoint returns the same dict. Task 8 JS reads `d.answer` and `d.source` — consistent.

**Placeholder scan:** No TBDs, no "implement later", no "add appropriate handling" without code.
