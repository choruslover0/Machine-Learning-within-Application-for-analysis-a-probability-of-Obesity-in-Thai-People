# Beast 1.0 — Obesity Q&A Chatbot Design Spec

**Date:** 2026-05-29  
**App:** O-Beast (FastAPI + server-side HTML)  
**Feature:** Floating chatbot widget named "Beast 1.0" embedded in all pages

---

## 1. Goals

- Answer user questions about obesity (causes, prevention, treatment, diet, exercise, sleep, BMI, genetics, risk factors) in English and Thai.
- Appear as a floating widget in the bottom-right corner of every page.
- On the Result page, personalise answers using the user's risk tier and probability score.
- Use a trained ML classifier (TF-IDF + Logistic Regression) to route questions to a knowledge base category, then return a pre-written rich answer with a WHO/CDC/MOPH citation.
- Show quick-reply chips so users can tap instead of type.

---

## 2. Architecture

### New files

| File | Purpose |
|---|---|
| `src/obesity_ml/chatbot.py` | Intent classifier training, language detection, answer lookup |
| `data/chatbot_training.json` | ~1,000 labelled bilingual training sentences (EN + TH) |
| `models/chatbot_model.joblib` | Trained TF-IDF + LogisticRegression artifact |

### Existing files modified

| File | Change |
|---|---|
| `src/obesity_ml/app.py` | Add `POST /chat` endpoint; inject chat widget HTML into `page_shell()` |
| `pyproject.toml` | Add `obesity_ml-train-chat` CLI entry point |
| `requirements.txt` | Add `langdetect` for language auto-detection |

### Data flow

```
User types message
      │
      ▼
POST /chat  { message, lang, context? }
      │
      ├─ langdetect → detected_lang (en / th)
      │
      ├─ TF-IDF vectorise message
      │
      ├─ LogisticRegression → intent category
      │
      ├─ Confidence < 0.45? → "I'm not sure, try rephrasing."
      │
      ├─ Look up answer[intent][detected_lang]
      │
      ├─ If context present → inject risk_tier + probability into answer template
      │
      └─ Return { answer, intent, detected_lang }
```

---

## 3. Intent Categories & Knowledge Base

Ten categories, each with ~100 training sentences (EN + TH combined) and one rich answer per language citing official sources.

| Intent | Topics covered | Sources |
|---|---|---|
| `greeting` | welcome, what can you do | — |
| `causes` | caloric surplus, sedentary lifestyle, genetics, sleep, stress, hormones, medications | WHO, CDC |
| `prevention` | caloric balance, 150 min/week exercise, screen time limits, sleep hygiene | WHO, CDC |
| `treatment` | dietary changes, exercise plans, behavioural therapy, clinical/surgical options | WHO Clinical Guidelines |
| `diet` | balanced meals, Thai dietary guidelines, foods to avoid, portion size | Thai MOPH, WHO |
| `exercise` | aerobic vs strength, duration, intensity, suitable activities for students | WHO, ACSM |
| `sleep` | sleep-obesity link, recommended hours, sleep hygiene tips | Sleep Foundation, CDC |
| `bmi` | BMI formula, normal ranges, limitations, Asian cut-off values | WHO, Thai MOPH |
| `genetics` | heritability, family history risk, gene-environment interaction | NIH, CDC |
| `risk_factors` | screen time, fast food, stress, socioeconomic factors, age, sex | WHO, CDC |

### Context-aware answer personalisation (Result page only)

When `context.risk_tier` is provided, the `treatment` and `prevention` answers prepend a personalised sentence:

- Very Low / Low risk → encouragement + maintain habits
- Moderate risk → specific actionable steps
- High / Very High risk → urgent WHO clinical recommendations + "consult a doctor"

---

## 4. Widget UI

**Trigger button (closed state):**
- Style: Rounded square, 68×68 px, border-radius 22px
- Background: O-Beast gradient (violet → hot pink → sun)
- Icon: `/static/beast1-logo.png` image (44px, object-fit cover), no emoji fallback
- On Result page only: green notification dot (badge) in top-right corner

**Chat window (open state):**
- Width: 310px, positioned bottom-right (16px from edges)
- Gradient header: live green dot + "Beast 1.0" title + EN/TH language toggle
- On Result page: pink context banner below header showing risk tier + probability
- Message area: bot bubbles (grey, left-aligned) + user bubbles (gradient, right-aligned)
- Each bot answer includes a small source citation line
- Quick-reply chips: 4 chips below messages, change label per page context
- Input row: text field + send button
- Fallback message for low-confidence predictions: "I'm not sure — try rephrasing, or tap a chip below."

**Language toggle:**
- Default: auto-detect from first user message using `langdetect`
- Toggle button in header lets user force EN or TH
- All chips, placeholder text, and bot answers switch to the selected language

---

## 5. Training Data Plan

**File:** `data/chatbot_training.json`

Structure:
```json
[
  { "text": "What causes obesity?", "intent": "causes", "lang": "en" },
  { "text": "สาเหตุของโรคอ้วนคืออะไร", "intent": "causes", "lang": "th" },
  ...
]
```

~100 sentences per category × 10 categories = ~1,000 rows total.  
Sentences cover varied phrasing, student vocabulary, and common misspellings.

**Training pipeline (`obesity_ml-train-chat` CLI):**
1. Load `data/chatbot_training.json`
2. TF-IDF vectorise (char n-grams 2–4 + word unigrams, sublinear TF, Thai-friendly tokenisation via whitespace)
3. Train LogisticRegression (C=1.0, max_iter=1000, multi-class)
4. 5-fold cross-validation → print per-class accuracy
5. Save TF-IDF + model to `models/chatbot_model.joblib`

---

## 6. API Endpoint

```
POST /chat
Content-Type: application/json

{
  "message": "how do I lose weight?",
  "lang": "auto",
  "context": {
    "risk_tier": "Very High Risk",
    "probability": 0.72
  }
}

→ 200 OK
{
  "answer": "With a Very High risk score (72%), WHO recommends...",
  "intent": "treatment",
  "detected_lang": "en"
}
```

Low-confidence fallback (confidence < 0.45):
```json
{ "answer": "I'm not sure I understood that — try rephrasing, or tap a chip below.", "intent": "unknown", "detected_lang": "en" }
```

---

## 7. Frontend Integration

`page_shell()` in `app.py` injects the chat widget HTML + inline JS at the bottom of every page's `<body>`.

The widget JS:
- Toggles open/close on FAB click
- Sends `fetch("POST /chat", ...)` on message submit or chip click
- Appends bot/user bubbles to the message area
- On Result page, reads `data-risk-tier` and `data-probability` from the FAB element and includes them in every `/chat` request

The Result page endpoint (`predict_form`) sets these attributes on the FAB when rendering.

---

## 8. Testing

- Unit tests for `chatbot.py`: intent detection returns correct category for sample EN + TH sentences
- Unit tests for `/chat` endpoint: correct response shape, fallback for unknown input
- Manual test: open all 4 pages, verify widget appears; submit form and verify context banner on Result page

---

## 9. Out of Scope

- No conversation history / multi-turn memory (stateless per request)
- No voice input
- No Claude API calls (fully offline after training)
- No admin panel for editing answers (edit `chatbot.py` directly)
