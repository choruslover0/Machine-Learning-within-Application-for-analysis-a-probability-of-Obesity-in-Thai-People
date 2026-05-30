# Beast 1.0 Chatbot — Update Summary

**Date:** 2026-05-30  
**Branch:** main  
**Total commits:** 29 commits on top of `a874392`

---

## What Was Built

### Beast 1.0 — Bilingual Obesity Q&A Chatbot

A floating chatbot widget named **Beast 1.0** was added to every page of the O-Beast app. It answers user questions about obesity in **English and Thai**, and gives personalised responses on the Result and Advice pages based on the user's risk score.

---

## Features Added

### 1. Trained Intent Classifier (`src/obesity_ml/chatbot.py`)

- A **TF-IDF + Logistic Regression** pipeline trained on 200 bilingual sentences across **10 intent categories**: `greeting`, `causes`, `prevention`, `treatment`, `diet`, `exercise`, `sleep`, `bmi`, `genetics`, `risk_factors`
- Language detection using `langdetect` — auto-detects Thai or English from the user's message
- Confidence threshold (0.20) — returns a graceful fallback if the message is unrecognisable
- The trained model is saved to `models/chatbot_model.joblib`
- CLI command: `obesity_ml-train-chat` retrains the model from `data/chatbot_training.json`

### 2. Knowledge Base

Full EN + TH answers for all 10 intents, each citing official sources (WHO, CDC, Thai MOPH, NIH, Sleep Foundation, ACSM).

**Context-aware answers** on the Result page: when a user's risk tier is known, the `treatment` and `prevention` answers are personalised — e.g. a Very High Risk user gets urgent WHO clinical recommendations, while a Low Risk user gets encouragement to maintain habits.

### 3. Training Data (`data/chatbot_training.json`)

200 labelled bilingual sentences (10 EN + 10 TH per intent) covering natural student phrasing, Thai vocabulary, and common question variations.

### 4. API Endpoint (`POST /chat`)

```
POST /chat
{ "message": "...", "lang": "auto" | "en" | "th", "context"?: { "risk_tier": "...", "probability": 0.72 } }
→ { "answer": "...", "intent": "...", "detected_lang": "...", "source": "..." }
```

### 5. Floating Chat Widget (all pages)

- **Style:** Rounded-square FAB (72 × 72 px) in the O-Beast gradient, `position: fixed` bottom-right — stays pinned during scroll
- **Logo:** Custom Beast 1.0 mascot image (`/static/beast1-logo.png`)
- **Chat window:** 310 px wide, capped at `max-height: calc(100vh - 140px)` so it never overflows the screen
- **EN / TH toggle** in the header — tight pill buttons; active state has a snug background that fits the text
- **Quick-reply chips** (Causes / Prevention / Diet / Exercise) that shrink to a compact row after the first message is sent
- **Source citations** shown below each bot answer
- **Close controls:** × button in the header, click-outside-to-close, FAB click toggles open/close

### 6. Result Page Context Awareness

After submitting the predictor form, the widget FAB receives `data-risk-tier` and `data-probability` attributes. The chat header shows a pink banner with the user's score. Every `/chat` request from that page includes the context so answers are personalised.

### 7. Notification System (Result + Advice pages only)

When a user receives a prediction result or wellness advice:

- After **1.5 seconds**, the Beast 1.0 FAB **shakes** (wiggle animation)
- A speech bubble appears above the FAB: **"If you want any extra answers, I am here 👋"**
- The bubble has its own **×** dismiss button; it only appears when the chat is closed
- Home, Predictor, and Methods pages are **not** affected

---

## Bugs Fixed

### Initial build bugs

| Bug | Fix |
|---|---|
| Chat widget scrolled with page instead of staying fixed | Moved `pageIn` animation from `<body>` to `<main>` — body transform broke `position: fixed` |
| Chat had no way to close | Added × button in header, FAB click toggles open/close, click-outside closes |
| Chips stretched to full width on Safari/iOS | Added `flex: 0 0 auto; width: auto; display: inline-flex` to `.beast-chip` |
| Chat window too tall — extended above top of screen | Added `max-height: calc(100vh - 140px)` + flex column layout |
| Chips stayed large after selecting a topic | Added `.compact` CSS class applied via JS on first message send |

### Post-release bugs (CSS specificity & JS)

| Bug | Root Cause | Fix |
|---|---|---|
| Chat could not be opened or closed at all | **JS syntax error**: Python f-string `\'` collapsed to a raw `'` inside a JS single-quoted string (`I'll`, `I'm`) — the parse error aborted the entire script before any event handlers registered | Rewrote strings to avoid apostrophes; added `node --check` regression test |
| Chat window not hiding when closed | **CSS specificity clash**: `.beast-chat { display: flex }` (specificity 0-1-0) overrode the browser's `[hidden] { display: none }` UA rule (same specificity, author sheet wins) — `chat.hidden = true` had no visual effect | Replaced `hidden` attribute with a `.open` class toggle: `display: none` by default, `display: flex` only on `.beast-chat.open` |
| Empty "cloud" bubble always visible | Same CSS specificity bug on `.beast-notify { display: flex }` | Converted notify to `display: none` default + `.show` class toggle; added guard so it never appears over an already-open chat |
| Widget buttons (EN/TH, chips, ×, send) stretched into tall ovals | App's global `button { min-height: 52px; width: 100%; margin-top: 18px }` rule leaked into all widget buttons | Added a scoped reset `.beast-fab, .beast-chat button, .beast-notify button { min-height: 0; margin-top: 0; box-shadow: none }` |
| EN/TH language toggle looked oversized | Same global button bleed + no explicit height | Added `height: 22px; padding: 0 9px; display: inline-flex; align-items: center` so the active background fits snugly around the text |

---

## Files Changed

| File | What Changed |
|---|---|
| `src/obesity_ml/chatbot.py` | New — knowledge base, classifier, language detection, `chat()`, `train_chatbot()` |
| `src/obesity_ml/app.py` | `POST /chat` endpoint, `page_shell()` updated, full chat widget HTML/CSS/JS, all bug fixes |
| `src/obesity_ml/static/beast1-logo.png` | New — Beast 1.0 mascot logo |
| `src/obesity_ml/config.py` | Added `CHATBOT_MODEL_PATH` |
| `data/chatbot_training.json` | New — 200 bilingual training sentences |
| `models/chatbot_model.joblib` | New — trained TF-IDF + LR model artifact |
| `tests/test_chatbot.py` | New — 42 tests covering config, data, knowledge base, classifier, endpoint, widget, smart-quote regression |
| `requirements.txt` | Added `langdetect>=1.0.9` |
| `pyproject.toml` | Added `langdetect` dependency + `obesity_ml-train-chat` CLI entry point |
| `.gitignore` | Added `!models/chatbot_model.joblib` and `.superpowers/` |
| `BEAST1_UPDATE.md` | This document |

---

## Test Coverage

**42 tests total, all passing.**

| Test class | What it covers |
|---|---|
| `ChatbotConfigTests` | `CHATBOT_MODEL_PATH` points to correct location |
| `ChatbotTrainingDataTests` | JSON structure, 10 intents, ≥100 rows |
| `ChatbotGetAnswerTests` | EN/TH answers, context injection, unknown lang fallback |
| `ChatbotTrainTests` | Training produces model file, CV > 50%, all 10 classes |
| `ChatbotChatTests` | EN/TH routing, auto-detect, low-confidence fallback, context personalisation |
| `ChatbotEndpointTests` | `/chat` returns correct JSON shape, handles unknown and context |
| `ChatbotWidgetTests` | FAB + chat window on all 4 pages; smart-quote regression check |
| `ChatbotResultContextTests` | Result page renders non-empty `data-risk-tier` and `data-probability` |

---

## How to Run

```bash
# Start the app
source .venv/bin/activate
pip install -e .
uvicorn obesity_ml.app:app --host 127.0.0.1 --port 8000

# Retrain the chatbot classifier (optional)
obesity_ml-train-chat
```

Open **http://127.0.0.1:8000**, then:

1. Click the **Beast 1.0** button (bottom-right) to open the chat; click again (or the × or outside) to close
2. Tap a chip or type a question in English or Thai
3. Submit a prediction — after 1.5 s the FAB shakes and a speech bubble invites you to ask follow-up questions
4. Click the bubble or the FAB to open the chat with your risk score personalising the answers
