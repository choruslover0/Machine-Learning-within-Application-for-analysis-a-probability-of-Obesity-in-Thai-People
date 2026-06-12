# O-Beast Premium Health UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform every O-Beast user-facing page into a calm premium-health experience while preserving existing FastAPI, prediction, advice, and chatbot behavior.

**Architecture:** Keep server-rendered HTML in `src/obesity_ml/app.py`, extend shared CSS and page components, and use a small client-side controller to turn the existing single prediction form into guided steps. Add one Higgsfield-generated decorative asset under `src/obesity_ml/static/`. Protect behavior with route-level regression tests and browser QA.

**Tech Stack:** Python, FastAPI, server-rendered HTML/CSS/JavaScript, unittest, Higgsfield GPT Image 2.

---

### Task 1: Protect Premium Health Page Contracts

**Files:**
- Modify: `tests/test_app_regressions.py`
- Test: `tests/test_app_regressions.py`

- [ ] **Step 1: Write failing tests**

Add route tests requiring:

```python
self.assertIn("Know your risk", home_response.text)
self.assertIn("premium-health-visual", home_response.text)
self.assertIn('class="form-step active"', predictor_response.text)
self.assertIn('data-step="2"', predictor_response.text)
self.assertIn("Three useful next steps", result_response.text)
self.assertIn("prefers-reduced-motion", home_response.text)
```

- [ ] **Step 2: Verify tests fail**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.test_app_regressions.RouteRegressionTests
```

Expected: failures for missing premium-health content and guided form structure.

- [ ] **Step 3: Preserve existing behavior tests**

Keep all existing route, field-name, chatbot, methods-page, and prediction-response assertions unchanged.

### Task 2: Generate Premium Health Visual

**Files:**
- Create: `src/obesity_ml/static/premium-health-pattern.png`

- [ ] **Step 1: Generate asset with Higgsfield GPT Image 2**

Run:

```bash
higgsfield generate create gpt_image_2 \
  --prompt "Premium abstract health pattern for O-Beast obesity-risk education app, flowing body and lifestyle signals becoming a clear organized pattern, warm white background, deep forest green, soft sage and restrained peach accents, tactile paper-like dimensional forms, calm trustworthy modern health design, no people, no body-shaming, no medical diagnosis imagery, no text, no letters, no UI screenshot, landscape composition" \
  --aspect_ratio 16:9 \
  --resolution 2k \
  --wait
```

- [ ] **Step 2: Save output**

Download generated image to `src/obesity_ml/static/premium-health-pattern.png`.

- [ ] **Step 3: Inspect image**

Confirm visual contains no text, unsafe health claims, distorted people, or conflicting palette.

### Task 3: Build Shared Premium Health Visual System

**Files:**
- Modify: `src/obesity_ml/app.py`
- Test: `tests/test_app_regressions.py`

- [ ] **Step 1: Add shared tokens and component styles**

Replace old Instagram-gradient treatment with:

```css
:root {
  --ink: #14241b;
  --forest: #153c29;
  --green: #347849;
  --sage: #e9f0e8;
  --paper: #f7faf6;
  --peach: #eaa377;
  --muted: #66746c;
}
```

Add nested shells, compact floating nav, pill CTAs, calm cards, accessible focus styles, mobile fallbacks, and reduced-motion rules.

- [ ] **Step 2: Update navigation and page shell**

Keep existing routes. Use concise user-facing labels and preserve chatbot widget injection.

- [ ] **Step 3: Run route tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.test_app_regressions.RouteRegressionTests
```

Expected: existing route tests pass except contracts waiting on later tasks.

### Task 4: Redesign Home and Learning Visuals

**Files:**
- Modify: `src/obesity_ml/app.py`
- Test: `tests/test_app_regressions.py`

- [ ] **Step 1: Redesign home hero**

Use headline `Know your risk. Shape tomorrow.`, one primary prediction action, a learning action, concise facts, and generated visual.

- [ ] **Step 2: Simplify algorithm visual cards**

Keep all algorithm names and theoretically correct explanations. Reduce visual labels, arrow size, and technical clutter.

- [ ] **Step 3: Preserve producer section**

Keep real names, photos, and existing contact links. Restyle into premium-health system.

- [ ] **Step 4: Run home tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.test_app_regressions.RouteRegressionTests.test_home_page_renders
```

Expected: pass.

### Task 5: Convert Predictor Into Guided Steps

**Files:**
- Modify: `src/obesity_ml/app.py`
- Test: `tests/test_app_regressions.py`

- [ ] **Step 1: Group existing fields**

Group unchanged input names into body profile, daily routine, and food/family sections.

- [ ] **Step 2: Add form-step controls**

Add accessible previous/continue buttons, current-step indicator, progress bar, and final submit button. JavaScript changes visibility only.

- [ ] **Step 3: Keep no-JavaScript submission viable**

All fields remain in one form and all required backend input names remain present in HTML.

- [ ] **Step 4: Run predictor tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.test_app_regressions.RouteRegressionTests
```

Expected: predictor route and form submission tests pass.

### Task 6: Redesign Result, Advice, and Methods

**Files:**
- Modify: `src/obesity_ml/app.py`
- Test: `tests/test_app_regressions.py`

- [ ] **Step 1: Reorder result content**

Show probability, five-level band, plain-language meaning, and top advice before model transparency.

- [ ] **Step 2: Restyle advice page**

Group transparent advice by habit area and retain trusted sources and educational disclaimer.

- [ ] **Step 3: Restyle methods page**

Keep all validation details and metric table. Use plain-language summary and visually separated research details.

- [ ] **Step 4: Run result and advice tests**

Run:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.test_app_regressions.RouteRegressionTests
```

Expected: all route regression tests pass.

### Task 7: Full Verification and Browser QA

**Files:**
- Modify if needed: `src/obesity_ml/app.py`
- Modify if needed: `tests/test_app_regressions.py`

- [ ] **Step 1: Run compile and full tests**

```bash
PYTHONPATH=src .venv/bin/python -m compileall src
chflags -R nohidden .venv 2>/dev/null || true
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

Expected: exit code `0`.

- [ ] **Step 2: Restart app and smoke test routes**

```bash
launchctl kickstart -k gui/$(id -u)/com.obeast.localserver
for path in / /predictor /advice /methods; do
  curl -sS -o /dev/null -w "$path %{http_code}\n" "http://127.0.0.1:8000$path"
done
```

Expected: HTTP `200` for every route.

- [ ] **Step 3: Browser QA**

Inspect Home, Predictor, Result, Advice, and Methods at desktop and mobile widths. Confirm:

- No clipping or overlaps
- Generated asset loads
- Predictor step controls work
- Form submits and result renders
- Chatbot opens and closes
- Text stays user-facing

- [ ] **Step 4: Final repository checks**

```bash
git diff --check
git status --short --branch
```

Expected: no whitespace errors. Report pre-existing unrelated changes separately.

