# Jinja2 Hybrid Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move O-Beast page rendering from f-string HTML inside the 3924-line `app.py` to Jinja2 templates, while keeping data-driven SVG charts as Python helpers.

**Architecture:** Hybrid. Page layouts/pages become Jinja2 `.html` templates served via FastAPI `Jinja2Templates`. SVG/metric chart fragments stay as Python functions and are injected into templates with the `|safe` filter. `app.py` shrinks to a FastAPI factory plus thin route modules under `routes/`.

**Tech Stack:** FastAPI, Starlette `Jinja2Templates`, Jinja2, existing scikit-learn predict/advice modules (unchanged).

## Global Constraints

- Tech floor: `jinja2>=3.1` added to `pyproject.toml` dependencies. Python `>=3.10`.
- **CSS stays inline.** Tests assert CSS substrings in `response.text` (`.section-grid > * { min-width: 0; }`, `overflow-wrap: anywhere`, `prefers-reduced-motion`). CSS is extracted out of Python into `templates/partials/styles.html` and `{% include %}`-d inline in `<head>` — NOT served as an external `<link>`. Moving CSS to an external file is OUT OF SCOPE because it breaks the test contract.
- Every existing assertion string must survive verbatim. Critical substrings: `O-Beast`, `Know your risk`, `premium-health-pattern.png`, `prefers-reduced-motion`, `class="form-step active"`, `data-step="2"`, `id="formProgress"`, `Three useful next steps`, `What this result means`, `How O-Beast Checks Results`, `Selected for the current training data`, `SMOTENC is not a prediction algorithm`, `Model tournament scores`, `<table class="metric-table"`, `class="evaluation-dashboard"`, `class="roc-model-curve"`, `Your Wellness Advice`, `advice-card`.
- No web UI text may mention conversations, prompts, or agents (CLAUDE.md rule).
- Chart helpers (`roc_curves_html`, `metric_comparison_chart_html`, `metric_bars_html`, `metric_table_html`, `selected_model_banner_html`, `evaluation_dashboard_html`) keep their current names and signatures so their direct unit tests (EvaluationDashboardTests) keep importing them.
- Run after every task: `PYTHONPATH=src python3 -m unittest discover -s tests` — must stay green (currently 58 tests).
- Commit after each task with Conventional Commits.

---

### Task 1: Add Jinja2 and a Templates singleton

**Files:**
- Modify: `pyproject.toml` (add `jinja2>=3.1`)
- Create: `src/obesity_ml/templating.py`
- Create: `src/obesity_ml/templates/_smoke.html`
- Test: `tests/test_templating.py`

**Interfaces:**
- Produces: `templating.templates` (a `Jinja2Templates` instance rooted at `src/obesity_ml/templates`), `templating.render(name: str, context: dict) -> str` returning rendered HTML as a string (no Request needed for unit tests).

- [ ] **Step 1: Write the failing test**
```python
def test_render_returns_html_string():
    from obesity_ml.templating import render
    html = render("_smoke.html", {"value": "hello-obeast"})
    assert "hello-obeast" in html
```
- [ ] **Step 2: Run it, expect ImportError/FileNotFound**
Run: `PYTHONPATH=src python3 -m unittest tests.test_templating -v`
- [ ] **Step 3: Add dep + implement**
`pyproject.toml`: add `"jinja2>=3.1",` to `dependencies`.
`_smoke.html`: `<p>{{ value }}</p>`
`templating.py`:
```python
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
_env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(["html"]),
)

def render(name: str, context: dict) -> str:
    return _env.get_template(name).render(**context)
```
- [ ] **Step 4: Run test, expect PASS**
- [ ] **Step 5: Commit** `feat: add jinja2 templating singleton`

---

### Task 2: Extract CSS into an inline partial

**Files:**
- Create: `src/obesity_ml/templates/partials/styles.html`
- Modify: `src/obesity_ml/app.py` (replace `STYLE`/`CYBERPUNK_OVERRIDE`/`PREMIUM_HEALTH_OVERRIDE` string literals with a single `render_styles()` that returns the partial)
- Test: reuse `RouteRegressionTests.test_methods_page_renders`

**Interfaces:**
- Produces: `templates/partials/styles.html` containing the three CSS blocks concatenated, each still wrapped in its own `<style>` tag.

- [ ] **Step 1:** Copy the exact contents of `STYLE`, `CYBERPUNK_OVERRIDE`, `PREMIUM_HEALTH_OVERRIDE` (currently in `app.py`) into `partials/styles.html`, preserving every character including `.section-grid > * { min-width: 0; }`, `overflow-wrap: anywhere`, and the `@media (prefers-reduced-motion: reduce)` block.
- [ ] **Step 2:** In `app.py`, replace the three module-level CSS constants with `from obesity_ml.templating import render` and a helper `def render_styles() -> str: return render("partials/styles.html", {})`. Update `page_shell` to interpolate `{render_styles()}` where `{STYLE}{CYBERPUNK_OVERRIDE}{PREMIUM_HEALTH_OVERRIDE}` was.
- [ ] **Step 3: Run the suite**
Run: `PYTHONPATH=src python3 -m unittest discover -s tests`
Expected: still 58 passing (CSS substrings now sourced from the partial, still inline).
- [ ] **Step 4: Commit** `refactor: extract css into inline jinja partial`

---

### Task 3: base.html layout replaces page_shell internals

**Files:**
- Create: `src/obesity_ml/templates/base.html`
- Modify: `src/obesity_ml/app.py` (`page_shell` delegates to `base.html`)
- Test: `RouteRegressionTests.test_home_page_renders`, `test_premium_health_ui_contract_renders_on_home_and_predictor`

**Interfaces:**
- Consumes: `render_styles()` from Task 2; existing `chat_widget_html(...)`.
- Produces: `base.html` with blocks `{% block body %}{% endblock %}`, an injected `{{ styles|safe }}`, `{{ nav|safe }}` (or static nav markup), and `{{ widget|safe }}`. `page_shell(title, body, chat_context)` keeps its current signature and now calls `render("base.html", {...})`.

- [ ] **Step 1:** Author `base.html` reproducing the current `page_shell` document exactly: doctype, `<head>` with `<title>{{ title }}</title>`, `{{ styles|safe }}`, the `js` class script, `<body><main>` with the nav block (Home/Predictor/Advice/Methods links and the `O-Beast` brand with `/static/obeast-logo.png`), `{{ body|safe }}`, then `{{ widget|safe }}`.
- [ ] **Step 2:** Rewrite `page_shell` to: compute `widget` as today, then `return render("base.html", {"title": title, "styles": render_styles(), "body": body, "widget": widget})`.
- [ ] **Step 3: Run the suite**, expect 58 passing.
- [ ] **Step 4: Commit** `refactor: render page shell via base.html template`

---

### Task 4: Page bodies to templates (home, predictor, methods, result, advice)

> One sub-task per route. Each: create `<page>.html`, change the route function to build its context dict and call `render`, wrapping the result in `page_shell`. Charts/fragments are passed in pre-rendered via `|safe`. Run the suite after each.

- [ ] **4a home** -> `templates/home.html`. Keep `Know your risk`, `premium-health-pattern.png`. Commit `refactor: render home via template`.
- [ ] **4b predictor** -> `templates/predictor.html`. Keep `class="form-step active"`, `data-step="2"`, `id="formProgress"`. Commit.
- [ ] **4c methods** -> `templates/methods.html`. Inject `{{ algorithms|safe }}` (`methods_html()`), `{{ dashboard|safe }}` (`evaluation_dashboard_html(...)`), `{{ data_prep|safe }}`. Keep `How O-Beast Checks Results`, `Model tournament scores`, `<table class="metric-table"`, `class="evaluation-dashboard"`, the metric-table-before-dashboard ordering, and the SMOTENC explanation strings. Commit.
- [ ] **4d result** -> `templates/result.html`. Keep `Estimated Probability`, `Three useful next steps`, `What this result means`; must NOT contain `Model tournament scores` or `<table class="metric-table"`. Commit.
- [ ] **4e advice** -> `templates/advice.html`. Keep `Your Wellness Advice`, `advice-card`. Commit.

---

### Task 5: Move chart/fragment helpers into charts.py

**Files:**
- Create: `src/obesity_ml/charts.py`
- Modify: `src/obesity_ml/app.py` (import the moved helpers), `tests/test_app_regressions.py` import sites only if needed
- Test: existing `EvaluationDashboardTests`, `BestModelReasonTests`

**Interfaces:**
- Produces (moved verbatim, same names/signatures): `metric_bars_html`, `metric_table_html`, `metric_comparison_chart_html`, `roc_curves_html`, `selected_model_banner_html`, `evaluation_dashboard_html`, `best_model_reason`, `_ranked_metrics`, `_finite_metric`, `_friendly_model_name`.

- [ ] **Step 1:** Cut those functions from `app.py` into `charts.py`; in `app.py` add `from obesity_ml.charts import (...)`.
- [ ] **Step 2:** EvaluationDashboardTests import `from obesity_ml.app import roc_curves_html` etc. Keep those names re-exported from `app` (the `import` in Step 1 satisfies this) so tests need no change.
- [ ] **Step 3: Run the suite**, expect 58 passing.
- [ ] **Step 4: Commit** `refactor: split chart helpers into charts.py`

---

### Task 6: Split routes out of app.py

**Files:**
- Create: `src/obesity_ml/routes/__init__.py`, `routes/pages.py`, `routes/api.py`
- Modify: `src/obesity_ml/app.py` (becomes factory: create `app`, mount static, include routers)
- Test: full suite + `test_app_imports_from_non_project_working_directory`

**Interfaces:**
- Consumes: `templating`, `charts`, `predict`, `advice`, page-template renderers from Task 4.
- Produces: `app` object importable as `obesity_ml.app:app` (uvicorn entrypoint unchanged).

- [ ] **Step 1:** Move page routes (`/`, `/predictor`, `/methods`, `/advice` GET, `/predict-form`, `/advice` POST) into `routes/pages.py` as an `APIRouter`. Move `/predict`, `/chat` JSON endpoints into `routes/api.py`.
- [ ] **Step 2:** `app.py`: build `app = FastAPI(...)`, mount `/static`, `app.include_router(pages.router)`, `app.include_router(api.router)`. Keep `from obesity_ml.charts import *` re-exports for the unit tests that import chart funcs from `app`.
- [ ] **Step 3: Run the suite**, expect 58 passing.
- [ ] **Step 4: Commit** `refactor: split fastapi routes into modules`

---

### Task 7: Verification and cleanup

- [ ] **Step 1:** `PYTHONPATH=src python3 -m unittest discover -s tests` -> 58 passing.
- [ ] **Step 2:** Manual smoke: `PYTHONPATH=src uvicorn obesity_ml.app:app` then GET `/`, `/predictor`, `/methods`, `/advice`; POST a valid form to `/predict-form`. Confirm pages render and styles load.
- [ ] **Step 3:** Confirm `app.py` is now under ~400 lines.
- [ ] **Step 4:** Update `docs/ARCHITECTURE.md` with the new `templates/`, `charts.py`, `routes/` layout.
- [ ] **Step 5: Commit** `docs: document jinja2 template architecture`

## Self-Review

- Spec coverage: CSS-inline constraint (Task 2), every page (Task 4), charts stay Python (Task 5), thin app.py (Task 6). Covered.
- Type consistency: chart helper names unchanged across Tasks 5-6; `page_shell` signature preserved through Tasks 2-3.
- Risk: the biggest is whitespace/escaping. Jinja2 `autoescape` will escape `{{ body }}` unless `|safe` is used — every injected HTML fragment MUST use `|safe`. Flagged in Tasks 3-4.
