# Methods Evaluation Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Premium Health performance dashboard beneath the Methods evaluation table using six real cross-validation metrics and real stored ROC-curve points.

**Architecture:** Extend the existing training evaluation to return scalar metrics plus optional out-of-fold ROC points, storing curves separately in the joblib artifact for backward compatibility. Add focused server-rendered HTML/SVG helpers in the existing FastAPI app and compose them below the current Methods table. Existing artifacts without curve data render an honest retraining-required state.

**Tech Stack:** Python 3, FastAPI server-rendered HTML, scikit-learn, unittest, inline accessible SVG/CSS, Playwright visual QA.

---

## File Structure

- Modify `src/obesity_ml/train.py`: calculate and store real cross-validation ROC points.
- Modify `src/obesity_ml/app.py`: render selected-model banner, six-metric comparison chart, ROC panel, and responsive styles.
- Modify `tests/test_app_regressions.py`: verify ROC data integrity and Methods dashboard behavior.

### Task 1: Store Real Cross-Validation ROC Curves

**Files:**
- Modify: `src/obesity_ml/train.py:12-24`
- Modify: `src/obesity_ml/train.py:196-243`
- Modify: `src/obesity_ml/train.py:288-318`
- Test: `tests/test_app_regressions.py`

- [ ] **Step 1: Write failing ROC-data tests**

Add tests that call a new `cross_validated_evaluation()` helper with a simple pipeline and balanced synthetic frame. Assert that it returns scalar metrics and a curve whose points remain inside `0.0–1.0`, begin at `(0,0)`, and end at `(1,1)`. Add a training artifact assertion for the `roc_curves` field.

```python
metrics, curve = cross_validated_evaluation(model, x, y)
self.assertIn("roc_auc", metrics)
self.assertEqual(curve["false_positive_rate"][0], 0.0)
self.assertEqual(curve["true_positive_rate"][0], 0.0)
self.assertEqual(curve["false_positive_rate"][-1], 1.0)
self.assertEqual(curve["true_positive_rate"][-1], 1.0)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=src /Users/phawichpilathong/.venvs/obeast/bin/python -m unittest \
  tests.test_app_regressions.TrainingPipelineTests.test_cross_validated_evaluation_returns_real_roc_points
```

Expected: failure because `cross_validated_evaluation` does not exist.

- [ ] **Step 3: Implement evaluation and curve storage**

Import `roc_curve`, then implement:

```python
def cross_validated_evaluation(model, x_train, y_train):
    cv = cross_validation_strategy(y_train)
    if cv is None:
        model.fit(x_train, y_train)
        return evaluate_model(model, x_train, y_train), None
    probability = cross_val_predict(model, x_train, y_train, cv=cv, method="predict_proba")[:, 1]
    metrics = evaluate_probabilities(y_train, probability)
    false_positive_rate, true_positive_rate, _ = roc_curve(y_train, probability)
    curve = {
        "false_positive_rate": [float(value) for value in false_positive_rate],
        "true_positive_rate": [float(value) for value in true_positive_rate],
    }
    return metrics, curve
```

Keep `cross_validated_metrics()` as a compatibility wrapper returning only metrics. In `train()`, collect valid curves separately:

```python
validation_results = {}
roc_curves = {}
for name, model in candidate_models(y_train).items():
    validation_results[name], curve = cross_validated_evaluation(model, x_train, y_train)
    if curve is not None:
        roc_curves[name] = curve
```

Store `"roc_curves": roc_curves` in the artifact.

- [ ] **Step 4: Run focused and existing training tests**

Run:

```bash
PYTHONPATH=src /Users/phawichpilathong/.venvs/obeast/bin/python -m unittest \
  tests.test_app_regressions.TrainingPipelineTests
```

Expected: all training tests pass.

### Task 2: Render the Dashboard Components

**Files:**
- Modify: `src/obesity_ml/app.py:1797-2366`
- Modify: `src/obesity_ml/app.py:2667-2754`
- Test: `tests/test_app_regressions.py`

- [ ] **Step 1: Write failing component tests**

Add tests for:

- `selected_model_banner_html()` renders the stored model, ROC-AUC, F1, and “current training data”.
- `metric_comparison_chart_html()` includes all six metric legend labels and treats NaN as unavailable.
- `roc_curves_html({})` contains the retraining-required message and no model curve path.
- `roc_curves_html(real_curves, metrics)` includes the real model name and accessible SVG.

- [ ] **Step 2: Run component tests and verify RED**

Run:

```bash
PYTHONPATH=src /Users/phawichpilathong/.venvs/obeast/bin/python -m unittest \
  tests.test_app_regressions.EvaluationDashboardTests
```

Expected: failure because dashboard helpers do not exist.

- [ ] **Step 3: Implement safe formatting and banner**

Add `_finite_metric()`, `_friendly_model_name()`, and `selected_model_banner_html()`. Render unavailable values as `Not available`; never convert them to zero.

- [ ] **Step 4: Implement grouped six-metric comparison**

Build accessible server-rendered grouped bars using HTML. Each bar receives:

```html
<span class="eval-bar" style="--metric-value:85%" title="Random Forest · ROC-AUC: 0.85"></span>
```

Render a visible legend for ROC-AUC, F1, Accuracy, Kappa, Sensitivity, and Specificity. Missing values use an unavailable marker.

- [ ] **Step 5: Implement real ROC SVG and unavailable state**

Map stored `false_positive_rate` and `true_positive_rate` points into an SVG plot area. Include axes, a dashed random-guess line, curve legend, and exact AUC labels. If no valid stored points exist, render only:

```html
<div class="roc-unavailable">ROC curves are available after retraining the models.</div>
```

- [ ] **Step 6: Add responsive Premium Health styles**

Add styles for `.evaluation-dashboard`, `.selected-model-banner`, `.evaluation-panels`, `.metric-comparison`, `.roc-panel`, legends, bars, SVG axes, and unavailable states. Use two columns on desktop and one column below `900px`, with no horizontal page overflow.

- [ ] **Step 7: Run component tests and verify GREEN**

Run the `EvaluationDashboardTests` command from Step 2.

Expected: all component tests pass.

### Task 3: Integrate Dashboard Beneath the Evaluation Table

**Files:**
- Modify: `src/obesity_ml/app.py:3069-3127`
- Test: `tests/test_app_regressions.py:283-316`

- [ ] **Step 1: Write failing Methods route tests**

Extend the Methods route regression test to assert:

```python
self.assertLess(response.text.index('class="metric-table"'), response.text.index('class="evaluation-dashboard"'))
self.assertIn("Selected for the current training data", response.text)
self.assertIn("ROC curves are available after retraining the models.", response.text)
```

Patch `load_artifact()` in a separate test with stored `roc_curves` and assert that the real ROC SVG is rendered instead of the unavailable message.

- [ ] **Step 2: Run route tests and verify RED**

Run:

```bash
PYTHONPATH=src /Users/phawichpilathong/.venvs/obeast/bin/python -m unittest \
  tests.test_app_regressions.RouteRegressionTests.test_model_tournament_details_live_on_methods_page_not_result_page
```

Expected: failure because the dashboard is not present.

- [ ] **Step 3: Integrate artifact data and renderers**

Read optional curves safely:

```python
roc_curves = artifact.get("roc_curves", {})
```

Build the dashboard:

```python
evaluation_dashboard = evaluation_dashboard_html(best, metrics, roc_curves)
```

Place `{evaluation_dashboard}` immediately after `{metric_table}` inside the Methods tournament section.

- [ ] **Step 4: Run route tests and verify GREEN**

Run all `RouteRegressionTests`.

Expected: all route tests pass.

### Task 4: Retrain, Verify, and Visually Inspect

**Files:**
- Update generated model artifact only if the repository’s sample-training workflow succeeds.
- No source changes expected unless verification exposes a bug.

- [ ] **Step 1: Retrain the sample model**

Run:

```bash
PYTHONPATH=src /Users/phawichpilathong/.venvs/obeast/bin/python -m obesity_ml.train \
  --data data/sample_obesity_training.csv
```

Expected: model artifact saves real `roc_curves`. If the sample dataset cannot support cross-validation, preserve the unavailable state and report it.

- [ ] **Step 2: Run full project verification**

Run:

```bash
PYTHONPATH=src /Users/phawichpilathong/.venvs/obeast/bin/python -m compileall -q src
PYTHONPATH=src /Users/phawichpilathong/.venvs/obeast/bin/python -m unittest discover -s tests
/usr/bin/git diff --check
```

Expected: compile succeeds, all tests pass, no whitespace errors.

- [ ] **Step 3: Restart and smoke-test the Methods route**

Run:

```bash
launchctl kickstart -k gui/$(id -u)/com.obeast.localserver
/usr/bin/curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/methods
```

Expected: HTTP `200`.

- [ ] **Step 4: Perform desktop and mobile rendered QA**

Using Browser plugin when callable, otherwise the existing Playwright runtime:

- Open `/methods` at desktop and mobile widths.
- Verify page identity, meaningful content, no framework overlay, no console errors, and no horizontal overflow.
- Verify dashboard follows the evaluation table.
- Verify selected-model values and chart legend are readable.
- Verify ROC panel shows only real stored curves or the retraining-required state.
- Save screenshots outside the repository.

- [ ] **Step 5: Report verification and remaining risks**

Report:

- Tests passed
- Route status
- Desktop/mobile visual findings
- Whether sample retraining produced real ROC curves
- Existing scikit-learn artifact/runtime version warning
- Exact code references

