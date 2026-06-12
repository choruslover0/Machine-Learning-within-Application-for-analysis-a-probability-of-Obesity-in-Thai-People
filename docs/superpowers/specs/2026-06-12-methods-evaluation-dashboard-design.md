# O-Beast Methods Evaluation Dashboard

## Goal

Add a clear visual performance dashboard directly beneath the existing evaluation table on the Methods page. The dashboard helps students, guests, and research reviewers understand why the current model was selected without presenting fabricated or overly technical results.

## Approved Direction

Use the existing **Premium Health** visual system rather than the dark technical style shown in the reference image.

The dashboard contains:

1. A selected-model summary banner
2. A grouped performance comparison chart
3. A real ROC-curve panel
4. The existing prototype-data warning

Every displayed number must come from the saved model artifact. No example metrics or illustrative ROC curves may appear as if they were evaluation results.

## Placement

The new dashboard appears on `/methods` directly after the existing evaluation metric table inside the “Model tournament scores” section.

The information order is:

1. Existing ROC-AUC comparison bars
2. Existing evaluation table
3. Selected-model banner
4. Performance comparison and ROC panels
5. Dataset warning

## Selected-Model Banner

The banner identifies the model selected from the current training data.

It displays:

- Selected model name
- Cross-validated ROC-AUC
- Cross-validated F1 score
- A short statement that selection applies only to the current training data

If metrics are unavailable, the banner displays a neutral “Train the model to see the selected result” state.

The banner must not call the selected model universally best or imply clinical validation.

## Performance Comparison Chart

The grouped chart compares every trained candidate model using six stored cross-validation metrics:

- ROC-AUC
- F1 score
- Accuracy
- Cohen’s Kappa
- Sensitivity
- Specificity

### Display Rules

- Metric values use a common `0.0–1.0` scale.
- Each metric has a stable, distinguishable color and visible legend label.
- Model names remain readable on desktop and mobile.
- Hover or focus text exposes the exact model, metric, and value.
- Missing or non-finite metric values render as unavailable, not zero.
- The chart includes a short plain-language explanation.

The chart should use accessible server-rendered HTML/SVG and must not require a new charting dependency.

## ROC-Curve Panel

ROC curves must use real out-of-fold cross-validation probabilities generated during model training.

### Training Data Flow

For each candidate model:

1. Use the existing stratified cross-validation strategy.
2. Generate out-of-fold predicted probabilities.
3. Calculate the model’s validation metrics from those probabilities.
4. Calculate false-positive-rate and true-positive-rate points from the same probabilities.
5. Store the curve points in the model artifact under a dedicated `roc_curves` field.

This keeps the ROC visualization aligned with the cross-validated metrics used for model selection.

### Display Rules

- Show one curve for each model with stored valid points.
- Include a labeled random-guess diagonal.
- Identify each curve using model name and ROC-AUC.
- Use a common `0.0–1.0` scale for both axes.
- Do not create or interpolate illustrative curves when stored points are unavailable.
- Existing artifacts without `roc_curves` show: “ROC curves are available after retraining the models.”

When cross-validation is impossible because the dataset is too small, the app must not describe training-set ROC points as cross-validated ROC curves.

## Visual Design

- Deep-green selected-model banner with subtle trophy/checkmark treatment
- Warm white chart surfaces with sage borders
- Green, peach, gold, blue, violet, and muted-red metric colors
- Clear legends and restrained typography
- Two-column chart layout on desktop
- Single-column layout on mobile
- No horizontal page overflow
- No dark dashboard theme that conflicts with the current O-Beast interface

## User-Facing Language

Use plain language suitable for students and general users.

Required messages:

- “Selected for the current training data”
- “Higher scores are generally better for these measures.”
- “ROC curves are available after retraining the models.” when curve data is absent
- The existing warning that prototype metrics are not suitable for research claims

Avoid:

- “Champion” when it could imply universal superiority
- “Clinical prediction inference”
- Private development instructions or conversation history
- Claims of medical accuracy or diagnosis

## Architecture

### Training

Update `src/obesity_ml/train.py` so the cross-validation evaluation can return both metrics and valid ROC points. Store curves separately from scalar metrics to preserve existing metric consumers.

### Rendering

Update `src/obesity_ml/app.py` with small focused helpers:

- Selected-model banner renderer
- Grouped metric-comparison renderer
- ROC-curve renderer
- Safe numeric formatting and unavailable states

The Methods route reads `metrics`, `base_model_name`, and optional `roc_curves` from the artifact and places the dashboard beneath the evaluation table.

### Backward Compatibility

- Existing model artifacts remain loadable.
- Missing `roc_curves` produces the retraining-required state.
- Existing prediction behavior, model selection rule, and result page remain unchanged.

## Testing

### Automated Tests

- Training evaluation stores real ROC points when cross-validation is available.
- Stored ROC points begin near `(0,0)`, end near `(1,1)`, and remain within `0.0–1.0`.
- Metrics remain scalar values and current model selection still works.
- Methods page places the new dashboard after the evaluation table.
- Selected-model banner uses stored model name and metrics.
- Six comparison metrics appear in the chart legend.
- Missing ROC points show the retraining-required state.
- No fabricated ROC path appears when curve data is absent.

### Rendered Verification

- Desktop and mobile Methods screenshots
- No clipping, overlap, or horizontal overflow
- Legends and model names remain readable
- Dashboard matches Premium Health styling
- Browser console has no relevant errors

### Project Verification

- Python compile check
- Full unit/regression suite
- Methods route HTTP `200`
- `git diff --check`

## Out of Scope

- Changing the model-selection ranking rule
- Changing risk tiers or advice logic
- Claiming research accuracy before the real dataset is collected
- Displaying hold-out test curves as cross-validation curves
- Adding a third-party chart library
