# Tasks

Current task board for O-Beast.

## Next

- [ ] Replace BMI-derived prototype target with doctor-diagnosis label when available.
- [ ] Re-import Google Form data after final dataset collection.
- [ ] Retrain lifestyle ML model after doctor labels arrive.
- [ ] Re-check metrics after target change: ROC-AUC, F1, Brier score, Accuracy, Kappa, Sensitivity, Specificity.
- [ ] Update research report tables with final model metrics.
- [ ] Add final references for BMI screening, obesity ML comparison, and wellness advice.

## Improve

- [ ] Add more positive examples to reduce class imbalance risk.
- [ ] Add confusion matrix display on Methods page.
- [ ] Add calibration curve when dataset is large enough.
- [ ] Add export button for prediction result summary.
- [ ] Add Thai/English language switch for main user pages.
- [ ] Review advice copy for Thai high-school audience.

## Maintenance

- [ ] Keep `README.md` as first-start doc.
- [ ] Keep `CONTEXT.md` glossary-only.
- [ ] Update `CHANGELOG.md` after meaningful changes.
- [ ] Run full tests before GitHub push.
- [ ] Check model artifact after retraining:
  - [ ] `feature_columns` excludes `height_cm`, `weight_kg`, `bmi`
  - [ ] `excluded_body_screen_features` includes body screen fields
  - [ ] `probability_blend_strategy` mentions 50/50 blend

## Done Recently

- [x] Separate BMI from lifestyle ML features.
- [x] Add final 50/50 probability blend.
- [x] Make BMI screen a config-driven graded curve (midpoint 25, steepness 3).
- [x] Add `httpx2` to remove Starlette TestClient deprecation warning.
- [x] Add workflow PNG/PDF assets for research report.
- [x] Add five risk tiers.
