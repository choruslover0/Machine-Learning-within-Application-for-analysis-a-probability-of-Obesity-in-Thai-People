# O-Beast Context

O-Beast is an educational obesity-risk research app. This glossary defines project language so UI, research writing, and code comments stay consistent.

## Language

**O-Beast**:
The educational app that estimates obesity-risk probability from lifestyle answers and BMI screening.
_Avoid_: SK Research ML, obesity checker, diagnosis app

**Risk Estimate**:
An educational probability output that describes how strongly a user's answers match current research data patterns.
_Avoid_: Diagnosis, medical result, certainty

**Lifestyle ML Probability**:
The probability produced by the machine-learning model using lifestyle, family-history, and survey-behavior features.
_Avoid_: BMI prediction, medical diagnosis, final risk by itself

**BMI Screen Score**:
A separate screening signal calculated from BMI and used after the lifestyle model.
_Avoid_: ML feature, diagnosis, body-fat measurement

**Final Risk Blend**:
The final probability created by combining lifestyle ML probability and BMI screen score with equal weight.
_Avoid_: Raw model output, BMI-only result

**Risk Tier**:
One of five user-facing probability bands: very low, low, moderate, high, or very high.
_Avoid_: Disease stage, clinical class

**Wellness Advice**:
Rule-based educational guidance generated from user answers, risk tier, and public-health references.
_Avoid_: Treatment plan, prescription, doctor's advice

**SMOTENC**:
A data-balancing method that helps train on uneven classes while respecting categorical answers.
_Avoid_: Prediction algorithm, model, classifier

**Model Tournament**:
The comparison of candidate algorithms on the current training data.
_Avoid_: Universal best model, guaranteed winner

**Pipeline Visualizer**:
An educational view showing how data moves through training, validation, model comparison, and result generation.
_Avoid_: Main app, production site

**Hold-out Test Set**:
Data kept separate from training and model selection for final reporting.
_Avoid_: Calibration data, training fold

**Doctor Diagnosis Label**:
Future target data supplied by a medical diagnosis rather than a BMI-derived prototype target.
_Avoid_: Self-report only, BMI target
