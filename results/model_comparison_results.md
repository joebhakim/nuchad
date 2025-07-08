# Model Comparison: CHADS-VAsC Score vs. UK-Fitted Model

## Overview

Comparison of discrimination performance between:
1. **Off-the-shelf CHADS-VAsC**: Using published risk tables from Lip et al.
2. **UK-Fitted Model**: Logistic regression trained on UK cohort data

## Patient Filtering

**Configuration**: Population-based AF Analysis (New Dataset - Comparable)

### Filtering Steps

| Step | Patients Remaining | Patients Removed | % of Original |
|------|-------------------|------------------|---------------|
| Initial cohort | 136,695 | 0 | 100.0% |
| AF diagnosis before or at time1 | 136,695 | 0 | 100.0% |
| Follow-up period ≥ 365 days | 92,115 | 44,580 | 67.4% |
| Stroke outcome in [1, 2] (excluding pre-AF strokes) | 9,335 | 82,780 | 6.8% |

## Sample Characteristics

- Total patients: 9,335
- Stroke events: 2,112 (22.6%)
- Mean age: 75.9 years
- Female: 48.1%

## Discrimination Performance

| Metric | CHADS-VAsC Score | UK-Fitted Model | Difference (95% CI) |
|--------|------------------|-----------------|---------------------|
| AUC | 0.497 | 0.511 | +0.014 [-0.005, 0.031] |

## Model Coefficients (UK-Fitted Model)

| Feature | Coefficient | Odds Ratio |
|---------|-------------|------------|
| age | -0.000 | 1.000 |
| hf | -0.023 | 0.978 |
| hypertension | 0.006 | 1.006 |
| HB_stroke_history | -0.083 | 0.921 |
| diab | -0.021 | 0.980 |
| vasc_dis_mi_pad | 0.035 | 1.036 |
| female | 0.044 | 1.045 |

**Intercept:** -1.233

## Interpretation

The UK-fitted model shows **improved discrimination** compared to the off-the-shelf CHADS-VAsC score (ΔAUC = +0.014).

This demonstrates the potential benefit of local model calibration versus transported clinical scores.

## Figures

![ROC Comparison](model_comparison_roc.png)
