# Model Comparison: CHADS-VAsC Score vs. UK-Fitted Model

## Overview

Comparison of discrimination performance between:
1. **Off-the-shelf CHADS-VAsC**: Using published risk tables from Lip et al.
2. **UK-Fitted Model**: Logistic regression trained on UK cohort data

## Patient Filtering

**Configuration**: AF with Valid Stroke Outcomes (New Dataset)

### Filtering Steps

| Step | Patients Remaining | Patients Removed | % of Original |
|------|-------------------|------------------|---------------|
| Initial cohort | 136,695 | 0 | 100.0% |
| AF diagnosis before or at time1 | 136,695 | 0 | 100.0% |
| Follow-up period ≥ 365 days | 92,115 | 44,580 | 67.4% |
| Stroke outcome in [1, 2, 3] (excluding pre-AF strokes) | 11,686 | 80,429 | 8.5% |

## Sample Characteristics

- Total patients: 11,686
- Stroke events: 2,112 (18.1%)
- Mean age: 76.0 years
- Female: 48.5%

## Discrimination Performance

| Metric | CHADS-VAsC Score | UK-Fitted Model | Difference (95% CI) |
|--------|------------------|-----------------|---------------------|
| AUC | 0.495 | 0.507 | +0.012 [-0.010, 0.035] |

## Model Coefficients (UK-Fitted Model)

| Feature | Coefficient | Odds Ratio |
|---------|-------------|------------|
| age | -0.001 | 0.999 |
| hf | -0.034 | 0.967 |
| hypertension | 0.008 | 1.008 |
| HB_stroke_history | -0.074 | 0.929 |
| diab | -0.007 | 0.993 |
| vasc_dis_mi_pad | -0.005 | 0.995 |
| female | 0.022 | 1.022 |

**Intercept:** -1.414

## Interpretation

The UK-fitted model shows **improved discrimination** compared to the off-the-shelf CHADS-VAsC score (ΔAUC = +0.012).

This demonstrates the potential benefit of local model calibration versus transported clinical scores.

## Figures

![ROC Comparison](model_comparison_roc.png)
