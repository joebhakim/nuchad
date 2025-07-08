# Apples-to-Apples Dataset Comparison

## Overview

This analysis compares modeling performance between the original and new datasets using comparable filtering configurations to answer the same medical question: **"What is the 1-year stroke risk for various levels of CHADS-VASc in AF patients?"**

## Configurations Used

### Original Dataset
- **Configuration**: `AF_FU365_nostroke_original`
- **Population**: Standard AF population with 1-year follow-up
- **Outcome**: stroke_1Y=1 (stroke) vs stroke_1Y=2 (no stroke)

### New Dataset  
- **Configuration**: `AF_FU365_population_new`
- **Population**: AF patients with definitive 1-year outcomes
- **Outcome**: stroke_1Y=1 (stroke within 1Y) vs stroke_1Y=2 (no stroke within 1Y)

## Patient Populations

| Metric | Original Dataset | New Dataset | Difference |
|--------|------------------|-------------|------------|
| **Total Patients** | 87,220 | 9,335 | -77,885 (-89.3%) |
| **Stroke Events** | 4,458 (5.1%) | 2,112 (22.6%) | +17.5 percentage points |
| **Mean Age** | 73.8 years | 75.9 years | +2.1 years |
| **Female %** | 47.6% | 48.1% | +0.5 percentage points |

## Discrimination Performance

| Metric | Original Dataset | New Dataset | Difference |
|--------|------------------|-------------|------------|
| **CHADS-VASc AUC** | 0.754 | 0.497 | -0.257 |
| **UK Model AUC** | 0.858 | 0.511 | -0.347 |
| **AUC Improvement** | +0.104 | +0.014 | -0.090 |

## Model Coefficients Comparison

| Feature | Original Dataset | New Dataset | Difference |
|---------|------------------|-------------|------------|
| **Age** | +0.038 (OR: 1.039) | -0.000 (OR: 1.000) | -0.038 |
| **Heart Failure** | -0.140 (OR: 0.869) | -0.023 (OR: 0.978) | +0.117 |
| **Hypertension** | +0.142 (OR: 1.152) | +0.006 (OR: 1.006) | -0.136 |
| **Previous Stroke** | +9.633 (OR: 15,259) | -0.083 (OR: 0.921) | -9.716 |
| **Diabetes** | +0.174 (OR: 1.191) | -0.021 (OR: 0.980) | -0.195 |
| **Vascular Disease** | -0.083 (OR: 0.920) | +0.035 (OR: 1.036) | +0.118 |
| **Female** | -0.059 (OR: 0.943) | +0.044 (OR: 1.045) | +0.103 |
| **Intercept** | -6.853 | -1.233 | +5.620 |

## Key Findings

### 1. Population Differences
- **New dataset has 89% fewer patients** after filtering for comparable outcomes
- **New dataset has 4.4x higher stroke rate** (22.6% vs 5.1%)
- **Patient characteristics are similar** (age, gender distribution)

### 2. Discrimination Performance
- **Original dataset shows excellent discrimination** (AUC 0.754-0.858)
- **New dataset shows poor discrimination** (AUC 0.497-0.511, barely better than random)
- **CHADS-VASc performs much worse** in new dataset (0.497 vs 0.754)

### 3. Model Coefficients
- **Previous stroke history**: Strongest predictor in original (OR: 15,259) but **negatively** associated in new dataset (OR: 0.921)
- **Age**: Strong positive predictor in original (OR: 1.039) but **no effect** in new dataset (OR: 1.000)
- **All traditional risk factors** have attenuated or reversed associations in new dataset

### 4. Clinical Interpretation
- **Original dataset**: Behaves as expected for population-based stroke risk prediction
- **New dataset**: Even with comparable filtering, still shows evidence of **selection bias**
- **New dataset likely represents a selected population** where traditional risk factors don't discriminate

## Conclusions

1. **The new dataset is fundamentally different** from the original despite comparable filtering
2. **Population-based risk prediction models cannot be validated** using the new dataset
3. **Traditional CHADS-VASc risk factors don't discriminate** in the new dataset population
4. **The new dataset appears to be a selected/enriched cohort** where usual risk stratification fails

## Recommendations

- **Use original dataset** for population-based risk prediction and model validation
- **Use new dataset only** for specific research questions about treatment effectiveness in high-risk populations
- **Do not directly compare results** between datasets due to fundamental population differences
- **Consider the new dataset as a case-control study** rather than a population-based cohort

---

*Analysis performed using filtering configurations to ensure apples-to-apples comparison of the same medical question across datasets.*