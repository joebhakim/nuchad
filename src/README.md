# Other docs

https://docs.google.com/document/d/1bc00rUIiBYlEEOH0K0QK6h4h4q1NZeUNjxw3L1zOLvc/edit?tab=t.0

# CHADS-VASc Analysis Project

This repository contains scripts for analyzing CHADS-VASc score performance in predicting stroke risk among patients with atrial fibrillation (AF). The analysis includes:

1. Exploratory data analysis
2. Follow-up time visualization
3. Patient characteristics (Table 2) generation
4. CHADS-VASc validation
5. Reweighting analysis for transportability to original development cohort

## Prerequisites

- Python 3.6+
- Required packages: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn
- Dataset: A patient-level dataset containing AF patients with the following key columns:
  - `time1`: Start of observation window
  - `earliest_af_date`: Date of AF diagnosis
  - `earliest_stroke_date`: Date of stroke (if applicable)
  - `end_fu`: End of follow-up date
  - Various risk factors (age, gender, hypertension, etc.)

## Script Descriptions and Expected Outputs

### 1. `eda.py`

**Purpose**: Core module containing data loading, patient filtering, and CHADS-VASc calculation functions.

**Functions**:
- `get_df()`: Loads and prepares the dataset
- `calculate_chadsvasc()`: Calculates CHADS-VASc score for each patient
- `filter_eligible_patients()`: Filters patients based on eligibility criteria
- `validate_chadsvasc()`: Calculates observed stroke rates by CHADS-VASc score

**Usage**:
```
python eda.py
```

**Expected Output**:
- Console: Patient filtering statistics, observed stroke rates by CHADS-VASc score
- Files: 
  - `results_df.md`: Table of observed vs. original CHADS-VASc stroke rates
  - `observed_vs_original_stroke_rates.png`: Plot comparing observed and original rates

### 2. `visualize_end_fu.py`

**Purpose**: Visualizes the distribution of follow-up end dates and durations.

**Usage**:
```
python visualize_end_fu.py
```

**Expected Output**:
- Console: Statistics about total records and follow-up duration
- File: `end_fu_distribution.png`: Four-panel visualization showing histograms and boxplots of follow-up dates and durations

### 3. `table2.py`

**Purpose**: Generates a descriptive statistics table ("Table 2") of patient characteristics.

**Usage**:
```
python table2.py
```

**Expected Output**:
- Console: CHADS-VASc score distribution summary
- Files:
  - `table2.md`: Markdown table with patient characteristics
  - `chadsvasc_distribution_eligible.md`: Distribution of CHADS-VASc scores in the eligible patient cohort

### 4. `table2_stratified.py`

**Purpose**: Creates a stratified version of Table 2, grouping patients by CHADS-VASc risk category (low, moderate, high).

**Usage**:
```
python table2_stratified.py
```

**Expected Output**:
- File: `table2_stratified.md`: Markdown table with patient characteristics stratified by risk group, with p-values for between-group comparisons

### 5. Reweighting Analysis Scripts

We provide two different implementations for performing transportability analysis through reweighting, targeting the key question: "How would the CHADS-VASc score perform if our cohort had the same covariate distribution as the original development cohort?"

#### 5a. `density_ratio_reweighting.py` (Recommended)

**Purpose**: Implements the density ratio approach for reweighting as directly specified in the original sketch.

**Features**:
- Direct implementation of the density ratio method using log-space calculations for numerical stability
- Handles both continuous (normal distribution) and binary (Bernoulli) variables
- Calculates effective sample size
- Produces weighted stroke rates by CHADS-VASc score
- Evaluates AUC before and after weighting

**Usage**:
```
python density_ratio_reweighting.py
```

**Expected Output**:
- Console: UK parameters, weight statistics, effective sample size (~28% of original), and AUC values
- Files:
  - `density_ratio_weight_distribution.png`: Histogram of weights
  - `density_ratio_weighted_rates.png`: Plot comparing weighted rates to original rates
  - `density_ratio_results.md`: Markdown summary of results

**Key Results**:
- Effective sample size: ~28% of original sample
- Weights range: 0.02-50.3 (normalized)
- The weighted AUC shows different discrimination compared to the original

#### 5b. `reweighting.py` (Alternative)

**Purpose**: Performs a reweighting analysis using an optimization-based approach.

**Features**:
- Maps variables between datasets
- Computes weights using likelihood optimization
- Trims extreme weights
- Calculates effective sample size
- Bootstraps confidence intervals (uses 15 iterations for faster computation)

**Usage**:
```
python reweighting.py
```

**Expected Output**:
- Console: Detailed statistics about parameters, weights, and performance
- Files:
  - `weight_distribution.png`: Histogram of original and trimmed weights
  - `weighted_vs_original_rates.png`: Plot comparing weighted rates to original rates
  - `weighted_chadsvasc_results.md`: Table of weighted results

## Execution Order

For a complete analysis, run the scripts in this order:

1. `eda.py` - Core data preparation and initial analysis
2. `visualize_end_fu.py` - Understand follow-up patterns
3. `table2.py` - Generate descriptive statistics
4. `table2_stratified.py` - Compare characteristics across risk groups
5. `density_ratio_reweighting.py` - Perform transportability analysis using the density ratio approach

## Notes

- All scripts use the patient filtering logic in `eda.py` for consistency
- Output files are generated in the `results` directory
- Both reweighting approaches address the transportation problem but use different mathematical formulations
- The density ratio method follows a more direct implementation of the original sketch and provides better numerical stability
