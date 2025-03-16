# CHADSVASC Stroke Prediction Project Specification

## Project Overview

This project aims to:
1. Replicate the analysis from the original CHADSVASC paper for predicting stroke risk in atrial fibrillation (AF) patients
2. Extend the predictive modeling using modern machine learning techniques, specifically Random Survival Forests
3. Compare the performance between traditional clinical scores and machine learning approaches

## Data Requirements

### Input Data
- Patient demographics (age, gender)
- AF diagnosis dates
- Stroke events and timing
- Comorbidities (heart failure, hypertension, diabetes, etc.)
- Follow-up information
- Anticoagulation status

### Data Structure
- `time1`: Start of observation window (date)
- `time2`: End of observation window (date)
- `earliest_af_date`: First date AF recorded
- `earliest_stroke_date`: First date of stroke
- `end_fu`: End of follow-up date
- `end_fu_due_to_death`: Death as reason for end of follow-up
- `stroke_1Y`: Stroke in first year after AF
- `stroke_time`: Time between AF diagnosis and stroke
- Binary indicators for risk factors (hypertension, diabetes, etc.)
- Continuous measurements (BMI, cholesterol, etc.)

### Data Preprocessing
- Convert dates to proper datetime format
- Calculate time-to-event for survival analysis
- Handle missing values appropriately
- Create eligible patient cohort based on criteria:
  - AF diagnosis before time1
  - Sufficient follow-up period (≥1 year)
  - Properly recorded stroke events

## Clinical Score Implementation

### CHADSVASC Score
- Calculate according to original paper methodology:
  - Congestive heart failure: +1 point
  - Hypertension: +1 point
  - Age ≥75: +2 points
  - Age 65-74: +1 point
  - Diabetes: +1 point
  - Stroke/TIA history: +2 points
  - Vascular disease: +1 point
  - Female gender: +1 point
- Validate score against published risk rates
  - Compare observed stroke rates per score with original publication
  - Calculate confidence intervals using Poisson distribution

## Machine Learning Models

### Random Survival Forest
- **Inputs**: Patient features (demographics, comorbidities)
- **Output**: Survival probabilities and risk scores
- **Parameters**:
  - Number of trees: 100 (default)
  - Min samples split: 10
  - Min samples leaf: 15
  - Feature selection: sqrt (default)
- **Evaluation**:
  - C-index (discrimination)
  - Calibration plots
  - Feature importance analysis

### Cox Proportional Hazards
- **Inputs**: Patient features (demographics, comorbidities)
- **Output**: Hazard ratios and risk scores
- **Parameters**:
  - Regularization (alpha): 0.1
- **Evaluation**:
  - C-index (discrimination)
  - Proportional hazards assumption testing
  - Feature coefficient analysis

## Model Comparison Framework

- Compare multiple models on the same test dataset
- Evaluation metrics:
  - C-index
  - Brier score
  - Calibration
  - Risk stratification capability
- Visualization of model performance:
  - Survival curves by risk group
  - Feature importance/effect plots
  - Patient risk tables

## Clinical Interpretation Tools

- Patient risk stratification into groups (low, medium, high)
- Personalized survival curves
- Feature effect visualization
- Individual patient risk tables
- Clinical decision support visualization

## Implementation Requirements

### Code Structure
- `eda.py`: Initial data exploration and validation
- `random_survival_forest.py`: RSF model implementation
- `model_comparison.py`: Framework for comparing multiple models
- `survival_insights.py`: Visualization and clinical interpretation tools

### Dependencies
- Python 3.12+
- Core libraries:
  - pandas
  - numpy
  - scipy
  - matplotlib
  - scikit-learn
  - scikit-survival

### Development Workflow
1. Data exploration and validation
2. Clinical score implementation
3. Machine learning model development
4. Model comparison and evaluation
5. Clinical interpretation tools
6. Documentation and reporting

## Expected Outputs

1. Validated CHADSVASC implementation
2. Random Survival Forest model for stroke prediction
3. Model comparison analysis
4. Clinical interpretation visualizations
5. Documentation of findings

## Future Extensions

1. External validation on additional datasets
2. Web-based clinical tool for risk calculation
3. Integration with electronic health records
4. Expansion to include additional risk factors
5. Dynamic risk updating based on changing patient characteristics 