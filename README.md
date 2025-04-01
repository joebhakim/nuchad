# CHADSVASC Replication and Extension

This project aims to replicate and extend the analysis from the original CHADSVASC paper for predicting stroke risk in atrial fibrillation patients.

## Project Overview

The CHADSVASC score is a clinical prediction rule for estimating the risk of stroke in patients with non-rheumatic atrial fibrillation. This project aims to:

1. Validate the original CHADSVASC scoring system using a large patient dataset
2. Extend the approach using modern machine learning techniques (Random Survival Forests)
3. Compare the performance of the original clinical score versus ML approaches

## Current State

- EDA (Exploratory Data Analysis) completed, identifying preliminary patterns
- Random Survival Forest modeling framework implemented
- Data preprocessing pipeline established
- Evaluation metrics defined (C-index, Brier score)

## Key Components

- `eda.py`: Initial data exploration and CHADSVASC score validation
- `random_survival_forest.py`: Implementation of survival analysis using Random Survival Forests
- `DATA_ISSUE.md`: Documentation of current data limitations

## Data Structure

The dataset includes:
- Atrial fibrillation diagnosis dates
- Stroke events and timing
- Patient demographics
- Comorbidities (heart failure, hypertension, diabetes, etc.)
- Follow-up information

## Model Implementation

The Random Survival Forest model offers several advantages over traditional clinical scores:
- Handles non-linear relationships
- Automatically discovers feature interactions
- Provides individual patient survival curves
- Quantifies feature importance

## Usage

To run the validation of the CHADSVASC score:
```bash
python eda.py
```

To run the Random Survival Forest analysis:
```bash
python random_survival_forest.py
```

## Next Steps

- Validate models with actual unscrambled data
- Compare performance of clinical scores vs. machine learning approaches
- Develop integrated risk prediction interface



