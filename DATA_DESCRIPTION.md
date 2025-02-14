# Dataset Description

This document provides a comprehensive overview of the medical dataset structure and variables, with a focus on the actual column names as they appear in the data file for use in modeling.

## Data Structure

The dataset is stored in CSV format with the following columns in order:

1. `time1` - Start of observation window (date)
2. `time2` - End of observation window (date)
3. `earliest_af_date` - First date AF recorded
4. `earliest_stroke_date` - First date of stroke
5. `end_fu` - End of follow-up date
6. `end_fu_due_to_death` - Death as reason for end of follow-up (1=yes)
7. `stroke_1Y` - Stroke in first year after AF (1=Yes, 2=No)
8. `stroke_time` - Time between AF diagnosis and stroke
9. `Anticoagulant` - Anticoagulation status ("No anticoagulant", "VKA", "DOAC")
10. `age` - Age at time of observation
11. `age_at_entry` - Age on entry into study
12. `gender` - Patient gender (1=male, 2=female)
13. `af` - Atrial fibrillation presence (1=yes)
14. `hypertension` - Hypertension presence (1=yes)
15. `diab` - Diabetes presence (1=yes)
16. `thrombo` - Thromboembolism presence (1=yes)
17. `hf` - Heart failure presence (1=yes)
18. `HB_stroke_history` - History of stroke prior to major bleed (1=yes)
19. `ckd` - Chronic kidney disease presence (1=yes)
20. `ethnic_group` - Ethnicity (e.g., "White", "Unknown")
21. `frailty_score` - Electronic frailty index (continuous)
22. `bmi` - Body Mass Index
23. `tc_mmol_L` - Total cholesterol
24. `acr_mg_mmol` - Albumin-Creatinine Ratio
25. `smoking_status` - Smoking status (e.g., "Non-smoker", "Ex-smoker")
26. `vasc_dis_mi_pad` - Vascular disease/MI/PAD presence (1=yes)
27. `aortic_plaq` - Aortic plaque presence (1=yes)

## Variable Types

### Categorical Variables
- `gender` (binary: 1=male, 2=female)
- `end_fu_due_to_death` (binary: 1=yes)
- `stroke_1Y` (binary: 1=yes, 2=no)
- `Anticoagulant` (nominal: "No anticoagulant", "VKA", "DOAC")
- `ethnic_group` (nominal)
- `smoking_status` (nominal)
- Binary indicators (1=yes):
  - `af`
  - `hypertension`
  - `diab`
  - `thrombo`
  - `hf`
  - `HB_stroke_history`
  - `ckd`
  - `vasc_dis_mi_pad`
  - `aortic_plaq`

### Continuous Variables
- `age`
- `age_at_entry`
- `frailty_score`
- `bmi`
- `tc_mmol_L`
- `acr_mg_mmol`
- `stroke_time`

### Date Variables
- `time1`
- `time2`
- `earliest_af_date`
- `earliest_stroke_date`
- `end_fu`

## Missing Values
The data contains missing values, represented by empty fields in the CSV file. This is particularly noticeable in variables like:
- `tc_mmol_L`
- `acr_mg_mmol`
- `earliest_stroke_date`

## Notes for Modeling
1. Date variables should be converted to appropriate datetime format
2. Missing values need to be handled appropriately based on the specific modeling requirements
3. Categorical variables may need encoding (e.g., one-hot encoding for `ethnic_group` and `smoking_status`)
4. Binary variables are generally coded as 1=yes, but some have different coding (e.g., `stroke_1Y` uses 1=yes, 2=no)
5. The dataset appears to be longitudinal with multiple observations per patient across different time windows 