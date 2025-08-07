# Dataset Description

This document provides a comprehensive overview of the medical dataset structure and variables, with a focus on the actual column names as they appear in the data file for use in modeling.

## ⚠️ CRITICAL DATA ENCODING INCOMPATIBILITY

**IMPORTANT**: This project contains two datasets with **fundamentally incompatible** `stroke_1Y` encodings:

### OLD Dataset (`random_nuchad.csv`):
- `stroke_1Y = 1`: Stroke within 1 year of AF diagnosis (6.8% of patients)
- `stroke_1Y = 2`: **NO STROKE** (control patients) (93.2% of patients)

### NEW Dataset (`random_nuchad_250623.csv`):
- `stroke_1Y = 1.0`: Stroke within 1 year of AF diagnosis (2.6% of patients)
- `stroke_1Y = 2.0`: **Stroke AFTER 1 year** of AF diagnosis (5.3% of patients)
- `stroke_1Y = 3.0`: Stroke on same day as AF diagnosis (3.8% of patients)
- `stroke_1Y = 4.0`: Stroke BEFORE AF diagnosis (5.9% of patients)
- **Missing `stroke_1Y`**: **NO STROKE** (control patients) (82.4% of patients)

**Critical Issue**: `stroke_1Y = 2` means "no stroke" in the old dataset but "stroke after 1 year" in the new dataset. This makes cross-dataset analysis invalid without proper recoding.

**Validation**: Run `python analyze_stroke_encoding.py` for definitive proof of these differences.

## Data Structure

The dataset is stored in CSV format with the following columns in order:

1. `time1` - Start of observation window (date)
2. `time2` - End of observation window (date)
3. `earliest_af_date` - First date AF recorded
4. `earliest_stroke_date` - First date of stroke (if applicable)
5. `end_fu` - End of follow-up date
6. `end_fu_due_to_death` - Death as reason for end of follow-up (1=yes)
7. `stroke_1Y` - **⚠️ CRITICAL**: Stroke outcome encoding (see incompatibility warning above)
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
- `stroke_1Y` (**⚠️ DATASET-DEPENDENT ENCODING**):
  - **OLD dataset**: Binary (1=stroke within 1Y, 2=no stroke)
  - **NEW dataset**: Multi-level (1=stroke within 1Y, 2=stroke after 1Y, 3=stroke same day, 4=stroke before AF, Missing=no stroke)
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
- **`stroke_1Y`** (82.4% missing in new dataset - these represent control patients)

## Notes for Modeling

1. **CRITICAL**: Always verify which dataset you're using and apply appropriate `stroke_1Y` encoding interpretation
2. Date variables should be converted to appropriate datetime format:
   - OLD dataset: `%d%b%Y` format (e.g., "01Jan2020")
   - NEW dataset: `%d-%b-%y` format (e.g., "01-Jan-20")
3. Missing values need to be handled appropriately based on the specific modeling requirements
4. Categorical variables may need encoding (e.g., one-hot encoding for `ethnic_group` and `smoking_status`)
5. **`stroke_1Y` encoding differences make cross-dataset analysis invalid without proper recoding**
6. For new dataset: Missing `stroke_1Y` should be treated as controls (no stroke patients)
7. The dataset appears to be longitudinal with multiple observations per patient across different time windows

## Dataset-Specific Event Rates

### OLD Dataset (`random_nuchad.csv`):
- Total patients: 128,590
- Stroke within 1 year: 8,748 (6.8%)
- Controls (no stroke): 119,842 (93.2%)

### NEW Dataset (`random_nuchad_250623.csv`):
- Total patients: 136,695
- Stroke within 1 year: 3,540 (2.6%)
- All stroke events: 24,081 (17.6%)
- Controls (no stroke/missing stroke_1Y): 112,614 (82.4%)

## Validation and Analysis Tools

- **`python analyze_stroke_encoding.py`**: Comprehensive analysis proving the encoding differences
- **Survival EDA**: Use `uv run run_survival_eda` with appropriate filter configurations
- **Filter configurations**: Located in `filtering_configs/` directory with dataset-specific settings 