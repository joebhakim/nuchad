# Patient Filtering Report

## Configuration

**Configuration**: AF with Valid Stroke Outcomes (New Dataset)

**Description**: AF patients from new dataset with valid stroke outcomes (excludes pre-AF strokes). Includes strokes within 1 year, after 1 year, and no strokes.

**Use Case**: Analysis of new dataset with proper stroke outcome filtering

**Notes**: Excludes stroke_1Y=4 (pre-AF strokes) and missing values. Includes: 1=stroke within 1Y, 2=no stroke within 1Y, 3=stroke after 1Y.

## Patient Flow

**Total patients at start**: 136,695
**Final eligible patients**: 11,686
**Overall retention rate**: 8.5%

## Filtering Steps

| Step | Patients Remaining | Patients Removed | % of Original |
|------|-------------------|------------------|---------------|
| Initial cohort | 136,695 | 0 | 100.0% |
| AF diagnosis before or at time1 | 136,695 | 0 | 100.0% |
| Follow-up period ≥ 365 days | 92,115 | 44,580 | 67.4% |
| Stroke outcome in [1, 2, 3] (excluding pre-AF strokes) | 11,686 | 80,429 | 8.5% |

## Variable Distributions

Comparison of variable distributions before and after filtering.

### Numeric Variables
| Variable | Original | Filtered | Change |
|----------|----------|----------|--------|
| end_fu_due_to_death | 0.41 ± 0.49 [0.00, 1.00] | 0.45 ± 0.50 [0.00, 1.00] | +10.0% |
| stroke_1Y | 2.74 ± 1.08 [1.00, 4.00] | 2.02 ± 0.62 [1.00, 3.00] | -26.3% |
| tia_1Y | 3.00 ± 1.14 [1.00, 4.00] | 2.96 ± 1.14 [1.00, 4.00] | -1.5% |
| stroke_time | 0.37 ± 3.80 [-17.10, 17.51] | 2.78 ± 3.06 [0.00, 17.51] | +654.2% |
| tia_time | -0.64 ± 4.25 [-17.71, 17.41] | -0.63 ± 4.06 [-16.07, 12.11] | -1.7% |
| age | 75.90 ± 10.96 [44.00, 112.00] | 76.01 ± 10.94 [45.00, 112.00] | +0.1% |
| gender | 1.49 ± 0.50 [1.00, 2.00] | 1.48 ± 0.50 [1.00, 2.00] | -0.5% |
| af | 1.00 ± 0.00 [1.00, 1.00] | 1.00 ± 0.00 [1.00, 1.00] | +0.0% |
| hypertension | 0.57 ± 0.49 [0.00, 1.00] | 0.58 ± 0.49 [0.00, 1.00] | +0.7% |
| diab | 0.15 ± 0.36 [0.00, 1.00] | 0.15 ± 0.36 [0.00, 1.00] | +0.9% |
| thrombo | 0.03 ± 0.17 [0.00, 1.00] | 0.03 ± 0.17 [0.00, 1.00] | +5.2% |
| hf | 0.21 ± 0.41 [0.00, 1.00] | 0.22 ± 0.41 [0.00, 1.00] | +1.3% |
| Stroke_TIA_hx | 0.08 ± 0.28 [0.00, 1.00] | 0.08 ± 0.27 [0.00, 1.00] | -2.1% |
| ckd | 0.16 ± 0.37 [0.00, 1.00] | 0.17 ± 0.37 [0.00, 1.00] | +4.9% |
| frailty_score | 0.16 ± 0.09 [0.00, 0.67] | 0.16 ± 0.09 [0.00, 0.61] | +0.3% |

### Categorical Variables
| Variable | Original | Filtered | Change |
|----------|----------|----------|--------|
| earliest_af_date | Unique: 6905 | Missing: 0 (0.0%) | Unique: 4793 | Missing: 0 (0.0%) | +0.0pp |
| earliest_stroke_date | Unique: 6365 | Missing: 112614 (82.4%) | Unique: 5088 | Missing: 0 (0.0%) | -82.4pp |
| earliest_tia_date | Unique: 4885 | Missing: 125860 (92.1%) | Unique: 833 | Missing: 10779 (92.2%) | +0.2pp |
| end_fu | Unique: 6823 | Missing: 0 (0.0%) | Unique: 4062 | Missing: 0 (0.0%) | +0.0pp |
| ethnic_group | Unique: 12 | Missing: 0 (0.0%) | Unique: 12 | Missing: 0 (0.0%) | +0.0pp |
| smoking_status | Unique: 3 | Missing: 5269 (3.9%) | Unique: 3 | Missing: 448 (3.8%) | -0.0pp |
| Anticoag3m_type | Unique: 3 | Missing: 0 (0.0%) | Unique: 3 | Missing: 0 (0.0%) | +0.0pp |
| first_OAC_date | Unique: 5051 | Missing: 95826 (70.1%) | Unique: 2472 | Missing: 8137 (69.6%) | -0.5pp |
| first_antiplatelet_date | Unique: 5288 | Missing: 77338 (56.6%) | Unique: 2963 | Missing: 6613 (56.6%) | +0.0pp |
| time1 | Unique: 6905 | Missing: 0 (0.0%) | Unique: 4793 | Missing: 0 (0.0%) | +0.0pp |

## Key Patient Characteristics

- **Age**: 76.0 ± 10.9 years
- **Gender**: 6,023 male (51.5%), 5,663 female (48.5%)
- **Stroke outcomes**:
  - stroke_1Y=1.0: 2,112 (18.1%)
  - stroke_1Y=2.0: 7,223 (61.8%)
  - stroke_1Y=3.0: 2,351 (20.1%)

## Combinatorial Filter Analysis

Effects of applying different combinations of filters.

| Filters Applied | Patients Remaining | % of Original |
|-----------------|-------------------|---------------|
| None (all patients) | 136,695 | 100.0% |
| AF diagnosis before or at time1 | 136,695 | 100.0% |
| Follow-up period ≥ 365 days | 92,115 | 67.4% |
| Stroke outcome in [1, 2, 3] (excluding pre-AF strokes) | 11,686 | 8.5% |
| AF diagnosis before or at time1 + Follow-up period ≥ 365 days | 92,115 | 67.4% |
| AF diagnosis before or at time1 + Stroke outcome in [1, 2, 3] (excluding pre-AF strokes) | 11,686 | 8.5% |
| Follow-up period ≥ 365 days + Stroke outcome in [1, 2, 3] (excluding pre-AF strokes) | 11,686 | 8.5% |
| AF diagnosis before or at time1 + Follow-up period ≥ 365 days + Stroke outcome in [1, 2, 3] (excluding pre-AF strokes) | 11,686 | 8.5% |
