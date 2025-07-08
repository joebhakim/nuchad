# Patient Filtering Report

## Configuration

**Configuration**: Standard AF Population Analysis (Original Dataset)

**Description**: Population-based AF patients with 1-year follow-up, no stroke requirement. This creates a representative AF population for risk prediction modeling.

**Use Case**: Population-based risk prediction and model validation

**Notes**: This is the standard configuration for population-based analysis. Maintains natural stroke risk distribution across CHADS-VASc scores.

## Patient Flow

**Total patients at start**: 128,590
**Final eligible patients**: 87,220
**Overall retention rate**: 67.8%

## Filtering Steps

| Step | Patients Remaining | Patients Removed | % of Original |
|------|-------------------|------------------|---------------|
| Initial cohort | 128,590 | 0 | 100.0% |
| AF diagnosis before or at time1 | 128,590 | 0 | 100.0% |
| Follow-up period ≥ 365 days | 87,220 | 41,370 | 67.8% |

## Variable Distributions

Comparison of variable distributions before and after filtering.

### Numeric Variables
| Variable | Original | Filtered | Change |
|----------|----------|----------|--------|
| end_fu_due_to_death | 0.41 ± 0.49 [0.00, 1.00] | 0.31 ± 0.46 [0.00, 1.00] | -23.5% |
| stroke_1Y | 1.93 ± 0.25 [1.00, 2.00] | 1.95 ± 0.22 [1.00, 2.00] | +0.9% |
| stroke_time | 2.06 ± 2.88 [0.00, 17.51] | 2.78 ± 3.06 [0.00, 17.51] | +34.9% |
| age | 75.71 ± 11.02 [44.00, 112.00] | 73.83 ± 10.72 [45.00, 104.00] | -2.5% |
| age_at_entry | 68.82 ± 11.45 [45.00, 105.00] | 67.34 ± 11.07 [45.00, 101.00] | -2.2% |
| gender | 1.49 ± 0.50 [1.00, 2.00] | 1.48 ± 0.50 [1.00, 2.00] | -1.1% |
| af | 1.00 ± 0.00 [1.00, 1.00] | 1.00 ± 0.00 [1.00, 1.00] | +0.0% |
| hypertension | 0.57 ± 0.50 [0.00, 1.00] | 0.57 ± 0.50 [0.00, 1.00] | -0.2% |
| diab | 0.15 ± 0.35 [0.00, 1.00] | 0.14 ± 0.35 [0.00, 1.00] | -5.7% |
| thrombo | 0.03 ± 0.16 [0.00, 1.00] | 0.03 ± 0.16 [0.00, 1.00] | -4.3% |
| hf | 0.21 ± 0.41 [0.00, 1.00] | 0.17 ± 0.38 [0.00, 1.00] | -20.3% |
| HB_stroke_history | 0.04 ± 0.20 [0.00, 1.00] | 0.03 ± 0.18 [0.00, 1.00] | -26.1% |
| ckd | 0.15 ± 0.36 [0.00, 1.00] | 0.13 ± 0.34 [0.00, 1.00] | -15.0% |
| frailty_score | 0.15 ± 0.09 [0.00, 0.64] | 0.15 ± 0.08 [0.00, 0.61] | -4.9% |
| bmi | 27.38 ± 5.30 [16.00, 45.00] | 27.74 ± 5.21 [16.00, 45.00] | +1.3% |

### Categorical Variables
| Variable | Original | Filtered | Change |
|----------|----------|----------|--------|
| time1 | Unique: 6903 | Missing: 0 (0.0%) | Unique: 6506 | Missing: 0 (0.0%) | +0.0pp |
| time2 | Unique: 6860 | Missing: 0 (0.0%) | Unique: 6471 | Missing: 0 (0.0%) | +0.0pp |
| earliest_af_date | Unique: 6903 | Missing: 0 (0.0%) | Unique: 6506 | Missing: 0 (0.0%) | +0.0pp |
| earliest_stroke_date | Unique: 5787 | Missing: 112614 (87.6%) | Unique: 5088 | Missing: 75534 (86.6%) | -1.0pp |
| end_fu | Unique: 6819 | Missing: 0 (0.0%) | Unique: 6168 | Missing: 0 (0.0%) | +0.0pp |
| Anticoagulant | Unique: 3 | Missing: 0 (0.0%) | Unique: 3 | Missing: 0 (0.0%) | +0.0pp |
| ethnic_group | Unique: 12 | Missing: 0 (0.0%) | Unique: 12 | Missing: 0 (0.0%) | +0.0pp |
| smoking_status | Unique: 3 | Missing: 5120 (4.0%) | Unique: 3 | Missing: 3363 (3.9%) | -0.1pp |

## Key Patient Characteristics

- **Age**: 73.8 ± 10.7 years
- **Gender**: 45,691 male (52.4%), 41,529 female (47.6%)
- **Stroke outcomes**:
  - stroke_1Y=1: 4,458 (5.1%)
  - stroke_1Y=2: 82,762 (94.9%)

## Combinatorial Filter Analysis

Effects of applying different combinations of filters.

| Filters Applied | Patients Remaining | % of Original |
|-----------------|-------------------|---------------|
| None (all patients) | 128,590 | 100.0% |
| AF diagnosis before or at time1 | 128,590 | 100.0% |
| Follow-up period ≥ 365 days | 87,220 | 67.8% |
| AF diagnosis before or at time1 + Follow-up period ≥ 365 days | 173,766 | 135.1% |
