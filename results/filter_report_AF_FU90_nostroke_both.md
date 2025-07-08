# Patient Filtering Report

## Configuration

**Configuration**: AF with Short Follow-up (Both Datasets)

**Description**: AF patients with 90-day follow-up requirement, no stroke requirement. Useful for early outcome studies where longer follow-up may introduce bias.

**Use Case**: Early outcome studies and sensitivity analysis

**Notes**: Shorter follow-up requirement increases sample size but may miss later events. Compatible with both datasets.

## Patient Flow

**Total patients at start**: 128,590
**Final eligible patients**: 106,408
**Overall retention rate**: 82.7%

## Filtering Steps

| Step | Patients Remaining | Patients Removed | % of Original |
|------|-------------------|------------------|---------------|
| Initial cohort | 128,590 | 0 | 100.0% |
| AF diagnosis before or at time1 | 128,590 | 0 | 100.0% |
| Follow-up period ≥ 90 days | 106,408 | 22,182 | 82.7% |

## Variable Distributions

Comparison of variable distributions before and after filtering.

### Numeric Variables
| Variable | Original | Filtered | Change |
|----------|----------|----------|--------|
| end_fu_due_to_death | 0.41 ± 0.49 [0.00, 1.00] | 0.34 ± 0.47 [0.00, 1.00] | -17.5% |
| stroke_1Y | 1.93 ± 0.25 [1.00, 2.00] | 1.94 ± 0.24 [1.00, 2.00] | +0.4% |
| stroke_time | 2.06 ± 2.88 [0.00, 17.51] | 2.41 ± 2.98 [0.00, 17.51] | +17.0% |
| age | 75.71 ± 11.02 [44.00, 112.00] | 74.64 ± 10.86 [44.00, 108.00] | -1.4% |
| age_at_entry | 68.82 ± 11.45 [45.00, 105.00] | 67.86 ± 11.25 [45.00, 105.00] | -1.4% |
| gender | 1.49 ± 0.50 [1.00, 2.00] | 1.48 ± 0.50 [1.00, 2.00] | -0.7% |
| af | 1.00 ± 0.00 [1.00, 1.00] | 1.00 ± 0.00 [1.00, 1.00] | +0.0% |
| hypertension | 0.57 ± 0.50 [0.00, 1.00] | 0.57 ± 0.50 [0.00, 1.00] | +0.4% |
| diab | 0.15 ± 0.35 [0.00, 1.00] | 0.14 ± 0.35 [0.00, 1.00] | -1.9% |
| thrombo | 0.03 ± 0.16 [0.00, 1.00] | 0.03 ± 0.16 [0.00, 1.00] | -2.4% |
| hf | 0.21 ± 0.41 [0.00, 1.00] | 0.19 ± 0.39 [0.00, 1.00] | -12.7% |
| HB_stroke_history | 0.04 ± 0.20 [0.00, 1.00] | 0.04 ± 0.19 [0.00, 1.00] | -17.6% |
| ckd | 0.15 ± 0.36 [0.00, 1.00] | 0.14 ± 0.35 [0.00, 1.00] | -7.2% |
| frailty_score | 0.15 ± 0.09 [0.00, 0.64] | 0.15 ± 0.09 [0.00, 0.64] | -2.1% |
| bmi | 27.38 ± 5.30 [16.00, 45.00] | 27.62 ± 5.26 [16.00, 45.00] | +0.9% |

### Categorical Variables
| Variable | Original | Filtered | Change |
|----------|----------|----------|--------|
| time1 | Unique: 6903 | Missing: 0 (0.0%) | Unique: 6801 | Missing: 0 (0.0%) | +0.0pp |
| time2 | Unique: 6860 | Missing: 0 (0.0%) | Unique: 6774 | Missing: 0 (0.0%) | +0.0pp |
| earliest_af_date | Unique: 6903 | Missing: 0 (0.0%) | Unique: 6801 | Missing: 0 (0.0%) | +0.0pp |
| earliest_stroke_date | Unique: 5787 | Missing: 112614 (87.6%) | Unique: 5456 | Missing: 92761 (87.2%) | -0.4pp |
| end_fu | Unique: 6819 | Missing: 0 (0.0%) | Unique: 6560 | Missing: 0 (0.0%) | +0.0pp |
| Anticoagulant | Unique: 3 | Missing: 0 (0.0%) | Unique: 3 | Missing: 0 (0.0%) | +0.0pp |
| ethnic_group | Unique: 12 | Missing: 0 (0.0%) | Unique: 12 | Missing: 0 (0.0%) | +0.0pp |
| smoking_status | Unique: 3 | Missing: 5120 (4.0%) | Unique: 3 | Missing: 4018 (3.8%) | -0.2pp |

## Key Patient Characteristics

- **Age**: 74.6 ± 10.9 years
- **Gender**: 55,229 male (51.9%), 51,179 female (48.1%)
- **Stroke outcomes**:
  - stroke_1Y=1: 6,419 (6.0%)
  - stroke_1Y=2: 99,989 (94.0%)

## Combinatorial Filter Analysis

Effects of applying different combinations of filters.

| Filters Applied | Patients Remaining | % of Original |
|-----------------|-------------------|---------------|
| None (all patients) | 128,590 | 100.0% |
| AF diagnosis before or at time1 | 128,590 | 100.0% |
| Follow-up period ≥ 90 days | 106,408 | 82.7% |
| AF diagnosis before or at time1 + Follow-up period ≥ 90 days | 211,905 | 164.8% |
