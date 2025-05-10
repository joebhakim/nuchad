# Patient Eligibility Filtering

This document describes the patient eligibility filtering functionality provided by the `eligibility_filters` module in the nuchad package.

## Overview

The `eligibility_filters` module provides tools for filtering patients based on various clinical criteria and generating detailed reports on the filtering process. These tools are essential for:

1. Creating well-defined cohorts for analysis
2. Ensuring reproducibility of results
3. Documenting filtering decisions
4. Understanding the impact of different inclusion/exclusion criteria

## Key Functions

The module provides two main functions:

### `filter_eligible_patients()`

This function filters a patient dataframe based on specified eligibility criteria.

```python
def filter_eligible_patients(
    df: pd.DataFrame,
    require_af: bool = True,
    require_follow_up: bool = True,
    require_stroke: bool = False,
    af_before_time1: bool = True,
    min_follow_up_days: int = 365,
    stroke_window_days: int = 365,
) -> Tuple[pd.DataFrame, Dict[str, Any]]
```

#### Parameters

- `df`: The input dataframe with patient data
- `require_af`: Whether to require patients to have AF diagnosis
- `require_follow_up`: Whether to require patients to have sufficient follow-up
- `require_stroke`: Whether to require patients to have a stroke diagnosis
- `af_before_time1`: If True, AF must be diagnosed before or at time1
- `min_follow_up_days`: Minimum follow-up period in days
- `stroke_window_days`: Window after time1 in which stroke must occur (if required)

#### Returns

- A tuple containing:
  - Filtered dataframe with only eligible patients
  - Dictionary with filter statistics for reporting

### `generate_filter_report()`

This function creates a detailed report of the filtering steps and statistics.

```python
def generate_filter_report(
    filter_stats: Dict[str, Any], 
    output_path: Optional[Union[str, Path]] = None,
    max_num_to_print_permutations: int = 4
) -> str
```

#### Parameters

- `filter_stats`: Dictionary with filter statistics from `filter_eligible_patients()`
- `output_path`: Path to save the report to, defaults to results/filter_report.md
- `max_num_to_print_permutations`: Maximum number of filters for which to print all combinations

#### Returns

- The report text as a string

## Usage Examples

### Basic Filtering with Default Criteria

```python
from nuchad.analysis.eda import get_df
from nuchad.data_processing.eligibility_filters import filter_eligible_patients, generate_filter_report

# Load data
df = get_df()

# Apply default filtering (AF required, 365 days follow-up, AF before time1)
filtered_df, stats = filter_eligible_patients(df)

# Generate a report
report = generate_filter_report(stats)
print(f"Eligible patients: {len(filtered_df)} out of {stats['total']}")
```

### Custom Filtering Criteria

```python
# Custom filtering: Require stroke within 180 days
filtered_df, stats = filter_eligible_patients(
    df,
    require_af=True,
    require_follow_up=True,
    require_stroke=True,
    af_before_time1=True,
    min_follow_up_days=90,  # Shorter follow-up required
    stroke_window_days=180  # Stroke must occur within 180 days
)

# Save report to custom path
report = generate_filter_report(stats, "results/stroke_patients_report.md")
```

### Using via Command Line Interface

The filtering functionality can also be used via the CLI:

```bash
# Default filtering
python -m nuchad --task filter

# Custom filtering
python -m nuchad --task filter --require-af --require-follow-up --require-stroke --min-follow-up-days 180

# Custom output
python -m nuchad --task filter --output custom_filter_analysis.md
```

## Filter Report Format

The filtering report contains:

1. Total number of patients at the start
2. A table showing each filter step applied
3. For each step: number of patients remaining, number removed, and percentage of original cohort
4. Combinatorial analysis showing the effect of applying different combinations of filters

Example report:

```
# Patient Eligibility Filtering Report

Total patients at start: 128590

## Filter Steps

| Step | Patients Remaining | Patients Removed | % of Original Cohort |
|------|-------------------|------------------|----------------------|
| Initial cohort | 128590 | 0 | 100.0% |
| 1. AF diagnosis before or at time1 | 128590 | 0 | 100.0% |
| 2. Follow-up period ≥ 365 days | 87220 | 41370 | 67.8% |

## Combinatorial Filter Analysis

| Filters Applied | Patients Remaining | % of Original Cohort |
|-----------------|-------------------|----------------------|
| None (all patients) | 128590 | 100.0% |
| AF diagnosis before or at time1 | 128590 | 100.0% |
| Follow-up period ≥ 365 days | 87220 | 67.8% |
| AF diagnosis before or at time1 + Follow-up period ≥ 365 days | 87220 | 67.8% |
```

## Output Path Handling

When specifying an output path for the report:

- If no path is provided, the report is saved to `results/filter_report.md`
- If a relative path without directory indicators (`/`, `./`, `../`) is provided, the file is saved in the results directory
- If an absolute or explicit relative path is provided, it is used as-is
- Parent directories are created automatically if they don't exist

## Implementation Notes

- The module uses the `Path` object from the `pathlib` module for handling file paths
- The `importlib.resources` feature is used instead of hard-coding paths
- Filter statistics include both sequential application and combinatorial effects
- The module supports both programmatic use and command-line interface 