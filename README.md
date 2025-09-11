# CHADS-VASc Analysis Package

This package provides tools for analyzing CHADS-VASc score performance in predicting stroke risk among patients with atrial fibrillation (AF). The analysis includes:

1. Exploratory data analysis
2. Patient characteristics (Table 1) generation
3. Stratified characteristics analysis by risk groups
4. Stroke rate calculation (Table 2) by CHADS-VASc score
5. Follow-up time visualization
6. Density ratio reweighting for transportability to original development cohort
7. Detailed patient eligibility filtering with customizable criteria and reporting

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nuchad.git
cd nuchad

# Set up a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package with uv
uv init --package .
uv add -e .
```

## NOtes on the target trial emulation

We have patients with lots of non conforming elig criteria

Look at filtering configs.
Do we inlcude in our analyses, yes or no,

patients with less than 365 days FU? yes, do careful censorship analysis, report discrete hazards
- normalize stroke rate by number remaining at risk
- For 1011 data, end_fu < or > stroke TIME (and TIA time for secondary metric, maybe death as end_fu, but no separate death time)

for stroke before AF: 

patid,AF_date,stroke_date
8676,29-Apr-10,06-Nov-09 # af after stroke: exlcude
9138,21-Apr-04,21-Apr-04 # af during stroke: (actualy loosen this to plus minus 7 days), include
9263,19-Oct-09,18-Sep-12 # af before stroke: include. this person had >1 year stroke free, then stroke
- HOWEVER, for stroke date >1 year after AF date, dont have this in direct stroke rate calculation (like study). include it as secondary outcome.

antiplatelets and anticoag: even if on them the whole time, TREAET THIS as another confoudning variable and use in weighting. 
- dates: 
- we need to BIN the anticoagulant dates to accomodate for the variable shift: do weighting based on different rates of anticoag at month long windows
-- dont have original dates, so sensitivty analysis like assumption: poisson process with average: 1 or 2 days and tail of like 1 2 months

And repeat this whole analysis by year. ALSO, the above moving poisson thing can be estimated by extrapolating how the time course of anticoag in repsonse to AF shifts year over year

Three sets of (rates, average time conditional on "yes")






## Usage

The package can be used in two ways:

### 1. Command Line Interface

```bash
# Run exploratory data analysis
python -m nuchad --task eda

# Generate Table 1 (patient characteristics)
python -m nuchad --task table1

# Generate Table 1 stratified by CHADS-VASc risk groups
python -m nuchad --task table1_stratified

# Generate Table 2 (stroke rates by CHADS-VASc score)
python -m nuchad --task table2

# Generate Table 2 stratified by anticoagulation status
python -m nuchad --task table2_stratified

# Visualize follow-up times
python -m nuchad --task visualize

# Perform density ratio reweighting analysis
python -m nuchad --task reweight

# Generate patient filtering report with default criteria
python -m nuchad --task filter

# Generate filtering report with custom criteria
python -m nuchad --task filter --require-af --require-follow-up --require-stroke --min-follow-up-days 180

# Save filtering report to a specific name in the results directory
python -m nuchad --task filter --output custom_filter_analysis.md
```

### 2. Python API

```python
from nuchad.analysis import eda, table1, table2
from nuchad.visualization import visualize_end_fu
from nuchad.data_processing.eligibility_filters import filter_eligible_patients, generate_filter_report

# Load and prepare data
df = eda.get_df()

# Filter eligible patients with custom criteria
eligible_df, filter_stats = filter_eligible_patients(
    df,
    require_af=True,
    require_follow_up=True,
    require_stroke=False,
    af_before_time1=True,
    min_follow_up_days=365,
    stroke_window_days=365
)

# Generate a filtering report (saved to results/custom_filter_report.md)
generate_filter_report(filter_stats, "custom_filter_report.md")

# Calculate CHADS-VASc scores
eligible_df['CHADS-Vasc'] = eligible_df.apply(eda.calculate_chadsvasc, axis=1)

# Generate Table 1
table1_results = table1.create_table1(eligible_df)

# Generate Table 2
table2_results = table2.generate_cohort_table(eligible_df)

# Visualize follow-up times
visualize_end_fu.plot_end_fu_distribution(eligible_df)
```

## Project Structure

```
nuchad/
├── src/
│   ├── nuchad/
│   │   ├── analysis/
│   │   │   ├── __init__.py
│   │   │   ├── eda.py                      # Exploratory data analysis
│   │   │   ├── table1.py                   # Patient characteristics table
│   │   │   ├── table1_stratified.py        # Patient characteristics stratified by risk
│   │   │   ├── table2.py                   # Stroke rate analysis
│   │   │   ├── table2_stratified.py        # Stroke rates stratified by risk and anticoagulation
│   │   │   └── density_ratio_reweighting.py # Transportability analysis
│   │   ├── data_processing/
│   │   │   ├── __init__.py
│   │   │   └── utils.py                    # Patient filtering utilities
│   │   ├── visualization/
│   │   │   ├── __init__.py
│   │   │   └── visualize_end_fu.py         # Follow-up time visualization
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   └── paths.py                    # Path utilities
│   │   ├── data.py                         # Data access functions
│   │   ├── __init__.py
│   │   └── __main__.py                     # CLI entry point
│   └── tests/
│       ├── __init__.py
│       └── test_eda.py
├── data/                                   # Data directory (not in repo)
├── results/                                # Output files directory
├── pyproject.toml
└── README.md
```

## ⚠️ Critical Data Encoding Issue

**IMPORTANT**: The stroke outcome variable (`stroke_1Y`) is encoded differently between datasets:

- **OLD dataset** (`random_nuchad.csv`): 
  - `stroke_1Y = 1`: Stroke within 1 year (6.8%)
  - `stroke_1Y = 2`: **NO STROKE** (controls) (93.2%)

- **NEW dataset** (`random_nuchad_250623.csv`):
  - `stroke_1Y = 1`: Stroke within 1 year (2.6%)  
  - `stroke_1Y = 2`: **STROKE AFTER 1 year** (5.3%)
  - `stroke_1Y = 3`: Stroke same day as AF (3.8%)
  - `stroke_1Y = 4`: Stroke before AF (5.9%)
  - **Missing `stroke_1Y`**: NO STROKE (controls) (82.4%)

**This fundamental incompatibility makes cross-dataset analysis invalid without proper recoding.**

### Data Validation Script

```bash
# Run comprehensive stroke encoding analysis
python analyze_stroke_encoding.py
```

This script provides definitive proof of the encoding differences and should be run to demonstrate the issue to collaborators.

## Data Requirements

The package expects a patient-level dataset containing AF patients with the following key columns:

- `time1`: Start of observation window
- `time2`: End of observation window
- `earliest_af_date`: Date of AF diagnosis
- `earliest_stroke_date`: Date of stroke (if applicable)
- `end_fu`: End of follow-up date
- `end_fu_due_to_death`: Whether follow-up ended due to death
- `stroke_1Y`: Depends on the dataset, see below (!)
- `stroke_time`: Time to stroke (years)
- `Anticoagulant`: Anticoagulation status
- Risk factors:
  - `age`: Age in years
  - `gender`: Gender (1=Male, 2=Female)
  - `hypertension`: Hypertension (1=Yes, 0=No)
  - `diab`: Diabetes (1=Yes, 0=No)
  - `hf`: Heart failure (1=Yes, 0=No)
  - `thrombo`: Thromboembolism (1=Yes, 0=No)
  - `HB_stroke_history`: Prior stroke (1=Yes, 0=No)
  - `vasc_dis_mi_pad`: Vascular disease (1=Yes, 0=No)
  - Other clinical variables (BMI, frailty score, cholesterol, etc.)

## Figure Generation

The package generates several types of visualizations and figures. Here are the main entry points:

### Command Line Interface

```bash
# Generate follow-up time distribution histogram
python -m nuchad --task visualize

# Generate density ratio reweighting plots (creates multiple figures)
python -m nuchad --task reweight
```

### Python API

```python
from nuchad.visualization import visualize_end_fu
from nuchad.analysis import density_ratio_reweighting

# Generate follow-up time distribution
df = eda.get_df()
visualize_end_fu.plot_end_fu_distribution(df)

# Generate density ratio reweighting analysis and plots
density_ratio_reweighting.perform_reweighting_analysis(df)
```

### Direct Module Execution

```bash
# Run visualization module directly
python -m nuchad.visualization.visualize_end_fu

# Run density ratio analysis directly
python -m nuchad.analysis.density_ratio_reweighting
```

### Generated Figures

The following figure files are created in the `results/` directory:

- **`end_fu_distribution.png`**: Histogram showing the distribution of patient follow-up times in years
- **`density_ratio_weighted_rates.png`**: Line plot comparing stroke rates by CHADS-VASc score between:
  - Original Lip et al. cohort rates
  - Observed rates in the current dataset
  - Weighted rates after density ratio reweighting
- **`density_ratio_weight_distribution.png`**: Histogram showing the distribution of density ratio weights applied to patients

## Output Files

All analysis tasks save their outputs to the `results/` directory:

- `table1.md`: Patient characteristics for eligible cohort
- `table1_stratified.md`: Patient characteristics stratified by CHADS-VASc risk groups
- `table2.md`: Stroke rates by CHADS-VASc score 
- `table2_stratified.md`: Stroke rates by CHADS-VASc score and anticoagulation status
- `end_fu_distribution.png`: Distribution of follow-up times
- `density_ratio_results.md`: Results of density ratio reweighting
- `density_ratio_weighted_rates.png`: Original vs. weighted stroke rates
- `density_ratio_weight_distribution.png`: Distribution of weights
- `filter_report.md`: Default report on patient filtering and eligibility criteria

## Patient Filtering

The package provides detailed patient filtering functionality through the `filter_eligible_patients` function in the `data_processing.eligibility_filters` module. This allows you to:

1. Specify custom eligibility criteria
2. Generate detailed reports on filtering steps
3. Track how many patients are removed at each filtering step

Default filtering criteria:
- Patients must have an AF diagnosis before or at the start of observation
- Patients must have at least 365 days of follow-up
- Stroke is not required as an eligibility criterion

All filtering reports are saved to the `results/` directory by default. When specifying an output filename without a path, the file will be placed in the results directory. You can also provide explicit relative or absolute paths.

For more detailed documentation on eligibility filtering functionality, see the [Eligibility Filters Documentation](docs/eligibility_filters.md).

### Filter Report Format

The filtering report contains:
1. Total number of patients at the start
2. A table showing each filter step applied
3. For each step: number of patients remaining, number removed, and percentage of original cohort

Example report:
```
# Patient Eligibility Filtering Report

Total patients at start: 128590

| Filter Step | Patients Remaining | Patients Removed | % of Original Cohort |
|-------------|-------------------|------------------|----------------------|
| Initial cohort | 128590 | 0 | 100.0% |
| 1. AF diagnosis before or at time1 | 128590 | 0 | 100.0% |
| 2. Follow-up period ≥ 365 days | 87220 | 41370 | 67.8% |
```

## Development

```bash
# Install development dependencies
uv add -e ".[dev]"

# Run tests
pytest src/tests/

# Run linting
ruff check src/
```

## Testing

The package includes comprehensive tests to ensure that all analysis tools work correctly. Tests verify that:

1. Required input data is available
2. Analysis scripts execute without errors
3. Expected output files are generated
4. Output files contain the expected content

### Running Tests

You can run tests using either pytest directly or the provided shell script:

```bash
# Run tests with pytest
pytest -xvs src/tests/

# Or use the shell script (which includes environment checks)
chmod +x run_tests.sh
./run_tests.sh
```

The test script will check for the required data file (`random_nuchad.csv`) and create necessary directories if they don't exist. All test outputs are saved with a `test_` prefix and are automatically cleaned up after tests complete.

### Test Coverage

Tests cover all major functionalities:
- Exploratory data analysis
- Table 1 (patient characteristics) generation
- Table 2 (stroke rates) generation 
- Stratified analyses
- Visualization generation
- Patient filtering
- Density ratio reweighting

# Other docs

https://docs.google.com/document/d/1bc00rUIiBYlEEOH0K0QK6h4h4q1NZeUNjxw3L1zOLvc/edit?tab=t.0

