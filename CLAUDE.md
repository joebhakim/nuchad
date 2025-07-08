# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CHADS-VASc analysis package for stroke risk prediction in atrial fibrillation (AF) patients. The package performs transportability analysis comparing stroke rates between the current dataset and the original Lip et al. cohort, using density ratio reweighting and model comparison techniques.

## ⚠️ CRITICAL DATA ENCODING ISSUE

**FUNDAMENTAL INCOMPATIBILITY DISCOVERED**: The stroke outcome variable (`stroke_1Y`) has completely different encodings between datasets, making cross-dataset analysis invalid without proper handling:

- **OLD dataset** (`random_nuchad.csv`): stroke_1Y = 1 (stroke), stroke_1Y = 2 (controls)
- **NEW dataset** (`random_nuchad_250623.csv`): stroke_1Y = 1-4 (different stroke types), Missing = controls

**Key incompatibility**: `stroke_1Y = 2` means "no stroke" in old dataset but "stroke after 1 year" in new dataset!

Run `python analyze_stroke_encoding.py` to see detailed proof of this issue. This analysis must be shared with collaborators before any cross-dataset work.

## Build and Development Commands

### Installation
```bash
# Install the package in development mode
uv add -e .

# Install with pip (fallback)
pip install -e .
```

### Testing
```bash
# Run tests using pytest
pytest -xvs src/tests/

# Run tests using the provided shell script
./run_tests.sh
```

### Running Analysis

#### Master Script (all analyses)
```bash
# Run all three key analyses (generates main figures)
python run_all_analyses.py
```

#### Individual Scripts (preferred approach)
```bash
# Run individual analyses via dedicated scripts
uv run make_table1                     # Patient characteristics
uv run make_table2                     # Stroke rates by CHADS-VASc
uv run make_table1_stratified          # Stratified patient characteristics
uv run make_table2_stratified          # Stratified stroke rates
uv run run_eda                         # EDA and validation
uv run run_survival_eda                # Survival-focused EDA with Kaplan-Meier curves (all datasets)
uv run run_survival_eda -d random_nuchad.csv  # Run on specific dataset
uv run run_survival_eda --list-configs        # List available filter configurations
uv run run_survival_eda -f AF_FU365_population_new  # Use specific JSON filter config
uv run visualize_followup              # Follow-up visualization
uv run density_reweighting             # Density ratio reweighting
uv run model_comparison                # Model comparison
uv run filter_patients                 # Patient filtering report (with options)

# Data validation and critical encoding analysis
python analyze_stroke_encoding.py          # ESSENTIAL: Proves stroke_1Y encoding incompatibility between datasets
```

#### Legacy CLI (still available)
```bash
# Run individual analyses via CLI
python -m nuchad --task eda                    # EDA and validation
python -m nuchad --task table1                 # Patient characteristics
python -m nuchad --task table2                 # Stroke rates by CHADS-VASc
python -m nuchad --task table1_stratified      # Stratified characteristics
python -m nuchad --task table2_stratified      # Stratified stroke rates
python -m nuchad --task visualize              # Follow-up visualization
python -m nuchad --task reweight               # Density ratio reweighting
python -m nuchad --task filter                 # Patient filtering report
python -m nuchad --task compare                # Model comparison
```

### Data Requirements
- The package expects `data/random_nuchad.csv` to exist
- Tests will be skipped if this file is not available
- The test script (`run_tests.sh`) will create necessary directories

## Code Architecture

### Package Structure
```
src/
├── nuchad/                  # Main package
│   ├── __init__.py              # Package entry point
│   ├── __main__.py              # CLI interface with argparse
│   ├── analysis/                # Core analysis modules
│   │   ├── eda.py              # Data loading, CHADS-VASc calculation, validation
│   │   ├── table1.py           # Patient characteristics table
│   │   ├── table2.py           # Stroke rates by CHADS-VASc score
│   │   ├── table1_stratified.py # Stratified patient characteristics
│   │   ├── table2_stratified.py # Stratified stroke rates
│   │   ├── density_ratio_reweighting.py # Transportability analysis
│   │   └── model_comparison.py # Model performance comparison
│   ├── data_processing/
│   │   └── eligibility_filters.py # Patient filtering utilities
│   ├── utils/
│   │   └── paths_data.py       # Path management and data access
│   └── visualization/
│       └── visualize_end_fu.py # Follow-up time visualization
└── scripts/                 # Individual analysis scripts
    ├── __init__.py
    ├── make_table1.py           # Generate Table 1
    ├── make_table2.py           # Generate Table 2
    ├── make_table1_stratified.py # Generate stratified Table 1
    ├── make_table2_stratified.py # Generate stratified Table 2
    ├── run_eda.py               # Run EDA and validation
    ├── run_survival_eda.py      # Run survival-focused EDA with Kaplan-Meier curves
    ├── visualize_followup.py    # Generate follow-up visualization
    ├── density_reweighting.py   # Perform density ratio reweighting
    ├── model_comparison.py      # Perform model comparison
    └── filter_patients.py       # Generate patient filtering report
```

### Key Components

**Data Loading and Processing (`eda.py`)**:
- `get_df()`: Loads and preprocesses the main dataset from CSV
- `calculate_chadsvasc()`: Calculates CHADS-VASc scores for patients
- `validate_chadsvasc()`: Validates scores against original Lip et al. cohort
- `run_survival_eda()`: Generates Kaplan-Meier curves, timeline visualizations, and survival statistics with embedded metadata
- `run_survival_eda_all_datasets()`: Runs survival EDA on all available datasets with data-specific output directories
- Handles date parsing and patient ID indexing

**Patient Filtering (`eligibility_filters.py`)**:
- `filter_eligible_patients()`: Applies multiple filtering criteria with detailed tracking
- `generate_filter_report()`: Creates markdown reports of filtering steps
- Supports customizable criteria (AF diagnosis, follow-up requirements, etc.)

**Analysis Workflow**:
1. **EDA**: Load data, calculate CHADS-VASc scores, compare to original cohort
2. **Tables**: Generate patient characteristics and stroke rate tables
3. **Reweighting**: Apply density ratio weights for transportability
4. **Comparison**: Compare model performance vs. CHADS-VASc scores

**Path Management (`utils/paths_data.py`)**:
- `get_project_root()`: Determines project root directory
- `get_data_file()`: Accesses data files with fallback handling
- `get_results_dir()`: Manages results directory creation

### Key Design Patterns

**Modular Analysis**: Each analysis task is in its own module with clear entry points

**Flexible CLI**: The `__main__.py` provides both CLI access and programmatic API usage

**Path Abstraction**: All file access goes through utility functions for portability

**Filter Tracking**: Patient filtering includes detailed statistics and reporting for reproducibility

**Output Organization**: All results saved to `results/` directory with consistent naming

**Survival EDA Features**: 
- Creates dataset-specific subdirectories (`results/survival_eda_<dataset_name>/` or `results/survival_eda_<dataset_name>_<filter_config>/`)
- Embeds rich metadata in HTML files including data source and filtering steps
- Supports both pre-filter and post-filter analysis
- **JSON Filter Configuration Support**: Use predefined filter configurations from `filtering_configs/` directory
- Generates Kaplan-Meier curves stratified by CHADS-VASc risk groups
- Creates interactive timeline visualizations with patient sampling
- Outputs complete JSON-formatted filtering metadata for reproducibility
- Graceful fallback to default filtering if configuration not found

### Testing Strategy

- Tests are in `src/tests/test_analysis_scripts.py`
- Tests verify data loading, analysis execution, and output file generation
- Test outputs are prefixed with `test_` and cleaned up after completion
- Tests require the `data/random_nuchad.csv` file to be present

### Entry Points

**Individual Scripts** (recommended): `uv run <script_name>` for each analysis phase
**Master Script**: `python run_all_analyses.py` (runs all three key analyses)
**Legacy CLI**: `python -m nuchad --task <task_name>`
**API**: Import modules directly from `nuchad.analysis`, `nuchad.visualization`, etc.

### Script Design Pattern

Each script in the `scripts/` folder follows a consistent pattern:
- Imports the necessary analysis modules from `nuchad.analysis`
- Loads and filters data using standard utilities
- Calls the appropriate analysis function
- Provides clear console output about what was generated
- Has a simple main() function that returns 0 on success

The package is designed for both interactive analysis and automated pipeline execution, with comprehensive filtering, reporting, and visualization capabilities.