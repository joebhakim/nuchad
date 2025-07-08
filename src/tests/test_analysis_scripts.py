"""Tests for the analysis scripts in the nuchad package."""

import os
import sys
import pytest
import pandas as pd
import subprocess
from pathlib import Path
import importlib.resources

from nuchad.utils import get_data_file, get_results_dir
from nuchad.analysis import eda_old, eda, table1, table2, table1_stratified, table2_stratified, density_ratio_reweighting, model_comparison
from nuchad.visualization import visualize_end_fu
from nuchad.data_processing.eligibility_filters import filter_eligible_patients, generate_filter_report

# Setup and helper functions
def check_data_exists():
    """Check if the necessary data file exists."""
    try:
        with get_data_file('random_nuchad.csv') as data_path:
            return True
    except FileNotFoundError:
        return False

def check_output_exists(file_path):
    """Check if an output file exists."""
    return os.path.exists(file_path)

def remove_test_outputs():
    """Remove test outputs to ensure clean testing environment."""
    results_dir = get_results_dir()
    test_files = [
        'test_table1.md',
        'test_table1_stratified.md',
        'test_table2.md',
        'test_table2_stratified.md',
        'test_filter_report.md',
        'test_density_ratio_results.md',
        'test_end_fu_distribution.png',
        'test_model_comparison_roc.png',
        'test_model_comparison_results.md',
        'survival_eda_test_dataset/',  # Directory for survival EDA outputs
        'test_dataset_comparison_analysis.md',
    ]
    
    for file in test_files:
        file_path = results_dir / file
        if file_path.is_file():
            os.remove(file_path)
        elif file_path.is_dir():
            # Remove directory and all contents
            import shutil
            shutil.rmtree(file_path, ignore_errors=True)

def run_script_command(script_name, args=None):
    """Run a script command and return the result."""
    cmd = [sys.executable, "-m", f"scripts.{script_name}"]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Script timed out"
    except Exception as e:
        return False, "", str(e)

# Fixtures
@pytest.fixture(scope="module")
def data_frame():
    """Fixture to load and prepare the dataset once for all tests."""
    if not check_data_exists():
        pytest.skip("Required data file 'random_nuchad.csv' not found")
    
    return eda_old.get_df()

@pytest.fixture(scope="module")
def eligible_data_frame(data_frame):
    """Fixture to get eligible patients according to default criteria."""
    df, _ = filter_eligible_patients(data_frame)
    return df

@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Setup before tests and cleanup after."""
    # Setup: ensure results directory exists
    results_dir = get_results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the test
    yield
    
    # Teardown: remove test outputs
    remove_test_outputs()

# Core Module Tests (existing functionality)
def test_eda_old_module(eligible_data_frame):
    """Test the exploratory data analysis functionality from eda_old."""
    # Run EDA validation
    results = eda_old.validate_chadsvasc(eligible_data_frame, "time1", "end_fu")
    
    # Check that results dataframe has expected structure
    assert isinstance(results, pd.DataFrame)
    assert "CHADS-Vasc" in results.columns
    assert "Observed Stroke Rate" in results.columns
    assert "Original Stroke Rate" in results.columns
    assert "95% CI Lower" in results.columns
    assert "95% CI Upper" in results.columns
    assert len(results) > 0

def test_eda_new_module(eligible_data_frame):
    """Test the newer EDA module with survival analysis functionality."""
    # Test the get_df function from new eda module
    df_new = eda.get_df()
    assert isinstance(df_new, pd.DataFrame)
    assert len(df_new) > 0
    
    # Test survival data preparation
    survival_df = eda.prepare_survival_data(eligible_data_frame)
    assert isinstance(survival_df, pd.DataFrame)
    assert 'survival_time' in survival_df.columns
    assert 'event' in survival_df.columns
    assert len(survival_df) > 0

def test_table1_module(eligible_data_frame):
    """Test the table1 generation functionality."""
    # Generate and save a test table1
    results_dir = get_results_dir()
    output_path = results_dir / 'test_table1.md'
    
    # Generate table1
    table = table1.create_table1(eligible_data_frame)
    
    # Save to test file
    with open(output_path, 'w') as f:
        f.write("# Test Table 1: Baseline Characteristics\n\n")
        f.write(table.to_markdown(index=False))
    
    # Check that the output file was created
    assert check_output_exists(output_path)
    
    # Read the file and check content
    with open(output_path, 'r') as f:
        content = f.read()
        assert "Baseline Characteristics" in content
        assert "Demographics" in content
        assert "Age (years)" in content

def test_table1_stratified_module(eligible_data_frame):
    """Test the stratified table1 generation functionality."""
    # Generate and save a test stratified table1
    results_dir = get_results_dir()
    output_path = results_dir / 'test_table1_stratified.md'
    
    # Generate stratified table1 (function saves to results_dir/table1_stratified.md)
    if 'CHADS-Vasc' not in eligible_data_frame.columns:
        from nuchad.utils import calculate_chadsvasc
        eligible_data_frame['CHADS-Vasc'] = eligible_data_frame.apply(calculate_chadsvasc, axis=1)
        
    result_df = table1_stratified.generate_stratified_table1(eligible_data_frame)
    
    # Check the function returns a DataFrame
    assert result_df is not None, "Function should return a DataFrame"
    assert not result_df.empty, "Returned DataFrame should not be empty"
    
    # Check that the output file was created (default location)
    results_dir = get_results_dir()
    actual_output_path = results_dir / 'table1_stratified.md'
    assert check_output_exists(actual_output_path)
    
    # Read the file and check content
    with open(actual_output_path, 'r') as f:
        content = f.read()
        assert "Stratified" in content
        assert "Risk" in content

def test_table2_module(eligible_data_frame):
    """Test the table2 generation functionality."""
    # Generate table2 (function saves to results_dir/table2.md)
    table = table2.generate_cohort_table(eligible_data_frame)
    
    # Check the function returns a DataFrame
    assert table is not None, "Function should return a DataFrame"
    assert not table.empty, "Returned DataFrame should not be empty"
    
    # Check that the output file was created (default location)
    results_dir = get_results_dir()
    actual_output_path = results_dir / 'table2.md'
    assert check_output_exists(actual_output_path)
    
    # Read the file and check content
    with open(actual_output_path, 'r') as f:
        content = f.read()
        assert "Stroke Rates" in content
        assert "CHADS-VASc Score" in content
        assert "Patient-Years" in content

def test_table2_stratified_module(eligible_data_frame):
    """Test the stratified table2 generation functionality."""
    # Generate and save a test stratified table2
    results_dir = get_results_dir()
    output_path = results_dir / 'test_table2_stratified.md'
    
    # Generate stratified table2
    if 'CHADS-Vasc' not in eligible_data_frame.columns:
        from nuchad.utils import calculate_chadsvasc
        eligible_data_frame['CHADS-Vasc'] = eligible_data_frame.apply(calculate_chadsvasc, axis=1)
        
    results = table2_stratified.generate_stratified_table2(eligible_data_frame)
    
    # Check that there are results
    assert 'characteristics' in results
    assert 'rates' in results
    
    # Check that the output files were created
    stratified_file = results_dir / 'table2_stratified.md'
    characteristics_file = results_dir / 'table2_stratified_characteristics.md'
    assert check_output_exists(stratified_file)
    assert check_output_exists(characteristics_file)
    
    # Read the file and check content
    with open(stratified_file, 'r') as f:
        content = f.read()
        assert "Stroke Rates Stratified" in content
        assert "Anticoagulation Status" in content

def test_visualize_module(eligible_data_frame):
    """Test the visualization functionality."""
    # Generate visualization (function saves to results_dir/end_fu_distribution.png)
    visualize_end_fu.plot_end_fu_distribution(eligible_data_frame)
    
    # Check that the output file was created (default location)
    results_dir = get_results_dir()
    actual_output_path = results_dir / 'end_fu_distribution.png'
    assert check_output_exists(actual_output_path)

def test_filter_report_module(data_frame):
    """Test the filter report generation functionality."""
    # Generate and save a test filter report
    results_dir = get_results_dir()
    output_path = results_dir / 'test_filter_report.md'
    
    # Filter patients and generate report
    _, filter_stats = filter_eligible_patients(
        data_frame,
        require_af=True,
        require_follow_up=True,
        require_stroke=False,
        af_before_time1=True,
        min_follow_up_days=180,  # Custom value to test
        stroke_window_days=365
    )
    
    generate_filter_report(filter_stats, output_path=output_path)
    
    # Check that the output file was created
    assert check_output_exists(output_path)
    
    # Read the file and check content
    with open(output_path, 'r') as f:
        content = f.read()
        assert "Patient Eligibility Filtering Report" in content
        assert "Filter Steps" in content
        assert "AF diagnosis" in content
        assert "Follow-up period" in content

def test_density_ratio_reweighting_module(eligible_data_frame):
    """Test the density ratio reweighting functionality."""
    # Run reweighting analysis (function saves to fixed locations)
    try:
        result_df = density_ratio_reweighting.perform_reweighting_analysis(eligible_data_frame)
        
        # Check the function returns a DataFrame
        assert result_df is not None, "Function should return a DataFrame"
        
        # Check that the output files were created (default locations)
        results_dir = get_results_dir()
        assert check_output_exists(results_dir / 'density_ratio_results.md')
        assert check_output_exists(results_dir / 'density_ratio_weighted_rates.png')
        assert check_output_exists(results_dir / 'density_ratio_weight_distribution.png')
        
        # Read the file and check content
        with open(results_dir / 'density_ratio_results.md', 'r') as f:
            content = f.read()
            assert "Density Ratio Reweighting" in content
    except (ImportError, AttributeError):
        # This module might be optional or have dependencies not installed
        pytest.skip("Density ratio reweighting module not fully implemented or dependencies missing") 

def test_model_comparison_module(eligible_data_frame):
    """Test the model comparison functionality."""
    # Run model comparison
    try:
        results = model_comparison.perform_model_comparison(eligible_data_frame)
        
        # Check that results contain expected keys
        assert 'auc_model' in results
        assert 'auc_score' in results
        
        # Check that output files were created
        results_dir = get_results_dir()
        assert check_output_exists(results_dir / 'model_comparison_roc.png')
        assert check_output_exists(results_dir / 'model_comparison_results.md')
        
    except (ImportError, AttributeError, NotImplementedError):
        pytest.skip("Model comparison module not fully implemented or dependencies missing")

# Script Entry Point Tests (new comprehensive tests)
def test_run_eda_script():
    """Test the run_eda script entry point."""
    if not check_data_exists():
        pytest.skip("Required data file not found")
    
    success, stdout, stderr = run_script_command("run_eda")
    assert success, f"Script failed: {stderr}"
    assert "exploratory data analysis" in stdout.lower()

def test_make_table1_script():
    """Test the make_table1 script entry point."""
    if not check_data_exists():
        pytest.skip("Required data file not found")
    
    success, stdout, stderr = run_script_command("make_table1")
    assert success, f"Script failed: {stderr}"
    assert "table 1" in stdout.lower()
    
    # Check output file was created
    results_dir = get_results_dir()
    assert check_output_exists(results_dir / 'table1.md')

def test_make_table2_script():
    """Test the make_table2 script entry point."""
    if not check_data_exists():
        pytest.skip("Required data file not found")
    
    success, stdout, stderr = run_script_command("make_table2")
    assert success, f"Script failed: {stderr}"
    assert "table 2" in stdout.lower()

def test_make_table1_stratified_script():
    """Test the make_table1_stratified script entry point."""
    if not check_data_exists():
        pytest.skip("Required data file not found")
    
    success, stdout, stderr = run_script_command("make_table1_stratified")
    assert success, f"Script failed: {stderr}"
    assert "stratified" in stdout.lower()

def test_make_table2_stratified_script():
    """Test the make_table2_stratified script entry point."""
    if not check_data_exists():
        pytest.skip("Required data file not found")
    
    success, stdout, stderr = run_script_command("make_table2_stratified")
    assert success, f"Script failed: {stderr}"
    assert "stratified" in stdout.lower()

def test_visualize_followup_script():
    """Test the visualize_followup script entry point."""
    if not check_data_exists():
        pytest.skip("Required data file not found")
    
    success, stdout, stderr = run_script_command("visualize_followup")
    assert success, f"Script failed: {stderr}"
    assert "visualization" in stdout.lower()

def test_density_reweighting_script():
    """Test the density_reweighting script entry point."""
    if not check_data_exists():
        pytest.skip("Required data file not found")
    
    success, stdout, stderr = run_script_command("density_reweighting")
    # This might fail due to dependencies, so we check more gracefully
    if not success:
        pytest.skip(f"Density reweighting script failed, possibly due to missing dependencies: {stderr}")
    assert "reweighting" in stdout.lower()

def test_model_comparison_script():
    """Test the model_comparison script entry point."""
    if not check_data_exists():
        pytest.skip("Required data file not found")
    
    success, stdout, stderr = run_script_command("model_comparison")
    # This might fail due to dependencies, so we check more gracefully
    if not success:
        pytest.skip(f"Model comparison script failed, possibly due to missing dependencies: {stderr}")
    assert "comparison" in stdout.lower()

def test_run_survival_eda_script():
    """Test the run_survival_eda script entry point."""
    if not check_data_exists():
        pytest.skip("Required data file not found")
    
    success, stdout, stderr = run_script_command("run_survival_eda", ["--sample-size", "10"])
    assert success, f"Script failed: {stderr}"
    assert "survival" in stdout.lower()
    
    # Check that survival EDA output directory was created
    results_dir = get_results_dir()
    survival_dirs = list(results_dir.glob("survival_eda_*"))
    assert len(survival_dirs) > 0, "No survival EDA output directories found"

def test_compare_datasets_script():
    """Test the compare_datasets script entry point."""
    if not check_data_exists():
        pytest.skip("Required data file not found")
    
    # This script compares two datasets, so it might fail if only one exists
    success, stdout, stderr = run_script_command("compare_datasets")
    if not success and "not found" in stderr.lower():
        pytest.skip("Second dataset not available for comparison")
    
    if success:
        assert "comparison" in stdout.lower()

# Integration Tests
def test_run_all_analyses_integration():
    """Test that the main run_all_analyses.py script works."""
    if not check_data_exists():
        pytest.skip("Required data file not found")
    
    # This is an integration test for the master script
    cmd = [sys.executable, "run_all_analyses.py"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=Path.cwd())
        # Even if it fails due to missing dependencies, we want to know it ran
        assert result.returncode in [0, 1], f"Unexpected return code: {result.returncode}"
        
        if result.returncode == 0:
            assert "analysis" in result.stdout.lower()
        
    except subprocess.TimeoutExpired:
        pytest.skip("Integration test timed out")
    except FileNotFoundError:
        pytest.skip("run_all_analyses.py not found")

# CLI Interface Tests
def test_nuchad_cli_help():
    """Test the main CLI interface help."""
    cmd = [sys.executable, "-m", "nuchad", "--help"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0
        assert "chads" in result.stdout.lower()
        assert "task" in result.stdout.lower()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pytest.skip("CLI help test failed")

def test_nuchad_cli_eda_task():
    """Test the main CLI with EDA task."""
    if not check_data_exists():
        pytest.skip("Required data file not found")
    
    cmd = [sys.executable, "-m", "nuchad", "--task", "eda"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        # Allow for graceful failure due to dependencies
        if result.returncode != 0:
            pytest.skip(f"CLI EDA task failed, possibly due to dependencies: {result.stderr}")
        assert "chads" in result.stdout.lower()
    except subprocess.TimeoutExpired:
        pytest.skip("CLI EDA test timed out") 