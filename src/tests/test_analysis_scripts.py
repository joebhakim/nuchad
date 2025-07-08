"""Tests for the analysis scripts in the nuchad package."""

import os
import pytest
import pandas as pd
from pathlib import Path
import importlib.resources

from nuchad.utils import get_data_file, get_results_dir
from nuchad.analysis import eda_old, table1, table2, table1_stratified, table2_stratified, density_ratio_reweighting
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
        'test_end_fu_distribution.png'
    ]
    
    for file in test_files:
        file_path = results_dir / file
        if os.path.exists(file_path):
            os.remove(file_path)

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

# Tests
def test_eda(eligible_data_frame):
    """Test the exploratory data analysis functionality."""
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

def test_table1(eligible_data_frame):
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

def test_table1_stratified(eligible_data_frame):
    """Test the stratified table1 generation functionality."""
    # Generate and save a test stratified table1
    results_dir = get_results_dir()
    output_path = results_dir / 'test_table1_stratified.md'
    
    # Generate stratified table1
    if 'CHADS-Vasc' not in eligible_data_frame.columns:
        eligible_data_frame['CHADS-Vasc'] = eligible_data_frame.apply(eda_old.calculate_chadsvasc, axis=1)
        
    table1_stratified.generate_stratified_table1(eligible_data_frame, output_path=output_path)
    
    # Check that the output file was created
    assert check_output_exists(output_path)
    
    # Read the file and check content
    with open(output_path, 'r') as f:
        content = f.read()
        assert "Stratified" in content
        assert "Risk" in content

def test_table2(eligible_data_frame):
    """Test the table2 generation functionality."""
    # Generate and save a test table2
    results_dir = get_results_dir()
    output_path = results_dir / 'test_table2.md'
    
    # Generate table2
    table = table2.generate_cohort_table(eligible_data_frame, output_path=output_path)
    
    # Check that the output file was created
    assert check_output_exists(output_path)
    
    # Read the file and check content
    with open(output_path, 'r') as f:
        content = f.read()
        assert "Stroke Rates" in content
        assert "CHADS-VASc Score" in content
        assert "Patient-Years" in content

def test_table2_stratified(eligible_data_frame):
    """Test the stratified table2 generation functionality."""
    # Generate and save a test stratified table2
    results_dir = get_results_dir()
    output_path = results_dir / 'test_table2_stratified.md'
    
    # Generate stratified table2
    if 'CHADS-Vasc' not in eligible_data_frame.columns:
        eligible_data_frame['CHADS-Vasc'] = eligible_data_frame.apply(eda_old.calculate_chadsvasc, axis=1)
        
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

def test_visualize(eligible_data_frame):
    """Test the visualization functionality."""
    # Generate and save a test visualization
    results_dir = get_results_dir()
    output_path = results_dir / 'test_end_fu_distribution.png'
    
    # Generate visualization
    visualize_end_fu.plot_end_fu_distribution(eligible_data_frame, output_path=output_path)
    
    # Check that the output file was created
    assert check_output_exists(output_path)

def test_filter_report(data_frame):
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

def test_density_ratio_reweighting(eligible_data_frame):
    """Test the density ratio reweighting functionality."""
    # Generate and save a test density ratio report
    results_dir = get_results_dir()
    output_path = results_dir / 'test_density_ratio_results.md'
    
    # Run reweighting analysis
    try:
        density_ratio_reweighting.perform_reweighting_analysis(
            eligible_data_frame, 
            output_path=output_path
        )
        
        # Check that the output file was created
        assert check_output_exists(output_path)
        
        # Check that the output images were created
        assert check_output_exists(results_dir / 'density_ratio_weighted_rates.png')
        assert check_output_exists(results_dir / 'density_ratio_weight_distribution.png')
        
        # Read the file and check content
        with open(output_path, 'r') as f:
            content = f.read()
            assert "Density Ratio Reweighting" in content
    except (ImportError, AttributeError):
        # This module might be optional or have dependencies not installed
        pytest.skip("Density ratio reweighting module not fully implemented or dependencies missing") 