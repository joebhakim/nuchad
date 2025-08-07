#!/usr/bin/env python3
"""
Create comprehensive correlation table summarizing all experiments.

This script consolidates results from all our correlation experiments:
1. Basic event definitions (test_event_definitions.py)
2. Date-based definitions (test_date_based_events.py)  
3. stroke_1Y combinations (test_stroke_1y_combinations.py)
4. Treatment-stratified analysis (test_treatment_stratified.py)
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path

from nuchad.utils import get_df, calculate_chadsvasc


def calculate_basic_correlations():
    """Calculate correlations for basic event definitions."""
    print("Calculating basic event definition correlations...")
    
    old_df = get_df("random_nuchad.csv")
    new_df = get_df("random_nuchad_250623.csv")
    
    results = []
    
    for dataset_name, df in [('OLD', old_df), ('NEW', new_df)]:
        # Add CHADS-VASc
        df['chadsvasc'] = df.apply(calculate_chadsvasc, axis=1)
        df['risk_group'] = pd.cut(
            df['chadsvasc'], 
            bins=[-0.5, 1.5, 3.5, 10], 
            labels=['Low (0-1)', 'Moderate (2-3)', 'High (4+)']
        )
        
        # Define events based on dataset
        if dataset_name == 'OLD':
            events = {
                'stroke_1Y = 1': df['stroke_1Y'] == 1,
                'stroke_1Y = 2': df['stroke_1Y'] == 2,
                'Has stroke date': df['earliest_stroke_date'].notna(),
                'stroke_1Y=1 AND has date': (df['stroke_1Y'] == 1) & df['earliest_stroke_date'].notna()
            }
        else:  # NEW
            events = {
                'stroke_1Y = 1': df['stroke_1Y'] == 1.0,
                'stroke_1Y = 2': df['stroke_1Y'] == 2.0,
                'stroke_1Y = 3': df['stroke_1Y'] == 3.0,
                'stroke_1Y = 4': df['stroke_1Y'] == 4.0,
                'stroke_1Y Missing': df['stroke_1Y'].isna(),
                'Has stroke date': df['earliest_stroke_date'].notna(),
                'Any stroke_1Y value': df['stroke_1Y'].notna(),
                'stroke_1Y=1 AND has date': (df['stroke_1Y'] == 1.0) & df['earliest_stroke_date'].notna()
            }
        
        for event_name, event_mask in events.items():
            age_corr, age_p, chads_corr, chads_p, gradient = calculate_correlations_for_mask(
                df, event_mask, event_name
            )
            
            results.append({
                'Analysis Type': 'Basic Event Definitions',
                'Dataset': dataset_name,
                'Event Definition': event_name,
                'Treatment Stratum': 'All patients',
                'N Events': int(event_mask.sum()),
                'N Total': len(df),
                'Event Rate (%)': event_mask.mean() * 100,
                'Age Correlation (r)': age_corr,
                'Age p-value': age_p,
                'Age Significance': get_significance_marker(age_p),
                'CHADS-VASc Correlation (r)': chads_corr,
                'CHADS-VASc p-value': chads_p,
                'CHADS-VASc Significance': get_significance_marker(chads_p),
                'Risk Gradient (%)': gradient,
                'Good Correlation': age_corr is not None and not np.isnan(age_corr) and age_corr > 0.02 and age_p < 0.05
            })
    
    return results


def calculate_date_based_correlations():
    """Calculate correlations for date-based event definitions."""
    print("Calculating date-based event definition correlations...")
    
    old_df = get_df("random_nuchad.csv")
    new_df = get_df("random_nuchad_250623.csv")
    
    results = []
    
    for dataset_name, df in [('OLD', old_df), ('NEW', new_df)]:
        # Add CHADS-VASc
        df['chadsvasc'] = df.apply(calculate_chadsvasc, axis=1)
        df['risk_group'] = pd.cut(
            df['chadsvasc'], 
            bins=[-0.5, 1.5, 3.5, 10], 
            labels=['Low (0-1)', 'Moderate (2-3)', 'High (4+)']
        )
        
        # Calculate timing for date-based definitions
        has_both_dates = df['earliest_af_date'].notna() & df['earliest_stroke_date'].notna()
        df['days_af_to_stroke'] = (df['earliest_stroke_date'] - df['earliest_af_date']).dt.days
        
        # Date-based event definitions
        date_events = {
            'Stroke 0-365 days after AF': has_both_dates & (df['days_af_to_stroke'] >= 0) & (df['days_af_to_stroke'] <= 365),
            'Stroke 1-365 days after AF': has_both_dates & (df['days_af_to_stroke'] > 0) & (df['days_af_to_stroke'] <= 365),
            'Stroke same day as AF': has_both_dates & (df['days_af_to_stroke'] == 0),
            'Any stroke after AF': has_both_dates & (df['days_af_to_stroke'] >= 0),
            'Stroke before AF': has_both_dates & (df['days_af_to_stroke'] < 0)
        }
        
        for event_name, event_mask in date_events.items():
            if event_mask.sum() > 0:  # Only process if there are events
                age_corr, age_p, chads_corr, chads_p, gradient = calculate_correlations_for_mask(
                    df, event_mask, event_name
                )
                
                results.append({
                    'Analysis Type': 'Date-Based Definitions',
                    'Dataset': dataset_name,
                    'Event Definition': event_name,
                    'Treatment Stratum': 'All patients',
                    'N Events': int(event_mask.sum()),
                    'N Total': len(df),
                    'Event Rate (%)': event_mask.mean() * 100,
                    'Age Correlation (r)': age_corr,
                    'Age p-value': age_p,
                    'Age Significance': get_significance_marker(age_p),
                    'CHADS-VASc Correlation (r)': chads_corr,
                    'CHADS-VASc p-value': chads_p,
                    'CHADS-VASc Significance': get_significance_marker(chads_p),
                    'Risk Gradient (%)': gradient,
                    'Good Correlation': age_corr is not None and not np.isnan(age_corr) and age_corr > 0.02 and age_p < 0.05
                })
    
    return results


def calculate_treatment_stratified_correlations():
    """Calculate correlations within treatment strata."""
    print("Calculating treatment-stratified correlations...")
    
    old_df = get_df("random_nuchad.csv")
    new_df = get_df("random_nuchad_250623.csv")
    
    results = []
    
    # OLD dataset treatment classification (simple)
    old_df['chadsvasc'] = old_df.apply(calculate_chadsvasc, axis=1)
    old_df['risk_group'] = pd.cut(
        old_df['chadsvasc'], 
        bins=[-0.5, 1.5, 3.5, 10], 
        labels=['Low (0-1)', 'Moderate (2-3)', 'High (4+)']
    )
    old_df['has_anticoag'] = old_df['Anticoagulant'].notna() & (old_df['Anticoagulant'] != 0)
    
    # Process OLD dataset (mostly homogeneous treatment)
    for treatment_status, treatment_mask in [('On OAC', old_df['has_anticoag']), ('No OAC', ~old_df['has_anticoag'])]:
        treatment_group = old_df[treatment_mask]
        if len(treatment_group) > 100:  # Only if reasonable sample size
            event_mask = treatment_group['stroke_1Y'] == 1
            if event_mask.sum() > 0:
                age_corr, age_p, chads_corr, chads_p, gradient = calculate_correlations_for_mask(
                    treatment_group, event_mask, 'stroke_1Y = 1'
                )
                
                results.append({
                    'Analysis Type': 'Treatment-Stratified',
                    'Dataset': 'OLD',
                    'Event Definition': 'stroke_1Y = 1',
                    'Treatment Stratum': treatment_status,
                    'N Events': int(event_mask.sum()),
                    'N Total': len(treatment_group),
                    'Event Rate (%)': event_mask.mean() * 100,
                    'Age Correlation (r)': age_corr,
                    'Age p-value': age_p,
                    'Age Significance': get_significance_marker(age_p),
                    'CHADS-VASc Correlation (r)': chads_corr,
                    'CHADS-VASc p-value': chads_p,
                    'CHADS-VASc Significance': get_significance_marker(chads_p),
                    'Risk Gradient (%)': gradient,
                    'Good Correlation': age_corr is not None and not np.isnan(age_corr) and age_corr > 0.02 and age_p < 0.05
                })
    
    # NEW dataset treatment classification (complex)
    new_df['chadsvasc'] = new_df.apply(calculate_chadsvasc, axis=1)
    new_df['risk_group'] = pd.cut(
        new_df['chadsvasc'], 
        bins=[-0.5, 1.5, 3.5, 10], 
        labels=['Low (0-1)', 'Moderate (2-3)', 'High (4+)']
    )
    
    # Create treatment categories
    new_df['has_oac'] = (
        new_df['first_OAC_date'].notna() &
        (new_df['first_OAC_date'] >= new_df['time1']) &
        (new_df['first_OAC_date'] <= new_df['end_fu'])
    )
    new_df['has_antiplatelet'] = (
        new_df['first_antiplatelet_date'].notna() &
        (new_df['first_antiplatelet_date'] >= new_df['time1']) &
        (new_df['first_antiplatelet_date'] <= new_df['end_fu'])
    )
    
    # Define treatment strata
    treatment_strata = {
        'No treatment': ~new_df['has_oac'] & ~new_df['has_antiplatelet'],
        'OAC only': new_df['has_oac'] & ~new_df['has_antiplatelet'],
        'Antiplatelet only': ~new_df['has_oac'] & new_df['has_antiplatelet'],
        'Both treatments': new_df['has_oac'] & new_df['has_antiplatelet']
    }
    
    # Process NEW dataset by treatment stratum
    for stratum_name, stratum_mask in treatment_strata.items():
        treatment_group = new_df[stratum_mask]
        if len(treatment_group) > 100:  # Only if reasonable sample size
            # Test main event definition
            event_mask = treatment_group['stroke_1Y'] == 1.0
            if event_mask.sum() > 0:
                age_corr, age_p, chads_corr, chads_p, gradient = calculate_correlations_for_mask(
                    treatment_group, event_mask, 'stroke_1Y = 1'
                )
                
                results.append({
                    'Analysis Type': 'Treatment-Stratified',
                    'Dataset': 'NEW',
                    'Event Definition': 'stroke_1Y = 1',
                    'Treatment Stratum': stratum_name,
                    'N Events': int(event_mask.sum()),
                    'N Total': len(treatment_group),
                    'Event Rate (%)': event_mask.mean() * 100,
                    'Age Correlation (r)': age_corr,
                    'Age p-value': age_p,
                    'Age Significance': get_significance_marker(age_p),
                    'CHADS-VASc Correlation (r)': chads_corr,
                    'CHADS-VASc p-value': chads_p,
                    'CHADS-VASc Significance': get_significance_marker(chads_p),
                    'Risk Gradient (%)': gradient,
                    'Good Correlation': age_corr is not None and not np.isnan(age_corr) and age_corr > 0.02 and age_p < 0.05
                })
    
    return results


def calculate_correlations_for_mask(df, event_mask, event_name):
    """Helper function to calculate correlations for a given event mask."""
    # Age correlation
    age_valid = df[df['age'].notna()]
    event_valid = event_mask[age_valid.index]
    
    if len(age_valid) > 10 and event_valid.sum() > 0 and event_valid.sum() < len(event_valid):
        try:
            age_corr, age_p = pearsonr(event_valid.astype(int), age_valid['age'])
        except:
            age_corr, age_p = np.nan, np.nan
    else:
        age_corr, age_p = np.nan, np.nan
    
    # CHADS-VASc correlation
    chads_valid = df[df['chadsvasc'].notna()]
    event_valid_chads = event_mask[chads_valid.index]
    
    if len(chads_valid) > 10 and event_valid_chads.sum() > 0 and event_valid_chads.sum() < len(event_valid_chads):
        try:
            chads_corr, chads_p = pearsonr(event_valid_chads.astype(int), chads_valid['chadsvasc'])
        except:
            chads_corr, chads_p = np.nan, np.nan
    else:
        chads_corr, chads_p = np.nan, np.nan
    
    # Risk group gradient
    risk_rates = []
    for group in ['Low (0-1)', 'Moderate (2-3)', 'High (4+)']:
        group_data = df[df['risk_group'] == group]
        if len(group_data) > 0:
            rate = event_mask[group_data.index].mean() * 100
            risk_rates.append(rate)
        else:
            risk_rates.append(0)
    
    gradient = max(risk_rates) - min(risk_rates)
    
    return age_corr, age_p, chads_corr, chads_p, gradient


def get_significance_marker(p_value):
    """Convert p-value to significance marker."""
    if pd.isna(p_value):
        return 'N/A'
    elif p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'


def create_comprehensive_table():
    """Create the comprehensive correlation table."""
    print("="*80)
    print("CREATING COMPREHENSIVE CORRELATION TABLE")
    print("="*80)
    
    # Collect results from all experiments
    all_results = []
    
    # Basic event definitions
    all_results.extend(calculate_basic_correlations())
    
    # Date-based definitions
    all_results.extend(calculate_date_based_correlations())
    
    # Treatment-stratified
    all_results.extend(calculate_treatment_stratified_correlations())
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Round numeric columns for display
    numeric_cols = ['Event Rate (%)', 'Age Correlation (r)', 'Age p-value', 
                   'CHADS-VASc Correlation (r)', 'CHADS-VASc p-value', 'Risk Gradient (%)']
    
    for col in numeric_cols:
        if col in results_df.columns:
            results_df[col] = results_df[col].round(4)
    
    # Sort for better organization
    results_df = results_df.sort_values(['Analysis Type', 'Dataset', 'Treatment Stratum', 'Event Definition'])
    
    # Create summary statistics
    summary_stats = create_summary_statistics(results_df)
    
    return results_df, summary_stats


def create_summary_statistics(results_df):
    """Create summary statistics from the comprehensive table."""
    summary = {}
    
    # Count significant correlations by dataset
    for dataset in ['OLD', 'NEW']:
        dataset_results = results_df[results_df['Dataset'] == dataset]
        
        total_tests = len(dataset_results)
        significant_age = (dataset_results['Age Significance'].isin(['*', '**', '***'])).sum()
        significant_chads = (dataset_results['CHADS-VASc Significance'].isin(['*', '**', '***'])).sum()
        good_correlations = dataset_results['Good Correlation'].sum()
        
        # Age correlation range
        age_corrs = dataset_results['Age Correlation (r)'].dropna()
        age_min = age_corrs.min() if len(age_corrs) > 0 else np.nan
        age_max = age_corrs.max() if len(age_corrs) > 0 else np.nan
        
        # Risk gradient range
        gradients = dataset_results['Risk Gradient (%)'].dropna()
        grad_min = gradients.min() if len(gradients) > 0 else np.nan
        grad_max = gradients.max() if len(gradients) > 0 else np.nan
        
        summary[dataset] = {
            'Total Tests': total_tests,
            'Significant Age Correlations': significant_age,
            'Significant CHADS-VASc Correlations': significant_chads,
            'Good Correlations (r>0.02, p<0.05)': good_correlations,
            'Age Correlation Range': f"{age_min:.3f} to {age_max:.3f}" if not pd.isna(age_min) else "N/A",
            'Risk Gradient Range (%)': f"{grad_min:.1f} to {grad_max:.1f}" if not pd.isna(grad_min) else "N/A"
        }
    
    return summary


def main():
    """Create and save comprehensive correlation table."""
    print("Creating Comprehensive Correlation Summary Table")
    print("="*80)
    
    # Create table
    results_df, summary_stats = create_comprehensive_table()
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    # Save comprehensive table
    comprehensive_path = output_dir / 'comprehensive_correlation_table.csv'
    results_df.to_csv(comprehensive_path, index=False)
    
    # Save summary statistics as separate CSV
    summary_df = pd.DataFrame(summary_stats).T
    summary_path = output_dir / 'correlation_summary_statistics.csv'
    summary_df.to_csv(summary_path)
    
    # Print results summary
    print(f"\n" + "="*60)
    print("COMPREHENSIVE CORRELATION TABLE COMPLETE")
    print("="*60)
    print(f"Saved to: {comprehensive_path}")
    print(f"Summary stats: {summary_path}")
    
    print(f"\nTOTAL RESULTS:")
    print(f"- {len(results_df)} correlation tests performed")
    print(f"- {len(results_df['Event Definition'].unique())} unique event definitions tested")
    print(f"- {len(results_df['Analysis Type'].unique())} analysis approaches used")
    
    print(f"\nSUMMARY BY DATASET:")
    for dataset, stats in summary_stats.items():
        print(f"\n{dataset} Dataset:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # Key findings
    old_good = summary_stats['OLD']['Good Correlations (r>0.02, p<0.05)']
    new_good = summary_stats['NEW']['Good Correlations (r>0.02, p<0.05)']
    
    print(f"\n" + "🚨" * 20)
    print("KEY FINDING:")
    print(f"OLD dataset: {old_good}/{summary_stats['OLD']['Total Tests']} tests show good correlations")
    print(f"NEW dataset: {new_good}/{summary_stats['NEW']['Total Tests']} tests show good correlations")
    
    if new_good == 0:
        print("NEW dataset shows ZERO good correlations across ALL experiments!")
    
    print("🚨" * 20)
    
    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()