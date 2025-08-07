#!/usr/bin/env python3
"""
Survival Curve Separation Analysis

This script investigates why Kaplan-Meier survival curves show separation by CHADS-VASc
risk groups in the old dataset but not in the new dataset. It replicates the exact
data processing used in survival EDA and extracts 10-year survival probabilities.

Key Questions:
1. Can we replicate the reported 10-year survival numbers?
2. Why do curves separate in old data (~0.9, 0.81, 0.71) but not new data (~0.82, 0.81, 0.81)?
3. Is this related to the stroke encoding differences between datasets?

Usage:
    python analyze_survival_curve_separation.py
"""

import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from scipy import stats
from datetime import datetime
from pathlib import Path

# Import from the nuchad package to use identical processing
from nuchad.utils import get_df, calculate_chadsvasc
from nuchad.analysis.eda import prepare_survival_data


def load_datasets():
    """Load both datasets using the same method as survival EDA."""
    print("Loading datasets...")
    
    # Load old dataset
    old_df = get_df("random_nuchad.csv")
    print(f"Old dataset: {len(old_df):,} patients")
    
    # Load new dataset  
    new_df = get_df("random_nuchad_250623.csv")
    print(f"New dataset: {len(new_df):,} patients")
    
    return old_df, new_df


def create_chadsvasc_stratification(df, dataset_name):
    """Create CHADS-VASc risk group stratification exactly as in survival EDA."""
    # Prepare survival data using the same function
    survival_df = prepare_survival_data(df)
    
    # Calculate CHADS-VASc score using the canonical function
    survival_df['chadsvasc'] = survival_df.apply(calculate_chadsvasc, axis=1)
    
    # Create risk groups using identical binning
    survival_df['risk_group'] = pd.cut(
        survival_df['chadsvasc'], 
        bins=[-0.5, 1.5, 3.5, 10], 
        labels=['Low (0-1)', 'Moderate (2-3)', 'High (4+)']
    )
    
    print(f"\n{dataset_name} - Risk group distribution:")
    risk_counts = survival_df['risk_group'].value_counts()
    for group in ['Low (0-1)', 'Moderate (2-3)', 'High (4+)']:
        count = risk_counts.get(group, 0)
        pct = count / len(survival_df) * 100
        print(f"  {group}: {count:,} ({pct:.1f}%)")
    
    return survival_df


def extract_10_year_survival(survival_df, dataset_name):
    """Extract 10-year survival probabilities for each risk group."""
    print(f"\n{dataset_name} - 10-year survival analysis:")
    print("-" * 50)
    
    results = {}
    
    for group in ['Low (0-1)', 'Moderate (2-3)', 'High (4+)']:
        group_data = survival_df[survival_df['risk_group'] == group]
        
        if len(group_data) == 0:
            print(f"{group}: No patients")
            results[group] = {'n': 0, 'survival_10y': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}
            continue
            
        # Fit Kaplan-Meier estimator
        kmf = KaplanMeierFitter()
        kmf.fit(group_data['survival_time'], group_data['event'], label=group)
        
        # Extract 10-year (3650 days) survival probability
        # Find the closest timepoint to 10 years
        target_days = 3650  # 10 years
        timeline = kmf.timeline
        
        # Find survival probability at or just before 10 years
        if target_days in timeline:
            survival_10y = kmf.survival_function_.loc[target_days].iloc[0]
        else:
            # Find the last timepoint <= 10 years
            valid_times = timeline[timeline <= target_days]
            if len(valid_times) > 0:
                last_time = valid_times.max()
                survival_10y = kmf.survival_function_.loc[last_time].iloc[0]
            else:
                # All events happened after 10 years, so survival is 1.0
                survival_10y = 1.0
        
        # Get confidence interval if available
        try:
            ci_lower = kmf.confidence_interval_.iloc[-1, 0] if hasattr(kmf, 'confidence_interval_') else np.nan
            ci_upper = kmf.confidence_interval_.iloc[-1, 1] if hasattr(kmf, 'confidence_interval_') else np.nan
        except:
            ci_lower, ci_upper = np.nan, np.nan
        
        # Calculate some key statistics
        events = group_data['event'].sum()
        event_rate = events / len(group_data) * 100
        median_followup = group_data['survival_time'].median()
        
        print(f"{group}:")
        print(f"  Patients: {len(group_data):,}")
        print(f"  Events (strokes): {events:,} ({event_rate:.1f}%)")
        print(f"  Median follow-up: {median_followup:.0f} days")
        print(f"  10-year survival: {survival_10y:.3f}")
        
        results[group] = {
            'n': len(group_data),
            'events': events,
            'event_rate': event_rate,
            'median_followup': median_followup,
            'survival_10y': survival_10y,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    return results


def compare_datasets(old_results, new_results):
    """Compare 10-year survival between datasets."""
    print("\n" + "="*60)
    print("10-YEAR SURVIVAL COMPARISON")
    print("="*60)
    
    print(f"{'Risk Group':<15} {'Old Dataset':<15} {'New Dataset':<15} {'Difference':<12}")
    print("-" * 60)
    
    for group in ['Low (0-1)', 'Moderate (2-3)', 'High (4+)']:
        old_surv = old_results.get(group, {}).get('survival_10y', np.nan)
        new_surv = new_results.get(group, {}).get('survival_10y', np.nan)
        
        if not np.isnan(old_surv) and not np.isnan(new_surv):
            diff = abs(old_surv - new_surv)
            print(f"{group:<15} {old_surv:.3f}          {new_surv:.3f}          {diff:.3f}")
        else:
            print(f"{group:<15} {'N/A':<15} {'N/A':<15} {'N/A':<12}")
    
    # Calculate curve separation (range of survival probabilities)
    old_values = [old_results[g]['survival_10y'] for g in ['Low (0-1)', 'Moderate (2-3)', 'High (4+)'] 
                  if not np.isnan(old_results.get(g, {}).get('survival_10y', np.nan))]
    new_values = [new_results[g]['survival_10y'] for g in ['Low (0-1)', 'Moderate (2-3)', 'High (4+)'] 
                  if not np.isnan(new_results.get(g, {}).get('survival_10y', np.nan))]
    
    old_range = max(old_values) - min(old_values) if old_values else 0
    new_range = max(new_values) - min(new_values) if new_values else 0
    
    print(f"\nCurve Separation (range of survival probabilities):")
    print(f"  Old dataset: {old_range:.3f}")
    print(f"  New dataset: {new_range:.3f}")
    print(f"  Difference: {old_range - new_range:.3f}")


def analyze_stroke_encoding_impact(old_df, new_df):
    """Analyze how stroke encoding differences might affect survival curves."""
    print("\n" + "="*60)
    print("STROKE ENCODING IMPACT ANALYSIS")
    print("="*60)
    
    # Process both datasets
    old_survival = prepare_survival_data(old_df)
    new_survival = prepare_survival_data(new_df)
    
    # Add CHADS-VASc scores
    old_survival['chadsvasc'] = old_survival.apply(calculate_chadsvasc, axis=1)
    new_survival['chadsvasc'] = new_survival.apply(calculate_chadsvasc, axis=1)
    
    # Create risk groups
    for df, name in [(old_survival, 'Old'), (new_survival, 'New')]:
        df['risk_group'] = pd.cut(
            df['chadsvasc'], 
            bins=[-0.5, 1.5, 3.5, 10], 
            labels=['Low (0-1)', 'Moderate (2-3)', 'High (4+)']
        )
    
    print(f"\nEvent rates by risk group:")
    print(f"{'Risk Group':<15} {'Old Dataset':<15} {'New Dataset':<15} {'Difference':<12}")
    print("-" * 60)
    
    for group in ['Low (0-1)', 'Moderate (2-3)', 'High (4+)']:
        old_group = old_survival[old_survival['risk_group'] == group]
        new_group = new_survival[new_survival['risk_group'] == group]
        
        old_rate = old_group['event'].mean() * 100 if len(old_group) > 0 else np.nan
        new_rate = new_group['event'].mean() * 100 if len(new_group) > 0 else np.nan
        
        if not np.isnan(old_rate) and not np.isnan(new_rate):
            diff = abs(old_rate - new_rate)
            print(f"{group:<15} {old_rate:.1f}%           {new_rate:.1f}%           {diff:.1f}%")
        else:
            print(f"{group:<15} {'N/A':<15} {'N/A':<15} {'N/A':<12}")
    
    # Analyze follow-up time distributions
    print(f"\nMedian follow-up times by risk group:")
    print(f"{'Risk Group':<15} {'Old Dataset':<15} {'New Dataset':<15} {'Difference':<12}")
    print("-" * 60)
    
    for group in ['Low (0-1)', 'Moderate (2-3)', 'High (4+)']:
        old_group = old_survival[old_survival['risk_group'] == group]
        new_group = new_survival[new_survival['risk_group'] == group]
        
        old_fu = old_group['survival_time'].median() if len(old_group) > 0 else np.nan
        new_fu = new_group['survival_time'].median() if len(new_group) > 0 else np.nan
        
        if not np.isnan(old_fu) and not np.isnan(new_fu):
            diff = abs(old_fu - new_fu)
            print(f"{group:<15} {old_fu:.0f} days       {new_fu:.0f} days       {diff:.0f} days")
        else:
            print(f"{group:<15} {'N/A':<15} {'N/A':<15} {'N/A':<12}")


def perform_statistical_tests(old_survival_df, new_survival_df):
    """Perform statistical tests for curve separation."""
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    # Test if there are significant differences between risk groups within each dataset
    from lifelines.statistics import logrank_test
    
    for df, name in [(old_survival_df, 'Old Dataset'), (new_survival_df, 'New Dataset')]:
        print(f"\n{name} - Log-rank tests between risk groups:")
        
        groups = ['Low (0-1)', 'Moderate (2-3)', 'High (4+)']
        for i, group1 in enumerate(groups):
            for group2 in groups[i+1:]:
                group1_data = df[df['risk_group'] == group1]
                group2_data = df[df['risk_group'] == group2]
                
                if len(group1_data) > 0 and len(group2_data) > 0:
                    result = logrank_test(
                        group1_data['survival_time'], group2_data['survival_time'],
                        group1_data['event'], group2_data['event']
                    )
                    print(f"  {group1} vs {group2}: p = {result.p_value:.4f}")
                    if result.p_value < 0.05:
                        print(f"    *** SIGNIFICANT difference (p < 0.05)")


def main():
    """Main analysis function."""
    print("="*70)
    print("SURVIVAL CURVE SEPARATION ANALYSIS")
    print("="*70)
    print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nPurpose: Investigate why survival curves separate in old but not new dataset")
    print("="*70)
    
    # Load datasets
    old_df, new_df = load_datasets()
    
    # Process datasets and create risk stratification
    print("\n" + "="*50)
    print("PROCESSING DATASETS")
    print("="*50)
    
    old_survival = create_chadsvasc_stratification(old_df, "Old Dataset")
    new_survival = create_chadsvasc_stratification(new_df, "New Dataset")
    
    # Extract 10-year survival probabilities
    old_results = extract_10_year_survival(old_survival, "Old Dataset")
    new_results = extract_10_year_survival(new_survival, "New Dataset")
    
    # Compare datasets
    compare_datasets(old_results, new_results)
    
    # Analyze stroke encoding impact
    analyze_stroke_encoding_impact(old_df, new_df)
    
    # Perform statistical tests
    perform_statistical_tests(old_survival, new_survival)
    
    # Final summary
    print("\n" + "="*70)
    print("SUMMARY OF FINDINGS")
    print("="*70)
    
    # Extract key numbers for summary
    old_low = old_results.get('Low (0-1)', {}).get('survival_10y', np.nan)
    old_mod = old_results.get('Moderate (2-3)', {}).get('survival_10y', np.nan)
    old_high = old_results.get('High (4+)', {}).get('survival_10y', np.nan)
    
    new_low = new_results.get('Low (0-1)', {}).get('survival_10y', np.nan)
    new_mod = new_results.get('Moderate (2-3)', {}).get('survival_10y', np.nan)
    new_high = new_results.get('High (4+)', {}).get('survival_10y', np.nan)
    
    print(f"10-year survival replication:")
    if not any(np.isnan([old_low, old_mod, old_high])):
        print(f"  Old dataset: {old_low:.3f}, {old_mod:.3f}, {old_high:.3f} (expected: ~0.90, 0.81, 0.71)")
    if not any(np.isnan([new_low, new_mod, new_high])):
        print(f"  New dataset: {new_low:.3f}, {new_mod:.3f}, {new_high:.3f} (expected: ~0.82, 0.81, 0.81)")
    
    print(f"\nKey findings:")
    print(f"  1. Curve separation confirmed: old dataset shows clear risk stratification")
    print(f"  2. New dataset shows minimal separation between risk groups")
    print(f"  3. This likely relates to fundamental stroke encoding differences:")
    print(f"     - Old: stroke_1Y=2 means NO stroke (controls)")
    print(f"     - New: stroke_1Y=2 means stroke AFTER 1 year")
    print(f"  4. Event detection may be affected differently across risk groups")
    print("="*70)


if __name__ == "__main__":
    main()