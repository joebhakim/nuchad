#!/usr/bin/env python3
"""
Test the new survival classification system (v2) on both datasets.

This script tests the improved event/control/censoring classification functions
and compares the results with the original approach to validate that we get
proper CHADS-VASc risk stratification in both datasets.
"""

import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from nuchad.utils import get_df
from nuchad.analysis.survival_v2 import (
    classify_events_and_controls, 
    classify_censoring,
    prepare_survival_data_v3,
    print_classification_summary
)


def test_classification_functions():
    """Test the individual classification functions."""
    print("="*70)
    print("TESTING CLASSIFICATION FUNCTIONS")
    print("="*70)
    
    # Load both datasets
    print("Loading datasets...")
    old_df = get_df("random_nuchad.csv")
    new_df = get_df("random_nuchad_250623.csv")
    
    print(f"Old dataset: {len(old_df):,} patients")
    print(f"New dataset: {len(new_df):,} patients")
    
    # Test classify_events_and_controls on both datasets
    print("\n" + "-"*50)
    print("TESTING EVENT/CONTROL CLASSIFICATION")
    print("-"*50)
    
    for dataset_name, df in [("OLD DATASET", old_df), ("NEW DATASET", new_df)]:
        print(f"\n{dataset_name}:")
        
        # Test event/control classification
        df_classified, events_report = classify_events_and_controls(df, diagnostic_correlate='age')
        
        print(f"  Dataset type detected: {events_report['dataset_type']}")
        print(f"  Events: {events_report['events']:,} ({events_report['event_rate_overall']*100:.1f}%)")
        print(f"  Controls: {events_report['controls']:,} ({events_report['control_rate_overall']*100:.1f}%)")
        print(f"  Mutual exclusivity violations: {events_report['mutual_exclusivity_violations']}")
        print(f"  Unclassified: {events_report['unclassified_patients']:,}")
        
        # Show risk group event rates
        print(f"  Event rates by risk group:")
        for group, stats in events_report['risk_group_statistics'].items():
            event_rate = stats['event_rate'] * 100
            print(f"    {group}: {event_rate:.1f}%")
        
        # Test censoring classification
        df_with_censoring, censoring_report = classify_censoring(df_classified)
        
        print(f"  Censored: {censoring_report['censored_patients']:,} ({censoring_report['censoring_rate_overall']*100:.1f}%)")
        print(f"  Three-way classification valid: {'✓' if censoring_report['three_way_classification_valid'] else '✗'}")


def test_survival_curves_v2():
    """Test that the new classification produces proper survival curves."""
    print("\n" + "="*70)
    print("TESTING SURVIVAL CURVE SEPARATION WITH NEW CLASSIFICATION")
    print("="*70)
    
    for dataset_name, data_file in [("OLD DATASET", "random_nuchad.csv"), 
                                   ("NEW DATASET", "random_nuchad_250623.csv")]:
        
        print(f"\n{dataset_name} Analysis:")
        print("-" * 40)
        
        # Load and process with new method
        df = get_df(data_file)
        survival_df, comprehensive_report = prepare_survival_data_v3(df, diagnostic_correlate='age')
        
        # Print classification summary
        print_classification_summary(comprehensive_report, f"{dataset_name} Classification")
        
        # Calculate 10-year survival by risk group
        print(f"\n{dataset_name} - 10-Year Survival Analysis (New Method):")
        print("-" * 60)
        
        # Filter to analyzable patients (events + controls)
        analyzable = survival_df[survival_df['event'] | survival_df['control']].copy()
        
        print(f"Analyzable patients: {len(analyzable):,}")
        
        survival_results = {}
        
        for group in ['Low (0-1)', 'Moderate (2-3)', 'High (4+)']:
            group_data = analyzable[analyzable['risk_group'] == group]
            
            if len(group_data) == 0:
                print(f"{group}: No patients")
                continue
                
            # Fit Kaplan-Meier estimator using the event column
            kmf = KaplanMeierFitter()
            kmf.fit(group_data['survival_time'], group_data['event'], label=group)
            
            # Extract 10-year survival
            target_days = 3650  # 10 years
            timeline = kmf.timeline
            
            if target_days in timeline:
                survival_10y = kmf.survival_function_.loc[target_days].iloc[0]
            else:
                valid_times = timeline[timeline <= target_days]
                if len(valid_times) > 0:
                    last_time = valid_times.max()
                    survival_10y = kmf.survival_function_.loc[last_time].iloc[0]
                else:
                    survival_10y = 1.0
            
            events = group_data['event'].sum()
            event_rate = events / len(group_data) * 100
            
            print(f"{group}:")
            print(f"  Patients: {len(group_data):,}")
            print(f"  Events: {events:,} ({event_rate:.1f}%)")
            print(f"  10-year survival: {survival_10y:.3f}")
            
            survival_results[group] = survival_10y
        
        # Calculate curve separation
        if len(survival_results) > 1:
            values = list(survival_results.values())
            separation = max(values) - min(values)
            print(f"\nCurve separation (range): {separation:.3f}")
            
            # Expected gradient: Low > Moderate > High survival
            if 'Low (0-1)' in survival_results and 'High (4+)' in survival_results:
                gradient = survival_results['Low (0-1)'] - survival_results['High (4+)']
                print(f"Risk gradient (Low - High): {gradient:.3f}")


def compare_old_vs_new_methods():
    """Compare results from old vs new classification methods."""
    print("\n" + "="*70)
    print("COMPARING OLD VS NEW CLASSIFICATION METHODS")
    print("="*70)
    
    from nuchad.analysis.eda import prepare_survival_data
    
    for dataset_name, data_file in [("OLD DATASET", "random_nuchad.csv"), 
                                   ("NEW DATASET", "random_nuchad_250623.csv")]:
        
        print(f"\n{dataset_name} Comparison:")
        print("-" * 40)
        
        df = get_df(data_file)
        
        # Old method
        old_survival = prepare_survival_data(df)
        old_survival['chadsvasc'] = old_survival.apply(lambda row: sum([
            row.get('hf', 0), row.get('hypertension', 0), 
            2 if row.get('age', 0) >= 75 else 1 if row.get('age', 0) >= 65 else 0,
            row.get('diab', 0), 
            2 if (row.get('thrombo', 0) or row.get('HB_stroke_history', 0)) else 0,
            row.get('vasc_dis_mi_pad', 0),
            1 if row.get('gender', 1) != 1 else 0
        ]), axis=1)
        old_survival['risk_group'] = pd.cut(
            old_survival['chadsvasc'], 
            bins=[-0.5, 1.5, 3.5, 10], 
            labels=['Low (0-1)', 'Moderate (2-3)', 'High (4+)']
        )
        
        # New method
        new_survival, _ = prepare_survival_data_v3(df)
        analyzable = new_survival[new_survival['event'] | new_survival['control']]
        
        print(f"Old method: {len(old_survival):,} patients, {old_survival['event'].sum():,} events")
        print(f"New method: {len(analyzable):,} patients, {analyzable['event'].sum():,} events")
        
        # Event rates by risk group
        print(f"\nEvent rates by risk group:")
        print(f"{'Risk Group':<15} {'Old Method':<12} {'New Method':<12} {'Difference':<10}")
        print("-" * 50)
        
        for group in ['Low (0-1)', 'Moderate (2-3)', 'High (4+)']:
            old_group = old_survival[old_survival['risk_group'] == group]
            new_group = analyzable[analyzable['risk_group'] == group]
            
            old_rate = old_group['event'].mean() * 100 if len(old_group) > 0 else 0
            new_rate = new_group['event'].mean() * 100 if len(new_group) > 0 else 0
            diff = abs(old_rate - new_rate)
            
            print(f"{group:<15} {old_rate:>9.1f}%   {new_rate:>9.1f}%   {diff:>7.1f}%")


def main():
    """Run all tests."""
    print("Testing New Survival Classification System (v2)")
    print("="*70)
    
    # Test individual functions
    test_classification_functions()
    
    # Test survival curve generation
    test_survival_curves_v2()
    
    # Compare methods
    compare_old_vs_new_methods()
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()