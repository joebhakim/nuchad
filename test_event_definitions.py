#!/usr/bin/env python3
"""
Simple test script to find the correct event definition for the new dataset.

The key insight: 'age' should correlate with stroke events (older patients have more strokes).
This works in the old dataset but not the new dataset with our current definitions.

Let's try different combinations of stroke_1Y values and stroke dates to find the right definition.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from nuchad.utils import get_df, calculate_chadsvasc


def test_event_definition(df, event_definition_name, event_mask, dataset_name):
    """Test a specific event definition and return correlation results."""
    
    # Apply the event definition
    df_test = df.copy()
    df_test['event'] = event_mask
    
    # Calculate CHADS-VASc scores
    df_test['chadsvasc'] = df_test.apply(calculate_chadsvasc, axis=1)
    df_test['risk_group'] = pd.cut(
        df_test['chadsvasc'], 
        bins=[-0.5, 1.5, 3.5, 10], 
        labels=['Low (0-1)', 'Moderate (2-3)', 'High (4+)']
    )
    
    # Calculate correlations with age
    age_corr = None
    age_p = None
    if 'age' in df_test.columns:
        valid_age = df_test[df_test['age'].notna()]
        if len(valid_age) > 0:
            age_corr, age_p = pearsonr(valid_age['event'].astype(int), valid_age['age'])
    
    # Calculate event rates by risk group
    risk_rates = {}
    for group in ['Low (0-1)', 'Moderate (2-3)', 'High (4+)']:
        group_data = df_test[df_test['risk_group'] == group]
        if len(group_data) > 0:
            rate = group_data['event'].mean() * 100
            risk_rates[group] = rate
    
    # Calculate gradient (should increase with risk)
    rates_list = [risk_rates.get(g, 0) for g in ['Low (0-1)', 'Moderate (2-3)', 'High (4+)']]
    gradient = max(rates_list) - min(rates_list)
    
    print(f"\n{dataset_name} - {event_definition_name}:")
    print(f"  Total events: {event_mask.sum():,} ({event_mask.mean()*100:.1f}%)")
    print(f"  Age correlation: r={age_corr:.3f}, p={age_p:.4f}" if age_corr is not None else "  Age correlation: N/A")
    print(f"  Risk group rates: Low={rates_list[0]:.1f}%, Mod={rates_list[1]:.1f}%, High={rates_list[2]:.1f}%")
    print(f"  Rate gradient: {gradient:.1f} percentage points")
    print(f"  Good stratification: {'✓' if gradient > 2.0 else '✗'}")
    
    return {
        'name': event_definition_name,
        'events': int(event_mask.sum()),
        'event_rate': event_mask.mean(),
        'age_correlation': age_corr,
        'age_p_value': age_p,
        'risk_rates': risk_rates,
        'gradient': gradient,
        'good_stratification': gradient > 2.0
    }


def test_all_event_definitions():
    """Test various event definitions on both datasets."""
    
    print("="*80)
    print("TESTING DIFFERENT EVENT DEFINITIONS")
    print("="*80)
    
    # Load datasets
    old_df = get_df("random_nuchad.csv")
    new_df = get_df("random_nuchad_250623.csv")
    
    print(f"Old dataset: {len(old_df):,} patients")
    print(f"New dataset: {len(new_df):,} patients")
    
    results = {}
    
    # Test on OLD dataset first (as baseline)
    print("\n" + "="*50)
    print("OLD DATASET (BASELINE)")
    print("="*50)
    
    results['OLD'] = []
    
    # Definition 1: Just stroke_1Y = 1 (our current approach)
    mask1 = (old_df['stroke_1Y'] == 1)
    results['OLD'].append(test_event_definition(old_df, "stroke_1Y = 1", mask1, "OLD"))
    
    # Definition 2: Has stroke date (regardless of stroke_1Y)
    mask2 = old_df['earliest_stroke_date'].notna()
    results['OLD'].append(test_event_definition(old_df, "Has stroke date", mask2, "OLD"))
    
    # Definition 3: stroke_1Y = 1 AND has stroke date
    mask3 = (old_df['stroke_1Y'] == 1) & old_df['earliest_stroke_date'].notna()
    results['OLD'].append(test_event_definition(old_df, "stroke_1Y=1 AND has date", mask3, "OLD"))
    
    # Definition 4: stroke_1Y = 1 AND has stroke date AND after time1
    time1_mask = old_df['earliest_stroke_date'] > old_df['time1']
    mask4 = (old_df['stroke_1Y'] == 1) & old_df['earliest_stroke_date'].notna() & time1_mask
    results['OLD'].append(test_event_definition(old_df, "stroke_1Y=1 AND date after time1", mask4, "OLD"))
    
    # Test on NEW dataset
    print("\n" + "="*50)
    print("NEW DATASET (TESTING)")
    print("="*50)
    
    results['NEW'] = []
    
    # Definition 1: Just stroke_1Y = 1 (our current approach)
    mask1 = (new_df['stroke_1Y'] == 1.0)
    results['NEW'].append(test_event_definition(new_df, "stroke_1Y = 1", mask1, "NEW"))
    
    # Definition 2: Has stroke date (regardless of stroke_1Y)
    mask2 = new_df['earliest_stroke_date'].notna()
    results['NEW'].append(test_event_definition(new_df, "Has stroke date", mask2, "NEW"))
    
    # Definition 3: stroke_1Y is not missing (any stroke type)
    mask3 = new_df['stroke_1Y'].notna()
    results['NEW'].append(test_event_definition(new_df, "Any stroke_1Y value", mask3, "NEW"))
    
    # Definition 4: stroke_1Y in [1,2,3,4] (all stroke types)
    mask4 = new_df['stroke_1Y'].isin([1.0, 2.0, 3.0, 4.0])
    results['NEW'].append(test_event_definition(new_df, "stroke_1Y in [1,2,3,4]", mask4, "NEW"))
    
    # Definition 5: Has stroke date AND after time1
    time1_mask = new_df['earliest_stroke_date'] > new_df['time1']
    mask5 = new_df['earliest_stroke_date'].notna() & time1_mask
    results['NEW'].append(test_event_definition(new_df, "Has date AND after time1", mask5, "NEW"))
    
    # Definition 6: stroke_1Y not missing AND has stroke date
    mask6 = new_df['stroke_1Y'].notna() & new_df['earliest_stroke_date'].notna()
    results['NEW'].append(test_event_definition(new_df, "Any stroke_1Y AND has date", mask6, "NEW"))
    
    # Definition 7: Just has stroke date after time1 and within 1 year
    within_1y = (new_df['earliest_stroke_date'] - new_df['time1']).dt.days <= 365
    mask7 = new_df['earliest_stroke_date'].notna() & time1_mask & within_1y
    results['NEW'].append(test_event_definition(new_df, "Date after time1 within 1Y", mask7, "NEW"))
    
    return results


def find_best_definition(results):
    """Analyze results to find the best event definition."""
    
    print("\n" + "="*80)
    print("ANALYSIS: FINDING BEST EVENT DEFINITION")
    print("="*80)
    
    print("\nCriteria for good event definition:")
    print("1. Positive correlation with age (older patients more events)")
    print("2. Clear risk stratification (Low < Moderate < High rates)")
    print("3. Reasonable event rates (not too high/low)")
    print("4. Statistical significance (p < 0.05)")
    
    print(f"\n{'Dataset':<8} {'Definition':<25} {'Age r':<8} {'Age p':<8} {'Gradient':<10} {'Good?':<6}")
    print("-" * 80)
    
    best_old = None
    best_new = None
    
    for dataset in ['OLD', 'NEW']:
        for result in results[dataset]:
            age_r = f"{result['age_correlation']:.3f}" if result['age_correlation'] is not None else "N/A"
            age_p = f"{result['age_p_value']:.4f}" if result['age_p_value'] is not None else "N/A"
            gradient = f"{result['gradient']:.1f}%"
            good = "✓" if result['good_stratification'] and result['age_correlation'] and result['age_correlation'] > 0 else "✗"
            
            print(f"{dataset:<8} {result['name']:<25} {age_r:<8} {age_p:<8} {gradient:<10} {good:<6}")
            
            # Track best options
            if dataset == 'OLD' and result['good_stratification'] and result['age_correlation'] and result['age_correlation'] > 0:
                if best_old is None or result['age_correlation'] > best_old['age_correlation']:
                    best_old = result
                    
            if dataset == 'NEW' and result['good_stratification'] and result['age_correlation'] and result['age_correlation'] > 0:
                if best_new is None or result['age_correlation'] > best_new['age_correlation']:
                    best_new = result
    
    print(f"\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if best_old:
        print(f"OLD dataset best definition: {best_old['name']}")
        print(f"  Age correlation: r={best_old['age_correlation']:.3f}")
        print(f"  Risk gradient: {best_old['gradient']:.1f}%")
    
    if best_new:
        print(f"NEW dataset best definition: {best_new['name']}")
        print(f"  Age correlation: r={best_new['age_correlation']:.3f}")
        print(f"  Risk gradient: {best_new['gradient']:.1f}%")
    else:
        print("NEW dataset: No definition shows good stratification!")
        print("Consider alternative approaches:")
        print("- Different time windows")
        print("- Different stroke_1Y value combinations")
        print("- Alternative outcome definitions")


def main():
    """Run the analysis."""
    print("Testing Event Definitions for Proper Risk Stratification")
    print("="*80)
    print("Goal: Find event definition where age correlates positively with events")
    print("and CHADS-VASc risk groups show proper gradient")
    
    # Test all definitions
    results = test_all_event_definitions()
    
    # Find best definition
    find_best_definition(results)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()