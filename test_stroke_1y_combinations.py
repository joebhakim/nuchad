#!/usr/bin/env python3
"""
Test specific stroke_1Y value combinations based on encoding analysis.

From analyze_stroke_encoding.py:
- stroke_1Y = 1.0: Stroke within 1 year of AF diagnosis  
- stroke_1Y = 2.0: Stroke AFTER 1 year of AF diagnosis
- stroke_1Y = 3.0: Stroke same day as AF diagnosis
- stroke_1Y = 4.0: Stroke BEFORE AF diagnosis
- Missing: Controls (no stroke)

Let's test different combinations to find proper risk stratification.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from nuchad.utils import get_df, calculate_chadsvasc


def test_stroke_1y_combination(df, stroke_values, combo_name, dataset_name):
    """Test a specific combination of stroke_1Y values."""
    
    # Create event mask
    if stroke_values == 'missing':
        event_mask = df['stroke_1Y'].isna()
        combo_name = f"{combo_name} (Missing values)"
    else:
        event_mask = df['stroke_1Y'].isin(stroke_values)
        value_str = '+'.join(map(str, stroke_values))
        combo_name = f"{combo_name} (stroke_1Y={value_str})"
    
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
    age_corr, age_p = None, None
    if 'age' in df_test.columns:
        valid_age = df_test[df_test['age'].notna()]
        if len(valid_age) > 0:
            try:
                # Check if there's variation in events
                if valid_age['event'].sum() > 0 and valid_age['event'].sum() < len(valid_age):
                    age_corr, age_p = pearsonr(valid_age['event'].astype(int), valid_age['age'])
                else:
                    age_corr, age_p = np.nan, np.nan
            except:
                age_corr, age_p = np.nan, np.nan
    
    # Calculate event rates by risk group
    risk_rates = {}
    for group in ['Low (0-1)', 'Moderate (2-3)', 'High (4+)']:
        group_data = df_test[df_test['risk_group'] == group]
        if len(group_data) > 0:
            rate = group_data['event'].mean() * 100
            risk_rates[group] = rate
    
    # Calculate gradient
    rates_list = [risk_rates.get(g, 0) for g in ['Low (0-1)', 'Moderate (2-3)', 'High (4+)']]
    gradient = max(rates_list) - min(rates_list)
    
    print(f"\n{combo_name}:")
    print(f"  Events: {event_mask.sum():,} ({event_mask.mean()*100:.1f}%)")
    print(f"  Age correlation: r={age_corr:.3f}, p={age_p:.4f}" if age_corr is not None else "  Age correlation: N/A")
    print(f"  Risk rates: Low={rates_list[0]:.1f}%, Mod={rates_list[1]:.1f}%, High={rates_list[2]:.1f}%")
    print(f"  Gradient: {gradient:.1f}% {'✓' if gradient > 2.0 else '✗'}")
    print(f"  Good stratification: {'✓' if gradient > 2.0 and age_corr and age_corr > 0.02 else '✗'}")
    
    return {
        'name': combo_name,
        'stroke_values': stroke_values,
        'events': int(event_mask.sum()),
        'event_rate': event_mask.mean(),
        'age_correlation': age_corr,
        'age_p_value': age_p,
        'gradient': gradient,
        'rates': rates_list,
        'good': gradient > 2.0 and age_corr and age_corr > 0.02
    }


def analyze_stroke_1y_values(df, dataset_name):
    """Analyze the distribution of stroke_1Y values."""
    print(f"\n{dataset_name} - stroke_1Y Value Distribution:")
    print("-" * 50)
    
    stroke_counts = df['stroke_1Y'].value_counts(dropna=False).sort_index()
    total = len(df)
    
    for value, count in stroke_counts.items():
        pct = count/total*100
        if pd.isna(value):
            interpretation = "Controls (no stroke)"
            print(f"Missing:     {count:8,} ({pct:5.1f}%) - {interpretation}")
        else:
            if value == 1.0:
                interpretation = "Stroke within 1 year"
            elif value == 2.0:
                interpretation = "Stroke after 1 year"
            elif value == 3.0:
                interpretation = "Stroke same day"
            elif value == 4.0:
                interpretation = "Stroke before AF"
            else:
                interpretation = "Unknown"
            print(f"Value {value}:   {count:8,} ({pct:5.1f}%) - {interpretation}")


def test_all_combinations():
    """Test all logical combinations of stroke_1Y values."""
    print("="*80)
    print("TESTING STROKE_1Y VALUE COMBINATIONS")
    print("="*80)
    
    # Load datasets
    old_df = get_df("random_nuchad.csv")
    new_df = get_df("random_nuchad_250623.csv")
    
    print(f"Old dataset: {len(old_df):,} patients")
    print(f"New dataset: {len(new_df):,} patients")
    
    # Analyze value distributions
    analyze_stroke_1y_values(old_df, "OLD DATASET")
    analyze_stroke_1y_values(new_df, "NEW DATASET")
    
    results = {'OLD': [], 'NEW': []}
    
    # Test combinations for both datasets
    for dataset_name, df in [("OLD", old_df), ("NEW", new_df)]:
        print(f"\n" + "="*60)
        print(f"{dataset_name} DATASET - COMBINATION TESTING")
        print("="*60)
        
        # Individual values
        results[dataset_name].append(test_stroke_1y_combination(df, [1.0], "Within 1 year only", dataset_name))
        results[dataset_name].append(test_stroke_1y_combination(df, [2.0], "After 1 year only", dataset_name))
        results[dataset_name].append(test_stroke_1y_combination(df, [3.0], "Same day only", dataset_name))
        results[dataset_name].append(test_stroke_1y_combination(df, [4.0], "Before AF only", dataset_name))
        
        # Logical combinations for events
        results[dataset_name].append(test_stroke_1y_combination(df, [1.0, 3.0], "Within 1Y + Same day", dataset_name))
        results[dataset_name].append(test_stroke_1y_combination(df, [1.0, 2.0], "Within + After 1Y", dataset_name))
        results[dataset_name].append(test_stroke_1y_combination(df, [1.0, 2.0, 3.0], "Any stroke after/during AF", dataset_name))
        results[dataset_name].append(test_stroke_1y_combination(df, [1.0, 2.0, 3.0, 4.0], "Any stroke (all values)", dataset_name))
        
        # Test controls (missing values)
        results[dataset_name].append(test_stroke_1y_combination(df, 'missing', "Controls", dataset_name))
        
        # Exclusion-based: anything NOT value 4 (excluding prior strokes) - skip this complex case for now
    
    return results


def find_working_combinations(results):
    """Find combinations that work for risk stratification."""
    print("\n" + "="*80)
    print("SUMMARY: FINDING WORKING COMBINATIONS")
    print("="*80)
    
    print(f"{'Dataset':<8} {'Combination':<30} {'Events':<8} {'Age r':<8} {'Gradient':<10} {'Good?':<6}")
    print("-" * 80)
    
    best_old = None
    best_new = None
    
    for dataset in ['OLD', 'NEW']:
        for result in results[dataset]:
            age_r = f"{result['age_correlation']:.3f}" if result['age_correlation'] is not None else "N/A"
            gradient = f"{result['gradient']:.1f}%"
            good = "✓" if result['good'] else "✗"
            
            # Truncate long names
            name = result['name'].split(' (')[0] if ' (' in result['name'] else result['name']
            name = name[:28] if len(name) > 28 else name
            
            print(f"{dataset:<8} {name:<30} {result['events']:<8} {age_r:<8} {gradient:<10} {good:<6}")
            
            # Track best
            if result['good']:
                if dataset == 'OLD':
                    if best_old is None or result['age_correlation'] > best_old['age_correlation']:
                        best_old = result
                elif dataset == 'NEW':
                    if best_new is None or result['age_correlation'] > best_new['age_correlation']:
                        best_new = result
    
    print(f"\n" + "="*60)
    print("BEST COMBINATIONS IDENTIFIED")
    print("="*60)
    
    if best_old:
        print(f"OLD dataset best: {best_old['name']}")
        print(f"  Age correlation: r={best_old['age_correlation']:.3f}")
        print(f"  Risk gradient: {best_old['gradient']:.1f}%")
        print(f"  Event rate: {best_old['event_rate']*100:.1f}%")
    
    if best_new:
        print(f"NEW dataset best: {best_new['name']}")
        print(f"  Age correlation: r={best_new['age_correlation']:.3f}")
        print(f"  Risk gradient: {best_new['gradient']:.1f}%")
        print(f"  Event rate: {best_new['event_rate']*100:.1f}%")
        print("\n🎉 SUCCESS! Found working combination for new dataset!")
        
        # Show the specific stroke_1Y values that work
        if best_new['stroke_values'] != 'missing':
            print(f"  Use stroke_1Y values: {best_new['stroke_values']}")
            print("  Interpretation:")
            for val in best_new['stroke_values']:
                if val == 1.0:
                    print("    - stroke_1Y=1: Stroke within 1 year")
                elif val == 2.0:
                    print("    - stroke_1Y=2: Stroke after 1 year")
                elif val == 3.0:
                    print("    - stroke_1Y=3: Stroke same day")
                elif val == 4.0:
                    print("    - stroke_1Y=4: Stroke before AF")
    else:
        print("NEW dataset: No combination shows good stratification")
        print("\nThis suggests fundamental data quality issues:")
        print("- Age/risk relationships may be scrambled")
        print("- Clinical variables may not reflect true risk")
        print("- Dataset may need additional cleaning/filtering")


def main():
    """Run the stroke_1Y combination analysis."""
    print("Testing stroke_1Y Value Combinations for Risk Stratification")
    print("="*80)
    print("Based on analyze_stroke_encoding.py interpretations:")
    print("  1 = Within 1 year | 2 = After 1 year | 3 = Same day | 4 = Before AF")
    
    # Test all combinations
    results = test_all_combinations()
    
    # Find working combinations
    find_working_combinations(results)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()