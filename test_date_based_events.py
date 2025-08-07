#!/usr/bin/env python3
"""
Test date-based event definitions using the dataset creator's guidance.

Key insights from dataset creator:
- Ignore stroke_1Y completely, use actual dates
- Day-of-diagnosis strokes might be coding artifacts  
- Exclude patients with prior strokes
- Focus on stroke from AF diagnosis to 1-year follow-up

This should give us clean temporal event definitions.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from nuchad.utils import get_df, calculate_chadsvasc


def analyze_stroke_timing(df, dataset_name):
    """Analyze the timing of strokes relative to AF diagnosis."""
    print(f"\n{dataset_name} - Stroke Timing Analysis:")
    print("-" * 50)
    
    # Patients with both AF and stroke dates
    both_dates = df[df['earliest_af_date'].notna() & df['earliest_stroke_date'].notna()].copy()
    print(f"Patients with both AF and stroke dates: {len(both_dates):,}")
    
    if len(both_dates) == 0:
        return
    
    # Calculate days between AF diagnosis and stroke
    both_dates['days_af_to_stroke'] = (both_dates['earliest_stroke_date'] - both_dates['earliest_af_date']).dt.days
    
    # Categorize timing
    same_day = (both_dates['days_af_to_stroke'] == 0).sum()
    within_1y = (both_dates['days_af_to_stroke'] <= 365).sum() 
    within_1y_excl_same = ((both_dates['days_af_to_stroke'] > 0) & (both_dates['days_af_to_stroke'] <= 365)).sum()
    after_1y = (both_dates['days_af_to_stroke'] > 365).sum()
    before_af = (both_dates['days_af_to_stroke'] < 0).sum()
    
    print(f"Same day as AF diagnosis: {same_day:,} ({same_day/len(both_dates)*100:.1f}%)")
    print(f"Within 1 year (including same day): {within_1y:,} ({within_1y/len(both_dates)*100:.1f}%)")
    print(f"Within 1 year (excluding same day): {within_1y_excl_same:,} ({within_1y_excl_same/len(both_dates)*100:.1f}%)")
    print(f"After 1 year: {after_1y:,} ({after_1y/len(both_dates)*100:.1f}%)")
    print(f"Before AF diagnosis: {before_af:,} ({before_af/len(both_dates)*100:.1f}%)")
    
    if len(both_dates) > 0:
        print(f"Mean days AF to stroke: {both_dates['days_af_to_stroke'].mean():.1f}")
        print(f"Median days AF to stroke: {both_dates['days_af_to_stroke'].median():.1f}")


def test_date_based_definitions(df, dataset_name):
    """Test various date-based event definitions."""
    print(f"\n{dataset_name} - Testing Date-Based Event Definitions:")
    print("=" * 60)
    
    results = []
    
    # Calculate CHADS-VASc for all tests
    df_test = df.copy()
    df_test['chadsvasc'] = df_test.apply(calculate_chadsvasc, axis=1)
    df_test['risk_group'] = pd.cut(
        df_test['chadsvasc'], 
        bins=[-0.5, 1.5, 3.5, 10], 
        labels=['Low (0-1)', 'Moderate (2-3)', 'High (4+)']
    )
    
    # Helper function to test definition
    def test_definition(mask, name, description=""):
        df_test['event'] = mask
        
        # Age correlation
        age_corr, age_p = None, None
        if 'age' in df_test.columns:
            valid_age = df_test[df_test['age'].notna()]
            if len(valid_age) > 0:
                age_corr, age_p = pearsonr(valid_age['event'].astype(int), valid_age['age'])
        
        # Risk group rates
        risk_rates = {}
        for group in ['Low (0-1)', 'Moderate (2-3)', 'High (4+)']:
            group_data = df_test[df_test['risk_group'] == group]
            if len(group_data) > 0:
                rate = group_data['event'].mean() * 100
                risk_rates[group] = rate
        
        rates_list = [risk_rates.get(g, 0) for g in ['Low (0-1)', 'Moderate (2-3)', 'High (4+)']]
        gradient = max(rates_list) - min(rates_list)
        
        print(f"\n{name}:")
        if description:
            print(f"  {description}")
        print(f"  Events: {mask.sum():,} ({mask.mean()*100:.1f}%)")
        print(f"  Age correlation: r={age_corr:.3f}, p={age_p:.4f}" if age_corr is not None else "  Age correlation: N/A")
        print(f"  Risk rates: Low={rates_list[0]:.1f}%, Mod={rates_list[1]:.1f}%, High={rates_list[2]:.1f}%")
        print(f"  Gradient: {gradient:.1f}% {'✓' if gradient > 2.0 else '✗'}")
        
        return {
            'name': name,
            'events': int(mask.sum()),
            'event_rate': mask.mean(),
            'age_correlation': age_corr,
            'age_p_value': age_p,
            'gradient': gradient,
            'rates': rates_list
        }
    
    # Calculate timing mask for reuse
    has_both_dates = df_test['earliest_af_date'].notna() & df_test['earliest_stroke_date'].notna()
    df_test['days_af_to_stroke'] = (df_test['earliest_stroke_date'] - df_test['earliest_af_date']).dt.days
    
    # Definition 1: Any stroke after AF diagnosis within 1 year (including same day)
    mask1 = has_both_dates & (df_test['days_af_to_stroke'] >= 0) & (df_test['days_af_to_stroke'] <= 365)
    results.append(test_definition(mask1, "1. Stroke 0-365 days after AF", "Including same-day strokes"))
    
    # Definition 2: Stroke after AF diagnosis within 1 year (excluding same day) 
    mask2 = has_both_dates & (df_test['days_af_to_stroke'] > 0) & (df_test['days_af_to_stroke'] <= 365)
    results.append(test_definition(mask2, "2. Stroke 1-365 days after AF", "Excluding same-day strokes"))
    
    # Definition 3: Stroke within 1 year, excluding prior strokes (before AF)
    no_prior_stroke = ~(has_both_dates & (df_test['days_af_to_stroke'] < 0))
    mask3 = no_prior_stroke & has_both_dates & (df_test['days_af_to_stroke'] >= 0) & (df_test['days_af_to_stroke'] <= 365)
    results.append(test_definition(mask3, "3. Stroke 0-365d, no prior stroke", "Excluding patients with strokes before AF"))
    
    # Definition 4: Stroke 1-365 days, excluding prior strokes
    mask4 = no_prior_stroke & has_both_dates & (df_test['days_af_to_stroke'] > 0) & (df_test['days_af_to_stroke'] <= 365)
    results.append(test_definition(mask4, "4. Stroke 1-365d, no prior stroke", "Clean 1-year follow-up events"))
    
    # Definition 5: Just has stroke date after AF date (any timeframe)
    mask5 = has_both_dates & (df_test['days_af_to_stroke'] >= 0)
    results.append(test_definition(mask5, "5. Any stroke after AF", "No time restriction"))
    
    # Definition 6: Stroke within 2 years
    mask6 = has_both_dates & (df_test['days_af_to_stroke'] >= 0) & (df_test['days_af_to_stroke'] <= 730)
    results.append(test_definition(mask6, "6. Stroke 0-730 days after AF", "2-year follow-up window"))
    
    # Definition 7: Same day strokes only (potential coding artifacts)
    mask7 = has_both_dates & (df_test['days_af_to_stroke'] == 0)
    results.append(test_definition(mask7, "7. Same-day strokes only", "Potential coding artifacts"))
    
    return results


def compare_datasets():
    """Compare both datasets with date-based definitions."""
    print("="*80)
    print("DATE-BASED EVENT DEFINITION TESTING")
    print("="*80)
    print("Using dataset creator's guidance:")
    print("- Ignore stroke_1Y coding")
    print("- Use actual AF and stroke dates") 
    print("- Consider excluding same-day and prior strokes")
    print("="*80)
    
    # Load datasets
    old_df = get_df("random_nuchad.csv")
    new_df = get_df("random_nuchad_250623.csv")
    
    print(f"Old dataset: {len(old_df):,} patients")
    print(f"New dataset: {len(new_df):,} patients")
    
    # Analyze stroke timing patterns
    analyze_stroke_timing(old_df, "OLD DATASET")
    analyze_stroke_timing(new_df, "NEW DATASET")
    
    # Test date-based definitions
    old_results = test_date_based_definitions(old_df, "OLD DATASET")
    new_results = test_date_based_definitions(new_df, "NEW DATASET")
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    print(f"{'Definition':<35} {'Dataset':<8} {'Events':<8} {'Age r':<8} {'Gradient':<10} {'Good?':<6}")
    print("-" * 80)
    
    # Find best definitions
    best_old = None
    best_new = None
    
    for i, (old_result, new_result) in enumerate(zip(old_results, new_results)):
        for result, dataset in [(old_result, "OLD"), (new_result, "NEW")]:
            age_r = f"{result['age_correlation']:.3f}" if result['age_correlation'] is not None else "N/A"
            gradient = f"{result['gradient']:.1f}%"
            good = "✓" if result['gradient'] > 2.0 and result['age_correlation'] and result['age_correlation'] > 0 else "✗"
            
            print(f"{result['name']:<35} {dataset:<8} {result['events']:<8} {age_r:<8} {gradient:<10} {good:<6}")
            
            # Track best
            if (dataset == "OLD" and result['gradient'] > 2.0 and result['age_correlation'] and result['age_correlation'] > 0):
                if best_old is None or result['age_correlation'] > best_old['age_correlation']:
                    best_old = result
                    
            if (dataset == "NEW" and result['gradient'] > 2.0 and result['age_correlation'] and result['age_correlation'] > 0):
                if best_new is None or result['age_correlation'] > best_new['age_correlation']:
                    best_new = result
    
    print(f"\n" + "="*60)
    print("BEST DEFINITIONS FOUND")
    print("="*60)
    
    if best_old:
        print(f"OLD dataset: {best_old['name']}")
        print(f"  Age correlation: r={best_old['age_correlation']:.3f}")
        print(f"  Risk gradient: {best_old['gradient']:.1f}%")
        print(f"  Event rate: {best_old['event_rate']*100:.1f}%")
    
    if best_new:
        print(f"NEW dataset: {best_new['name']}")
        print(f"  Age correlation: r={best_new['age_correlation']:.3f}")
        print(f"  Risk gradient: {best_new['gradient']:.1f}%")
        print(f"  Event rate: {best_new['event_rate']*100:.1f}%")
        print("\n🎉 SUCCESS! Found working definition for new dataset!")
    else:
        print("NEW dataset: Still no working definition found")
        print("Consider:")
        print("- Different time windows")
        print("- Additional exclusion criteria")
        print("- Data quality investigation")


def main():
    """Run the date-based analysis."""
    print("Testing Date-Based Event Definitions")
    print("="*80)
    print("Following dataset creator's recommendations")
    
    compare_datasets()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()