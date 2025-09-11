#!/usr/bin/env python3
"""
Test if treatment regimes are confounding the risk relationships.

Hypothesis: Different treatment patterns between old/new datasets are masking
the age/CHADS-VASc correlations. By controlling for treatment (OAC, antiplatelet),
we might recover proper risk stratification within treatment levels.

Treatment categories:
- No treatment (both OAC and antiplatelet missing)
- OAC only (during time1 to end_fu window)
- Antiplatelet only (during time1 to end_fu window)
- Both treatments
- Treatment outside window
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from nuchad.utils import get_df, calculate_chadsvasc


def classify_treatment(df, dataset_name):
    """Classify patients by treatment regime."""
    print(f"\n{dataset_name} - Treatment Classification:")
    print("-" * 50)
    
    df_treat = df.copy()
    
    # Different column names for different datasets
    if dataset_name == "OLD DATASET":
        # Old dataset only has Anticoagulant (categorical, not dates)
        # We'll classify based on presence of anticoagulant treatment
        has_anticoag = df_treat['Anticoagulant'].notna() & (df_treat['Anticoagulant'] != 0)
        
        # No antiplatelet info in old dataset, so simple classification
        oac_during_fu = has_anticoag
        antiplatelet_during_fu = pd.Series(False, index=df_treat.index)  # No antiplatelet data
        
    else:  # NEW DATASET
        # New dataset has date-based treatment info
        # OAC during follow-up window
        oac_during_fu = (
            df_treat['first_OAC_date'].notna() &
            (df_treat['first_OAC_date'] >= df_treat['time1']) &
            (df_treat['first_OAC_date'] <= df_treat['end_fu'])
        )
        
        # Antiplatelet during follow-up window
        antiplatelet_during_fu = (
            df_treat['first_antiplatelet_date'].notna() &
            (df_treat['first_antiplatelet_date'] >= df_treat['time1']) &
            (df_treat['first_antiplatelet_date'] <= df_treat['end_fu'])
        )
    
    # Create treatment categories
    df_treat['treatment_category'] = 'No treatment'
    
    # Single treatments
    df_treat.loc[oac_during_fu & ~antiplatelet_during_fu, 'treatment_category'] = 'OAC only'
    df_treat.loc[~oac_during_fu & antiplatelet_during_fu, 'treatment_category'] = 'Antiplatelet only'
    df_treat.loc[oac_during_fu & antiplatelet_during_fu, 'treatment_category'] = 'Both treatments'
    
    # Check for treatments outside window (only for new dataset)
    if dataset_name != "OLD DATASET":
        oac_outside = df_treat['first_OAC_date'].notna() & ~oac_during_fu
        antiplatelet_outside = df_treat['first_antiplatelet_date'].notna() & ~antiplatelet_during_fu
        
        df_treat.loc[(oac_outside | antiplatelet_outside) & 
                     (df_treat['treatment_category'] == 'No treatment'), 
                     'treatment_category'] = 'Treatment outside window'
    
    # Report distribution
    treatment_counts = df_treat['treatment_category'].value_counts()
    total = len(df_treat)
    
    print("Treatment distribution:")
    for category, count in treatment_counts.items():
        pct = count / total * 100
        print(f"  {category}: {count:,} ({pct:.1f}%)")
    
    return df_treat


def test_within_treatment_stratum(df_treat, treatment_category, stroke_definition, dataset_name):
    """Test age/CHADS-VASc correlation within a specific treatment stratum."""
    
    # Filter to treatment group
    treatment_group = df_treat[df_treat['treatment_category'] == treatment_category].copy()
    
    if len(treatment_group) == 0:
        return None
    
    # Apply stroke definition
    if stroke_definition == 'stroke_1Y_1':
        if dataset_name == 'OLD':
            event_mask = treatment_group['stroke_1Y'] == 1
        else:  # NEW
            event_mask = treatment_group['stroke_1Y'] == 1.0
    elif stroke_definition == 'stroke_1Y_1_3':
        if dataset_name == 'OLD':
            event_mask = treatment_group['stroke_1Y'].isin([1])  # No 3 in old dataset
        else:  # NEW
            event_mask = treatment_group['stroke_1Y'].isin([1.0, 3.0])
    elif stroke_definition == 'has_stroke_date':
        event_mask = treatment_group['earliest_stroke_date'].notna()
    else:
        return None
    
    treatment_group['event'] = event_mask
    
    # Calculate CHADS-VASc
    treatment_group['chadsvasc'] = treatment_group.apply(calculate_chadsvasc, axis=1)
    treatment_group['risk_group'] = pd.cut(
        treatment_group['chadsvasc'], 
        bins=[-0.5, 1.5, 3.5, 10], 
        labels=['Low (0-1)', 'Moderate (2-3)', 'High (4+)']
    )
    
    # Age correlation
    age_corr, age_p = None, None
    if 'age' in treatment_group.columns:
        valid_age = treatment_group[treatment_group['age'].notna()]
        if len(valid_age) > 10:  # Need minimum sample size
            try:
                if valid_age['event'].sum() > 0 and valid_age['event'].sum() < len(valid_age):
                    age_corr, age_p = pearsonr(valid_age['event'].astype(int), valid_age['age'])
                else:
                    age_corr, age_p = np.nan, np.nan
            except:
                age_corr, age_p = np.nan, np.nan
    
    # Risk group rates
    risk_rates = {}
    for group in ['Low (0-1)', 'Moderate (2-3)', 'High (4+)']:
        group_data = treatment_group[treatment_group['risk_group'] == group]
        if len(group_data) > 0:
            rate = group_data['event'].mean() * 100
            risk_rates[group] = rate
        else:
            risk_rates[group] = 0
    
    rates_list = [risk_rates[g] for g in ['Low (0-1)', 'Moderate (2-3)', 'High (4+)']]
    gradient = max(rates_list) - min(rates_list)
    
    return {
        'treatment_category': treatment_category,
        'stroke_definition': stroke_definition,
        'n_patients': len(treatment_group),
        'n_events': int(event_mask.sum()),
        'event_rate': event_mask.mean(),
        'age_correlation': age_corr,
        'age_p_value': age_p,
        'risk_rates': risk_rates,
        'gradient': gradient,
        'good_correlation': age_corr is not None and not np.isnan(age_corr) and age_corr > 0.02 and age_p < 0.05,
        'good_gradient': gradient > 2.0
    }


def test_treatment_stratified_analysis():
    """Test risk stratification within treatment levels."""
    print("="*80)
    print("TREATMENT-STRATIFIED RISK ANALYSIS")
    print("="*80)
    print("Testing if treatment differences explain lack of risk stratification")
    print("="*80)
    
    # Load datasets
    old_df = get_df("random_nuchad.csv")
    new_df = get_df("random_nuchad_250623.csv")
    
    # Classify treatments
    old_treated = classify_treatment(old_df, "OLD DATASET")
    new_treated = classify_treatment(new_df, "NEW DATASET")
    
    results = {}
    
    # Test different stroke definitions within treatment strata
    stroke_definitions = [
        'stroke_1Y_1',     # stroke_1Y = 1 (within 1 year)
        'stroke_1Y_1_3',   # stroke_1Y = 1 or 3 (within 1Y + same day)
        'has_stroke_date'  # Has any stroke date
    ]
    
    for dataset_name, df_treated in [("OLD", old_treated), ("NEW", new_treated)]:
        print(f"\n" + "="*60)
        print(f"{dataset_name} DATASET - TREATMENT-STRATIFIED ANALYSIS")
        print("="*60)
        
        results[dataset_name] = []
        
        # Get unique treatment categories
        treatment_categories = df_treated['treatment_category'].unique()
        
        for stroke_def in stroke_definitions:
            print(f"\nStroke Definition: {stroke_def}")
            print("-" * 40)
            
            for treatment_cat in treatment_categories:
                result = test_within_treatment_stratum(
                    df_treated, treatment_cat, stroke_def, dataset_name
                )
                
                if result is not None:
                    results[dataset_name].append(result)
                    
                    age_r = f"{result['age_correlation']:.3f}" if result['age_correlation'] is not None and not np.isnan(result['age_correlation']) else "N/A"
                    age_p = f"{result['age_p_value']:.4f}" if result['age_p_value'] is not None and not np.isnan(result['age_p_value']) else "N/A"
                    
                    rates = result['risk_rates']
                    rates_str = f"L={rates['Low (0-1)']:.1f}%, M={rates['Moderate (2-3)']:.1f}%, H={rates['High (4+)']:.1f}%"
                    
                    good_marker = "✓" if result['good_correlation'] and result['good_gradient'] else "✗"
                    
                    print(f"  {treatment_cat:<25} n={result['n_patients']:<6} events={result['n_events']:<5} "
                          f"age_r={age_r:<7} gradient={result['gradient']:<5.1f}% {rates_str} {good_marker}")
    
    return results


def summarize_treatment_findings(results):
    """Summarize findings from treatment-stratified analysis."""
    print("\n" + "="*80)
    print("TREATMENT-STRATIFIED FINDINGS SUMMARY")
    print("="*80)
    
    # Find best results for each dataset
    best_old = []
    best_new = []
    
    for dataset in ['OLD', 'NEW']:
        good_results = []
        for result in results[dataset]:
            if result['good_correlation'] and result['good_gradient']:
                good_results.append(result)
        
        if dataset == 'OLD':
            best_old = good_results
        else:
            best_new = good_results
    
    print(f"\nSUCCESSFUL STRATIFICATIONS:")
    print("-" * 50)
    
    if best_old:
        print(f"OLD dataset - {len(best_old)} successful stratifications found:")
        for result in best_old:
            print(f"  {result['treatment_category']} + {result['stroke_definition']}")
            print(f"    Age correlation: r={result['age_correlation']:.3f}, p={result['age_p_value']:.4f}")
            print(f"    Risk gradient: {result['gradient']:.1f}%")
    
    if best_new:
        print(f"NEW dataset - {len(best_new)} successful stratifications found:")
        for result in best_new:
            print(f"  {result['treatment_category']} + {result['stroke_definition']}")
            print(f"    Age correlation: r={result['age_correlation']:.3f}, p={result['age_p_value']:.4f}")
            print(f"    Risk gradient: {result['gradient']:.1f}%")
        print(f"\n🎉 SUCCESS! Found {len(best_new)} working stratifications for new dataset!")
    else:
        print("NEW dataset - No successful stratifications found")
        print("Even controlling for treatment doesn't recover risk relationships")
    
    # Compare treatment distributions
    print(f"\n" + "="*60)
    print("TREATMENT DISTRIBUTION COMPARISON")
    print("="*60)
    
    print("This analysis reveals whether treatment differences explain")
    print("the lack of risk stratification in the new dataset.")
    
    if best_new:
        print(f"\n✓ HYPOTHESIS CONFIRMED: Treatment confounding was masking risk relationships")
        print("The new dataset shows proper risk stratification within treatment strata")
    else:
        print(f"\n✗ HYPOTHESIS REJECTED: Treatment confounding doesn't explain the issue")
        print("The fundamental lack of risk relationships persists even within treatment levels")


def main():
    """Run treatment-stratified analysis."""
    print("Testing Treatment-Stratified Risk Analysis")
    print("="*80)
    print("Hypothesis: Treatment regime differences are confounding risk relationships")
    
    # Run analysis
    results = test_treatment_stratified_analysis()
    
    # Summarize findings
    summarize_treatment_findings(results)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()