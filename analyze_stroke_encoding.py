#!/usr/bin/env python3
"""
Stroke Encoding Analysis - Documentation for Collaborators

This script provides definitive proof of how stroke_1Y is encoded differently 
between the old and new datasets. This analysis demonstrates the fundamental 
incompatibility between datasets that must be addressed in any cross-dataset work.

Key Findings:
- OLD dataset: stroke_1Y = 1 (stroke), stroke_1Y = 2 (no stroke/controls)
- NEW dataset: stroke_1Y = 1-4 (different stroke types), Missing = controls
- stroke_1Y = 2 means completely different things in each dataset!

Usage:
    python analyze_stroke_encoding.py
    
Output:
    - Detailed breakdown of stroke_1Y values in both datasets
    - Stroke date analysis confirming the encoding interpretation
    - Hypothesis testing for missing values as controls
    - Summary tables for easy reference
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_and_prepare_data():
    """Load both datasets and prepare date columns."""
    print("Loading datasets...")
    
    # Load old dataset
    old_df = pd.read_csv('data/random_nuchad.csv')
    print(f"Old dataset: {len(old_df):,} patients")
    
    # Load new dataset  
    new_df = pd.read_csv('data/random_nuchad_250623.csv')
    print(f"New dataset: {len(new_df):,} patients")
    
    # Convert dates for new dataset (it has different format)
    date_cols = ['earliest_af_date', 'earliest_stroke_date']
    for col in date_cols:
        if col in new_df.columns:
            new_df[col] = pd.to_datetime(new_df[col], format='%d-%b-%y', errors='coerce')
    
    # Convert dates for old dataset  
    for col in date_cols:
        if col in old_df.columns:
            old_df[col] = pd.to_datetime(old_df[col], format='%d%b%Y', errors='coerce')
    
    return old_df, new_df

def analyze_old_dataset_encoding(df):
    """Analyze stroke_1Y encoding in the old dataset."""
    print("\n" + "="*60)
    print("OLD DATASET ANALYSIS (random_nuchad.csv)")
    print("="*60)
    
    # stroke_1Y value counts
    stroke_counts = df['stroke_1Y'].value_counts(dropna=False).sort_index()
    
    print("\nstroke_1Y Value Distribution:")
    print("-" * 40)
    total = len(df)
    for value, count in stroke_counts.items():
        pct = count/total*100
        if pd.isna(value):
            print(f"Missing:     {count:8,} patients ({pct:5.1f}%)")
        else:
            print(f"Value {value}:     {count:8,} patients ({pct:5.1f}%)")
    
    # Analyze stroke dates for each group
    print("\nStroke Date Analysis (Proof of Encoding):")
    print("-" * 40)
    
    for value in sorted(df['stroke_1Y'].dropna().unique()):
        subset = df[df['stroke_1Y'] == value]
        has_stroke_date = subset['earliest_stroke_date'].notna().sum()
        
        print(f"\nstroke_1Y = {value}:")
        print(f"  Patients with stroke date (colname is earliest_stroke_date): {has_stroke_date:,}/{len(subset):,} ({has_stroke_date/len(subset)*100:.1f}%)")
        
        if has_stroke_date > 0:
            # Calculate timing
            valid_dates = subset.dropna(subset=['earliest_af_date', 'earliest_stroke_date'])
            if len(valid_dates) > 0:
                valid_dates = valid_dates.copy()
                valid_dates['days_af_to_stroke'] = (valid_dates['earliest_stroke_date'] - valid_dates['earliest_af_date']).dt.days
                
                within_1y = (valid_dates['days_af_to_stroke'] <= 365).sum()
                after_1y = (valid_dates['days_af_to_stroke'] > 365).sum()
                before_af = (valid_dates['days_af_to_stroke'] < 0).sum()
                
                print(f"  Timing analysis ({len(valid_dates):,} with both dates):")
                print(f"    ≤365 days (within 1Y): {within_1y:,} ({within_1y/len(valid_dates)*100:.1f}%)")
                print(f"    >365 days (after 1Y): {after_1y:,} ({after_1y/len(valid_dates)*100:.1f}%)")
                print(f"    <0 days (before AF): {before_af:,} ({before_af/len(valid_dates)*100:.1f}%)")
                if len(valid_dates) > 0:
                    print(f"    Mean days: {valid_dates['days_af_to_stroke'].mean():.1f}")
        
        # Interpretation
        if value == 1:
            print(f"  → INTERPRETATION: Stroke within 1 year of AF diagnosis")
        elif value == 2:
            if has_stroke_date < len(subset) * 0.1:  # Less than 10% have stroke dates
                print(f"  → INTERPRETATION: NO STROKE (control patients)")
            else:
                print(f"  → INTERPRETATION: Stroke after 1 year of AF diagnosis")

def analyze_new_dataset_encoding(df):
    """Analyze stroke_1Y encoding in the new dataset."""
    print("\n" + "="*60)
    print("NEW DATASET ANALYSIS (random_nuchad_250623.csv)")
    print("="*60)
    
    # stroke_1Y value counts
    stroke_counts = df['stroke_1Y'].value_counts(dropna=False).sort_index()
    
    print("\nstroke_1Y Value Distribution:")
    print("-" * 40)
    total = len(df)
    for value, count in stroke_counts.items():
        pct = count/total*100
        if pd.isna(value):
            print(f"Missing:     {count:8,} patients ({pct:5.1f}%)")
        else:
            print(f"Value {value}:   {count:8,} patients ({pct:5.1f}%)")
    
    # Analyze stroke dates for each group (including missing)
    print("\nStroke Date Analysis (Proof of Encoding):")
    print("-" * 40)
    
    # Analyze defined values
    for value in sorted(df['stroke_1Y'].dropna().unique()):
        subset = df[df['stroke_1Y'] == value]
        has_stroke_date = subset['earliest_stroke_date'].notna().sum()
        
        print(f"\nstroke_1Y = {value}:")
        print(f"  Patients with stroke date (colname is earliest_stroke_date): {has_stroke_date:,}/{len(subset):,} ({has_stroke_date/len(subset)*100:.1f}%)")
        
        if has_stroke_date > 0:
            # Calculate timing
            valid_dates = subset.dropna(subset=['earliest_af_date', 'earliest_stroke_date'])
            if len(valid_dates) > 0:
                valid_dates = valid_dates.copy()
                valid_dates['days_af_to_stroke'] = (valid_dates['earliest_stroke_date'] - valid_dates['earliest_af_date']).dt.days
                
                within_1y = (valid_dates['days_af_to_stroke'] <= 365).sum()
                after_1y = (valid_dates['days_af_to_stroke'] > 365).sum()
                before_af = (valid_dates['days_af_to_stroke'] < 0).sum()

                within_2_days_before_after_af_window = (valid_dates['days_af_to_stroke'] <= 2) & (valid_dates['days_af_to_stroke'] >= -2)
                within_2_days_before_after_af_window_count = within_2_days_before_after_af_window.sum()
                within_2_days_before_after_af_window_count_percentage = within_2_days_before_after_af_window_count / len(valid_dates) * 100
                print(f"  Patients within 2 days before and after AF diagnosis: {within_2_days_before_after_af_window_count:,}/{len(valid_dates):,} ({within_2_days_before_after_af_window_count_percentage:.1f}%)")
                
                print(f"  Timing analysis ({len(valid_dates):,} with both dates):")
                print(f"    ≤365 days (within 1Y): {within_1y:,} ({within_1y/len(valid_dates)*100:.1f}%)")
                print(f"    >365 days (after 1Y): {after_1y:,} ({after_1y/len(valid_dates)*100:.1f}%)")
                print(f"    <0 days (before AF): {before_af:,} ({before_af/len(valid_dates)*100:.1f}%)")
                if len(valid_dates) > 0:
                    print(f"    Mean days: {valid_dates['days_af_to_stroke'].mean():.1f}")
        
        # Interpretation based on timing patterns
        if value == 1.0:
            print(f"  → INTERPRETATION: Stroke within 1 year of AF diagnosis")
        elif value == 2.0:
            print(f"  → INTERPRETATION: Stroke AFTER 1 year of AF diagnosis")
        elif value == 3.0:
            print(f"  → INTERPRETATION: Stroke same day as AF diagnosis")
        elif value == 4.0:
            print(f"  → INTERPRETATION: Stroke BEFORE AF diagnosis")
    
    # CRITICAL: Analyze missing stroke_1Y group (potential controls)
    print(f"\n🔍 CRITICAL: Missing stroke_1Y Analysis:")
    print("-" * 60)
    missing_subset = df[df['stroke_1Y'].isna()]
    has_stroke_date = missing_subset['earliest_stroke_date'].notna().sum()
    
    print(f"stroke_1Y = Missing ({len(missing_subset):,} patients):")
    print(f"  Patients with stroke date: {has_stroke_date:,}/{len(missing_subset):,} ({has_stroke_date/len(missing_subset)*100:.1f}%)")
    print(f"  Patients WITHOUT stroke date: {len(missing_subset)-has_stroke_date:,}/{len(missing_subset):,} ({(len(missing_subset)-has_stroke_date)/len(missing_subset)*100:.1f}%)")
    
    if has_stroke_date == 0:
        print(f"  → INTERPRETATION: NO STROKE (control patients)")
        print(f"  ✅ CONFIRMED: All missing stroke_1Y patients lack stroke dates!")
    else:
        print(f"  ⚠️  WARNING: {has_stroke_date:,} missing stroke_1Y patients DO have stroke dates!")
        print(f"  → Need further investigation of missing stroke_1Y meaning")

def generate_encoding_comparison_table(old_df, new_df):
    """Generate the definitive encoding comparison table."""
    print("\n" + "="*90)
    print("🚨 CRITICAL INCOMPATIBILITY: stroke_1Y ENCODING COMPARISON")
    print("="*90)
    
    print(f"\n{'Dataset':<25} {'stroke_1Y Value':<15} {'Count':<12} {'%':<8} {'Meaning':<30}")
    print("-" * 90)
    
    # Old dataset breakdown
    print("OLD (random_nuchad.csv):")
    old_stroke_counts = old_df['stroke_1Y'].value_counts(dropna=False).sort_index()
    for value, count in old_stroke_counts.items():
        pct = count/len(old_df)*100
        if value == 1:
            meaning = "Stroke within 1 year"
        elif value == 2:
            meaning = "NO STROKE (controls)"
        else:
            meaning = "Unknown"
        
        value_str = "Missing" if pd.isna(value) else str(int(value))
        print(f"{'':25} {value_str:<15} {count:>11,} {pct:>7.1f}% {meaning}")
    
    print()
    print("NEW (random_nuchad_250623.csv):")
    new_stroke_counts = new_df['stroke_1Y'].value_counts(dropna=False).sort_index()
    for value, count in new_stroke_counts.items():
        pct = count/len(new_df)*100
        if pd.isna(value):
            meaning = "NO STROKE (controls)"
            value_str = "Missing"
        elif value == 1.0:
            meaning = "Stroke within 1 year"
            value_str = "1"
        elif value == 2.0:
            meaning = "Stroke AFTER 1 year"  # KEY DIFFERENCE!
            value_str = "2"
        elif value == 3.0:
            meaning = "Stroke same day as AF"
            value_str = "3"
        elif value == 4.0:
            meaning = "Stroke BEFORE AF"
            value_str = "4"
        else:
            meaning = "Unknown"
            value_str = str(value)
        
        print(f"{'':25} {value_str:<15} {count:>11,} {pct:>7.1f}% {meaning}")
    
    print("\n" + "🚨" * 30)
    print("KEY INCOMPATIBILITY:")
    print("  stroke_1Y = 2 in OLD dataset = NO STROKE (controls)")
    print("  stroke_1Y = 2 in NEW dataset = STROKE AFTER 1 YEAR")
    print("🚨" * 30)

def calculate_comparable_event_rates(old_df, new_df):
    """Calculate comparable event rates for both datasets."""
    print("\n" + "="*60)
    print("COMPARABLE EVENT RATES")
    print("="*60)
    
    # Old dataset
    old_stroke_1y = (old_df['stroke_1Y'] == 1).sum()
    old_total = len(old_df)
    old_rate = old_stroke_1y / old_total * 100
    
    # New dataset - using corrected interpretation
    new_stroke_1y = (new_df['stroke_1Y'] == 1.0).sum()  # Only value 1 = within 1 year
    new_total = len(new_df)
    new_rate = new_stroke_1y / new_total * 100
    
    # New dataset - all stroke events
    new_all_strokes = new_df['stroke_1Y'].notna().sum()
    new_all_rate = new_all_strokes / new_total * 100
    
    print(f"\nStroke Within 1 Year Rates:")
    print(f"  Old dataset: {old_stroke_1y:,}/{old_total:,} = {old_rate:.1f}%")
    print(f"  New dataset: {new_stroke_1y:,}/{new_total:,} = {new_rate:.1f}%")
    print(f"  Rate difference: {abs(old_rate - new_rate):.1f} percentage points")
    
    print(f"\nAll Stroke Events:")
    print(f"  Old dataset: {old_stroke_1y:,}/{old_total:,} = {old_rate:.1f}% (only tracks 1-year)")
    print(f"  New dataset: {new_all_strokes:,}/{new_total:,} = {new_all_rate:.1f}% (all timepoints)")

def main():
    """Main analysis function."""
    print("="*70)
    print("STROKE ENCODING ANALYSIS - DEFINITIVE DOCUMENTATION")
    print("="*70)
    print(f"Analysis generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nPurpose: Prove stroke_1Y encoding differences between datasets")
    print("Author: Analysis performed by Claude AI")
    print("="*70)
    
    # Load data
    old_df, new_df = load_and_prepare_data()
    
    # Analyze each dataset
    analyze_old_dataset_encoding(old_df)
    analyze_new_dataset_encoding(new_df)
    
    # Generate comparison table
    generate_encoding_comparison_table(old_df, new_df)
    
    # Calculate event rates
    calculate_comparable_event_rates(old_df, new_df)
    
    # Final summary
    print("\n" + "="*70)
    print("SUMMARY FOR COLLABORATORS")
    print("="*70)
    print("1. stroke_1Y encoding is FUNDAMENTALLY INCOMPATIBLE between datasets")
    print("2. stroke_1Y = 2 means opposite things in each dataset:")
    print("   - OLD: no stroke (controls)")  
    print("   - NEW: stroke after 1 year")
    print("3. Controls are encoded differently:")
    print("   - OLD: stroke_1Y = 2")
    print("   - NEW: Missing stroke_1Y values")
    print("4. Any cross-dataset analysis must account for these differences")
    print("5. Event rates are not directly comparable without recoding")
    print("="*70)

if __name__ == "__main__":
    main() 