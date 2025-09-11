#!/usr/bin/env python3
"""Demonstrate lack of separation for 1-year stroke outcome.

This minimal script computes the outcome "stroke from AF diagnosis (time1)
to 1-year follow-up (yes/no)" using dates, then reports:
- Overall event rate
- Event rates by CHADS-VASc score
- Correlation with age and with CHADS-VASc

By default it uses the newer dataset (random_nuchad_250623.csv) and applies
the standard eligibility filters. Results are printed and saved to results/.
"""

import sys
import argparse
import numpy as np
import pandas as pd

from nuchad.utils import get_df, calculate_chadsvasc, get_results_dir
from nuchad.data_processing import eligibility_filters as data_utils
from nuchad.data_processing.eligibility_filters import (
    filter_patients_from_config, 
    get_available_configs,
    detect_anticoagulant
)


def compute_event_within_1y(df: pd.DataFrame) -> pd.Series:
    """Return boolean Series: stroke occurs between time1 and 365 days inclusive."""
    if 'time1' not in df.columns:
        if 'earliest_af_date' in df.columns:
            df = df.copy()
            df['time1'] = df['earliest_af_date']
        else:
            return pd.Series(False, index=df.index)

    if 'earliest_stroke_date' not in df.columns:
        return pd.Series(False, index=df.index)

    valid = df['time1'].notna() & df['earliest_stroke_date'].notna()
    days = (df.loc[valid, 'earliest_stroke_date'] - df.loc[valid, 'time1']).dt.days
    event = pd.Series(False, index=df.index)
    event.loc[valid] = (days >= 0) & (days <= 365)
    return event


def compute_correlations(df: pd.DataFrame) -> dict:
    """Compute correlations between predictors and stroke events."""
    def corr(x: pd.Series, y_bool: pd.Series) -> float:
        """Point-biserial correlation (Pearson with binary)."""
        xv = x.to_numpy(dtype=float)
        yv = y_bool.astype(int).to_numpy()
        x_c = xv - np.nanmean(xv)
        y_c = yv - np.nanmean(yv)
        num = np.nansum(x_c * y_c)
        den = (np.nansum(x_c ** 2) ** 0.5) * (np.nansum(y_c ** 2) ** 0.5)
        return float(num / den) if den != 0 else 0.0

    # Prepare age data
    age = pd.to_numeric(df['age'], errors='coerce') if 'age' in df.columns else pd.Series(np.nan, index=df.index)
    age_filled = age.fillna(age.median()) if len(age.dropna()) else pd.Series(0.0, index=df.index)
    
    return {
        'age': corr(age_filled, df['event_1y']),
        'chads_vasc': corr(df['CHADS-Vasc'], df['event_1y'])
    }


def analyze_population(df: pd.DataFrame, population_name: str) -> dict:
    """Analyze a population and return summary statistics."""
    if len(df) == 0:
        return {
            'name': population_name,
            'n_patients': 0,
            'event_rate': 0.0,
            'correlations': {'age': 0.0, 'chads_vasc': 0.0},
            'event_rates_by_score': pd.Series(dtype=float)
        }
    
    # Compute stroke events
    event = compute_event_within_1y(df)
    df = df.copy()
    df['event_1y'] = event
    
    # Ensure CHADS-VASc scores
    if 'CHADS-Vasc' not in df.columns:
        df['CHADS-Vasc'] = df.apply(calculate_chadsvasc, axis=1)
    
    # Compute statistics
    event_rate = float(df['event_1y'].mean()) if len(df) else 0.0
    correlations = compute_correlations(df)
    by_score = df.groupby('CHADS-Vasc')['event_1y'].mean().fillna(0.0)
    
    return {
        'name': population_name,
        'n_patients': len(df),
        'event_rate': event_rate,
        'correlations': correlations,
        'event_rates_by_score': by_score
    }


def compute_stratified_analysis(df: pd.DataFrame, dataset_type: str) -> dict:
    """Perform stratified analysis by anticoagulant use."""
    results = {}
    
    # Overall population
    results['overall'] = analyze_population(df, 'Overall Population')
    
    # Detect anticoagulant use
    anticoag_use = detect_anticoagulant(df, dataset_type)
    
    # Anticoagulant users
    anticoag_df = df[anticoag_use]
    results['anticoag'] = analyze_population(anticoag_df, 'Anticoagulant Users')
    
    # Non-anticoagulant users
    no_anticoag_df = df[~anticoag_use]
    results['no_anticoag'] = analyze_population(no_anticoag_df, 'Non-Anticoagulant Users')
    
    return results


def save_stratified_results(results: dict, data_file: str, results_dir) -> None:
    """Save stratified analysis results to markdown files."""
    
    # Main stratified summary
    out_path = results_dir / 'no_separation_summary_stratified.md'
    with open(out_path, 'w') as f:
        f.write("# No-Separation Analysis Stratified by Anticoagulant Use\n\n")
        f.write(f"**Dataset:** {data_file}\n\n")
        f.write("**Outcome:** Stroke between time1 (AF dx) and 365 days (by dates)\n\n")
        
        # Summary table
        f.write("## Summary Comparison\n\n")
        f.write("| Population | N Patients | Event Rate (%) | Corr(age, event) | Corr(CHADS-VASc, event) |\n")
        f.write("|------------|------------|----------------|------------------|------------------------|\n")
        
        for key, result in results.items():
            f.write(f"| {result['name']} | {result['n_patients']:,} | {result['event_rate']*100:.2f} | {result['correlations']['age']:.4f} | {result['correlations']['chads_vasc']:.4f} |\n")
        
        # Detailed results for each population
        for key, result in results.items():
            f.write(f"\n## {result['name']}\n\n")
            f.write(f"**N patients:** {result['n_patients']:,}\n\n")
            f.write(f"**Overall event rate:** {result['event_rate']*100:.2f}%\n\n")
            f.write(f"**Corr(age, event):** {result['correlations']['age']:.4f}\n\n")
            f.write(f"**Corr(CHADS-VASc, event):** {result['correlations']['chads_vasc']:.4f}\n\n")
            
            if len(result['event_rates_by_score']) > 0:
                f.write("### Event Rate by CHADS-VASc Score\n\n")
                f.write("| Score | Event Rate (%) |\n|-------|-----------------|\n")
                for score in sorted(result['event_rates_by_score'].index.tolist())[:10]:
                    f.write(f"| {score} | {result['event_rates_by_score'].loc[score]*100:.2f} |\n")
            f.write("\n")
    
    print(f"Stratified summary saved to {out_path}")
    
    # Individual population files
    for key, result in results.items():
        if key == 'overall':
            continue
            
        filename = f"no_separation_{key}.md"
        individual_path = results_dir / filename
        
        with open(individual_path, 'w') as f:
            f.write(f"# No-Separation Analysis: {result['name']}\n\n")
            f.write(f"**Dataset:** {data_file}\n\n")
            f.write(f"**Population:** {result['name']}\n\n")
            f.write(f"**N patients:** {result['n_patients']:,}\n\n")
            f.write(f"**Overall event rate:** {result['event_rate']*100:.2f}%\n\n")
            f.write(f"**Corr(age, event):** {result['correlations']['age']:.4f}\n\n")
            f.write(f"**Corr(CHADS-VASc, event):** {result['correlations']['chads_vasc']:.4f}\n\n")
            
            if len(result['event_rates_by_score']) > 0:
                f.write("## Event Rate by CHADS-VASc Score\n\n")
                f.write("| Score | Event Rate (%) |\n|-------|-----------------|\n")
                for score in sorted(result['event_rates_by_score'].index.tolist())[:10]:
                    f.write(f"| {score} | {result['event_rates_by_score'].loc[score]*100:.2f} |\n")
        
        print(f"{result['name']} analysis saved to {individual_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description='Demonstrate no separation for 1Y stroke outcome')
    parser.add_argument('--dataset', choices=['new', 'old', 'new_unscrambled'], default='new',
                        help='Dataset to use (default: new)')
    parser.add_argument('--data-file', type=str,
                        help='Specific data file to use (overrides --dataset)')
    parser.add_argument('--no-filter', action='store_true',
                        help='Skip eligibility filtering')
    parser.add_argument('--stratify-anticoag', action='store_true',
                        help='Perform stratified analysis by anticoagulant use')
    parser.add_argument('--config', type=str,
                        help='Use specific filtering configuration (e.g., AF_FU365_population_new)')
    parser.add_argument('--list-configs', action='store_true',
                        help='List available filtering configurations and exit')
    args = parser.parse_args()

    # List available configurations if requested
    if args.list_configs:
        configs = get_available_configs()
        print("Available filtering configurations:")
        for config in configs:
            print(f"  - {config}")
        return 0

    if args.dataset == 'new_unscrambled':
        data_file = 'random_nuchad_1011.csv'
    elif args.dataset == 'new':
        data_file = 'random_nuchad_250623.csv'
    else:
        data_file = 'random_nuchad.csv'
    # Map dataset type for anticoagulant detection
    dataset_type = 'new' if args.dataset in ['new', 'new_unscrambled'] else 'old'

    print(f"Loading dataset: {data_file}")
    df = get_df(data_file)

    # Apply filtering
    if args.config:
        print(f"Applying filtering configuration: {args.config}")
        try:
            df, filter_stats = filter_patients_from_config(df, f"{args.config}.json")
            print(f"Filtered to {len(df):,} patients using configuration '{args.config}'")
        except Exception as e:
            print(f"Error loading configuration '{args.config}': {e}")
            print("Falling back to default filtering...")
            df, _ = data_utils.filter_eligible_patients(df)
            print(f"Filtered to {len(df):,} patients using default filters")
    elif not args.no_filter:
        print("Applying default eligibility filters...")
        df, _ = data_utils.filter_eligible_patients(df)
        print(f"Filtered to {len(df):,} patients")
    else:
        print(f"Using all {len(df):,} patients (no filtering)")

    # Perform analysis
    results_dir = get_results_dir()
    
    if args.stratify_anticoag:
        # Stratified analysis
        print("\n=== Stratified No-Separation Analysis ===")
        results = compute_stratified_analysis(df, dataset_type)
        
        # Print summary comparison
        print(f"\nSummary Comparison:")
        print(f"{'Population':<25} {'N Patients':<12} {'Event Rate':<12} {'Corr(age)':<10} {'Corr(CHADS)':<12}")
        print("-" * 75)
        for key, result in results.items():
            print(f"{result['name']:<25} {result['n_patients']:<12,} {result['event_rate']*100:<12.2f} {result['correlations']['age']:<10.4f} {result['correlations']['chads_vasc']:<12.4f}")
        
        # Save stratified results
        save_stratified_results(results, data_file, results_dir)
    else:
        # Original single-population analysis
        result = analyze_population(df, 'Overall Population')
        
        # Print concise summary
        print("\n=== No-Separation Demonstration ===")
        print(f"Overall 1Y event rate (by dates): {result['event_rate']*100:.2f}%")
        print(f"Corr(age, event): {result['correlations']['age']:.4f}")
        print(f"Corr(CHADS-VASc, event): {result['correlations']['chads_vasc']:.4f}")
        print("\nEvent rate by CHADS-VASc (first 10 scores):")
        for score in range(10):
            if score in result['event_rates_by_score'].index:
                print(f"  {score}: {result['event_rates_by_score'].loc[score]*100:.2f}%")

        # Save original format summary
        out_path = results_dir / 'no_separation_summary.md'
        with open(out_path, 'w') as f:
            f.write("# No-Separation Demonstration\n\n")
            f.write(f"**Dataset:** {data_file}\n\n")
            f.write("**Outcome:** stroke between time1 (AF dx) and 365 days (by dates)\n\n")
            f.write(f"**N patients:** {result['n_patients']}\n\n")
            f.write(f"**Overall event rate:** {result['event_rate']*100:.2f}%\n\n")
            f.write(f"**Corr(age, event):** {result['correlations']['age']:.4f}\n\n")
            f.write(f"**Corr(CHADS-VASc, event):** {result['correlations']['chads_vasc']:.4f}\n\n")
            f.write("## Event Rate by CHADS-VASc\n\n")
            f.write("| Score | Event Rate (%) |\n|-------|-----------------|\n")
            for score in sorted(result['event_rates_by_score'].index.tolist())[:10]:
                f.write(f"| {score} | {result['event_rates_by_score'].loc[score]*100:.2f} |\n")

        print(f"\nSummary saved to {out_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
