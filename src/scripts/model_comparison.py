#!/usr/bin/env python3
"""Perform model comparison analysis."""

import sys
import argparse
import numpy as np
from nuchad.analysis import eda, model_comparison
from nuchad.data_processing import eligibility_filters as data_utils

def main():
    """Perform model comparison analysis."""
    parser = argparse.ArgumentParser(description='Model comparison analysis')
    parser.add_argument('--dataset', choices=['original', 'new'], default='original',
                       help='Which dataset to use')
    parser.add_argument('--config', type=str, default=None,
                       help='Filtering configuration file name (without .json extension)')
    args = parser.parse_args()
    
    print(f"Performing model comparison analysis on {args.dataset} dataset...")
    
    # Load and filter data
    if args.dataset == 'new':
        # Use the new dataset
        df = eda.get_df(data_file="random_nuchad_250623.csv")
    else:
        df = eda.get_df()
    
    # Apply filtering
    if args.config:
        # Use specified configuration
        config_file = f"{args.config}.json"
        eligible_df, filter_stats = data_utils.filter_patients_from_config(df, config_file)
        print(f"Using filtering configuration: {args.config}")
    else:
        # Use default filtering
        eligible_df, filter_stats = data_utils.filter_eligible_patients(df)
        print("Using default filtering")
    
    print(f"Filtered to {len(eligible_df):,} eligible patients")
    
    # Perform comparison
    results = model_comparison.perform_model_comparison(eligible_df, filter_stats=filter_stats)
    auc_diff = results['auc_model'] - results['auc_score']
    
    print(f"Model comparison completed. AUC difference: {auc_diff:+.3f}")
    
    # Print coefficients for comparison
    print("\n=== MODEL COEFFICIENTS ===")
    for feature, coef in zip(results['feature_names'], results['coefficients']):
        or_val = np.exp(coef)
        print(f"{feature:15s}: {coef:+.3f} (OR: {or_val:.3f})")
    print(f"Intercept: {results['intercept']:+.3f}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())