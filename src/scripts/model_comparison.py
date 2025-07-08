#!/usr/bin/env python3
"""Perform model comparison analysis."""

import sys
from nuchad.analysis import eda, model_comparison
from nuchad.data_processing import eligibility_filters as data_utils

def main():
    """Perform model comparison analysis."""
    print("Performing model comparison analysis...")
    
    # Load and filter data
    df = eda.get_df()
    eligible_df, _ = data_utils.filter_eligible_patients(df)
    
    # Perform comparison
    results = model_comparison.perform_model_comparison(eligible_df)
    auc_diff = results['auc_model'] - results['auc_score']
    
    print(f"Model comparison completed. AUC difference: {auc_diff:+.3f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())