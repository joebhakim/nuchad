#!/usr/bin/env python3
"""Perform density ratio reweighting analysis."""

import sys
from nuchad.analysis import density_ratio_reweighting, eda_old
from nuchad.data_processing import eligibility_filters as data_utils

def main():
    """Perform density ratio reweighting analysis."""
    print("Performing density ratio reweighting analysis...")
    
    # Load and filter data
    df = eda_old.get_df()
    eligible_df, _ = data_utils.filter_eligible_patients(df)
    
    # Perform reweighting
    density_ratio_reweighting.perform_reweighting_analysis(eligible_df)
    
    print("Reweighting analysis has been completed")
    return 0

if __name__ == "__main__":
    sys.exit(main())