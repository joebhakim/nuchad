#!/usr/bin/env python3
"""Generate Table 1 stratified by CHADS-VASc risk groups."""

import sys
from nuchad.analysis import eda_old, table1_stratified
from nuchad.data_processing import eligibility_filters as data_utils

def main():
    """Generate Table 1 stratified by CHADS-VASc risk groups."""
    print("Generating Table 1 stratified by CHADS-VASc risk groups...")
    
    # Load and filter data
    df = eda_old.get_df()
    eligible_df, _ = data_utils.filter_eligible_patients(df)
    
    # Generate table
    table1_stratified.generate_stratified_table1(eligible_df)
    
    print("Stratified Table 1 has been generated")
    return 0

if __name__ == "__main__":
    sys.exit(main())