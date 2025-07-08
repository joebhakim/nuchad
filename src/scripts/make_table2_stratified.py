#!/usr/bin/env python3
"""Generate Table 2 stratified by CHADS-VASc risk groups and anticoagulation."""

import sys
from nuchad.analysis import eda, table2_stratified
from nuchad.data_processing import eligibility_filters as data_utils

def main():
    """Generate Table 2 stratified by CHADS-VASc risk groups and anticoagulation."""
    print("Generating Table 2 stratified by CHADS-VASc risk groups and anticoagulation...")
    
    # Load and filter data
    df = eda.get_df()
    eligible_df, _ = data_utils.filter_eligible_patients(df)
    
    # Generate table
    table2_stratified.generate_stratified_table2(eligible_df)
    
    print("Stratified Table 2 has been generated")
    return 0

if __name__ == "__main__":
    print("THIS IS SUPPOSED TO BE RUN AS A SCRIPT, NOT DIRECTLY, USE SOMETHING LIKE UV RUN")
    sys.exit(main())