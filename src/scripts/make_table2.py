#!/usr/bin/env python3
"""Generate Table 2 - Stroke rates by CHADS-VASc score."""

import sys
from nuchad.analysis import eda, table2
from nuchad.data_processing import eligibility_filters as data_utils

def main():
    """Generate Table 2 with stroke rates by CHADS-VASc score."""
    print("Generating Table 2 - Stroke rates by CHADS-VASc score...")
    
    # Load and filter data
    df = eda.get_df()
    eligible_df, _ = data_utils.filter_eligible_patients(df)
    
    # Generate table
    table2.generate_cohort_table(eligible_df)
    
    print("Table 2 has been generated")
    return 0

if __name__ == "__main__":
    print("THIS IS SUPPOSED TO BE RUN AS A SCRIPT, NOT DIRECTLY, USE SOMETHING LIKE UV RUN")
    sys.exit(main())