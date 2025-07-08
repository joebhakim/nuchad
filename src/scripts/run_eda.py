#!/usr/bin/env python3
"""Run exploratory data analysis and CHADS-VASc validation."""

import sys
from nuchad.analysis import eda_old
from nuchad.data_processing import eligibility_filters as data_utils

def main():
    """Run exploratory data analysis and CHADS-VASc validation."""
    print("Running exploratory data analysis and CHADS-VASc validation...")
    
    # Load and filter data
    df = eda_old.get_df()
    eligible_df, _ = data_utils.filter_eligible_patients(df)
    
    # Run validation
    results = eda_old.validate_chadsvasc(eligible_df, "time1", "end_fu")
    print(results.to_markdown(index=False))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())