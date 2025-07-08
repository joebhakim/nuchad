#!/usr/bin/env python3
"""Generate follow-up time visualization."""

import sys
from nuchad.analysis import eda_old
from nuchad.visualization import visualize_end_fu
from nuchad.data_processing import eligibility_filters as data_utils

def main():
    """Generate follow-up time visualization."""
    print("Generating follow-up time visualization...")
    
    # Load and filter data
    df = eda_old.get_df()
    eligible_df, _ = data_utils.filter_eligible_patients(df)
    
    # Generate visualization
    visualize_end_fu.plot_end_fu_distribution(eligible_df)
    
    print("Follow-up time visualization has been generated")
    return 0

if __name__ == "__main__":
    sys.exit(main())