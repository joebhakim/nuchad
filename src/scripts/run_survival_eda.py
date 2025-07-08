#!/usr/bin/env python3
"""Run survival-focused exploratory data analysis."""

import sys
from nuchad.analysis.eda import run_survival_eda

def main():
    """Run survival-focused exploratory data analysis."""
    print("Running survival-focused exploratory data analysis...")
    print("This will generate Kaplan-Meier curves, timeline visualizations, and survival statistics.")
    
    # Run survival EDA with default parameters
    run_survival_eda(pre_filter=True, post_filter=True, sample_size=100)
    
    print("Survival EDA completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
