#!/usr/bin/env python3
"""Run survival-focused exploratory data analysis."""

import sys
import argparse
from nuchad.analysis.eda import run_survival_eda, run_survival_eda_all_datasets

def main():
    """Run survival-focused exploratory data analysis."""
    parser = argparse.ArgumentParser(description="Run survival-focused EDA with Kaplan-Meier curves and timeline visualizations")
    parser.add_argument("--data-file", "-d", type=str, default=None,
                       help="Specific data file to analyze (e.g., random_nuchad.csv)")
    parser.add_argument("--all-datasets", "-a", action="store_true",
                       help="Run analysis on all available datasets")
    parser.add_argument("--no-pre-filter", action="store_true",
                       help="Skip pre-filter analysis")
    parser.add_argument("--no-post-filter", action="store_true",
                       help="Skip post-filter analysis")
    parser.add_argument("--sample-size", "-s", type=int, default=100,
                       help="Number of patients to sample for timeline visualization (default: 100)")
    
    args = parser.parse_args()
    
    print("Running survival-focused exploratory data analysis...")
    print("This will generate Kaplan-Meier curves, timeline visualizations, and survival statistics.")
    print("Outputs include HTML metadata with data file info and filtering steps in JSON format.")
    
    pre_filter = not args.no_pre_filter
    post_filter = not args.no_post_filter
    
    if args.all_datasets:
        print("\n=== Running analysis on ALL datasets ===")
        run_survival_eda_all_datasets(pre_filter=pre_filter, post_filter=post_filter, sample_size=args.sample_size)
    elif args.data_file:
        print(f"\n=== Running analysis on {args.data_file} ===")
        run_survival_eda(data_file=args.data_file, pre_filter=pre_filter, post_filter=post_filter, sample_size=args.sample_size)
    else:
        # Default behavior - run on all datasets
        print("\n=== Running analysis on ALL datasets (default behavior) ===")
        run_survival_eda_all_datasets(pre_filter=pre_filter, post_filter=post_filter, sample_size=args.sample_size)
    
    print("Survival EDA completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
