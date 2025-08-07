#!/usr/bin/env python3
"""Run survival-focused exploratory data analysis."""

import sys
import argparse
from nuchad.analysis.eda import run_survival_eda, run_survival_eda_all_datasets
from nuchad.data_processing.eligibility_filters import get_available_configs

def main():
    """Run survival-focused exploratory data analysis."""
    
    # Get available filter configurations
    available_configs = get_available_configs()
    configs_text = "Available filter configurations: " + ", ".join(available_configs) if available_configs else "No filter configurations found"
    
    parser = argparse.ArgumentParser(
        description="Run survival-focused EDA with Kaplan-Meier curves and timeline visualizations",
        epilog=f"{configs_text}\n\nExample usage:\n"
               f"  {sys.argv[0]} --all-datasets --filter-config AF_FU365_population_new\n"
               f"  {sys.argv[0]} -d random_nuchad.csv --filter-config AF_FU90_nostroke_both",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--data-file", "-d", type=str, default=None,
                       help="Specific data file to analyze (e.g., random_nuchad.csv)")
    parser.add_argument("--all-datasets", "-a", action="store_true",
                       help="Run analysis on all available datasets")
    parser.add_argument("--filter-config", "-f", type=str, default=None,
                       help="JSON filter configuration to use (without .json extension). "
                            "If not specified, uses default filtering criteria.")
    parser.add_argument("--list-configs", "-l", action="store_true",
                       help="List available filter configurations and exit")
    parser.add_argument("--no-pre-filter", action="store_true",
                       help="Skip pre-filter analysis")
    parser.add_argument("--no-post-filter", action="store_true",
                       help="Skip post-filter analysis")
    parser.add_argument("--sample-size", "-s", type=int, default=100,
                       help="Number of patients to sample for timeline visualization (default: 100)")
    
    args = parser.parse_args()
    
    # Handle list configs option
    if args.list_configs:
        print("Available filter configurations:")
        if available_configs:
            for i, config in enumerate(available_configs, 1):
                print(f"  {i:2d}. {config}")
        else:
            print("  No configurations found in filtering_configs/ directory")
        return 0
    
    print("Running survival-focused exploratory data analysis...")
    print("This will generate Kaplan-Meier curves, timeline visualizations, and survival statistics.")
    print("Outputs include HTML metadata with data file info and filtering steps in JSON format.")
    
    if args.filter_config:
        if args.filter_config not in available_configs:
            print(f"\nWARNING: Filter configuration '{args.filter_config}' not found!")
            print(f"Available configurations: {', '.join(available_configs)}")
            print("Proceeding anyway (will fall back to default filtering)...")
        else:
            print(f"\nUsing filter configuration: {args.filter_config}")
    
    no_pre_filter = not args.no_pre_filter
    no_post_filter = not args.no_post_filter
    
    if args.all_datasets:
        print("\n=== Running analysis on ALL datasets ===")
        run_survival_eda_all_datasets(
            no_pre_filter=no_pre_filter, 
            no_post_filter=no_post_filter, 
            sample_size=args.sample_size,
            filter_config=args.filter_config
        )
    elif args.data_file:
        print(f"\n=== Running analysis on {args.data_file} ===")
        run_survival_eda(
            data_file=args.data_file, 
            no_pre_filter=no_pre_filter, 
            no_post_filter=no_post_filter, 
            sample_size=args.sample_size,
            filter_config=args.filter_config
        )
    else:
        raise ValueError("No data file specified! Either use --all-datasets or --data-file <filename>")
    
    print("Survival EDA completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
