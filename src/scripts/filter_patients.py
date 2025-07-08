#!/usr/bin/env python3
"""Generate patient filtering report."""

import sys
import argparse
from pathlib import Path
from nuchad.analysis import eda
from nuchad.data_processing import eligibility_filters as data_utils
from nuchad.utils import get_results_dir

def main():
    """Generate patient filtering report."""
    parser = argparse.ArgumentParser(
        description="Generate patient filtering report with customizable criteria"
    )
    
    # Add filter-specific arguments
    parser.add_argument(
        "--require-af",
        action="store_true",
        help="Require patients to have AF diagnosis",
    )
    
    parser.add_argument(
        "--require-follow-up",
        action="store_true",
        help="Require patients to have sufficient follow-up",
    )
    
    parser.add_argument(
        "--require-stroke",
        action="store_true",
        help="Require patients to have a stroke diagnosis",
    )
    
    parser.add_argument(
        "--af-before-time1",
        action="store_true",
        help="If set, AF must be diagnosed before or at time1",
    )
    
    parser.add_argument(
        "--min-follow-up-days",
        type=int,
        default=365,
        help="Minimum follow-up period in days",
    )
    
    parser.add_argument(
        "--stroke-window-days",
        type=int,
        default=365,
        help="Window after time1 in which stroke must occur (if required)",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for the report (defaults to results/filter_report.md)",
    )
    
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=4,
        help="Maximum number of filters for which to print all combinations",
    )
    
    args = parser.parse_args()
    
    # Set default arguments if none are provided
    if not any([args.require_af, args.require_follow_up, args.require_stroke]):
        args.require_af = True
        args.require_follow_up = True
        args.af_before_time1 = True
        
    print("Generating patient filtering report...")
    
    # Load data
    df = eda.get_df()
    
    # Run filtering
    filtered_df, filter_stats = data_utils.filter_eligible_patients(
        df,
        require_af=args.require_af,
        require_follow_up=args.require_follow_up,
        require_stroke=args.require_stroke,
        af_before_time1=args.af_before_time1,
        min_follow_up_days=args.min_follow_up_days,
        stroke_window_days=args.stroke_window_days
    )
    
    # Determine output path
    output_path = args.output
    
    # Generate the report
    report = data_utils.generate_filter_report(
        filter_stats, 
        output_path,
        max_num_to_print_permutations=args.max_combinations
    )
    
    # Get the resolved path for output message
    if output_path is None:
        output_path = get_results_dir() / "filter_report.md"
    elif not Path(output_path).is_absolute() and not str(output_path).startswith('./') and not str(output_path).startswith('../'):
        output_path = get_results_dir() / output_path
    
    print(f"Filtering complete. Eligible patients: {len(filtered_df)} out of {filter_stats['total']}")
    print(f"Report saved to: {output_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())