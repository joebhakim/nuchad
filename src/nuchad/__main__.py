"""Command line interface for the nuchad package."""

import argparse
import sys
from pathlib import Path

from nuchad.analysis import eda, table1, table2, table1_stratified, table2_stratified, density_ratio_reweighting
from nuchad.visualization import visualize_end_fu
from nuchad.data_processing import eligibility_filters as data_utils
from nuchad.utils import get_results_dir

def main():
    """Main entry point for the package."""
    parser = argparse.ArgumentParser(
        description="CHADS-VASc Analysis Package for stroke risk prediction in AF patients"
    )
    
    parser.add_argument(
        "--task",
        choices=["eda", "table1", "table2", "table1_stratified", "table2_stratified", "visualize", "reweight", "filter"],
        required=True,
        help="Analysis task to perform",
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
    
    # Run the requested analysis task
    if args.task == "filter":
        # Set default arguments if none are provided
        if not any([args.require_af, args.require_follow_up, args.require_stroke]):
            args.require_af = True
            args.require_follow_up = True
            args.af_before_time1 = True
            
        # Run the filtering utility directly
        df = eda.get_df()
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
    
    # Load the data once for other tasks
    df = eda.get_df()
    eligible_df, _ = data_utils.filter_eligible_patients(df)
    
    # Run the requested analysis task
    if args.task == "eda":
        results = eda.validate_chadsvasc(eligible_df, "time1", "end_fu")
        print(results.to_markdown(index=False))
    
    elif args.task == "table1":
        # Process from table1.py
        table = table1.create_table1(eligible_df)
        results_dir = get_results_dir()
        print(f"Table 1 has been generated and saved to {results_dir / 'table1.md'}")
    
    elif args.task == "table2":
        # Process from table2.py
        table2.generate_cohort_table(eligible_df)
        print("Table 2 has been generated")
    
    elif args.task == "table1_stratified":
        # Process from table1_stratified.py
        table1_stratified.generate_stratified_table1(eligible_df)
        print("Stratified Table 1 has been generated")
    
    elif args.task == "table2_stratified":
        # Process from table2_stratified.py
        table2_stratified.generate_stratified_table2(eligible_df)
        print("Stratified Table 2 has been generated")
    
    elif args.task == "visualize":
        # Process from visualize_end_fu.py
        visualize_end_fu.plot_end_fu_distribution(eligible_df)
        print("Follow-up time visualization has been generated")
    
    elif args.task == "reweight":
        # Process from density_ratio_reweighting.py
        density_ratio_reweighting.perform_reweighting_analysis(eligible_df)
        print("Reweighting analysis has been completed")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 