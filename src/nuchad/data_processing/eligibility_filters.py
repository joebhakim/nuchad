"""Utilities for data processing and filtering patients in nuchad package."""

import pandas as pd
import argparse
import sys
from pathlib import Path
import itertools
from typing import Dict, List, Tuple, Optional, Any, Union

from nuchad.utils import get_results_dir

# Removed import from eda to avoid circular dependency

def filter_eligible_patients(
    df: pd.DataFrame,
    require_af: bool = True,
    require_follow_up: bool = True,
    require_stroke: bool = False,
    af_before_time1: bool = True,
    min_follow_up_days: int = 365,
    stroke_window_days: int = 365,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Filter dataframe to eligible patients based on criteria.
    
    Args:
        df: The input dataframe with patient data
        require_af: Whether to require patients to have AF diagnosis
        require_follow_up: Whether to require patients to have sufficient follow-up
        require_stroke: Whether to require patients to have a stroke diagnosis
        af_before_time1: If True, AF must be diagnosed before or at time1
        min_follow_up_days: Minimum follow-up period in days
        stroke_window_days: Window after time1 in which stroke must occur (if required)
        
    Returns:
        Tuple containing:
            - Filtered dataframe with only eligible patients
            - Dictionary with filter statistics for reporting
    """
    # Track each filter step and the number of patients remaining
    filter_stats = {
        "total": len(df),
        "steps": [],
        "filter_masks": {},
        "filter_descriptions": {},
    }
    
    # Start with all patients
    current_df = df.copy()
    
    # Apply AF diagnosis filter if requested
    if require_af:
        filter_description = f"AF diagnosis {'before or at time1' if af_before_time1 else 'any time'}"
        if af_before_time1:
            mask = current_df["af_before_time1"] == True
        else:
            mask = current_df["has_af"] == True
            
        filter_stats["filter_masks"]["af"] = mask
        filter_stats["filter_descriptions"]["af"] = filter_description
        
        patients_before = len(current_df)
        current_df = current_df[mask]
        patients_after = len(current_df)
        
        filter_stats["steps"].append({
            "description": filter_description,
            "remaining": patients_after,
            "removed": patients_before - patients_after,
            "percent_remaining": round(patients_after / filter_stats["total"] * 100, 1)
        })
    
    # Apply follow-up period filter if requested
    if require_follow_up:
        filter_description = f"Follow-up period ≥ {min_follow_up_days} days"
        mask = (current_df["end_fu"] - current_df["time1"]).dt.days >= min_follow_up_days
        
        filter_stats["filter_masks"]["follow_up"] = mask
        filter_stats["filter_descriptions"]["follow_up"] = filter_description
        
        patients_before = len(current_df)
        current_df = current_df[mask]
        patients_after = len(current_df)
        
        filter_stats["steps"].append({
            "description": filter_description,
            "remaining": patients_after,
            "removed": patients_before - patients_after,
            "percent_remaining": round(patients_after / filter_stats["total"] * 100, 1)
        })
    
    # Apply stroke diagnosis filter if requested
    if require_stroke:
        filter_description = f"Stroke within {stroke_window_days} days after time1"
        
        # Calculate days between stroke and time1
        # Make sure to handle missing stroke dates
        stroke_to_time1_days = pd.Series(float('nan'), index=current_df.index)
        mask_has_stroke = ~current_df["first_stroke_date"].isna()
        stroke_to_time1_days[mask_has_stroke] = (
            current_df.loc[mask_has_stroke, "first_stroke_date"] - 
            current_df.loc[mask_has_stroke, "time1"]
        ).dt.days
        
        # Apply the stroke window filter
        mask = (stroke_to_time1_days >= 0) & (stroke_to_time1_days <= stroke_window_days)
        
        filter_stats["filter_masks"]["stroke"] = mask
        filter_stats["filter_descriptions"]["stroke"] = filter_description
        
        patients_before = len(current_df)
        current_df = current_df[mask]
        patients_after = len(current_df)
        
        filter_stats["steps"].append({
            "description": filter_description,
            "remaining": patients_after,
            "removed": patients_before - patients_after,
            "percent_remaining": round(patients_after / filter_stats["total"] * 100, 1)
        })
    
    return current_df, filter_stats


def generate_filter_report(
    filter_stats: Dict[str, Any], 
    output_path: Optional[Union[str, Path]] = None,
    max_num_to_print_permutations: int = 4
) -> str:
    """Generate a report of the filter steps and statistics.
    
    Args:
        filter_stats: Dictionary with filter statistics from filter_eligible_patients()
        output_path: Path to save the report to, defaults to results/filter_report.md
        max_num_to_print_permutations: Maximum number of filters for which to print all combinations
        
    Returns:
        The report text as a string
    """
    # Create the report header
    report = [
        "# Patient Eligibility Filtering Report\n",
        f"Total patients at start: {filter_stats['total']}\n",
        "## Filter Steps\n",
        "| Step | Patients Remaining | Patients Removed | % of Original Cohort |\n",
        "|------|-------------------|------------------|----------------------|\n",
        f"| Initial cohort | {filter_stats['total']} | 0 | 100.0% |\n"
    ]
    
    # Add each filter step
    for step in filter_stats["steps"]:
        report.append(
            f"| {step['description']} | {step['remaining']} | {step['removed']} | {step['percent_remaining']}% |\n"
        )
    
    # Add combinatorial analysis if we have fewer than the max permitted filters
    if 'filter_masks' in filter_stats and len(filter_stats['filter_masks']) <= max_num_to_print_permutations:
        report.append("\n## Combinatorial Filter Analysis\n")
        report.append("| Filters Applied | Patients Remaining | % of Original Cohort |\n")
        report.append("|-----------------|-------------------|----------------------|\n")
        
        # Start with the full dataframe for reference
        report.append(f"| None (all patients) | {filter_stats['total']} | 100.0% |\n")
        
        # Get all combinations of filters
        filter_names = list(filter_stats['filter_masks'].keys())
        
        # For each length of combination (1, 2, ..., all filters)
        for r in range(1, len(filter_names) + 1):
            # For each specific combination of that length
            for combo in itertools.combinations(filter_names, r):
                # Create a description of this combination
                filter_desc = " + ".join([filter_stats['filter_descriptions'][f] for f in combo])
                
                # Combine the masks for this combination
                # Start with all True mask to capture all patients
                full_df_size = filter_stats['total']
                combined_mask = pd.Series(True, index=range(full_df_size))
                
                # Apply each filter mask
                for filter_name in combo:
                    combined_mask = combined_mask & filter_stats['filter_masks'][filter_name]
                
                # Count how many patients remain
                remaining = combined_mask.sum()
                percent = round(remaining / full_df_size * 100, 1)
                
                # Add to the report
                report.append(f"| {filter_desc} | {remaining} | {percent}% |\n")
    
    # Join the report into a single string
    report_text = "".join(report)
    
    # Save the report if an output path is provided
    if output_path is not None:
        path_obj: Path
        
        # Handle paths - if not absolute, put in results dir
        if isinstance(output_path, str):
            if not output_path.startswith('/') and not output_path.startswith('./') and not output_path.startswith('../'):
                path_obj = get_results_dir() / output_path
            else:
                path_obj = Path(output_path)
        else:
            # output_path is already a Path object
            path_obj = output_path
            
        # Create the directory if it doesn't exist
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path_obj, "w") as f:
            f.write(report_text)
    
    return report_text


def main():
    """
    Command-line interface for filtering patients and generating a report.
    """
    parser = argparse.ArgumentParser(
        description="Filter eligible patients from the dataset and generate a filtering report"
    )
    
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
    
    # Import here to avoid circular import
    from nuchad.analysis.eda import get_df
    
    # Load the data
    df = get_df()
    
    # Set default arguments if none are provided
    if not any([args.require_af, args.require_follow_up, args.require_stroke]):
        args.require_af = True
        args.require_follow_up = True
        args.af_before_time1 = True
    
    # Filter patients
    filtered_df, filter_stats = filter_eligible_patients(
        df,
        require_af=args.require_af,
        require_follow_up=args.require_follow_up,
        require_stroke=args.require_stroke,
        af_before_time1=args.af_before_time1,
        min_follow_up_days=args.min_follow_up_days,
        stroke_window_days=args.stroke_window_days
    )
    
    # Generate and save the report
    output_path = args.output
    report = generate_filter_report(
        filter_stats, 
        output_path, 
        max_num_to_print_permutations=args.max_combinations
    )
    
    # Get the resolved path for output message
    if output_path is None:
        results_dir = get_results_dir()
        output_path = results_dir / "filter_report.md"
    
    print(f"Filtering complete. Eligible patients: {len(filtered_df)} out of {filter_stats['total']}")
    print(f"Report saved to: {output_path}")
    
    # Print counts of key patient characteristics
    if len(filtered_df) > 0:
        print("\nFiltered patient characteristics:")
        if 'gender' in filtered_df.columns:
            gender_counts = filtered_df['gender'].value_counts()
            print(f"- Gender: {gender_counts.get(1, 0)} male, {gender_counts.get(2, 0)} female")
        
        if 'age' in filtered_df.columns:
            print(f"- Age: {filtered_df['age'].mean():.1f} years (mean), {filtered_df['age'].median():.1f} years (median)")
        
        if 'stroke_1Y' in filtered_df.columns:
            stroke_counts = filtered_df['stroke_1Y'].value_counts()
            print(f"- Stroke within 1 year: {stroke_counts.get(1, 0)} yes, {stroke_counts.get(2, 0)} no")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 