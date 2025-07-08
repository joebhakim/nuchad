#!/usr/bin/env python3
"""Generate comprehensive patient filtering report using JSON configurations."""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from nuchad.analysis import eda
from nuchad.data_processing import eligibility_filters as data_utils
from nuchad.utils import get_results_dir

def generate_variable_distribution_table(df_original, df_filtered, filter_stats):
    """Generate a condensed variable distribution table comparing original vs filtered data.
    
    Args:
        df_original: Original unfiltered DataFrame
        df_filtered: Filtered DataFrame
        filter_stats: Filter statistics dictionary
        
    Returns:
        String containing the formatted table
    """
    # Identify numeric and categorical columns
    numeric_cols = df_original.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_original.select_dtypes(include=['object', 'datetime']).columns.tolist()
    
    # Remove problematic columns
    cols_to_remove = ['patient_id', 'Unnamed: 0']
    numeric_cols = [col for col in numeric_cols if col not in cols_to_remove]
    categorical_cols = [col for col in categorical_cols if col not in cols_to_remove]
    
    report_lines = []
    
    # Numeric columns
    if numeric_cols:
        report_lines.append("### Numeric Variables\n")
        report_lines.append("| Variable | Original | Filtered | Change |\n")
        report_lines.append("|----------|----------|----------|--------|\n")
        
        for col in numeric_cols[:15]:  # Limit to first 15 columns
            if col in df_original.columns and col in df_filtered.columns:
                orig_mean = df_original[col].mean()
                orig_std = df_original[col].std()
                orig_min = df_original[col].min()
                orig_max = df_original[col].max()
                
                filt_mean = df_filtered[col].mean()
                filt_std = df_filtered[col].std()
                filt_min = df_filtered[col].min()
                filt_max = df_filtered[col].max()
                
                orig_desc = f"{orig_mean:.2f} ± {orig_std:.2f} [{orig_min:.2f}, {orig_max:.2f}]"
                filt_desc = f"{filt_mean:.2f} ± {filt_std:.2f} [{filt_min:.2f}, {filt_max:.2f}]"
                
                # Calculate percent change in mean
                if orig_mean != 0:
                    change = ((filt_mean - orig_mean) / orig_mean) * 100
                    change_str = f"{change:+.1f}%"
                else:
                    change_str = "N/A"
                
                report_lines.append(f"| {col} | {orig_desc} | {filt_desc} | {change_str} |\n")
    
    # Categorical columns
    if categorical_cols:
        report_lines.append("\n### Categorical Variables\n")
        report_lines.append("| Variable | Original | Filtered | Change |\n")
        report_lines.append("|----------|----------|----------|--------|\n")
        
        for col in categorical_cols[:10]:  # Limit to first 10 columns
            if col in df_original.columns and col in df_filtered.columns:
                orig_unique = df_original[col].nunique()
                orig_missing = df_original[col].isnull().sum()
                orig_missing_pct = (orig_missing / len(df_original)) * 100
                
                filt_unique = df_filtered[col].nunique()
                filt_missing = df_filtered[col].isnull().sum()
                filt_missing_pct = (filt_missing / len(df_filtered)) * 100
                
                orig_desc = f"Unique: {orig_unique} | Missing: {orig_missing} ({orig_missing_pct:.1f}%)"
                filt_desc = f"Unique: {filt_unique} | Missing: {filt_missing} ({filt_missing_pct:.1f}%)"
                
                # Calculate change in missing percentage
                change = filt_missing_pct - orig_missing_pct
                change_str = f"{change:+.1f}pp"
                
                report_lines.append(f"| {col} | {orig_desc} | {filt_desc} | {change_str} |\n")
    
    return "".join(report_lines)

def generate_enhanced_filter_report(df_original, df_filtered, filter_stats, config_info=None):
    """Generate an enhanced filtering report with variable distributions.
    
    Args:
        df_original: Original unfiltered DataFrame
        df_filtered: Filtered DataFrame
        filter_stats: Filter statistics dictionary
        config_info: Configuration information dictionary
        
    Returns:
        String containing the full report
    """
    report_lines = []
    
    # Header
    report_lines.append("# Patient Filtering Report\n\n")
    
    # Configuration information
    if config_info:
        report_lines.append("## Configuration\n\n")
        report_lines.append(f"**Configuration**: {config_info['name']}\n\n")
        report_lines.append(f"**Description**: {config_info['description']}\n\n")
        if 'use_case' in config_info.get('metadata', {}):
            report_lines.append(f"**Use Case**: {config_info['metadata']['use_case']}\n\n")
        if 'notes' in config_info.get('metadata', {}):
            report_lines.append(f"**Notes**: {config_info['metadata']['notes']}\n\n")
    
    # Patient flow
    report_lines.append("## Patient Flow\n\n")
    report_lines.append(f"**Total patients at start**: {filter_stats['total']:,}\n")
    report_lines.append(f"**Final eligible patients**: {len(df_filtered):,}\n")
    report_lines.append(f"**Overall retention rate**: {len(df_filtered) / filter_stats['total'] * 100:.1f}%\n\n")
    
    # Filtering steps
    report_lines.append("## Filtering Steps\n\n")
    report_lines.append("| Step | Patients Remaining | Patients Removed | % of Original |\n")
    report_lines.append("|------|-------------------|------------------|---------------|\n")
    report_lines.append(f"| Initial cohort | {filter_stats['total']:,} | 0 | 100.0% |\n")
    
    for step in filter_stats["steps"]:
        report_lines.append(
            f"| {step['description']} | {step['remaining']:,} | {step['removed']:,} | {step['percent_remaining']}% |\n"
        )
    
    # Variable distributions
    report_lines.append("\n## Variable Distributions\n\n")
    report_lines.append("Comparison of variable distributions before and after filtering.\n\n")
    
    distribution_table = generate_variable_distribution_table(df_original, df_filtered, filter_stats)
    report_lines.append(distribution_table)
    
    # Key characteristics
    report_lines.append("\n## Key Patient Characteristics\n\n")
    
    if 'age' in df_filtered.columns:
        report_lines.append(f"- **Age**: {df_filtered['age'].mean():.1f} ± {df_filtered['age'].std():.1f} years\n")
    
    if 'gender' in df_filtered.columns:
        gender_counts = df_filtered['gender'].value_counts()
        male_count = gender_counts.get(1, 0)
        female_count = gender_counts.get(2, 0)
        total_gender = male_count + female_count
        if total_gender > 0:
            male_pct = (male_count / total_gender) * 100
            female_pct = (female_count / total_gender) * 100
            report_lines.append(f"- **Gender**: {male_count:,} male ({male_pct:.1f}%), {female_count:,} female ({female_pct:.1f}%)\n")
    
    if 'stroke_1Y' in df_filtered.columns:
        stroke_counts = df_filtered['stroke_1Y'].value_counts().sort_index()
        report_lines.append("- **Stroke outcomes**:\n")
        for value, count in stroke_counts.items():
            pct = (count / len(df_filtered)) * 100
            report_lines.append(f"  - stroke_1Y={value}: {count:,} ({pct:.1f}%)\n")
    
    # Combinatorial analysis if applicable
    if 'filter_masks' in filter_stats and len(filter_stats['filter_masks']) <= 4:
        report_lines.append("\n## Combinatorial Filter Analysis\n\n")
        report_lines.append("Effects of applying different combinations of filters.\n\n")
        report_lines.append("| Filters Applied | Patients Remaining | % of Original |\n")
        report_lines.append("|-----------------|-------------------|---------------|\n")
        report_lines.append(f"| None (all patients) | {filter_stats['total']:,} | 100.0% |\n")
        
        # Get all combinations of filters
        import itertools
        filter_names = list(filter_stats['filter_masks'].keys())
        
        for r in range(1, len(filter_names) + 1):
            for combo in itertools.combinations(filter_names, r):
                filter_desc = " + ".join([filter_stats['filter_descriptions'][f] for f in combo])
                
                # Combine the masks
                combined_mask = pd.Series(True, index=range(filter_stats['total']))
                for filter_name in combo:
                    combined_mask = combined_mask & filter_stats['filter_masks'][filter_name]
                
                remaining = combined_mask.sum()
                percent = (remaining / filter_stats['total']) * 100
                
                report_lines.append(f"| {filter_desc} | {remaining:,} | {percent:.1f}% |\n")
    
    return "".join(report_lines)

def main():
    """Generate comprehensive patient filtering report using JSON configurations."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive patient filtering report using JSON configurations"
    )
    
    parser.add_argument(
        "config",
        type=str,
        help="Configuration file name (without .json extension) or path to JSON file"
    )
    
    parser.add_argument(
        "--dataset",
        choices=["original", "new"],
        default="original",
        help="Which dataset to use"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for the report (defaults to results/filter_report_<config>.md)"
    )
    
    args = parser.parse_args()
    
    print(f"Generating patient filtering report using configuration: {args.config}")
    
    # Load data
    if args.dataset == "new":
        df = eda.get_df(data_file="random_nuchad_250623.csv")
    else:
        df = eda.get_df()
    
    # Store original dataframe for comparison
    df_original = df.copy()
    
    # Load configuration and apply filtering
    config_file = args.config if args.config.endswith('.json') else f"{args.config}.json"
    try:
        config_info = data_utils.load_filtering_config(config_file)
        filtered_df, filter_stats = data_utils.filter_patients_from_config(df, config_file)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        config_name = args.config.replace('.json', '')
        output_path = get_results_dir() / f"filter_report_{config_name}.md"
    
    # Make sure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate enhanced report
    report = generate_enhanced_filter_report(
        df_original, 
        filtered_df, 
        filter_stats, 
        config_info
    )
    
    # Save the report
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Filtering complete. Eligible patients: {len(filtered_df):,} out of {filter_stats['total']:,}")
    print(f"Report saved to: {output_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())