#!/usr/bin/env python3
"""Compare two datasets to understand their structure and distributions."""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from nuchad.utils import get_data_path

def load_dataset(filename):
    """Load a dataset with basic preprocessing."""
    data_path = get_data_path() / filename
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"Loading {filename} from: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Basic cleanup
    if 'patid' in df.columns:
        df = df.rename(columns={"patid": "patient_id"})
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {len(df.columns)}")
    
    return df

def analyze_column(series, col_name):
    """Analyze a single column and return summary statistics."""
    result = {
        'column': col_name,
        'dtype': str(series.dtype),
        'missing': series.isnull().sum(),
        'missing_pct': (series.isnull().sum() / len(series)) * 100,
        'unique': series.nunique()
    }
    
    if pd.api.types.is_numeric_dtype(series):
        result['type'] = 'numeric'
        result['mean'] = series.mean()
        result['std'] = series.std()
        result['min'] = series.min()
        result['max'] = series.max()
        result['summary'] = f"{result['mean']:.2f} ± {result['std']:.2f} [{result['min']:.2f}, {result['max']:.2f}]"
    else:
        result['type'] = 'categorical'
        # Get top values
        value_counts = series.value_counts()
        top_values = value_counts.head(3)
        result['top_values'] = dict(top_values)
        
        # Create summary string
        if len(top_values) > 0:
            top_str = ", ".join([f"{val}: {count} ({count/len(series)*100:.1f}%)" for val, count in top_values.items()])
            result['summary'] = f"Top: {top_str}"
        else:
            result['summary'] = "No values"
    
    return result

def compare_datasets(df1, df2, name1, name2):
    """Compare two datasets and return a summary."""
    print(f"\n=== DATASET COMPARISON: {name1} vs {name2} ===")
    
    # Basic comparison
    print(f"\nShape comparison:")
    print(f"  {name1}: {df1.shape}")
    print(f"  {name2}: {df2.shape}")
    
    # Column comparison
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    common_cols = cols1 & cols2
    only_in_1 = cols1 - cols2
    only_in_2 = cols2 - cols1
    
    print(f"\nColumn comparison:")
    print(f"  Common columns: {len(common_cols)}")
    print(f"  Only in {name1}: {len(only_in_1)} - {list(only_in_1) if only_in_1 else 'None'}")
    print(f"  Only in {name2}: {len(only_in_2)} - {list(only_in_2) if only_in_2 else 'None'}")
    
    # Analyze common columns
    if common_cols:
        print(f"\n=== COMMON COLUMNS ANALYSIS ===")
        
        comparison_data = []
        
        for col in sorted(common_cols):
            if col in df1.columns and col in df2.columns:
                analysis1 = analyze_column(df1[col], col)
                analysis2 = analyze_column(df2[col], col)
                
                comparison_data.append({
                    'column': col,
                    'type': analysis1['type'],
                    f'{name1}_missing': f"{analysis1['missing']} ({analysis1['missing_pct']:.1f}%)",
                    f'{name2}_missing': f"{analysis2['missing']} ({analysis2['missing_pct']:.1f}%)",
                    f'{name1}_unique': analysis1['unique'],
                    f'{name2}_unique': analysis2['unique'],
                    f'{name1}_summary': analysis1['summary'],
                    f'{name2}_summary': analysis2['summary']
                })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Print numeric columns
        numeric_cols = comparison_df[comparison_df['type'] == 'numeric']
        if not numeric_cols.empty:
            print(f"\nNumeric columns ({len(numeric_cols)}):")
            for _, row in numeric_cols.iterrows():
                print(f"  {row['column']:25} | {row[f'{name1}_summary']:25} | {row[f'{name2}_summary']:25}")
        
        # Print categorical columns
        categorical_cols = comparison_df[comparison_df['type'] == 'categorical']
        if not categorical_cols.empty:
            print(f"\nCategorical columns ({len(categorical_cols)}):")
            for _, row in categorical_cols.iterrows():
                print(f"  {row['column']:25} | Unique: {row[f'{name1}_unique']:6} vs {row[f'{name2}_unique']:6} | Missing: {row[f'{name1}_missing']:15} vs {row[f'{name2}_missing']:15}")
        
        return comparison_df
    
    return None

def main():
    """Compare two datasets."""
    parser = argparse.ArgumentParser(
        description="Compare two datasets to understand their structure and distributions"
    )
    
    parser.add_argument(
        "--dataset1",
        type=str,
        default="random_nuchad.csv",
        help="First dataset to compare (default: random_nuchad.csv)",
    )
    
    parser.add_argument(
        "--dataset2",
        type=str,
        default="random_nuchad_250623.csv",
        help="Second dataset to compare (default: random_nuchad_250623.csv)",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file for detailed comparison (optional)",
    )
    
    args = parser.parse_args()
    
    print("Dataset Comparison Tool")
    print("=" * 50)
    
    try:
        # Load datasets
        df1 = load_dataset(args.dataset1)
        df2 = load_dataset(args.dataset2)
        
        # Compare datasets
        comparison_df = compare_datasets(df1, df2, args.dataset1, args.dataset2)
        
        # Save detailed comparison if requested
        if args.output and comparison_df is not None:
            comparison_df.to_csv(args.output, index=False)
            print(f"\nDetailed comparison saved to: {args.output}")
        
        print("\nComparison complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())