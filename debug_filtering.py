#!/usr/bin/env python3
"""Debug the filtering process."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from nuchad.analysis import eda
from nuchad.data_processing import eligibility_filters as data_utils

def debug_filtering():
    """Debug the filtering process step by step."""
    
    print("Loading new dataset...")
    df_new = eda.get_df("random_nuchad_250623.csv")
    print(f"Shape: {df_new.shape}")
    
    # Check time1 and earliest_af_date
    print("\nChecking time1 vs earliest_af_date:")
    print(f"time1 first 5 values: {df_new['time1'].head()}")
    print(f"earliest_af_date first 5 values: {df_new['earliest_af_date'].head()}")
    print(f"Are they equal? {(df_new['time1'] == df_new['earliest_af_date']).all()}")
    
    # Try filtering step by step
    print("\nTesting AF filter condition:")
    mask = df_new["earliest_af_date"] <= df_new["time1"]
    print(f"AF before time1: {mask.sum()} out of {len(df_new)} patients")
    
    print("\nTesting follow-up filter:")
    if 'Follow_Up_Years' not in df_new.columns:
        df_new['Follow_Up_Years'] = (df_new['end_fu'] - df_new['time1']).dt.days / 365.25
    
    follow_up_mask = df_new['Follow_Up_Years'] >= (365 / 365.25)
    print(f"Follow-up >= 365 days: {follow_up_mask.sum()} out of {len(df_new)} patients")
    
    print(f"Follow-up years stats: min={df_new['Follow_Up_Years'].min():.2f}, max={df_new['Follow_Up_Years'].max():.2f}, mean={df_new['Follow_Up_Years'].mean():.2f}")
    
    # Check for null values
    print("\nChecking for null values:")
    print(f"time1 nulls: {df_new['time1'].isnull().sum()}")
    print(f"end_fu nulls: {df_new['end_fu'].isnull().sum()}")
    print(f"earliest_af_date nulls: {df_new['earliest_af_date'].isnull().sum()}")
    
    # Try running the actual filter function with debug info
    print("\nRunning actual filter function...")
    try:
        eligible_df, filter_stats = data_utils.filter_eligible_patients(df_new)
        print(f"Result: {len(eligible_df)} eligible patients")
        print(f"Filter stats: {filter_stats}")
    except Exception as e:
        print(f"Error in filtering: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_filtering()