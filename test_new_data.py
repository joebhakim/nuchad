#!/usr/bin/env python3
"""Test script to verify new data loading works."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from nuchad.analysis import eda
from nuchad.data_processing import eligibility_filters as data_utils

def test_data_loading():
    """Test loading both datasets."""
    
    print("Testing original dataset...")
    try:
        df_old = eda.get_df("random_nuchad.csv")
        print(f"Original dataset shape: {df_old.shape}")
        print(f"Original dataset columns: {list(df_old.columns)}")
        print(f"Has time1: {'time1' in df_old.columns}")
        print(f"Has time2: {'time2' in df_old.columns}")
        print()
    except Exception as e:
        print(f"Error loading original dataset: {e}")
        return False
    
    print("Testing new dataset...")
    try:
        df_new = eda.get_df("random_nuchad_250623.csv")
        print(f"New dataset shape: {df_new.shape}")
        print(f"New dataset columns: {list(df_new.columns)}")
        print(f"Has time1: {'time1' in df_new.columns}")
        print(f"Has time2: {'time2' in df_new.columns}")
        print(f"Has Anticoag3m_type: {'Anticoag3m_type' in df_new.columns}")
        
        if 'Anticoag3m_type' in df_new.columns:
            print(f"Anticoag3m_type categories: {df_new['Anticoag3m_type'].value_counts()}")
        print()
    except Exception as e:
        print(f"Error loading new dataset: {e}")
        return False
    
    print("Testing eligibility filtering on new dataset...")
    try:
        eligible_df, filter_stats = data_utils.filter_eligible_patients(df_new)
        print(f"Eligible patients from new dataset: {len(eligible_df)} out of {filter_stats['total']}")
        print("Filtering test passed!")
        return True
    except Exception as e:
        print(f"Error filtering new dataset: {e}")
        return False

if __name__ == "__main__":
    success = test_data_loading()
    sys.exit(0 if success else 1)