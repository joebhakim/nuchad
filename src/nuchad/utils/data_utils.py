"""
Consolidated data utility functions.

This module contains canonical versions of commonly-used utility functions
that were previously duplicated across multiple analysis modules.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional
from .paths_data import get_data_file


def get_df(data_file: str = "random_nuchad.csv") -> pd.DataFrame:
    """
    Load and prepare the dataset.
    
    This is the canonical version that consolidates the best features from
    the previous implementations in eda_old.py, eda.py, and density_ratio_reweighting.py.
    
    Args:
        data_file: Name of the CSV file to load from the data directory
        
    Returns:
        DataFrame with cleaned and prepared data
    """
    # Load data using the data access module
    with get_data_file(data_file) as data_path:
        df = pd.read_csv(data_path)
        
        # Handle patid column if present
        if 'patid' in df.columns:
            df = df.rename(columns={"patid": "patient_id"}).set_index("patient_id")
        
        # Remove unnamed columns
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        # Convert date columns to datetime objects - handle both old and new formats
        date_cols = ['time1', 'time2', 'earliest_af_date', 'earliest_stroke_date', 'earliest_tia_date', 
                     'end_fu', 'first_OAC_date', 'first_antiplatelet_date']
        
        # Store original data for re-parsing if needed
        original_data = {}
        for col in date_cols:
            if col in df.columns:
                original_data[col] = df[col].copy()
        
        for col in date_cols:
            if col in df.columns:
                if col in ['time1', 'time2']:
                    df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")
                else:
                    # Try multiple date formats for flexibility
                    # First try the old format
                    df[col] = pd.to_datetime(original_data[col], format="%d%b%Y", errors="coerce")
                    null_count = df[col].isnull().sum()
                    
                    # If most are null, try the new format
                    if null_count > len(df) * 0.8:
                        df[col] = pd.to_datetime(original_data[col], format="%d-%b-%y", errors="coerce")
                        null_count = df[col].isnull().sum()
                    
                    # If still mostly null, fallback to automatic parsing
                    if null_count > len(df) * 0.8:
                        df[col] = pd.to_datetime(original_data[col], errors="coerce")

        # Handle dataset compatibility: create time1 and time2 equivalents for new dataset
        if 'time1' not in df.columns and 'earliest_af_date' in df.columns:
            # Use AF diagnosis date as time1 equivalent
            df['time1'] = df['earliest_af_date']
            print("Created time1 from earliest_af_date")
        
        if 'time2' not in df.columns and 'end_fu' in df.columns:
            # For time2, we'll use a window after time1 (e.g., 3 months)
            if 'time1' in df.columns:
                df['time2'] = df['time1'] + pd.Timedelta(days=90)  # 3 months after AF diagnosis
                print("Created time2 as 3 months after time1")

        return df


def calculate_chadsvasc(row: Union[pd.Series, dict]) -> int:
    """
    Calculate the CHADS-VASc score for a single patient.
    
    This is the canonical version that consolidates the best features from
    the previous implementations, adding defensive programming for missing data.
    
    Args:
        row: A pandas Series (DataFrame row) or dictionary containing patient data
        
    Returns:
        Integer CHADS-VASc score (0-9)
        
    Note:
        CHADS-VASc scoring:
        - Congestive heart failure: 1 point
        - Hypertension: 1 point  
        - Age ≥75: 2 points
        - Age 65-74: 1 point
        - Diabetes mellitus: 1 point
        - Stroke/TIA/Thromboembolism: 2 points
        - Vascular disease: 1 point
        - Sex (Female): 1 point
    """
    score = 0
    
    # Helper function to safely get and convert values
    def safe_get_int(key: str, default: int = 0) -> int:
        if isinstance(row, dict):
            value = row.get(key, default)
        else:  # pandas Series
            value = row.get(key, default) if hasattr(row, 'get') else getattr(row, key, default)
        
        # Handle missing/null values
        if pd.isna(value):
            return default
        return int(value)
    
    def safe_get_float(key: str, default: float = 0.0) -> float:
        if isinstance(row, dict):
            value = row.get(key, default)
        else:  # pandas Series
            value = row.get(key, default) if hasattr(row, 'get') else getattr(row, key, default)
        
        # Handle missing/null values
        if pd.isna(value):
            return default
        return float(value)
    
    # Congestive heart failure (1 point)
    score += safe_get_int("hf")
    
    # Hypertension (1 point)
    score += safe_get_int("hypertension")
    
    # Age scoring (1-2 points)
    age = safe_get_float("age")
    if age >= 75:
        score += 2  # Age ≥75: 2 points
    elif age >= 65:
        score += 1  # Age 65-74: 1 point
    
    # Diabetes mellitus (1 point)
    score += safe_get_int("diab")
    
    # Stroke/TIA/Thromboembolism (2 points)
    thrombo = safe_get_int("thrombo")
    stroke_history = safe_get_int("HB_stroke_history")
    if thrombo or stroke_history:
        score += 2
    
    # Vascular disease (1 point)
    score += safe_get_int("vasc_dis_mi_pad")
    
    # Sex - Female (1 point)
    # Assuming: 1 = male, 2 = female (common coding)
    gender = safe_get_int("gender", 1)  # Default to male if missing
    if gender != 1:  # Not male
        score += 1
    
    return score 