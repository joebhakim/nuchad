#!/usr/bin/env python
# Visualize the distribution of end_fu times

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

from nuchad.utils import get_results_dir
from nuchad.analysis.eda_old import get_df
from nuchad.data_processing.eligibility_filters import filter_eligible_patients

def plot_end_fu_distribution(df=None):
    """
    Plot the distribution of end follow-up times.
    
    Args:
        df: Optional pre-loaded DataFrame. If None, will load the data.
    """
    # Get the dataframe if not provided
    if df is None:
        df = get_df()
        df, _ = filter_eligible_patients(df)

    # Calculate follow-up time in years
    df['follow_up_years'] = (df['end_fu'] - df['time1']).dt.days / 365.25

    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['follow_up_years'], bins=50, edgecolor='black')
    plt.xlabel('Follow-up Time (years)')
    plt.ylabel('Number of Patients')
    plt.title('Distribution of End Follow-up Times')

    # Layout adjustments
    plt.tight_layout()
    
    # Save to results directory
    results_dir = get_results_dir()
    plt.savefig(results_dir / 'end_fu_distribution.png', dpi=300)
    
    print(f"Visualization saved as '{results_dir / 'end_fu_distribution.png'}'")

def main():
    """Main entry point for the visualization module."""
    plot_end_fu_distribution()

if __name__ == "__main__":
    main() 