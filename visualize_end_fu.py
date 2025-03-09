#!/usr/bin/env python
# Visualize the distribution of end_fu times

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from eda import get_df

# Get the dataframe
df = get_df()

# Check if end_fu exists and has valid data
if 'end_fu' not in df.columns:
    print("Error: 'end_fu' column not found in the dataframe")
    exit(1)

# Count non-null values
valid_end_fu_count = df['end_fu'].notna().sum()
print(f"Total records: {len(df)}")
print(f"Valid end_fu records: {valid_end_fu_count}")

# Create a figure with multiple visualizations
plt.figure(figsize=(15, 10))

# 1. Histogram of end_fu dates
plt.subplot(2, 2, 1)
plt.hist(df['end_fu'].dropna(), bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of End Follow-Up Dates')
plt.xlabel('End Follow-Up Date')
plt.ylabel('Frequency')
plt.xticks(rotation=45)

# 2. Boxplot of end_fu
plt.subplot(2, 2, 2)
sns.boxplot(y=df['end_fu'].dropna())
plt.title('Boxplot of End Follow-Up Dates')
plt.ylabel('End Follow-Up Date')

# 3. Calculate follow-up duration in days from time1 to end_fu
if 'time1' in df.columns:
    df['follow_up_days'] = (df['end_fu'] - df['time1']).dt.days
    
    plt.subplot(2, 2, 3)
    plt.hist(df['follow_up_days'].dropna(), bins=30, color='lightgreen', edgecolor='black')
    plt.title('Distribution of Follow-Up Duration (Days)')
    plt.xlabel('Duration (Days)')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 2, 4)
    sns.boxplot(y=df['follow_up_days'].dropna())
    plt.title('Boxplot of Follow-Up Duration (Days)')
    plt.ylabel('Duration (Days)')
    
    # Print basic statistics about follow-up duration
    print("\nFollow-up duration statistics (days):")
    print(df['follow_up_days'].describe())

# Layout adjustments
plt.tight_layout()
plt.savefig('end_fu_distribution.png')
plt.show()

print("\nVisualization saved as 'end_fu_distribution.png'") 