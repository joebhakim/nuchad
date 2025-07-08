"""
EDA module with survival modeling and longitudinal analysis.

This module provides comprehensive exploratory data analysis including:
- Kaplan-Meier survival curves
- Timeline visualization with horizontal bar charts
- Categorical and continuous variable grid views
- Survival statistics and stroke timing analysis
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
import warnings
warnings.filterwarnings('ignore')

from nuchad.utils.paths_data import get_data_file, get_results_dir
from nuchad.data_processing.eligibility_filters import filter_eligible_patients


def get_df(data_file: str = "random_nuchad.csv") -> pd.DataFrame:
    """Load and prepare the dataset."""
    with get_data_file(data_file) as data_path:
        df = pd.read_csv(data_path)
        
        # Handle patid column if present
        if 'patid' in df.columns:
            df = df.rename(columns={"patid": "patient_id"}).set_index("patient_id")
        
        # Remove unnamed columns
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        # Convert date columns to datetime objects
        date_cols = ['time1', 'time2', 'earliest_af_date', 'earliest_stroke_date', 
                     'earliest_tia_date', 'end_fu', 'first_OAC_date', 'first_antiplatelet_date']
        
        original_data = {}
        for col in date_cols:
            if col in df.columns:
                original_data[col] = df[col].copy()
        
        for col in date_cols:
            if col in df.columns:
                if col in ['time1', 'time2']:
                    df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors="coerce")
                else:
                    # Try multiple date formats
                    df[col] = pd.to_datetime(original_data[col], format="%d%b%Y", errors="coerce")
                    null_count = df[col].isnull().sum()
                    
                    if null_count > len(df) * 0.8:
                        df[col] = pd.to_datetime(original_data[col], format="%d-%b-%y", errors="coerce")
                        null_count = df[col].isnull().sum()
                    
                    if null_count > len(df) * 0.8:
                        df[col] = pd.to_datetime(original_data[col], errors="coerce")

        # Handle dataset compatibility
        if 'time1' not in df.columns and 'earliest_af_date' in df.columns:
            df['time1'] = df['earliest_af_date']
            print("Created time1 from earliest_af_date")
        
        if 'time2' not in df.columns and 'end_fu' in df.columns:
            if 'time1' in df.columns:
                df['time2'] = df['time1'] + pd.Timedelta(days=90)
                print("Created time2 as 3 months after time1")

        return df


def prepare_survival_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for survival analysis."""
    survival_df = df.copy()
    
    # Calculate survival time (time to event or censoring)
    survival_df['survival_time'] = (survival_df['end_fu'] - survival_df['time1']).dt.days
    
    # Create event indicator (1 if stroke occurred, 0 if censored)
    survival_df['event'] = 0
    stroke_mask = pd.notna(survival_df['earliest_stroke_date'])
    survival_df.loc[stroke_mask, 'event'] = 1
    
    # For patients with stroke, use time to stroke as survival time
    survival_df.loc[stroke_mask, 'survival_time'] = (
        survival_df.loc[stroke_mask, 'earliest_stroke_date'] - 
        survival_df.loc[stroke_mask, 'time1']
    ).dt.days
    
    # Remove patients with negative or zero survival time
    survival_df = survival_df[survival_df['survival_time'] > 0]
    
    return survival_df


def create_kaplan_meier_curves(df: pd.DataFrame) -> go.Figure:
    """Create Kaplan-Meier survival curves."""
    survival_df = prepare_survival_data(df)
    
    # Calculate CHADS-VASc score for stratification
    def calculate_chadsvasc(row):
        score = 0
        score += int(row.get("hf", 0))
        score += int(row.get("hypertension", 0))
        score += 2 * int(row.get("age", 0) >= 75)
        score += int(65 <= row.get("age", 0) < 75)
        score += int(row.get("diab", 0))
        score += 2 * int(row.get("thrombo", 0) or row.get("HB_stroke_history", 0))
        score += int(row.get("vasc_dis_mi_pad", 0))
        score += int(row.get("gender", 1) != 1)
        return score
    
    survival_df['chadsvasc'] = survival_df.apply(calculate_chadsvasc, axis=1)
    
    # Create risk groups
    survival_df['risk_group'] = pd.cut(
        survival_df['chadsvasc'], 
        bins=[-0.5, 1.5, 3.5, 10], 
        labels=['Low (0-1)', 'Moderate (2-3)', 'High (4+)']
    )
    
    # Create figure
    fig = go.Figure()
    
    colors = ['#1f77b4', '#ff7f0e', '#d62728']
    
    # Fit KM curves for each risk group
    for i, (group, color) in enumerate(zip(['Low (0-1)', 'Moderate (2-3)', 'High (4+)'], colors)):
        group_data = survival_df[survival_df['risk_group'] == group]
        
        if len(group_data) > 0:
            kmf = KaplanMeierFitter()
            kmf.fit(group_data['survival_time'], group_data['event'], label=group)
            
            # Add survival curve
            fig.add_trace(go.Scatter(
                x=kmf.timeline,
                y=kmf.survival_function_.iloc[:, 0],
                mode='lines',
                name=f'{group} (n={len(group_data)})',
                line=dict(color=color, width=2),
                hovertemplate='Time: %{x} days<br>Survival: %{y:.3f}<extra></extra>'
            ))
    
    # Update layout
    fig.update_layout(
        title='Kaplan-Meier Survival Curves by CHADS-VASc Risk Group',
        xaxis_title='Time (days)',
        yaxis_title='Survival Probability',
        legend=dict(x=0.7, y=0.9),
        hovermode='x unified',
        height=500
    )
    
    return fig


def create_timeline_visualization(df: pd.DataFrame, n_sample: int = 100) -> go.Figure:
    """Create horizontal bar chart timeline visualization."""
    # Sample patients for visualization
    sample_df = df.sample(n=min(n_sample, len(df)), random_state=42).copy()
    sample_df = sample_df.reset_index(drop=True)
    sample_df['patient_id'] = range(len(sample_df))
    
    # Create subplots: timeline + categorical + continuous
    fig = make_subplots(
        rows=1, cols=3,
        column_widths=[0.4, 0.3, 0.3],
        specs=[[{"type": "scatter"}, {"type": "heatmap"}, {"type": "heatmap"}]],
        horizontal_spacing=0.02,
        subplot_titles=("Patient Timelines", "Categorical Variables", "Continuous Variables")
    )
    
    # Timeline visualization
    for idx, (_, patient) in enumerate(sample_df.iterrows()):
        # Base observation period
        start_time = patient['time1']
        end_time = patient['end_fu']
        
        # Handle missing end_fu
        if pd.isna(end_time):
            end_time = start_time + pd.Timedelta(days=365)  # Default 1 year
            line_style = dict(color='gray', width=3, dash='dash')
        else:
            line_style = dict(color='lightblue', width=3)
        
        # Add observation period line
        fig.add_trace(
            go.Scatter(
                x=[start_time, end_time],
                y=[idx, idx],
                mode='lines',
                line=line_style,
                showlegend=False,
                hovertemplate=f'Patient {idx}<br>Start: %{{x}}<br>End: {end_time}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add stroke event if present
        if pd.notna(patient['earliest_stroke_date']):
            fig.add_trace(
                go.Scatter(
                    x=[patient['earliest_stroke_date']],
                    y=[idx],
                    mode='markers',
                    marker=dict(symbol='star', size=8, color='red'),
                    showlegend=False,
                    hovertemplate=f'Patient {idx}<br>Stroke: %{{x}}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Add missing end_fu marker
        if pd.isna(patient['end_fu']):
            fig.add_trace(
                go.Scatter(
                    x=[end_time],
                    y=[idx],
                    mode='markers',
                    marker=dict(symbol='x', size=8, color='black'),
                    showlegend=False,
                    hovertemplate=f'Patient {idx}<br>Missing end_fu<extra></extra>'
                ),
                row=1, col=1
            )
    
    # Categorical variables heatmap
    categorical_vars = ['gender', 'hypertension', 'diab', 'hf', 'ckd', 'vasc_dis_mi_pad', 'af']
    cat_matrix = np.zeros((len(sample_df), len(categorical_vars)))
    
    for i, var in enumerate(categorical_vars):
        if var in sample_df.columns:
            if var == 'gender':
                cat_matrix[:, i] = (sample_df[var] == 1).astype(int)  # 1 = Male
            else:
                cat_matrix[:, i] = sample_df[var].fillna(0).astype(int)
    
    fig.add_trace(
        go.Heatmap(
            z=cat_matrix,
            x=['Male', 'HTN', 'DM', 'HF', 'CKD', 'Vasc', 'AF'],
            y=list(range(len(sample_df))),
            colorscale=[[0, 'white'], [1, 'red']],
            showscale=False,
            hovertemplate='Patient %{y}<br>%{x}: %{z}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Continuous variables heatmap
    continuous_vars = ['age', 'bmi', 'tc_mmol_L', 'frailty_score']
    available_vars = [var for var in continuous_vars if var in sample_df.columns]
    
    if available_vars:
        cont_matrix = np.zeros((len(sample_df), len(available_vars)))
        for i, var in enumerate(available_vars):
            values = sample_df[var].fillna(sample_df[var].median())
            # Normalize to 0-1 range
            min_val, max_val = values.min(), values.max()
            if max_val > min_val:
                cont_matrix[:, i] = (values - min_val) / (max_val - min_val)
        
        fig.add_trace(
            go.Heatmap(
                z=cont_matrix,
                x=available_vars,
                y=list(range(len(sample_df))),
                colorscale=[[0, 'white'], [1, 'blue']],
                showscale=True,
                hovertemplate='Patient %{y}<br>%{x}: %{z:.2f}<extra></extra>'
            ),
            row=1, col=3
        )
    
    # Update layout
    fig.update_layout(
        title=f'Patient Timeline Analysis (n={len(sample_df)} sampled patients)',
        height=800,
        showlegend=False
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Patient ID", row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=3)
    
    return fig


def create_survival_statistics_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics table for survival analysis."""
    survival_df = prepare_survival_data(df)
    
    # Calculate basic statistics
    stats = {
        'Total Patients': len(survival_df),
        'Events (Strokes)': survival_df['event'].sum(),
        'Censored': len(survival_df) - survival_df['event'].sum(),
        'Median Follow-up (days)': survival_df['survival_time'].median(),
        'Mean Follow-up (days)': survival_df['survival_time'].mean(),
        'Min Follow-up (days)': survival_df['survival_time'].min(),
        'Max Follow-up (days)': survival_df['survival_time'].max(),
        'Event Rate (%)': (survival_df['event'].sum() / len(survival_df)) * 100
    }
    
    return pd.DataFrame(list(stats.items()), columns=['Statistic', 'Value'])


def run_survival_eda(pre_filter: bool = True, post_filter: bool = True, 
                     sample_size: int = 100) -> None:
    """
    Run comprehensive survival-focused EDA.
    
    Args:
        pre_filter: Generate visualizations for pre-filtered data
        post_filter: Generate visualizations for post-filtered data
        sample_size: Number of patients to sample for timeline visualization
    """
    results_dir = get_results_dir()
    
    # Load data
    df = get_df()
    print(f"Loaded {len(df)} patients")
    
    # Pre-filter analysis
    if pre_filter:
        print("\n=== PRE-FILTER ANALYSIS ===")
        
        # Create survival statistics
        pre_stats = create_survival_statistics_table(df)
        pre_stats.to_csv(results_dir / "pre_filter_survival_stats.csv", index=False)
        print("Pre-filter survival statistics:")
        print(pre_stats.to_string(index=False))
        
        # Create Kaplan-Meier curves
        km_fig = create_kaplan_meier_curves(df)
        km_fig.write_html(results_dir / "pre_filter_kaplan_meier.html")
        print(f"Saved Kaplan-Meier curves to {results_dir / 'pre_filter_kaplan_meier.html'}")
        
        # Create timeline visualization
        timeline_fig = create_timeline_visualization(df, sample_size)
        timeline_fig.write_html(results_dir / "pre_filter_timeline.html")
        print(f"Saved timeline visualization to {results_dir / 'pre_filter_timeline.html'}")
    
    # Post-filter analysis
    if post_filter:
        print("\n=== POST-FILTER ANALYSIS ===")
        
        # Filter patients
        filtered_df, filter_stats = filter_eligible_patients(
            df,
            require_af=True,
            require_follow_up=True,
            require_stroke=False,
            af_before_time1=True,
            min_follow_up_days=365,
            stroke_window_days=365
        )
        
        print(f"Filtered to {len(filtered_df)} patients")
        
        # Create survival statistics
        post_stats = create_survival_statistics_table(filtered_df)
        post_stats.to_csv(results_dir / "post_filter_survival_stats.csv", index=False)
        print("Post-filter survival statistics:")
        print(post_stats.to_string(index=False))
        
        # Create Kaplan-Meier curves
        km_fig = create_kaplan_meier_curves(filtered_df)
        km_fig.write_html(results_dir / "post_filter_kaplan_meier.html")
        print(f"Saved Kaplan-Meier curves to {results_dir / 'post_filter_kaplan_meier.html'}")
        
        # Create timeline visualization
        timeline_fig = create_timeline_visualization(filtered_df, sample_size)
        timeline_fig.write_html(results_dir / "post_filter_timeline.html")
        print(f"Saved timeline visualization to {results_dir / 'post_filter_timeline.html'}")
    
    print(f"\nAll outputs saved to {results_dir}")


if __name__ == "__main__":
    run_survival_eda()