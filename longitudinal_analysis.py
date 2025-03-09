import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import plotly.express as px
from typing import List, Dict, Tuple
from plotly.subplots import make_subplots

def load_and_preprocess_data(file_path: str, sample_size: int = 50) -> pd.DataFrame:
    """
    Load and preprocess the stroke/AF dataset
    
    Args:
        file_path: Path to the CSV file
        sample_size: Number of patients to sample
    
    Returns:
        Preprocessed DataFrame with a sample of patients
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert date columns to datetime
    date_columns = ['time1', 'time2', 'earliest_af_date', 'earliest_stroke_date', 'end_fu']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
    
    # Map categorical variables
    df['gender'] = df['gender'].map({1: 'Male', 2: 'Female'})
    
    # Map binary variables
    binary_vars = ['hypertension', 'diab', 'hf', 'ckd', 'vasc_dis_mi_pad']
    for var in binary_vars:
        df[var] = df[var].map({1: 'Yes', 0: 'No'})
    
    # Ensure anticoagulant is treated as category
    df['Anticoagulant'] = pd.Categorical(df['Anticoagulant'])
    
    # Take a random sample
    sampled_df = df.sample(n=sample_size, random_state=42)
    
    # Reset index to create patient IDs
    sampled_df = sampled_df.reset_index(drop=True)
    sampled_df['patient_id'] = sampled_df.index
    
    return sampled_df

def get_comorbidity_text(patient: pd.Series) -> str:
    """Create formatted text for comorbidities"""
    comorbidities = []
    if patient['hypertension'] == 'Yes':
        comorbidities.append("Hypertension")
    if patient['diab'] == 'Yes':
        comorbidities.append("Diabetes")
    if patient['hf'] == 'Yes':
        comorbidities.append("Heart Failure")
    if patient['ckd'] == 'Yes':
        comorbidities.append("CKD")
    if patient['vasc_dis_mi_pad'] == 'Yes':
        comorbidities.append("Vascular Disease")
    
    return "<br>".join([f"• {c}" for c in comorbidities]) if comorbidities else "None"

def create_binary_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Create a matrix of binary covariates"""
    # Define the binary covariates we want to show
    covariate_mapping = {
        'gender': lambda x: 1 if x == 'Male' else 0,
        'hypertension': lambda x: 1 if x == 'Yes' else 0,
        'diab': lambda x: 1 if x == 'Yes' else 0,
        'hf': lambda x: 1 if x == 'Yes' else 0,
        'ckd': lambda x: 1 if x == 'Yes' else 0,
        'vasc_dis_mi_pad': lambda x: 1 if x == 'Yes' else 0,
        'HB_stroke_history': lambda x: 1 if x == 1 else 0,
        'aortic_plaq': lambda x: 1 if x == 1 else 0,
        'af': lambda x: 1 if x == 1 else 0,
        'thrombo': lambda x: 1 if x == 1 else 0,
        'end_fu_due_to_death': lambda x: 1 if x == 1 else 0,
        'stroke_1Y': lambda x: 1 if x == 1 else 0
    }
    
    # Create matrix
    matrix = np.zeros((len(df), len(covariate_mapping)))
    for i, (_, patient) in enumerate(df.iterrows()):
        for j, (col, func) in enumerate(covariate_mapping.items()):
            matrix[i, j] = func(patient[col])
    
    # Create column names with more descriptive labels
    col_names = [
        'Male', 'HTN', 'DM', 'HF', 'CKD', 'Vasc. Dis.',
        'Prior Stroke', 'Aortic Plaque', 'AF', 'Thromboemb.',
        'Death', 'Stroke 1Y'
    ]
            
    return matrix, col_names

def create_continuous_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[Tuple[float, float]]]:
    """Create a matrix of normalized continuous variables"""
    # Define continuous variables and their display names
    continuous_vars = {
        'age': 'Age (years)',
        'age_at_entry': 'Entry Age',
        'bmi': 'BMI',
        'frailty_score': 'Frailty',
        'tc_mmol_L': 'Total Chol.',
        'acr_mg_mmol': 'ACR',
        'stroke_time': 'Stroke Time'
    }
    
    # Initialize matrix and ranges list
    matrix = np.zeros((len(df), len(continuous_vars)))
    ranges = []
    col_names = []
    
    # Fill matrix with normalized values
    for j, (var, name) in enumerate(continuous_vars.items()):
        values = df[var].values
        valid_values = values[~np.isnan(values)]
        if len(valid_values) > 0:
            min_val = np.min(valid_values)
            max_val = np.max(valid_values)
            # Store the range for hover information
            ranges.append((min_val, max_val))
            # Normalize values between 0 and 1
            matrix[:, j] = np.nan_to_num((values - min_val) / (max_val - min_val) if max_val > min_val else 0)
            # Format ranges based on the variable type
            if var in ['age', 'age_at_entry', 'bmi']:
                col_names.append(f"{name}<br>[{min_val:.0f}-{max_val:.0f}]")
            elif var == 'stroke_time':
                col_names.append(f"{name}<br>[{min_val:.1f}-{max_val:.1f}] days")
            else:
                col_names.append(f"{name}<br>[{min_val:.2f}-{max_val:.2f}]")
        else:
            ranges.append((np.nan, np.nan))
            col_names.append(name)
    
    return matrix, col_names, ranges

def analyze_stroke_timing(df: pd.DataFrame) -> dict:
    """
    Analyze the timing of stroke events relative to time1
    
    Args:
        df: Preprocessed DataFrame with patient data
    
    Returns:
        Dictionary containing stroke timing statistics
    """
    # Calculate time to stroke for patients with stroke
    stroke_patients = df[pd.notna(df['earliest_stroke_date'])]
    
    # Calculate time difference in days
    stroke_patients['time_to_stroke'] = (stroke_patients['earliest_stroke_date'] - 
                                       stroke_patients['time1']).dt.total_seconds() / (24 * 3600)
    
    # Categorize strokes by different time periods
    strokes_within_two_days = stroke_patients[stroke_patients['time_to_stroke'] <= 2]
    strokes_within_week = stroke_patients[stroke_patients['time_to_stroke'] <= 7]
    strokes_within_year = stroke_patients[stroke_patients['time_to_stroke'] <= 365]
    strokes_after_year = stroke_patients[stroke_patients['time_to_stroke'] > 365]
    
    return {
        'total_patients': len(df),
        'total_strokes': len(stroke_patients),
        'strokes_within_two_days': len(strokes_within_two_days),
        'strokes_within_week': len(strokes_within_week),
        'strokes_within_year': len(strokes_within_year),
        'strokes_after_year': len(strokes_after_year),
        'median_time_to_stroke': stroke_patients['time_to_stroke'].median(),
        'mean_time_to_stroke': stroke_patients['time_to_stroke'].mean()
    }

def create_longitudinal_visualization(df: pd.DataFrame) -> go.Figure:
    """
    Create an interactive visualization with timeline and covariate grids
    
    Args:
        df: Preprocessed DataFrame with patient data
    
    Returns:
        Plotly Figure object
    """
    # Calculate stroke timing statistics
    stroke_stats = analyze_stroke_timing(df)
    
    # Create figure with three subplots
    fig = make_subplots(
        rows=1, cols=3,
        column_widths=[0.4, 0.3, 0.3],
        specs=[[{"type": "scatter"}, {"type": "heatmap"}, {"type": "heatmap"}]],
        horizontal_spacing=0.01,
        subplot_titles=("Timeline", "Binary Variables", "Continuous Variables")
    )
    
    # Plot timelines
    for _, patient in df.iterrows():
        # Determine line color based on stroke timing
        line_color = 'lightgray'  # Default for no stroke
        if pd.notna(patient['earliest_stroke_date']):
            time_to_stroke = (patient['earliest_stroke_date'] - patient['time1']).total_seconds() / (24 * 3600)
            if time_to_stroke <= 2:
                line_color = 'lightblue'  # Within 2 days
            elif time_to_stroke <= 7:
                line_color = 'lightgray'  # Within week
            elif time_to_stroke <= 365:
                line_color = 'mistyrose'  # Within year
            else:
                line_color = 'pink'  # After year
        
        # Add observation period line
        fig.add_trace(
            go.Scatter(
                x=[patient['time1'], patient['end_fu']],
                y=[patient['patient_id'], patient['patient_id']],
                mode='lines',
                line=dict(color=line_color, width=2),
                showlegend=False,
                hoverinfo='text',
                text=[f"Observation Period<br>"
                      f"Start: {patient['time1'].strftime('%Y-%m-%d')}<br>"
                      f"End: {patient['end_fu'].strftime('%Y-%m-%d')}<br>"
                      f"{'No stroke' if pd.isna(patient['earliest_stroke_date']) else f'Stroke after {time_to_stroke:.1f} days'}"]
            ),
            row=1, col=1
        )
        
        # Add AF event if present
        if pd.notna(patient['earliest_af_date']):
            fig.add_trace(
                go.Scatter(
                    x=[patient['earliest_af_date']],
                    y=[patient['patient_id']],
                    mode='markers',
                    marker=dict(symbol='diamond', size=10, color='purple'),
                    name='AF Diagnosis',
                    showlegend=False,
                    text=[f"AF Diagnosis<br>Date: {patient['earliest_af_date'].strftime('%Y-%m-%d')}"],
                    hoverinfo='text'
                ),
                row=1, col=1
            )
        
        # Add stroke event if present
        if pd.notna(patient['earliest_stroke_date']):
            # Calculate time to stroke
            time_to_stroke = (patient['earliest_stroke_date'] - patient['time1']).total_seconds() / (24 * 3600)
            
            # Color coding based on timing
            if time_to_stroke <= 2:
                stroke_color = 'darkblue'  # Within 2 days
            elif time_to_stroke <= 7:
                stroke_color = 'black'  # Within week
            elif time_to_stroke <= 365:
                stroke_color = 'darkred'  # Within year
            else:
                stroke_color = 'red'  # After year
            
            fig.add_trace(
                go.Scatter(
                    x=[patient['earliest_stroke_date']],
                    y=[patient['patient_id']],
                    mode='markers',
                    marker=dict(symbol='star', size=12, color=stroke_color),
                    name='Stroke',
                    showlegend=False,
                    text=[f"Stroke<br>Date: {patient['earliest_stroke_date'].strftime('%Y-%m-%d')}<br>"
                          f"Time from entry: {time_to_stroke:.1f} days<br>"
                          f"{'Within first 2 days' if time_to_stroke <= 2 else 'Within first week' if time_to_stroke <= 7 else 'Within first year' if time_to_stroke <= 365 else 'After first year'}"],
                    hoverinfo='text'
                ),
                row=1, col=1
            )
    
    # Create and add binary covariate heatmap
    binary_matrix, binary_names = create_binary_matrix(df)
    fig.add_trace(
        go.Heatmap(
            z=binary_matrix,
            x=binary_names,
            y=list(range(len(df))),
            colorscale=[[0, 'white'], [1, '#ff4d4d']],  # Slightly adjusted red color
            showscale=False,
            hoverongaps=False,
            hovertemplate="Patient %{y}<br>" +
                         "%{x}: %{z:d}<br>" +  # Format as integer
                         "<extra></extra>"
        ),
        row=1, col=2
    )
    
    # Create and add continuous covariate heatmap
    continuous_matrix, continuous_names, ranges = create_continuous_matrix(df)
    fig.add_trace(
        go.Heatmap(
            z=continuous_matrix,
            x=continuous_names,
            y=list(range(len(df))),
            colorscale=[[0, 'white'], [1, '#2E86C1']],  # Changed to blue for continuous vars
            showscale=True,
            hoverongaps=False,
            hovertemplate="Patient %{y}<br>" +
                         "%{x}<br>" +
                         "Normalized value: %{z:.2f}<br>" +
                         "<extra></extra>"
        ),
        row=1, col=3
    )
    
    # Update layout with enhanced stroke statistics
    title = (f'Longitudinal Analysis of Stroke Events (n={stroke_stats["total_patients"]})<br>'
             f'Strokes: {stroke_stats["total_strokes"]} total '
             f'({stroke_stats["strokes_within_two_days"]} within 2 days, '
             f'{stroke_stats["strokes_within_week"]} within week, '
             f'{stroke_stats["strokes_within_year"]} within year, '
             f'{stroke_stats["strokes_after_year"]} after)')
    
    fig.update_layout(
        title=title,
        showlegend=False,
        height=1600,
        hovermode='closest'
    )
    
    # Update x-axes with rotated labels for better readability
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(tickangle=45, row=1, col=2)
    fig.update_xaxes(tickangle=45, row=1, col=3)
    
    # Update y-axes
    fig.update_yaxes(
        title_text="Patient ID",
        range=[-1, len(df)],
        tickmode='array',
        ticktext=[f"Patient {i}" for i in range(len(df))],
        tickvals=list(range(len(df))),
        row=1, col=1
    )
    
    # Hide y-axis labels for heatmaps
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=3)
    
    return fig

def main(data_path: str, sample_size: int = 500):
    # Load and preprocess data
    df = load_and_preprocess_data(data_path, sample_size)
    
    # Analyze stroke timing
    stroke_stats = analyze_stroke_timing(df)
    
    # Print detailed statistics
    print("\nStroke Timing Analysis:")
    print(f"Total patients: {stroke_stats['total_patients']}")
    print(f"Total strokes: {stroke_stats['total_strokes']}")
    print(f"Strokes within first 2 days: {stroke_stats['strokes_within_two_days']}")
    print(f"Strokes within first week: {stroke_stats['strokes_within_week']}")
    print(f"Strokes within first year: {stroke_stats['strokes_within_year']}")
    print(f"Strokes after first year: {stroke_stats['strokes_after_year']}")
    print(f"Median time to stroke: {stroke_stats['median_time_to_stroke']:.1f} days")
    print(f"Mean time to stroke: {stroke_stats['mean_time_to_stroke']:.1f} days")
    
    # Create visualization
    fig = create_longitudinal_visualization(df)
    
    # Save the interactive HTML file
    fig.write_html('longitudinal_analysis.html')
    
    print("\nAnalysis complete! Open 'longitudinal_analysis.html' to view the interactive visualization.")

if __name__ == "__main__":
    main(data_path='dummy_data.csv', sample_size=500) 