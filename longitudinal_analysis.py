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

def create_longitudinal_visualization(df: pd.DataFrame) -> go.Figure:
    """
    Create an interactive visualization with timeline and covariate grids
    
    Args:
        df: Preprocessed DataFrame with patient data
    
    Returns:
        Plotly Figure object
    """
    # Create figure with three subplots
    fig = make_subplots(
        rows=1, cols=3,
        column_widths=[0.4, 0.3, 0.3],  # Adjusted widths to accommodate more variables
        specs=[[{"type": "scatter"}, {"type": "heatmap"}, {"type": "heatmap"}]],
        horizontal_spacing=0.01,
        subplot_titles=("Timeline", "Binary Variables", "Continuous Variables")
    )
    
    # Plot timelines
    for _, patient in df.iterrows():
        # Add observation period line
        fig.add_trace(
            go.Scatter(
                x=[patient['time1'], patient['end_fu']],
                y=[patient['patient_id'], patient['patient_id']],
                mode='lines',
                line=dict(color='saddlebrown', width=1),
                showlegend=False,
                hoverinfo='skip'
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
                    text=[f"AF Diagnosis<br>Date: {patient['earliest_af_date'].strftime('%Y-%m-%d')}"],
                    hoverinfo='text'
                ),
                row=1, col=1
            )
        
        # Add stroke event if present
        if pd.notna(patient['earliest_stroke_date']):
            fig.add_trace(
                go.Scatter(
                    x=[patient['earliest_stroke_date']],
                    y=[patient['patient_id']],
                    mode='markers',
                    marker=dict(symbol='star', size=12, color='red'),
                    name='Stroke',
                    text=[f"Stroke<br>Date: {patient['earliest_stroke_date'].strftime('%Y-%m-%d')}"],
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
    
    # Update layout
    fig.update_layout(
        title='Longitudinal Analysis of Stroke Events with Patient Characteristics',
        showlegend=True,
        height=800,
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

def main():
    # Load and preprocess data
    df = load_and_preprocess_data('dummy_data.csv', sample_size=50)
    
    # Create visualization
    fig = create_longitudinal_visualization(df)
    
    # Save the interactive HTML file
    fig.write_html('longitudinal_analysis.html')
    
    print("Analysis complete! Open 'longitudinal_analysis.html' to view the interactive visualization.")

if __name__ == "__main__":
    main() 