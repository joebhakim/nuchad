import pandas as pd
import numpy as np
from nuchad.utils import get_df
from nuchad.data_processing.eligibility_filters import filter_eligible_patients
from nuchad.utils import get_results_dir

def create_table1(df):
    """
    Create a "Table 1" summary of the dataset with appropriate statistics for each variable type:
    - Continuous variables: mean ± standard deviation
    - Categorical variables: count (percentage)
    
    Args:
        df: The dataframe containing the patient data
        
    Returns:
        A pandas DataFrame formatted as Table 1
    """
    # Initialize empty lists to store results
    rows = []
    
    # Define groups of variables
    demographics = {
        'age': 'continuous',
        'gender': 'categorical',
        'ethnic_group': 'categorical',
    }
    
    clinical_factors = {
        'bmi': 'continuous',
        'frailty_score': 'continuous',
        'tc_mmol_L': 'continuous',
        'acr_mg_mmol': 'continuous',
        'smoking_status': 'categorical'
    }
    
    comorbidities = {
        'af': 'categorical',
        'hypertension': 'categorical',
        'diab': 'categorical',
        'thrombo': 'categorical',
        'hf': 'categorical',
        'HB_stroke_history': 'categorical',
        'ckd': 'categorical',
        'vasc_dis_mi_pad': 'categorical',
        'aortic_plaq': 'categorical',
    }
    
    medications = {
        'Anticoagulant': 'categorical'
    }
    
    outcomes = {
        'stroke_1Y': 'categorical',
        'stroke_time': 'continuous',
        'end_fu_due_to_death': 'categorical'
    }
    
    # Define display names
    display_names = {
        'age': 'Age (years)',
        'gender': 'Gender',
        'ethnic_group': 'Ethnicity',
        'bmi': 'BMI (kg/m²)',
        'frailty_score': 'Frailty Score',
        'tc_mmol_L': 'Total Cholesterol (mmol/L)',
        'acr_mg_mmol': 'Albumin-Creatinine Ratio (mg/mmol)',
        'smoking_status': 'Smoking Status',
        'af': 'Atrial Fibrillation',
        'hypertension': 'Hypertension',
        'diab': 'Diabetes',
        'thrombo': 'Thromboembolism',
        'hf': 'Heart Failure',
        'HB_stroke_history': 'Prior Stroke',
        'ckd': 'Chronic Kidney Disease',
        'vasc_dis_mi_pad': 'Vascular Disease',
        'aortic_plaq': 'Aortic Plaque',
        'Anticoagulant': 'Anticoagulant Use',
        'stroke_1Y': 'Stroke within 1 Year',
        'stroke_time': 'Time to Stroke (years)',
        'end_fu_due_to_death': 'Death during Follow-up'
    }
    
    # Gender mapping
    gender_mapping = {1: 'Male', 2: 'Female'}
    
    # Binary mapping
    binary_mapping = {0: 'No', 1: 'Yes', 2: 'No'}
    
    # Process each group of variables
    variable_groups = [
        ('Demographics', demographics),
        ('Clinical Factors', clinical_factors),
        ('Comorbidities', comorbidities),
        ('Medications', medications),
        ('Outcomes', outcomes)
    ]
    
    for group_name, variables in variable_groups:
        # Add group header
        rows.append({'Variable': f'**{group_name}**', 'Statistics': ''})
        
        for var, var_type in variables.items():
            if var not in df.columns:
                continue
                
            display_name = display_names.get(var, var)
            
            # Handle different variable types
            if var_type == 'continuous':
                # Compute statistics only on non-missing values
                valid_values = df[var].dropna()
                if len(valid_values) > 0:
                    mean = valid_values.mean()
                    std = valid_values.std()
                    stat_text = f"{mean:.2f} ± {std:.2f}"
                    rows.append({'Variable': display_name, 'Statistics': stat_text})
                else:
                    rows.append({'Variable': display_name, 'Statistics': 'No data'})
            
            elif var_type == 'categorical':
                # Apply mappings if needed
                series = df[var].copy()
                
                if var == 'gender':
                    series = series.map(gender_mapping)
                elif var in ['af', 'hypertension', 'diab', 'thrombo', 'hf', 'HB_stroke_history', 
                           'ckd', 'vasc_dis_mi_pad', 'aortic_plaq', 'stroke_1Y', 'end_fu_due_to_death']:
                    series = series.map(binary_mapping)
                
                # Calculate counts and percentages
                value_counts = series.value_counts().sort_index()
                total = len(df)
                
                # Add a row for each category
                rows.append({'Variable': display_name, 'Statistics': ''})
                for val, count in value_counts.items():
                    if pd.notna(val):  # Skip NaN values
                        percentage = (count / total) * 100
                        stat_text = f"{count} ({percentage:.1f}%)"
                        rows.append({'Variable': f'  {val}', 'Statistics': stat_text})
    
    # Create DataFrame from rows
    table1_df = pd.DataFrame(rows)
    
    return table1_df

def calculate_chadsvasc_distribution(df):
    """
    Calculate the distribution of CHADS-VASc scores in the dataset
    
    Args:
        df: DataFrame with required variables for CHADS-VASc calculation
        
    Returns:
        DataFrame with distribution of CHADS-VASc scores
    """
    # Calculate CHADS-VASc scores if not already present
    if 'CHADS-Vasc' not in df.columns:
        from nuchad.utils import calculate_chadsvasc
        df['CHADS-Vasc'] = df.apply(calculate_chadsvasc, axis=1)
    
    # Get distribution
    score_counts = df['CHADS-Vasc'].value_counts().sort_index()
    
    # Calculate percentages
    total = len(df)
    percentages = score_counts / total * 100
    
    # Create result DataFrame
    result = pd.DataFrame({
        'Score': score_counts.index,
        'Count': score_counts.values,
        'Percentage': percentages.values
    })
    
    # Add formatting
    result['Formatted'] = result.apply(lambda x: f"{int(x['Count'])} ({x['Percentage']:.1f}%)", axis=1)
    
    return result

if __name__ == "__main__":
    # Load data
    df = get_df()
    
    # Filter eligible patients
    eligible_df = filter_eligible_patients(df)
    
    # Generate Table 1
    table1 = create_table1(eligible_df)
    
    # Save as markdown file
    results_dir = get_results_dir()
    with open(results_dir / 'table1.md', 'w') as f:
        f.write("# Table 1: Baseline Characteristics\n\n")
        f.write(table1.to_markdown(index=False))
    
    # Print success message
    print(f"Table 1 has been generated and saved as '{results_dir / 'table1.md'}'")
    
    # Calculate and display CHADS-VASc distribution
    print("\nCHADS-VASc Score Distribution:")
    chadsvasc_dist = calculate_chadsvasc_distribution(eligible_df)
    print(chadsvasc_dist[['Score', 'Formatted']].to_markdown(index=False))
    
    # Save CHADS-VASc distribution to markdown
    with open(results_dir / 'chadsvasc_distribution.md', 'w') as f:
        f.write("# CHADS-VASc Score Distribution\n\n")
        f.write(chadsvasc_dist[['Score', 'Formatted']].to_markdown(index=False)) 