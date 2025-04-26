import pandas as pd
import numpy as np
from scipy import stats
from eda import get_df, filter_eligible_patients, calculate_chadsvasc

def stratify_by_chadsvasc(df):
    """
    Stratify patients by CHADS-VASc score groups:
    - Low risk: 0-1
    - Moderate risk: 2-3
    - High risk: ≥4
    
    Args:
        df: DataFrame with patient data
        
    Returns:
        Tuple of (low_risk_df, moderate_risk_df, high_risk_df)
    """
    # Calculate CHADS-VASc scores if not already present
    if 'CHADS-Vasc' not in df.columns:
        df['CHADS-Vasc'] = df.apply(calculate_chadsvasc, axis=1)
    
    # Create stratified groups
    low_risk = df[df['CHADS-Vasc'].between(0, 1)]
    moderate_risk = df[df['CHADS-Vasc'].between(2, 3)]
    high_risk = df[df['CHADS-Vasc'] >= 4]
    
    return (low_risk, moderate_risk, high_risk)

def compute_p_value(groups, var, var_type):
    """
    Compute p-value for comparison between groups
    
    Args:
        groups: List of DataFrames for each group
        var: Variable name to compare
        var_type: Type of variable ('continuous' or 'categorical')
        
    Returns:
        p-value as string with appropriate formatting
    """
    if var_type == 'continuous':
        # Use ANOVA for continuous variables
        valid_groups = []
        for group in groups:
            valid_values = group[var].dropna()
            if len(valid_values) > 0:
                valid_groups.append(valid_values)
        
        if len(valid_groups) >= 2:
            try:
                f_stat, p_val = stats.f_oneway(*valid_groups)
                return format_p_value(p_val)
            except:
                return "N/A"
        else:
            return "N/A"
    
    elif var_type == 'categorical':
        # Use Chi-squared test for categorical variables
        contingency_table = pd.DataFrame()
        for i, group in enumerate(groups):
            value_counts = group[var].value_counts().sort_index()
            contingency_table[f'Group_{i+1}'] = value_counts
        
        # Fill NaN with 0
        contingency_table.fillna(0, inplace=True)
        
        if contingency_table.shape[0] >= 2 and contingency_table.shape[1] >= 2:
            try:
                chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)
                return format_p_value(p_val)
            except:
                return "N/A"
        else:
            return "N/A"
    
    return "N/A"

def format_p_value(p_val):
    """Format p-value with appropriate notation"""
    if p_val < 0.001:
        return "<0.001"
    elif p_val < 0.01:
        return f"{p_val:.3f}"
    else:
        return f"{p_val:.2f}"
    
def create_stratified_table1(df):
    """
    Create a Table 1 with statistics stratified by CHADS-VASc score groups
    
    Args:
        df: DataFrame with patient data
        
    Returns:
        DataFrame formatted as Table 1 with stratification
    """
    # Stratify patients
    low_risk, moderate_risk, high_risk = stratify_by_chadsvasc(df)
    
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
    
    # Add table header
    n_total = len(df)
    n_low = len(low_risk)
    n_moderate = len(moderate_risk)
    n_high = len(high_risk)
    
    total_header = f"Total\n(N={n_total})"
    low_header = f"Low Risk\n(CHA₂DS₂-VASc 0-1)\n(n={n_low}, {n_low/n_total*100:.1f}%)"
    moderate_header = f"Moderate Risk\n(CHA₂DS₂-VASc 2-3)\n(n={n_moderate}, {n_moderate/n_total*100:.1f}%)"
    high_header = f"High Risk\n(CHA₂DS₂-VASc ≥4)\n(n={n_high}, {n_high/n_total*100:.1f}%)"
    
    # Column names for the DataFrame
    columns = ["Variable", total_header, low_header, moderate_header, high_header, "P-value"]
    
    for group_name, variables in variable_groups:
        # Add group header
        rows.append({
            "Variable": f"**{group_name}**", 
            total_header: "", 
            low_header: "", 
            moderate_header: "", 
            high_header: "", 
            "P-value": ""
        })
        
        for var, var_type in variables.items():
            if var not in df.columns:
                continue
                
            display_name = display_names.get(var, var)
            
            # Handle different variable types
            if var_type == 'continuous':
                # For continuous variables, compute mean ± SD for each group
                stats_total = compute_continuous_stats(df, var)
                stats_low = compute_continuous_stats(low_risk, var)
                stats_moderate = compute_continuous_stats(moderate_risk, var)
                stats_high = compute_continuous_stats(high_risk, var)
                
                # Compute p-value
                p_value = compute_p_value([low_risk, moderate_risk, high_risk], var, var_type)
                
                rows.append({
                    "Variable": display_name,
                    total_header: stats_total,
                    low_header: stats_low,
                    moderate_header: stats_moderate,
                    high_header: stats_high,
                    "P-value": p_value
                })
                
            elif var_type == 'categorical':
                # Add header row for categorical variable
                rows.append({
                    "Variable": display_name,
                    total_header: "",
                    low_header: "",
                    moderate_header: "",
                    high_header: "",
                    "P-value": ""
                })
                
                # Apply mappings if needed
                df_mapped = df[var].copy()
                low_mapped = low_risk[var].copy()  
                moderate_mapped = moderate_risk[var].copy()
                high_mapped = high_risk[var].copy()
                
                if var == 'gender':
                    df_mapped = df_mapped.map(gender_mapping)
                    low_mapped = low_mapped.map(gender_mapping)
                    moderate_mapped = moderate_mapped.map(gender_mapping)
                    high_mapped = high_mapped.map(gender_mapping)
                elif var in ['af', 'hypertension', 'diab', 'thrombo', 'hf', 'HB_stroke_history', 
                           'ckd', 'vasc_dis_mi_pad', 'aortic_plaq', 'stroke_1Y', 'end_fu_due_to_death']:
                    df_mapped = df_mapped.map(binary_mapping)
                    low_mapped = low_mapped.map(binary_mapping)
                    moderate_mapped = moderate_mapped.map(binary_mapping)
                    high_mapped = high_mapped.map(binary_mapping)
                
                # Get all possible categories across all groups
                all_categories = set()
                for series in [df_mapped, low_mapped, moderate_mapped, high_mapped]:
                    for cat in series.dropna().unique():
                        all_categories.add(cat)
                
                all_categories = sorted(all_categories)
                
                # Compute p-value for the overall variable
                p_value = compute_p_value([low_risk, moderate_risk, high_risk], var, var_type)
                
                # Calculate counts for each category in each group
                for category in all_categories:
                    count_total = (df_mapped == category).sum()
                    pct_total = count_total / len(df) * 100
                    
                    count_low = (low_mapped == category).sum()
                    pct_low = count_low / len(low_risk) * 100 if len(low_risk) > 0 else 0
                    
                    count_moderate = (moderate_mapped == category).sum()
                    pct_moderate = count_moderate / len(moderate_risk) * 100 if len(moderate_risk) > 0 else 0
                    
                    count_high = (high_mapped == category).sum()
                    pct_high = count_high / len(high_risk) * 100 if len(high_risk) > 0 else 0
                    
                    stats_total = f"{count_total} ({pct_total:.1f}%)"
                    stats_low = f"{count_low} ({pct_low:.1f}%)"
                    stats_moderate = f"{count_moderate} ({pct_moderate:.1f}%)"
                    stats_high = f"{count_high} ({pct_high:.1f}%)"
                    
                    # Add row for this category
                    rows.append({
                        "Variable": f"  {category}",
                        total_header: stats_total,
                        low_header: stats_low,
                        moderate_header: stats_moderate,
                        high_header: stats_high,
                        "P-value": p_value if category == all_categories[0] else ""  # Only show p-value on first category
                    })
    
    # Create DataFrame from rows
    table1_df = pd.DataFrame(rows, columns=columns)
    
    return table1_df

def compute_continuous_stats(df, var):
    """Compute mean ± SD for continuous variable"""
    valid_values = df[var].dropna()
    if len(valid_values) > 0:
        mean = valid_values.mean()
        std = valid_values.std()
        return f"{mean:.2f} ± {std:.2f}"
    else:
        return "No data"

if __name__ == "__main__":
    # Load data
    df = get_df()
    
    # Filter eligible patients
    eligible_df = filter_eligible_patients(df)
    
    # Generate stratified Table 1
    table1 = create_stratified_table1(eligible_df)
    
    # Save as markdown file
    with open('table1_stratified.md', 'w') as f:
        f.write("# Table 1: Baseline Characteristics Stratified by CHA₂DS₂-VASc Risk Groups\n\n")
        f.write(table1.to_markdown(index=False))
    
    # Print success message
    print("Stratified Table 1 has been generated and saved as 'table1_stratified.md'") 