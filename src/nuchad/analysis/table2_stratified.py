import pandas as pd
import numpy as np
import os
from scipy import stats
from nuchad.analysis.eda_old import get_df, calculate_chadsvasc, calculate_stroke_rate, confidence_interval
from nuchad.data_processing.eligibility_filters import filter_eligible_patients
from nuchad.utils import get_results_dir

# Create directories if they don't exist
os.makedirs('../results', exist_ok=True)

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
    
def create_stratified_table2(df):
    """
    Create a Table 2 with statistics stratified by CHADS-VASc score groups for eligible patients
    
    Args:
        df: DataFrame with eligible patient data
        
    Returns:
        DataFrame formatted as Table 2 with stratification
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
                
                # Get unique values across all groups
                all_values = set()
                for series in [df_mapped, low_mapped, moderate_mapped, high_mapped]:
                    all_values.update(series.dropna().unique())
                all_values = sorted(all_values)
                
                # Compute p-value
                p_value = compute_p_value([low_risk, moderate_risk, high_risk], var, var_type)
                
                # For each category value, calculate counts and percentages
                for val in all_values:
                    # Total counts
                    total_count = df_mapped[df_mapped == val].count()
                    total_pct = total_count / n_total * 100 if n_total > 0 else 0
                    total_str = f"{total_count} ({total_pct:.1f}%)" if n_total > 0 else "0 (0.0%)"
                    
                    # Low risk counts
                    low_count = low_mapped[low_mapped == val].count()
                    low_pct = low_count / n_low * 100 if n_low > 0 else 0
                    low_str = f"{low_count} ({low_pct:.1f}%)" if n_low > 0 else "0 (0.0%)"
                    
                    # Moderate risk counts
                    moderate_count = moderate_mapped[moderate_mapped == val].count()
                    moderate_pct = moderate_count / n_moderate * 100 if n_moderate > 0 else 0
                    moderate_str = f"{moderate_count} ({moderate_pct:.1f}%)" if n_moderate > 0 else "0 (0.0%)"
                    
                    # High risk counts
                    high_count = high_mapped[high_mapped == val].count()
                    high_pct = high_count / n_high * 100 if n_high > 0 else 0
                    high_str = f"{high_count} ({high_pct:.1f}%)" if n_high > 0 else "0 (0.0%)"
                    
                    rows.append({
                        "Variable": f"  {val}",
                        total_header: total_str,
                        low_header: low_str,
                        moderate_header: moderate_str,
                        high_header: high_str,
                        "P-value": p_value if val == all_values[0] else ""  # Only show p-value for first category
                    })
    
    # Create DataFrame from rows
    table2_df = pd.DataFrame(rows)
    
    return table2_df

def compute_continuous_stats(df, var):
    """Compute mean ± SD for continuous variables"""
    valid_values = df[var].dropna()
    if len(valid_values) > 0:
        mean = valid_values.mean()
        std = valid_values.std()
        return f"{mean:.2f} ± {std:.2f}"
    else:
        return "No data"

def get_df_local():
    """
    Load the dataset, with columns properly formatted.
    """
    # load data
    df = pd.read_csv("../data/random_nuchad.csv").rename(columns={"patid": "patient_id"}).set_index("patient_id")
    df = df.drop(columns=["Unnamed: 0"])

    # convert time1 and time2 to datetime objects
    df["time1"] = pd.to_datetime(df["time1"], format="%Y-%m-%d", errors="raise")
    df["time2"] = pd.to_datetime(df["time2"], format="%Y-%m-%d", errors="raise")

    df["earliest_af_date"] = pd.to_datetime(df["earliest_af_date"], format="%d%b%Y", errors="raise")
    df["earliest_stroke_date"] = pd.to_datetime(df["earliest_stroke_date"], format="%d%b%Y", errors="raise")
    df["end_fu"] = pd.to_datetime(df["end_fu"], format="%d%b%Y", errors="raise")

    return df

def generate_stratified_table2(df=None):
    """
    Generate stratified cohort tables showing characteristics and stroke rates 
    by CHADS-VASc score and save to the results directory.
    
    Args:
        df: Optional pre-loaded DataFrame. If None, will load and prepare the data.
        
    Returns:
        Dictionary with both stratified tables
    """
    # Load data if not provided
    if df is None:
        df = get_df()
        # Use the filtering function from data_processing.eligibility_filters
        df, _ = filter_eligible_patients(df)
    
    # Generate detailed stratified Table 2 with characteristics
    detailed_table = create_stratified_table2(df)
    
    # Save detailed table as markdown file in results directory
    results_dir = get_results_dir()
    with open(results_dir / 'table2_stratified_characteristics.md', 'w') as f:
        f.write("# Table 2: Characteristics of Eligible Patients Stratified by CHADS-VASc Risk\n\n")
        f.write(detailed_table.to_markdown(index=False))
    
    # Generate stratified rates table by anticoagulation status
    rates_table = generate_stratified_rates_table(df)
    
    # Save rates table as markdown file
    with open(results_dir / 'table2_stratified.md', 'w') as f:
        f.write("# Table 2: Stroke Rates Stratified by CHADS-VASc Score and Anticoagulation Status\n\n")
        f.write(rates_table.to_markdown(index=False))
    
    # Print success message
    print(f"Stratified Table 2 has been generated and saved as '{results_dir / 'table2_stratified.md'}'")
    
    return {
        'characteristics': detailed_table,
        'rates': rates_table
    }

def generate_stratified_rates_table(df):
    """
    Generate a stratified cohort table showing stroke rates by CHADS-VASc score and anticoagulation status.
    
    Args:
        df: DataFrame with patient data
        
    Returns:
        DataFrame with stratified rates
    """
    # Calculate CHADS-VASc score if not already present
    if 'chadsvasc' not in df.columns and 'CHADS-Vasc' not in df.columns:
        df['chadsvasc'] = df.apply(calculate_chadsvasc, axis=1)
    
    # Use the appropriate column name
    score_col = 'chadsvasc' if 'chadsvasc' in df.columns else 'CHADS-Vasc'
    
    # Create a binary anticoagulation variable if not already present
    if 'anticoag_binary' not in df.columns:
        df['anticoag_binary'] = df['Anticoagulant'].apply(
            lambda x: 0 if x == 'No anticoagulant' else 1
        )
    
    # Group by CHADS-VASc score and anticoagulation status
    grouped = df.groupby([score_col, 'anticoag_binary'])
    
    # Create results dataframe
    results = []
    
    for (score, anticoag), group in grouped:
        num_patients = len(group)
        
        # Calculate patient years (difference between time1 and end_fu in years)
        if 'follow_up_years' in group.columns:
            total_patient_years = group['follow_up_years'].sum()
        else:
            group_copy = group.copy()
            group_copy['follow_up_years'] = (group_copy['end_fu'] - group_copy['time1']).dt.days / 365.25
            total_patient_years = group_copy['follow_up_years'].sum()
        
        # Count strokes
        strokes = group['stroke_1Y'].sum()
        
        # Calculate stroke rate per 100 patient-years
        stroke_rate = (strokes / total_patient_years) * 100 if total_patient_years > 0 else 0
        
        # Calculate 95% confidence interval
        if strokes > 0 and total_patient_years > 0:
            ci_lower = (stroke_rate * np.exp(-1.96 / np.sqrt(strokes)))
            ci_upper = (stroke_rate * np.exp(1.96 / np.sqrt(strokes)))
        else:
            ci_lower = 0
            ci_upper = 0
        
        results.append({
            'CHADS-VASc Score': score,
            'Anticoagulation': 'Yes' if anticoag == 1 else 'No',
            'Number of Patients': num_patients,
            'Total Patient-Years': round(total_patient_years, 1),
            'Number of Strokes': int(strokes),
            'Stroke Rate (per 100 patient-years)': round(stroke_rate, 2),
            '95% CI Lower': round(ci_lower, 2),
            '95% CI Upper': round(ci_upper, 2)
        })
    
    # Convert to DataFrame and sort by CHADS-VASc score and anticoagulation status
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(['CHADS-VASc Score', 'Anticoagulation'])
    
    return results_df

if __name__ == "__main__":
    # Generate and save stratified tables
    results = generate_stratified_table2()
    
    # Print the rates table
    print(results['rates'].to_markdown(index=False)) 