import os
import pandas as pd
import numpy as np
from eda import get_df, calculate_chadsvasc, filter_eligible_patients

def generate_cohort_table():
    """
    Generate a cohort table showing stroke rates by CHADS-VASc score.
    Save results to a markdown file.
    """
    # Ensure results directory exists
    os.makedirs("../results", exist_ok=True)
    
    # Get the dataframe using the existing function from eda
    df = get_df()
    
    # Filter eligible patients
    eligible_patients_df = filter_eligible_patients(df)
    
    # Calculate CHADS-VASc score for each patient
    eligible_patients_df['chadsvasc'] = eligible_patients_df.apply(calculate_chadsvasc, axis=1)
    
    # Group by CHADS-VASc score
    grouped = eligible_patients_df.groupby('chadsvasc')
    
    # Create results dataframe
    results = []
    
    for score, group in grouped:
        num_patients = len(group)
        
        # Calculate patient years (difference between time1 and end_fu in years)
        group['follow_up_years'] = (group['end_fu'] - group['time1']).dt.days / 365.25
        total_patient_years = group['follow_up_years'].sum()
        
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
            'Number of Patients': num_patients,
            'Total Patient-Years': round(total_patient_years, 1),
            'Number of Strokes': int(strokes),
            'Stroke Rate (per 100 patient-years)': round(stroke_rate, 2),
            '95% CI Lower': round(ci_lower, 2),
            '95% CI Upper': round(ci_upper, 2)
        })
    
    # Convert to DataFrame and sort by CHADS-VASc score
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('CHADS-VASc Score')
    
    # Save to markdown
    results_df.to_markdown('../results/table2.md', index=False)
    
    return results_df

if __name__ == "__main__":
    results_table = generate_cohort_table()
    print(results_table.to_markdown(index=False)) 