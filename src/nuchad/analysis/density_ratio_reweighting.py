import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from nuchad.utils import get_data_path, get_results_dir
from nuchad.analysis.eda import get_df, calculate_chadsvasc
from nuchad.data_processing.eligibility_filters import filter_eligible_patients

# Create results directory if it doesn't exist
results_dir = get_results_dir()
# Don't need to explicitly create it as get_results_dir() already does this

def get_df(data_path=None):
    """Load and prepare the dataset"""
    if data_path is None:
        data_path = str(get_data_path('random_nuchad.csv'))
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    if 'patid' in df.columns:
        df = df.rename(columns={"patid": "patient_id"}).set_index("patient_id")
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Convert date columns to datetime
    date_cols = ['time1', 'time2', 'earliest_af_date', 'earliest_stroke_date', 'end_fu']
    for col in date_cols:
        if col in df.columns:
            if col in ['time1', 'time2']:
                df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors='coerce')
            else:
                df[col] = pd.to_datetime(df[col], format="%d%b%Y", errors='coerce')

    return df

def calculate_chadsvasc(row):
    """Calculates the CHADS-VASc score for a single patient"""
    score = 0
    # Congestive heart failure
    if 'hf' in row and pd.notna(row["hf"]):
        score += int(row["hf"])
    # Hypertension
    if 'hypertension' in row and pd.notna(row["hypertension"]):
        score += int(row["hypertension"])
    # Age >= 75 (2 points)
    if 'age' in row and pd.notna(row["age"]):
        score += 2 * int(row["age"] >= 75)
        # Age 65-74 (1 point)
        score += int(65 <= row["age"] < 75)
    # Diabetes mellitus
    if 'diab' in row and pd.notna(row["diab"]):
        score += int(row["diab"])
    # Stroke/TIA/Thromboembolism (2 points)
    if 'thrombo' in row and 'HB_stroke_history' in row:
        if pd.notna(row["thrombo"]) and pd.notna(row["HB_stroke_history"]):
            score += 2 * int(row["thrombo"] or row["HB_stroke_history"])
    # Vascular disease
    if 'vasc_dis_mi_pad' in row and pd.notna(row["vasc_dis_mi_pad"]):
        score += int(row["vasc_dis_mi_pad"])
    # Sex (Female)
    if 'gender' in row and pd.notna(row["gender"]):
        score += int(row["gender"] != 1)  # 1 = male, 2 = female
    return score

def filter_eligible_patients_legacy(df):
    """
    Filters patients who are eligible for the study
    """
    # Filter patients who have an AF diagnosis before time1
    af_diagnosis_mask = (df["earliest_af_date"] <= df["time1"])
    
    # Filter patients who have a follow-up period of at least 1 year
    follow_up_mask = df["end_fu"] >= df["time1"] + pd.Timedelta(days=365)
    
    # Eligibility: with AF and with enough followup
    eligible_mask = af_diagnosis_mask & follow_up_mask
    
    # Apply the mask to the DataFrame
    eligible_patients_df = df[eligible_mask].copy()
    
    # Calculate CHADS-VASc score
    eligible_patients_df['CHADS-Vasc'] = eligible_patients_df.apply(calculate_chadsvasc, axis=1)
    
    # Calculate follow-up time in years
    eligible_patients_df['Follow_Up_Years'] = (
        (eligible_patients_df['end_fu'] - eligible_patients_df['time1']).dt.days / 365.25
    )
    
    return eligible_patients_df

def prepare_data_for_weighting(df):
    """
    Prepare data for weighting by creating necessary variables
    
    Args:
        df: DataFrame with patient data
        
    Returns:
        DataFrame with prepared variables
    """
    # Create a copy
    df_prep = df.copy()
    
    # Create binary variables according to the sketch
    df_prep['Age'] = df_prep['age']  # Continuous variable
    df_prep['Female'] = (df_prep['gender'] == 2).astype(int)
    df_prep['Hypertension'] = df_prep['hypertension']
    df_prep['Diabetes'] = df_prep['diab']
    df_prep['HeartFailure'] = df_prep['hf']
    df_prep['StrokeTIA'] = df_prep['HB_stroke_history']
    df_prep['VascularDx'] = df_prep['vasc_dis_mi_pad']
    df_prep['CurrentSmoker'] = (df_prep['smoking_status'] == 'Current smoker').astype(int)
    
    return df_prep

def density_ratio_weighting(df):
    """
    Implement the density ratio weighting approach from the sketch
    
    Args:
        df: DataFrame with prepared variables
        
    Returns:
        DataFrame with weights added
    """
    # Deep copy
    df_weighted = df.copy()
    
    # Define original vs. UK parameters
    # These are from the sketch but updated using our actual UK data
    orig_params = {
        'Age':       {'mean': 63.9,  'sd': 10.6},
        'Female':    {'p': 0.4},
        'Hypertension': {'p': 4964/9722},  # ~0.51
        'Diabetes':     {'p': 1032/9722},  # ~0.11
        'HeartFailure': {'p': 1944/9722},  # ~0.20
        'StrokeTIA':    {'p': 1166/9722},  # ~0.12
        'VascularDx':   {'p': 1944/9722},  # ~0.20
        'CurrentSmoker':{'p': 1944/9722},  # ~0.20
    }
    
    # Compute UK parameters from the data
    uk_params = {
        'Age':       {
            'mean': df_weighted['Age'].mean(), 
            'sd': df_weighted['Age'].std()
        },
        'Female':    {'p': df_weighted['Female'].mean()},
        'Hypertension': {'p': df_weighted['Hypertension'].mean()},
        'Diabetes':     {'p': df_weighted['Diabetes'].mean()},
        'HeartFailure': {'p': df_weighted['HeartFailure'].mean()},
        'StrokeTIA':    {'p': df_weighted['StrokeTIA'].mean()},
        'VascularDx':   {'p': df_weighted['VascularDx'].mean()},
        'CurrentSmoker':{'p': df_weighted['CurrentSmoker'].mean()},
    }
    
    # Print parameters for comparison
    print("UK parameters computed from data:")
    for var, param in uk_params.items():
        if 'mean' in param:
            print(f"{var}: mean={param['mean']:.2f}, sd={param['sd']:.2f}")
        else:
            print(f"{var}: p={param['p']:.3f}")
    
    # Compute weights as in the sketch
    def compute_weight(row):
        # Initialize weight
        w = 1.0
        
        try:
            # Age (continuous variable)
            μo, σo = orig_params['Age']['mean'], orig_params['Age']['sd']
            μu, σu = uk_params['Age']['mean'], uk_params['Age']['sd']
            
            # Use log-space to avoid numerical issues
            log_w = np.log(norm.pdf(row['Age'], loc=μo, scale=σo)) - np.log(norm.pdf(row['Age'], loc=μu, scale=σu))
            
            # Binary variables
            binary_vars = ['Female', 'Hypertension', 'Diabetes', 'HeartFailure', 'StrokeTIA', 'VascularDx', 'CurrentSmoker']
            
            for var in binary_vars:
                p_o = orig_params[var]['p']
                p_u = uk_params[var]['p']
                
                # Skip if proportions are 0 or 1 to avoid division by zero
                if p_o == 0 or p_o == 1 or p_u == 0 or p_u == 1:
                    continue
                    
                x = row[var]
                
                # Bernoulli pmf ratio in log space
                if x == 1:
                    log_w += np.log(p_o) - np.log(p_u)
                else:
                    log_w += np.log(1-p_o) - np.log(1-p_u)
            
            # Convert from log space
            w = np.exp(log_w)
            
            # Cap weights to avoid extreme values
            MAX_WEIGHT = 50.0  # Cap weights at 50
            if w > MAX_WEIGHT:
                w = MAX_WEIGHT
            
            return w
            
        except Exception as e:
            print(f"Error computing weight: {e}")
            return 1.0  # Default weight on error
    
    # Calculate weights
    df_weighted['weight'] = df_weighted.apply(compute_weight, axis=1)
    
    # Normalize weights to sum to n
    n = len(df_weighted)
    sum_w = df_weighted['weight'].sum()
    df_weighted['weight'] = df_weighted['weight'] * (n / sum_w)
    
    # Print weight statistics
    print("\nWeight statistics:")
    print(df_weighted['weight'].describe())
    
    # Calculate effective sample size
    ess = (df_weighted['weight'].sum() ** 2) / (df_weighted['weight'] ** 2).sum()
    print(f"Effective sample size: {ess:.1f} ({ess/n*100:.1f}% of original)")
    
    # Make sure Follow_Up_Years is calculated if not already present
    if 'Follow_Up_Years' not in df_weighted.columns and 'time1' in df_weighted.columns and 'end_fu' in df_weighted.columns:
        df_weighted['Follow_Up_Years'] = (df_weighted['end_fu'] - df_weighted['time1']).dt.days / 365.25
    
    return df_weighted

def evaluate_chadsvasc(df_weighted):
    """
    Evaluate the performance of CHADS-VASc score with and without weighting
    
    Args:
        df_weighted: DataFrame with patient data and weights
        
    Returns:
        Dictionary with evaluation results
    """
    from sklearn.metrics import roc_auc_score
    
    # Check the data to understand stroke rates
    n_patients = len(df_weighted)
    n_strokes = (df_weighted['stroke_1Y'] == 1).sum()
    total_py = df_weighted['Follow_Up_Years'].sum()
    overall_rate = (n_strokes / total_py) * 100
    
    print(f"\nData inspection:")
    print(f"Total patients: {n_patients}")
    print(f"Total strokes (stroke_1Y = 1): {n_strokes}")
    print(f"Total patient-years: {total_py:.1f}")
    print(f"Overall stroke rate per 100 person-years: {overall_rate:.2f}")
    
    # Prepare results
    results = {}
    
    # Original (unweighted) AUC
    original_auc = roc_auc_score(df_weighted['stroke_1Y'] == 1, df_weighted['CHADS-Vasc'])
    results['original_auc'] = original_auc
    
    # Weighted AUC
    weighted_auc = roc_auc_score(
        df_weighted['stroke_1Y'] == 1, 
        df_weighted['CHADS-Vasc'], 
        sample_weight=df_weighted['weight']
    )
    results['weighted_auc'] = weighted_auc
    
    print(f"Original AUC: {original_auc:.3f}")
    print(f"Weighted AUC: {weighted_auc:.3f}")
    
    # Lip et al. original rates (per 100 person-years)
    lip_original_rates = {
        0: 0.2,
        1: 0.6,
        2: 2.2,
        3: 3.2,
        4: 4.8,
        5: 7.2,
        6: 9.7,
        7: 11.2,
        8: 10.8,
        9: 12.2
    }
    
    # Calculate stroke rates by CHADS-VASc score
    original_rates = {}
    weighted_rates = {}
    
    for score in sorted(df_weighted['CHADS-Vasc'].unique()):
        # Filter for this score
        score_df = df_weighted[df_weighted['CHADS-Vasc'] == score]
        
        # Original rate (per 100 person-years)
        n_patients = len(score_df)
        n_strokes = sum(score_df['stroke_1Y'] == 1)  # Count where stroke_1Y is 1 (Yes)
        patient_years = score_df['Follow_Up_Years'].sum()
        
        if patient_years > 0:
            orig_rate = (n_strokes / patient_years) * 100
            original_rates[score] = {
                'rate': orig_rate,
                'n_strokes': n_strokes,
                'n_patients': n_patients,
                'patient_years': patient_years
            }
        
        # Weighted rate (per 100 person-years)
        weighted_patient_years = (score_df['Follow_Up_Years'] * score_df['weight']).sum()
        weighted_strokes = ((score_df['stroke_1Y'] == 1) * score_df['weight']).sum()
        
        if weighted_patient_years > 0:
            weighted_rate = (weighted_strokes / weighted_patient_years) * 100
            weighted_rates[score] = {
                'rate': weighted_rate,
                'weighted_strokes': weighted_strokes,
                'weighted_patient_years': weighted_patient_years
            }
    
    results['lip_original_rates'] = lip_original_rates
    results['original_rates'] = original_rates
    results['weighted_rates'] = weighted_rates
    
    return results

def plot_results(results, save_path):
    """
    Plot the results of the evaluation
    
    Args:
        results: Dictionary with evaluation results
        save_path: Path to save the plot
    """
    # Extract data
    scores = sorted(results['original_rates'].keys())
    lip_rates = [results['lip_original_rates'].get(s, 0) for s in scores]
    orig_rates = [results['original_rates'][s]['rate'] for s in scores]
    weighted_rates = [results['weighted_rates'][s]['rate'] for s in scores]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(scores, lip_rates, 'D-', label=f"Lip et al. (Original)")
    plt.plot(scores, orig_rates, 'o-', label=f"Observed (AUC={results['original_auc']:.3f})")
    plt.plot(scores, weighted_rates, 's--', label=f"Weighted (AUC={results['weighted_auc']:.3f})")
    
    plt.xlabel('CHADS-VASc Score')
    plt.ylabel('Stroke Rate (per 100 person-years)')
    plt.title('Stroke Rates by CHADS-VASc Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(scores)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_results_to_markdown(results, df_weighted, filepath):
    """
    Save the results to a markdown file
    
    Args:
        results: Dictionary with evaluation results
        df_weighted: DataFrame with patient data and weights
        filepath: Path to save the markdown file
    """
    with open(filepath, 'w') as f:
        f.write("# CHADS-VASc Density Ratio Reweighting Results\n\n")
        
        # Weight statistics
        f.write("## Weight Statistics\n\n")
        f.write(f"- Number of patients: {len(df_weighted)}\n")
        f.write(f"- Weight range: {df_weighted['weight'].min():.2f} - {df_weighted['weight'].max():.2f}\n")
        f.write(f"- Weight mean: {df_weighted['weight'].mean():.2f}\n")
        
        ess = (df_weighted['weight'].sum() ** 2) / (df_weighted['weight'] ** 2).sum()
        f.write(f"- Effective sample size: {ess:.1f} ({ess/len(df_weighted)*100:.1f}% of original)\n\n")
        
        # Performance metrics
        f.write("## Performance Metrics\n\n")
        f.write(f"- Original AUC: {results['original_auc']:.3f}\n")
        f.write(f"- Weighted AUC: {results['weighted_auc']:.3f}\n\n")
        
        # Stroke rates by score
        f.write("## Stroke Rates by CHADS-VASc Score (per 100 person-years)\n\n")
        f.write("| CHADS-VASc | Lip et al. Rate | Observed Rate | Weighted Rate |\n")
        f.write("|------------|-----------------|---------------|---------------|\n")
        
        for score in sorted(results['original_rates'].keys()):
            lip = results['lip_original_rates'].get(score, "N/A")
            orig = results['original_rates'][score]['rate']
            weighted = results['weighted_rates'][score]['rate']
            
            f.write(f"| {score} | {lip} | {orig:.2f} | {weighted:.2f} |\n")
            
        # Figures
        f.write("\n## Figures\n\n")
        f.write("![Stroke Rates](density_ratio_weighted_rates.png)\n")

def plot_weight_distribution(df_weighted, save_path):
    """
    Plot the distribution of weights
    
    Args:
        df_weighted: DataFrame with weight column
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df_weighted['weight'], bins=50, edgecolor='black')
    plt.xlabel('Weight')
    plt.ylabel('Count')
    plt.title('Distribution of Density Ratio Weights')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=1, color='red', linestyle='--', label='Weight=1')
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Weight distribution plot saved to {save_path}")
    plt.close()

def perform_reweighting_analysis(df=None):
    """
    Perform the full density ratio reweighting analysis.
    
    Args:
        df: Optional pre-loaded DataFrame. If None, will load and prepare the data.
        
    Returns:
        DataFrame with the weighted results
    """
    try:
        # If df is not provided, load and prepare it
        if df is None:
            df = get_df()
            df, _ = filter_eligible_patients(df)
        
        # Make sure we have the Follow_Up_Years column
        if 'Follow_Up_Years' not in df.columns and 'time1' in df.columns and 'end_fu' in df.columns:
            df['Follow_Up_Years'] = (df['end_fu'] - df['time1']).dt.days / 365.25
        
        # Calculate CHADS-VASc score if not already present
        if 'CHADS-Vasc' not in df.columns:
            df['CHADS-Vasc'] = df.apply(calculate_chadsvasc, axis=1)
        
        # Prepare data for weighting
        df_prep = prepare_data_for_weighting(df)
        
        # Make sure CHADS-Vasc score is preserved in prepared data
        if 'CHADS-Vasc' not in df_prep.columns and 'CHADS-Vasc' in df.columns:
            df_prep['CHADS-Vasc'] = df['CHADS-Vasc']
        
        # Compute weights
        df_weighted = density_ratio_weighting(df_prep)
        
        # Make sure we have the Follow_Up_Years column in the weighted dataframe too
        if 'Follow_Up_Years' not in df_weighted.columns and 'time1' in df_weighted.columns and 'end_fu' in df_weighted.columns:
            df_weighted['Follow_Up_Years'] = (df_weighted['end_fu'] - df_weighted['time1']).dt.days / 365.25
        elif 'Follow_Up_Years' in df.columns and 'Follow_Up_Years' not in df_weighted.columns:
            df_weighted['Follow_Up_Years'] = df['Follow_Up_Years']
        
        # Make sure CHADS-Vasc score is preserved in weighted data
        if 'CHADS-Vasc' not in df_weighted.columns and 'CHADS-Vasc' in df.columns:
            df_weighted['CHADS-Vasc'] = df['CHADS-Vasc']
        
        # Evaluate CHADS-VASc with and without weights
        results = evaluate_chadsvasc(df_weighted)
        
        # Save results
        save_path = results_dir / 'density_ratio_weighted_rates.png'
        plot_results(results, save_path)
        
        # Save weight distribution plot
        weight_plot_path = results_dir / 'density_ratio_weight_distribution.png'
        plot_weight_distribution(df_weighted, weight_plot_path)
        
        # Save results to markdown
        results_markdown_path = results_dir / 'density_ratio_results.md'
        save_results_to_markdown(results, df_weighted, results_markdown_path)
        
        print(f"Density ratio weighting analysis completed. Results saved to {results_dir}")
        
        return df_weighted
    
    except Exception as e:
        print(f"Error in density ratio reweighting analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run the density ratio weighting analysis"""
    df = get_df()
    df, _ = filter_eligible_patients(df)
    
    # Prepare data for weighting
    df_prep = prepare_data_for_weighting(df)
    
    # Apply density ratio weighting
    df_weighted = density_ratio_weighting(df_prep)
    
    # Evaluate CHADS-VASc performance
    results = evaluate_chadsvasc(df_weighted)
    
    # Plot weight distribution
    weight_dist_path = results_dir / 'density_ratio_weight_distribution.png'
    plot_weight_distribution(df_weighted, str(weight_dist_path))
    print(f"Weight distribution plot saved to '{weight_dist_path}'")
    
    # Plot results
    plot_path = results_dir / 'density_ratio_weighted_rates.png'
    plot_results(results, str(plot_path))
    print(f"Results plot saved to '{plot_path}'")
    
    # Save results to markdown
    results_path = results_dir / 'density_ratio_results.md'
    save_results_to_markdown(results, df_weighted, str(results_path))
    print(f"Results saved to '{results_path}'")

if __name__ == "__main__":
    main() 