import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import os
import sys

# Adjust path for both running from root or src directory
if os.path.basename(os.getcwd()) == 'src':
    # We're running from src directory
    DATA_DIR = '../data'
    RESULTS_DIR = '../results'
else:
    # We're running from root directory
    DATA_DIR = 'data'
    RESULTS_DIR = 'results'

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_df(data_path=None):
    """Load and prepare the dataset"""
    if data_path is None:
        data_path = os.path.join(DATA_DIR, 'random_nuchad.csv')
    
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
    if ('thrombo' in row and pd.notna(row["thrombo"])) and ('HB_stroke_history' in row and pd.notna(row["HB_stroke_history"])):
        score += 2 * int(row["thrombo"] or row["HB_stroke_history"])
    # Vascular disease
    if 'vasc_dis_mi_pad' in row and pd.notna(row["vasc_dis_mi_pad"]):
        score += int(row["vasc_dis_mi_pad"])
    # Sex (Female)
    if 'gender' in row and pd.notna(row["gender"]):
        score += int(row["gender"] != 1)  # 1 = male, 2 = female
    return score

def filter_eligible_patients(df):
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
    
    # Add a flag indicating complete follow-up
    eligible_patients_df['fu_complete'] = 1
    
    return eligible_patients_df

def map_variables(df):
    """
    Map variables to align with the original study
    
    Args:
        df: DataFrame with UK study data
        
    Returns:
        DataFrame with mapped variables
    """
    # Create a deep copy to avoid modifying the original
    new_df = df.copy()
    
    # Map gender (1 = male, 2 = female in our data)
    # We'll create a binary indicator for female
    new_df['is_female'] = (new_df['gender'] == 2).astype(int)
    
    # Map binary variables (0 = No, 1 = Yes in our data)
    new_df['hypertension_bin'] = new_df['hypertension']
    new_df['diabetes_bin'] = new_df['diab']
    new_df['heart_failure_bin'] = new_df['hf']
    new_df['stroke_hist_bin'] = new_df['HB_stroke_history']
    new_df['vascular_disease_bin'] = new_df['vasc_dis_mi_pad']
    
    # Map smoking (assumption: 'Current smoker' is the only category we count as smoking)
    new_df['is_smoker'] = (new_df['smoking_status'] == 'Current smoker').astype(int)
    
    return new_df

def compute_parameters(df, variable_keys):
    """
    Compute parameters (means and standard deviations) for the variables
    
    Args:
        df: DataFrame with mapped variables
        variable_keys: List of variable keys to compute parameters for
        
    Returns:
        Dict with variable parameters (mean and std)
    """
    params = {}
    
    for var in variable_keys:
        if var in df.columns:
            params[f"{var}_mean"] = df[var].mean()
            params[f"{var}_std"] = df[var].std()
    
    return params

def calculate_weights(df, original_params, variables):
    """
    Calculate weights using the density ratio approach
    
    Args:
        df: DataFrame with mapped variables (UK cohort)
        original_params: Dict with original study parameters
        variables: List of variables to use for weighting
        
    Returns:
        Array of weights for each patient
    """
    def negative_log_likelihood(beta):
        # Initialize log likelihood
        log_lik = 0
        
        # For each variable, add its contribution
        for i, var in enumerate(variables):
            # Get the mean and std from original study
            mean_orig = original_params[f"{var}_mean"]
            std_orig = original_params[f"{var}_std"]
            
            # Calculate contribution to log likelihood
            z = (df[var] - mean_orig) / std_orig
            log_lik += beta[i] * z.mean()
            log_lik += 0.5 * (beta[i] ** 2)
        
        return -log_lik
    
    # Initial guess for beta (zero for all variables)
    beta_init = np.zeros(len(variables))
    
    # Minimize negative log likelihood
    result = minimize(negative_log_likelihood, beta_init, method='BFGS')
    beta_opt = result.x
    
    # Calculate weights using the optimal beta
    weights = np.ones(len(df))
    
    for i, var in enumerate(variables):
        mean_orig = original_params[f"{var}_mean"]
        std_orig = original_params[f"{var}_std"]
        z = (df[var] - mean_orig) / std_orig
        weights *= np.exp(beta_opt[i] * z)
    
    # Normalize weights to sum to n
    weights = weights * len(df) / sum(weights)
    
    return weights

def trim_weights(weights, quantile=0.99):
    """
    Trim extreme weights to reduce variance
    
    Args:
        weights: Array of weights
        quantile: Quantile to trim at (e.g., 0.99 trims the top 1%)
        
    Returns:
        Trimmed weights
    """
    threshold = np.quantile(weights, quantile)
    trimmed_weights = np.minimum(weights, threshold)
    
    # Re-normalize to sum to n
    trimmed_weights = trimmed_weights * len(weights) / sum(trimmed_weights)
    
    return trimmed_weights

def plot_weight_distribution(weights, trimmed_weights, save_path):
    """
    Plot the distribution of weights
    
    Args:
        weights: Original weights
        trimmed_weights: Trimmed weights
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    plt.hist(weights, bins=50, alpha=0.5, label='Original Weights')
    plt.hist(trimmed_weights, bins=50, alpha=0.5, label='Trimmed Weights')
    
    # Add vertical lines for mean values
    plt.axvline(weights.mean(), color='blue', linestyle='--', 
                label=f'Mean Original: {weights.mean():.4f}')
    plt.axvline(trimmed_weights.mean(), color='orange', linestyle='--', 
                label=f'Mean Trimmed: {trimmed_weights.mean():.4f}')
    
    # Add labels and title
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Weights')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def effective_sample_size(weights):
    """
    Calculate the effective sample size of weighted data
    
    Args:
        weights: Array of weights
        
    Returns:
        Effective sample size and percentage of original
    """
    n = len(weights)
    ess = (sum(weights) ** 2) / sum(weights ** 2)
    pct = ess / n * 100
    
    return ess, pct

def reweight_chadsvasc_performance(df, weights, outcome_col='stroke_1Y', n_bootstrap=15):
    """
    Evaluate CHADS-VASc performance with and without weighting
    
    Args:
        df: DataFrame with CHADS-VASc scores and outcome
        weights: Array of weights for each patient
        outcome_col: Column name for the outcome variable
        n_bootstrap: Number of bootstrap iterations for confidence intervals
    
    Returns:
        DataFrame with performance metrics
    """
    # Make sure CHADS-VASc scores are calculated
    if 'CHADS-Vasc' not in df.columns:
        df['CHADS-Vasc'] = df.apply(calculate_chadsvasc, axis=1)
    
    # Original (unweighted) AUC
    original_auc = roc_auc_score(df[outcome_col], df['CHADS-Vasc'])
    
    # Weighted AUC
    weighted_auc = roc_auc_score(df[outcome_col], df['CHADS-Vasc'], sample_weight=weights)
    
    # Bootstrap confidence intervals (reduced iterations to speed up computation)
    original_aucs = []
    weighted_aucs = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample indices
        indices = np.random.choice(len(df), len(df), replace=True)
        
        # Bootstrap sample
        boot_df = df.iloc[indices].copy()
        boot_weights = weights[indices]
        
        # AUCs
        if sum(boot_df[outcome_col]) > 0:  # Ensure positive cases exist
            original_aucs.append(roc_auc_score(boot_df[outcome_col], boot_df['CHADS-Vasc']))
            weighted_aucs.append(roc_auc_score(boot_df[outcome_col], boot_df['CHADS-Vasc'], 
                                              sample_weight=boot_weights))
    
    # Calculate confidence intervals if we have bootstrap samples
    if original_aucs:
        original_ci_lower = np.percentile(original_aucs, 2.5)
        original_ci_upper = np.percentile(original_aucs, 97.5)
    else:
        original_ci_lower = original_ci_upper = np.nan
        
    if weighted_aucs:
        weighted_ci_lower = np.percentile(weighted_aucs, 2.5)
        weighted_ci_upper = np.percentile(weighted_aucs, 97.5)
    else:
        weighted_ci_lower = weighted_ci_upper = np.nan
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Metric': ['AUC'],
        'Original': [f"{original_auc:.3f} ({original_ci_lower:.3f}-{original_ci_upper:.3f})"],
        'Weighted': [f"{weighted_auc:.3f} ({weighted_ci_lower:.3f}-{weighted_ci_upper:.3f})"]
    })
    
    # Calculate stroke rates by CHADS-VASc score
    score_results = calculate_stroke_rates_by_score(df, weights, outcome_col)
    
    # Combine results
    results = pd.concat([results, score_results], ignore_index=True)
    
    return results

def calculate_stroke_rates_by_score(df, weights, outcome_col):
    """
    Calculate stroke rates by CHADS-VASc score
    
    Args:
        df: DataFrame with CHADS-VASc scores and outcome
        weights: Array of weights for each patient
        outcome_col: Column name for the outcome variable
    
    Returns:
        DataFrame with stroke rates by score
    """
    # Initialize results
    rows = []
    
    # Get unique CHADS-VASc scores
    scores = sorted(df['CHADS-Vasc'].unique())
    
    # For each score, calculate rates
    for score in scores:
        # Filter patients with this score
        mask = df['CHADS-Vasc'] == score
        
        # Original rate
        n_patients = sum(mask)
        n_strokes = sum(df.loc[mask, outcome_col])
        orig_rate = n_strokes / n_patients * 100 if n_patients > 0 else 0
        
        # Weighted rate
        # Sum of weights for patients with this score
        total_weight = sum(weights[mask])
        # Sum of weights for patients with stroke and this score
        stroke_weight = sum(weights[mask] * df.loc[mask, outcome_col])
        
        weighted_rate = stroke_weight / total_weight * 100 if total_weight > 0 else 0
        
        # Add to results
        rows.append({
            'Metric': f"CHADS-VASc {score}",
            'Original': f"{orig_rate:.2f}% ({n_strokes}/{n_patients})",
            'Weighted': f"{weighted_rate:.2f}%"
        })
    
    # Create DataFrame
    return pd.DataFrame(rows)

def plot_weighted_vs_original(df, weights, outcome_col):
    """
    Plot weighted versus original stroke rates by CHADS-VASc score
    
    Args:
        df: DataFrame with CHADS-VASc scores and outcome
        weights: Array of weights for each patient
        outcome_col: Column name for the outcome variable
    """
    # Get unique CHADS-VASc scores
    scores = sorted(df['CHADS-Vasc'].unique())
    
    # Calculate rates for each score
    orig_rates = []
    weighted_rates = []
    orig_errors = []
    
    for score in scores:
        # Filter patients with this score
        mask = df['CHADS-Vasc'] == score
        
        # Original rate
        n_patients = sum(mask)
        n_strokes = sum(df.loc[mask, outcome_col])
        orig_rate = n_strokes / n_patients * 100 if n_patients > 0 else 0
        orig_rates.append(orig_rate)
        
        # Error for original rate (binomial standard error)
        if n_patients > 0:
            p = orig_rate / 100
            se = 100 * np.sqrt(p * (1 - p) / n_patients)
            orig_errors.append(se)
        else:
            orig_errors.append(0)
        
        # Weighted rate
        total_weight = sum(weights[mask])
        stroke_weight = sum(weights[mask] * df.loc[mask, outcome_col])
        weighted_rate = stroke_weight / total_weight * 100 if total_weight > 0 else 0
        weighted_rates.append(weighted_rate)
    
    # Convert to arrays
    scores = np.array(scores)
    orig_rates = np.array(orig_rates)
    weighted_rates = np.array(weighted_rates)
    orig_errors = np.array(orig_errors)
    
    # Calculate lower and upper error bounds, ensuring they don't go below 0
    lower_errors = np.minimum(orig_rates, orig_errors)  # Don't go below 0
    upper_errors = orig_errors
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Original rates with error bars
    plt.errorbar(scores, orig_rates, yerr=[lower_errors, upper_errors], 
                 fmt='o-', label='Original', capsize=5)
    
    # Weighted rates
    plt.plot(scores, weighted_rates, 's--', label='Weighted')
    
    # Add labels and title
    plt.xlabel('CHADS-VASc Score')
    plt.ylabel('Stroke Rate (%)')
    plt.title('Stroke Rates by CHADS-VASc Score: Original vs. Weighted')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set x-ticks to integer values
    plt.xticks(scores)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'weighted_vs_original_rates.png'))
    plt.close()

if __name__ == "__main__":
    # Load data
    print("Loading and preparing data...")
    df = get_df()  # Use default path constructed from DATA_DIR
    
    # Display dataset statistics
    print(f"Total patients in dataset: {len(df)}")
    print(f"Patients with AF: {sum(df['af'])}")
    print(f"Patients with stroke: {sum(df['stroke_1Y'])}")
    
    # Filter eligible patients
    eligible_df = filter_eligible_patients(df)
    print(f"Eligible patients after filtering: {len(eligible_df)}")
    
    # Check smoking status values
    print(f"Unique smoking status values: {eligible_df['smoking_status'].unique()}")
    
    # Map variables for reweighting
    mapped_df = map_variables(eligible_df)
    
    # Calculate follow-up years if not present
    if 'Follow_Up_Years' not in mapped_df.columns:
        # Calculate follow-up time in years
        mapped_df['Follow_Up_Years'] = (
            (mapped_df['end_fu'] - mapped_df['time1']).dt.days / 365.25
        )
    
    # Define variables for weighting
    weight_variables = [
        'age',                  # Age 
        'is_female',            # Female
        'hypertension_bin',     # Hypertension
        'diabetes_bin',         # Diabetes
        'heart_failure_bin',    # Heart failure
        'stroke_hist_bin',      # Prior stroke
        'vascular_disease_bin', # Vascular disease
        'is_smoker'             # Current smoking
    ]
    
    # Parameters from the original study (based on literature)
    # These are hypothetical values - replace with actual values
    original_params = {
        'age_mean': 63.90,
        'age_std': 10.60,
        'is_female_mean': 0.40,
        'is_female_std': 0.49,
        'hypertension_bin_mean': 0.67,
        'hypertension_bin_std': 0.47,
        'diabetes_bin_mean': 0.22,
        'diabetes_bin_std': 0.41,
        'heart_failure_bin_mean': 0.21,
        'heart_failure_bin_std': 0.41,
        'stroke_hist_bin_mean': 0.12,
        'stroke_hist_bin_std': 0.33,
        'vascular_disease_bin_mean': 0.30,
        'vascular_disease_bin_std': 0.46,
        'is_smoker_mean': 0.22,
        'is_smoker_std': 0.41
    }
    
    # Compute parameters for UK dataset
    uk_params = compute_parameters(mapped_df, weight_variables)
    
    # Display parameter comparison
    print("\nComparing parameters:")
    print(f"{'Variable':<20} {'Original':<20} {'UK':<20}")
    print("-" * 60)
    
    for var in weight_variables:
        orig_mean = original_params[f"{var}_mean"]
        orig_std = original_params[f"{var}_std"]
        uk_mean = uk_params[f"{var}_mean"]
        uk_std = uk_params[f"{var}_std"]
        
        print(f"{var:<20} {orig_mean:.2f} ± {orig_std:.2f} {uk_mean:.2f} ± {uk_std:.2f}")
    
    # Calculate weights
    print("\nComputing weights...")
    weights = calculate_weights(mapped_df, original_params, weight_variables)
    
    # Trim extreme weights
    trimmed_weights = trim_weights(weights)
    
    # Display weight statistics
    print(f"Original weights: Min: {weights.min():.4f}, Max: {weights.max():.4f}, Mean: {weights.mean():.4f}")
    print(f"Trimmed weights: Min: {trimmed_weights.min():.4f}, Max: {trimmed_weights.max():.4f}, Mean: {trimmed_weights.mean():.4f}")
    
    # Calculate effective sample size
    orig_ess, orig_pct = effective_sample_size(weights)
    trim_ess, trim_pct = effective_sample_size(trimmed_weights)
    
    print(f"Original effective sample size: {orig_ess:.1f} ({orig_pct:.1f}% of actual)")
    print(f"Trimmed effective sample size: {trim_ess:.1f} ({trim_pct:.1f}% of actual)")
    
    # Plot weight distribution
    plot_weight_distribution(weights, trimmed_weights, os.path.join(RESULTS_DIR, 'weight_distribution.png'))
    print(f"Weight distribution plot saved to '{os.path.join(RESULTS_DIR, 'weight_distribution.png')}'")
    
    # Compute reweighted CHADS-VASc performance
    print("\nComputing reweighted CHADS-VASc performance...")
    performance = reweight_chadsvasc_performance(mapped_df, trimmed_weights)
    
    # Plot weighted vs original rates
    plot_weighted_vs_original(mapped_df, trimmed_weights, 'stroke_1Y')
    print(f"Rate comparison plot saved to '{os.path.join(RESULTS_DIR, 'weighted_vs_original_rates.png')}'")
    
    # Save results to markdown
    with open(os.path.join(RESULTS_DIR, 'weighted_chadsvasc_results.md'), 'w') as f:
        f.write("# Reweighted CHADS-VASc Performance Results\n\n")
        f.write("## Parameter Comparison\n\n")
        
        # Parameter comparison table
        f.write("| Variable | Original | UK |\n")
        f.write("|----------|----------|----|\n")
        
        for var in weight_variables:
            orig_mean = original_params[f"{var}_mean"]
            orig_std = original_params[f"{var}_std"]
            uk_mean = uk_params[f"{var}_mean"]
            uk_std = uk_params[f"{var}_std"]
            
            f.write(f"| {var} | {orig_mean:.2f} ± {orig_std:.2f} | {uk_mean:.2f} ± {uk_std:.2f} |\n")
        
        # Weight statistics
        f.write("\n## Weight Statistics\n\n")
        f.write(f"- Original weights: Min: {weights.min():.4f}, Max: {weights.max():.4f}, Mean: {weights.mean():.4f}\n")
        f.write(f"- Trimmed weights: Min: {trimmed_weights.min():.4f}, Max: {trimmed_weights.max():.4f}, Mean: {trimmed_weights.mean():.4f}\n")
        f.write(f"- Original effective sample size: {orig_ess:.1f} ({orig_pct:.1f}% of actual)\n")
        f.write(f"- Trimmed effective sample size: {trim_ess:.1f} ({trim_pct:.1f}% of actual)\n\n")
        
        # Performance results
        f.write("## CHADS-VASc Performance\n\n")
        f.write(performance.to_markdown(index=False))
        
        # Figures
        f.write("\n\n## Figures\n\n")
        f.write("### Weight Distribution\n\n")
        f.write("![Weight Distribution](weight_distribution.png)\n\n")
        f.write("### Stroke Rates by CHADS-VASc Score\n\n")
        f.write("![Stroke Rates](weighted_vs_original_rates.png)\n")
    
    print(f"\nResults saved to '{os.path.join(RESULTS_DIR, 'weighted_chadsvasc_results.md')}'") 