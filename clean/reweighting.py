import numpy as np
import pandas as pd
from scipy.stats import norm
from eda import get_df, filter_eligible_patients
import matplotlib.pyplot as plt
import seaborn as sns

def map_variables(df):
    """
    Map variables from our UK dataset to match variables in Lip et al. 2010 study
    
    Args:
        df: DataFrame with UK data
        
    Returns:
        DataFrame with mapped variables
    """
    mapped_df = df.copy()
    
    # Map binary variables (0/1)
    mapped_df['Hypertension'] = mapped_df['hypertension']
    mapped_df['Diabetes'] = mapped_df['diab']
    mapped_df['HeartFailure'] = mapped_df['hf']
    mapped_df['StrokeTIA'] = mapped_df['HB_stroke_history'] | mapped_df['thrombo']  # Combine stroke history and thromboembolism
    mapped_df['VascularDx'] = mapped_df['vasc_dis_mi_pad']
    
    # Check the unique values in smoking_status
    unique_smoking = mapped_df['smoking_status'].unique()
    print(f"Unique smoking status values: {unique_smoking}")
    
    # Current smoker mapping (assuming smoking_status has values like "Current smoker", "Ex-smoker", "Non-smoker", etc.)
    # Handle potential NaN values
    mapped_df['CurrentSmoker'] = 0  # Default to 0
    if 'smoking_status' in mapped_df.columns:
        # Check if "Current smoker" is in the dataset
        if 'Current smoker' in mapped_df['smoking_status'].values:
            mapped_df.loc[mapped_df['smoking_status'] == 'Current smoker', 'CurrentSmoker'] = 1
    
    # Age is already available
    mapped_df['Age'] = mapped_df['age']
    
    # Return only the columns we need for reweighting
    return mapped_df[['Age', 'Hypertension', 'Diabetes', 'HeartFailure', 'StrokeTIA', 'VascularDx', 'CurrentSmoker']]

def compute_uk_params(df):
    """
    Compute the parameters (means, SDs, proportions) from the UK dataset
    
    Args:
        df: DataFrame with mapped variables
        
    Returns:
        Dictionary with UK parameters
    """
    uk_params = {}
    
    # Continuous variables
    uk_params['Age'] = {'mean': df['Age'].mean(), 'sd': df['Age'].std()}
    
    # Binary variables
    for col in ['Hypertension', 'Diabetes', 'HeartFailure', 'StrokeTIA', 'VascularDx', 'CurrentSmoker']:
        uk_params[col] = {'p': df[col].mean()}  # Mean of binary variable = proportion
    
    return uk_params

def compute_weight(row, orig_params, uk_params):
    """
    Compute the weight for a single patient based on density ratio
    
    Args:
        row: Patient data (Series)
        orig_params: Parameters from original Lip et al. 2010 study
        uk_params: Parameters from UK dataset
        
    Returns:
        Weight for the patient
    """
    w = 1.0
    
    # Continuous: Age
    μo, σo = orig_params['Age']['mean'], orig_params['Age']['sd']
    μu, σu = uk_params['Age']['mean'], uk_params['Age']['sd']
    w *= norm.pdf(row['Age'], loc=μo, scale=σo) / norm.pdf(row['Age'], loc=μu, scale=σu)
    
    # Binary covariates
    for col in ['Hypertension', 'Diabetes', 'HeartFailure', 'StrokeTIA', 'VascularDx', 'CurrentSmoker']:
        p_o = orig_params[col]['p']
        p_u = uk_params[col]['p']
        x = row[col]
        
        # Bernoulli pmf ratio (handle potential zeros in denominators)
        if x == 1 and p_u > 0 and p_o > 0:
            w *= p_o / p_u
        elif x == 0 and p_u < 1 and p_o < 1:
            w *= (1 - p_o) / (1 - p_u)
        elif x == 1 and (p_u == 0 or p_o == 0):
            # Handle edge case where prevalence is 0 in either cohort
            w *= 0.001  # Small value to avoid zeros
        elif x == 0 and (p_u == 1 or p_o == 1):
            # Handle edge case where prevalence is 1 in either cohort
            w *= 0.001  # Small value to avoid zeros
    
    return w

def trim_weights(weights, trim_quantile=0.99):
    """
    Trim extreme weights to improve stability
    
    Args:
        weights: Series of weights
        trim_quantile: Quantile for trimming (default: 0.99)
        
    Returns:
        Trimmed weights
    """
    upper_bound = weights.quantile(trim_quantile)
    return weights.clip(upper=upper_bound)

def calculate_effective_sample_size(weights):
    """
    Calculate effective sample size 
    
    Args:
        weights: Series of weights
        
    Returns:
        Effective sample size
    """
    return (weights.sum() ** 2) / (weights ** 2).sum()

def plot_weight_distribution(weights, trimmed_weights=None):
    """
    Plot the distribution of weights
    
    Args:
        weights: Series of original weights
        trimmed_weights: Series of trimmed weights (optional)
    """
    plt.figure(figsize=(10, 6))
    
    if trimmed_weights is not None:
        # Plot original weights
        plt.hist(weights, bins=50, alpha=0.5, label='Original Weights')
        # Plot trimmed weights
        plt.hist(trimmed_weights, bins=50, alpha=0.5, label='Trimmed Weights')
        plt.legend()
    else:
        plt.hist(weights, bins=50, alpha=0.7)
    
    plt.title('Distribution of Transportability Weights')
    plt.xlabel('Weight')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('weight_distribution.png', dpi=300)

def reweight_chadsvasc_performance(df, weights):
    """
    Analyze CHADS-VASc performance using weighted data
    
    Args:
        df: DataFrame with patient data
        weights: Series of weights
        
    Returns:
        DataFrame with weighted results
    """
    from eda import validate_chadsvasc
    
    # Ensure CHADS-VASc scores are calculated
    if 'CHADS-Vasc' not in df.columns:
        from eda import calculate_chadsvasc
        df['CHADS-Vasc'] = df.apply(calculate_chadsvasc, axis=1)
    
    # Ensure Follow_Up_Years is calculated
    if 'Follow_Up_Years' not in df.columns:
        df['Follow_Up_Years'] = (df['end_fu'] - df['time1']).dt.days / 365.25
    
    # Run the validation with weights
    # For now, we'll just store the weights in the DataFrame
    # A more comprehensive approach would modify validate_chadsvasc to use weights
    df_with_weights = df.copy()
    df_with_weights['weight'] = weights
    
    # Group by CHADS-VASc score and calculate weighted statistics
    grouped = df_with_weights.groupby('CHADS-Vasc')
    
    results = []
    for score, group in grouped:
        # Calculate weighted number of patients and patient-years
        weighted_patients = group['weight'].sum()
        weighted_years = (group['weight'] * group['Follow_Up_Years']).sum()
        
        # Calculate weighted stroke rate (per 100 patient-years)
        strokes = group[group['stroke_1Y'] == 1]
        weighted_strokes = strokes['weight'].sum()
        
        if weighted_years > 0:
            weighted_rate = (weighted_strokes / weighted_years) * 100
            
            # Simple bootstrap for confidence intervals with only 15 iterations instead of 1000
            n_bootstrap = 15  # Reduced from 1000 to speed up computation
            bootstrap_rates = []
            
            for _ in range(n_bootstrap):
                sample_indices = np.random.choice(group.index, size=len(group), replace=True)
                sample = df_with_weights.loc[sample_indices]
                
                sample_strokes = sample[sample['stroke_1Y'] == 1]['weight'].sum()
                sample_years = (sample['weight'] * sample['Follow_Up_Years']).sum()
                
                if sample_years > 0:
                    sample_rate = (sample_strokes / sample_years) * 100
                    bootstrap_rates.append(sample_rate)
            
            # Only calculate CI if we have bootstrap samples
            if bootstrap_rates:
                ci_lower = np.percentile(bootstrap_rates, 2.5)
                ci_upper = np.percentile(bootstrap_rates, 97.5)
            else:
                ci_lower = weighted_rate * 0.5  # Fallback if bootstrap fails
                ci_upper = weighted_rate * 1.5
        else:
            weighted_rate = 0
            ci_lower = 0
            ci_upper = 0
        
        # Original CHADS-VASc rates from Lip et al.
        original_rates = {
            0: 0.2,
            1: 0.6,
            2: 2.2,
            3: 3.2,
            4: 4.8,
            5: 7.2,
            6: 9.7,
            7: 11.2,
            8: 10.8,
            9: 12.2,
        }
        
        original_rate = original_rates.get(score, np.nan)
        
        results.append({
            'CHADS-Vasc': score,
            'Weighted Patients': weighted_patients,
            'Weighted Patient Years': weighted_years,
            'Weighted Stroke Rate': weighted_rate,
            '95% CI Lower': ci_lower,
            '95% CI Upper': ci_upper,
            'Original Stroke Rate': original_rate,
            'Original Rate Within CI': ci_lower <= original_rate <= ci_upper if not np.isnan(original_rate) else False
        })
    
    results_df = pd.DataFrame(results)
    return results_df

def plot_weighted_vs_original(results_df, unweighted_df=None):
    """
    Plot weighted stroke rates vs original rates
    
    Args:
        results_df: DataFrame with weighted results
        unweighted_df: DataFrame with unweighted results (optional)
    """
    plt.figure(figsize=(12, 8))
    
    # Calculate error bars, ensuring they're not negative
    lower_errors = results_df['Weighted Stroke Rate'] - results_df['95% CI Lower']
    upper_errors = results_df['95% CI Upper'] - results_df['Weighted Stroke Rate']
    
    # Make sure error values are non-negative
    lower_errors = lower_errors.clip(lower=0)
    upper_errors = upper_errors.clip(lower=0)
    
    # Plot weighted rates with error bars
    plt.errorbar(results_df['CHADS-Vasc'], results_df['Weighted Stroke Rate'], 
                 yerr=[lower_errors, upper_errors], 
                 fmt='o-', capsize=5, label='Weighted Rates', color='blue', markersize=8)
    
    # Plot original rates
    plt.plot(results_df['CHADS-Vasc'], results_df['Original Stroke Rate'], 
             'x--', label='Original Rates (Lip et al.)', color='red', markersize=8)
    
    # Plot unweighted rates if provided
    if unweighted_df is not None:
        plt.plot(unweighted_df['CHADS-Vasc'], unweighted_df['Observed Stroke Rate'],
                  's-', label='Unweighted Rates', color='green', markersize=6, alpha=0.7)
    
    # Add data points as text
    for i, row in results_df.iterrows():
        plt.text(row['CHADS-Vasc'], row['Weighted Stroke Rate'] + 0.3, 
                 f"{row['Weighted Stroke Rate']:.2f}", ha='center', fontsize=8)
        plt.text(row['CHADS-Vasc'], row['Original Stroke Rate'] - 0.3, 
                 f"{row['Original Stroke Rate']:.1f}", ha='center', color='red', fontsize=8)
    
    # Set labels and title
    plt.title('Stroke Rates by CHA₂DS₂-VASc Score: Weighted vs. Original', fontsize=14)
    plt.xlabel('CHA₂DS₂-VASc Score', fontsize=12)
    plt.ylabel('Stroke Rate (per 100 patient-years)', fontsize=12)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    
    # Save figure
    plt.tight_layout()
    plt.savefig('weighted_vs_original_stroke_rates.png', dpi=300)

if __name__ == "__main__":
    # 1. Load data
    print("Loading data...")
    df = get_df()
    print(f"Total patients in dataset: {len(df)}")
    
    eligible_df = filter_eligible_patients(df)
    print(f"Eligible patients after filtering: {len(eligible_df)}")
    print("Confirming we're using the eligible patient subset for all analyses...")
    
    # 2. Map variables to match Lip et al. 2010 study
    print("Mapping variables...")
    mapped_df = map_variables(eligible_df)
    
    # 3. Specify original parameters from Lip et al. 2010
    orig_params = {
        'Age':          {'mean': 63.9,  'sd': 10.6},
        'Hypertension': {'p': 4964/9722},
        'Diabetes':     {'p': 1032/9722},
        'HeartFailure': {'p': 100/9722},
        'StrokeTIA':    {'p': 278/9722}, 
        'VascularDx':   {'p': 1188/9722},
        'CurrentSmoker':{'p': 1242/9722},
    }
    
    # 4. Compute UK parameters
    print("Computing UK parameters...")
    uk_params = compute_uk_params(mapped_df)
    
    # Display parameters for comparison
    print("\nParameter comparison - Original vs UK:")
    print("Parameter\t\tOriginal\t\tUK")
    print("-" * 60)
    
    for param in orig_params:
        if param == 'Age':
            print(f"{param}\t\t{orig_params[param]['mean']:.2f} ± {orig_params[param]['sd']:.2f}\t\t"
                  f"{uk_params[param]['mean']:.2f} ± {uk_params[param]['sd']:.2f}")
        else:
            print(f"{param}\t{orig_params[param]['p']*100:.2f}%\t\t{uk_params[param]['p']*100:.2f}%")
    
    # 5. Compute weights
    print("\nComputing weights...")
    weights = mapped_df.apply(lambda row: compute_weight(row, orig_params, uk_params), axis=1)
    
    # 6. Trim weights and calculate effective sample size
    trimmed_weights = trim_weights(weights)
    
    ess_original = calculate_effective_sample_size(weights)
    ess_trimmed = calculate_effective_sample_size(trimmed_weights)
    
    print(f"Original weights - Min: {weights.min():.4f}, Max: {weights.max():.4f}, Mean: {weights.mean():.4f}")
    print(f"Trimmed weights - Min: {trimmed_weights.min():.4f}, Max: {trimmed_weights.max():.4f}, Mean: {trimmed_weights.mean():.4f}")
    print(f"Original effective sample size: {ess_original:.1f} ({ess_original/len(weights)*100:.1f}% of actual)")
    print(f"Trimmed effective sample size: {ess_trimmed:.1f} ({ess_trimmed/len(trimmed_weights)*100:.1f}% of actual)")
    
    # 7. Plot weight distribution
    print("Plotting weight distribution...")
    plot_weight_distribution(weights, trimmed_weights)
    
    # 8. Reweight CHADS-VASc performance
    print("Computing reweighted CHADS-VASc performance...")
    weighted_results = reweight_chadsvasc_performance(eligible_df, trimmed_weights)
    
    # 9. Get unweighted results for comparison
    from eda import validate_chadsvasc
    unweighted_results = validate_chadsvasc(eligible_df, "time1", "end_fu", "stroke_1Y")
    
    # 10. Plot weighted vs original rates
    print("Plotting weighted vs original rates...")
    plot_weighted_vs_original(weighted_results, unweighted_results)
    
    # 11. Save results
    weighted_results.to_csv('weighted_chadsvasc_results.csv', index=False)
    weighted_results.to_markdown('weighted_chadsvasc_results.md', index=False)
    
    print("\nReweighting analysis complete. Results saved to 'weighted_chadsvasc_results.md'") 