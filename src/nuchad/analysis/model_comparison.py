"""
Model comparison analysis: CHADS-VAsC score vs. data-fitted model.

This module implements the comparison between:
1. Off-the-shelf CHADS-VAsC score using published risk tables
2. Data-driven logistic regression fitted to UK cohort

The analysis demonstrates whether local calibration improves discrimination
performance compared to transported clinical scores.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
import json

from nuchad.utils import get_results_dir
from nuchad.analysis.eda_old import get_df, calculate_chadsvasc
from nuchad.data_processing.eligibility_filters import filter_eligible_patients


def compute_chadsvasc_score(df):
    """
    Compute CHADS-VAsC score for each patient using existing function.
    
    Args:
        df: DataFrame with patient data
        
    Returns:
        Series with CHADS-VAsC scores
    """
    return df.apply(calculate_chadsvasc, axis=1)


def get_published_risk_table():
    """
    Get published 1-year stroke risk rates by CHADS-VAsC score.
    
    Using rates from Lip et al. (converted to proportions).
    
    Returns:
        Dict mapping CHADS-VAsC score to 1-year stroke risk
    """
    # Lip et al. rates per 100 person-years, converted to 1-year risks
    # These are approximate 1-year risks (rate/100)
    risk_table = {
        0: 0.002,   # 0.2%
        1: 0.006,   # 0.6%
        2: 0.022,   # 2.2%1
        3: 0.032,   # 3.2%
        4: 0.048,   # 4.8%
        5: 0.072,   # 7.2%
        6: 0.097,   # 9.7%
        7: 0.112,   # 11.2%
        8: 0.108,   # 10.8%
        9: 0.122,   # 12.2%
    }
    return risk_table


def embed_png_metadata(png_path, metadata):
    """
    Embed metadata in PNG file using PIL.
    
    Args:
        png_path: Path to PNG file
        metadata: Dictionary with metadata to embed
    """
    try:
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo
        
        # Convert metadata to JSON string
        metadata_json = json.dumps(metadata, indent=2, default=str)
        
        # Open the PNG file
        with Image.open(png_path) as img:
            # Create PngInfo object
            png_info = PngInfo()
            
            # Add individual metadata fields
            for key, value in metadata.items():
                png_info.add_text(f"model_comparison.{key}", str(value))
            
            # Add full metadata as JSON
            png_info.add_text("model_comparison.metadata", metadata_json)
            
            # Save with metadata
            img.save(png_path, pnginfo=png_info)
            
    except ImportError:
        print("Warning: PIL not available, skipping PNG metadata embedding")
    except Exception as e:
        print(f"Warning: Failed to embed PNG metadata: {e}")


def prepare_modeling_data(df):
    """
    Prepare data for modeling by creating proper feature matrix.
    
    Args:
        df: DataFrame with patient data
        
    Returns:
        Tuple of (X, y, feature_names) where:
        - X: Feature matrix
        - y: Binary outcome (stroke_1Y == 1)
        - feature_names: List of feature names
    """
    # Define covariates that match CHADS-VAsC components
    feature_cols = ['age', 'gender', 'hf', 'hypertension', 'HB_stroke_history', 'diab', 'vasc_dis_mi_pad']
    
    # Handle column compatibility between datasets
    df_working = df.copy()
    if 'Stroke_TIA_hx' in df.columns and 'HB_stroke_history' not in df.columns:
        df_working['HB_stroke_history'] = df['Stroke_TIA_hx']
        print("Note: Using 'Stroke_TIA_hx' as 'HB_stroke_history' for compatibility")
    
    # Check which columns exist
    available_cols = [col for col in feature_cols if col in df_working.columns]
    missing_cols = [col for col in feature_cols if col not in df_working.columns]
    
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}")
    
    # Create feature matrix
    X = df_working[available_cols].copy()
    
    # Handle gender: convert to binary (female=1)
    if 'gender' in X.columns:
        X['female'] = (X['gender'] != 1).astype(int)  # 1=male, 2=female -> 0=male, 1=female
        X = X.drop('gender', axis=1)
    
    # Ensure all features are numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Handle missing values
    X = X.fillna(0)
    
    # Create outcome variable
    y = (df_working['stroke_1Y'] == 1).astype(int)
    
    return X, y, list(X.columns)


def fit_logistic_model(X, y):
    """
    Fit logistic regression model to predict stroke risk.
    
    Args:
        X: Feature matrix
        y: Binary outcome
        
    Returns:
        Fitted LogisticRegression model
    """
    # Use regularization to handle potential collinearity
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,  # Regularization strength
        random_state=42
    )
    
    # Fit the model
    model.fit(X, y)
    
    return model




def bootstrap_auc_difference(y_true, scores_1, scores_2, n_bootstrap=1000, random_state=42):
    """
    Bootstrap confidence interval for AUC difference.
    
    Args:
        y_true: True binary labels
        scores_1: Risk scores from method 1
        scores_2: Risk scores from method 2
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed
        
    Returns:
        Tuple of (mean_diff, ci_lower, ci_upper)
    """
    np.random.seed(random_state)
    
    n_samples = len(y_true)
    bootstrap_diffs = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        y_boot = y_true.iloc[indices] if hasattr(y_true, 'iloc') else y_true[indices]
        scores_1_boot = scores_1.iloc[indices] if hasattr(scores_1, 'iloc') else scores_1[indices]
        scores_2_boot = scores_2.iloc[indices] if hasattr(scores_2, 'iloc') else scores_2[indices]
        
        # Compute AUCs
        try:
            auc_1 = roc_auc_score(y_boot, scores_1_boot)
            auc_2 = roc_auc_score(y_boot, scores_2_boot)
            bootstrap_diffs.append(auc_2 - auc_1)
        except ValueError:
            # Skip if bootstrap sample doesn't have both classes
            continue
    
    bootstrap_diffs = np.array(bootstrap_diffs)
    mean_diff = np.mean(bootstrap_diffs)
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)
    
    return mean_diff, ci_lower, ci_upper


def create_comparison_plot(df, save_path, filter_stats=None):
    """
    Create ROC curve comparison plot.
    
    Args:
        df: DataFrame with score_risk and model_risk columns
        save_path: Path to save the plot  
        filter_stats: Optional filtering statistics for metadata
    """
    y_true = (df['stroke_1Y'] == 1).astype(int)
    
    # Compute ROC curves
    fpr_score, tpr_score, _ = roc_curve(y_true, df['score_risk'])
    fpr_model, tpr_model, _ = roc_curve(y_true, df['model_risk'])
    
    # Compute AUCs
    auc_score = roc_auc_score(y_true, df['score_risk'])
    auc_model = roc_auc_score(y_true, df['model_risk'])
    
    # Create plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_score, tpr_score, label=f'CHADS-VAsC Score (AUC = {auc_score:.3f})', linewidth=2)
    plt.plot(fpr_model, tpr_model, label=f'UK-Fitted Model (AUC = {auc_model:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: CHADS-VAsC Score vs. UK-Fitted Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Embed metadata in PNG if filter_stats provided
    if filter_stats:
        embed_png_metadata(save_path, filter_stats)


def save_results_table(results, save_path, filter_stats=None):
    """
    Save comparison results to markdown table.
    
    Args:
        results: Dictionary with comparison results
        save_path: Path to save the markdown file
        filter_stats: Optional filtering statistics to include in metadata
    """
    with open(save_path, 'w') as f:
        f.write("# Model Comparison: CHADS-VAsC Score vs. UK-Fitted Model\n\n")
        
        f.write("## Overview\n\n")
        f.write("Comparison of discrimination performance between:\n")
        f.write("1. **Off-the-shelf CHADS-VAsC**: Using published risk tables from Lip et al.\n")
        f.write("2. **UK-Fitted Model**: Logistic regression trained on UK cohort data\n\n")
        
        # Add filtering information if available
        if filter_stats:
            f.write("## Patient Filtering\n\n")
            if 'config_name' in filter_stats:
                f.write(f"**Configuration**: {filter_stats['config_name']}\n\n")
            
            f.write("### Filtering Steps\n\n")
            f.write("| Step | Patients Remaining | Patients Removed | % of Original |\n")
            f.write("|------|-------------------|------------------|---------------|\n")
            f.write(f"| Initial cohort | {filter_stats['total']:,} | 0 | 100.0% |\n")
            
            for step in filter_stats['steps']:
                f.write(f"| {step['description']} | {step['remaining']:,} | {step['removed']:,} | {step['percent_remaining']}% |\n")
            f.write("\n")
        
        f.write("## Sample Characteristics\n\n")
        f.write(f"- Total patients: {results['n_patients']:,}\n")
        f.write(f"- Stroke events: {results['n_strokes']:,} ({results['stroke_rate']:.1f}%)\n")
        f.write(f"- Mean age: {results['mean_age']:.1f} years\n")
        f.write(f"- Female: {results['female_pct']:.1f}%\n\n")
        
        f.write("## Discrimination Performance\n\n")
        f.write("| Metric | CHADS-VAsC Score | UK-Fitted Model | Difference (95% CI) |\n")
        f.write("|--------|------------------|-----------------|---------------------|\n")
        
        auc_diff = results['auc_model'] - results['auc_score']
        auc_ci = f"[{results['auc_diff_ci'][0]:.3f}, {results['auc_diff_ci'][1]:.3f}]"
        f.write(f"| AUC | {results['auc_score']:.3f} | {results['auc_model']:.3f} | {auc_diff:+.3f} {auc_ci} |\n")
        
        if 'c_index_score' in results:
            c_diff = results['c_index_model'] - results['c_index_score']
            f.write(f"| C-index | {results['c_index_score']:.3f} | {results['c_index_model']:.3f} | {c_diff:+.3f} |\n")
        
        f.write("\n## Model Coefficients (UK-Fitted Model)\n\n")
        f.write("| Feature | Coefficient | Odds Ratio |\n")
        f.write("|---------|-------------|------------|\n")
        
        for feature, coef in zip(results['feature_names'], results['coefficients']):
            or_val = np.exp(coef)
            f.write(f"| {feature} | {coef:.3f} | {or_val:.3f} |\n")
        
        f.write(f"\n**Intercept:** {results['intercept']:.3f}\n\n")
        
        f.write("## Interpretation\n\n")
        if auc_diff > 0:
            f.write(f"The UK-fitted model shows **improved discrimination** compared to the off-the-shelf CHADS-VAsC score (ΔAUC = +{auc_diff:.3f}).\n")
        else:
            f.write(f"The CHADS-VAsC score shows comparable discrimination to the UK-fitted model (ΔAUC = {auc_diff:.3f}).\n")
        
        f.write("\nThis demonstrates the potential benefit of local model calibration versus transported clinical scores.\n\n")
        f.write("## Figures\n\n")
        f.write("![ROC Comparison](model_comparison_roc.png)\n")


def perform_model_comparison(df=None, filter_stats=None):
    """
    Perform the complete model comparison analysis.
    
    Args:
        df: Optional pre-loaded DataFrame. If None, will load and prepare the data.
        filter_stats: Optional filtering statistics from eligibility filtering
        
    Returns:
        Dictionary with comparison results
    """
    print("Starting model comparison analysis...")
    
    # Load and prepare data if not provided
    if df is None:
        df = get_df()
        df, _ = filter_eligible_patients(df)
    
    # Ensure we have required columns (handle column compatibility)
    required_cols = ['stroke_1Y', 'age', 'gender', 'hf', 'hypertension', 'diab', 'vasc_dis_mi_pad']
    stroke_history_col = 'HB_stroke_history' if 'HB_stroke_history' in df.columns else 'Stroke_TIA_hx'
    required_cols.append(stroke_history_col)
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"Working with {len(df):,} patients")
    
    # 1. Compute CHADS-VAsC scores
    print("Computing CHADS-VAsC scores...")
    df = df.copy()
    # Handle column compatibility for CHADS-VAsC calculation
    if 'Stroke_TIA_hx' in df.columns and 'HB_stroke_history' not in df.columns:
        df['HB_stroke_history'] = df['Stroke_TIA_hx']
    df['chadsvasc_score'] = compute_chadsvasc_score(df)
    
    # 2. Map scores to published risk tables
    print("Mapping scores to published risk rates...")
    risk_table = get_published_risk_table()
    df['score_risk'] = df['chadsvasc_score'].map(risk_table)
    
    # Handle scores not in table (use max available)
    max_score = max(risk_table.keys())
    df['score_risk'] = df['score_risk'].fillna(risk_table[max_score])
    
    # 3. Prepare modeling data
    print("Preparing modeling data...")
    X, y, feature_names = prepare_modeling_data(df)
    
    # 4. Fit logistic regression
    print("Fitting logistic regression model...")
    logistic_model = fit_logistic_model(X, y)
    
    # Get logistic model predictions
    df['model_risk'] = logistic_model.predict_proba(X)[:, 1]
    
    # 5. Compute discrimination metrics
    print("Computing discrimination metrics...")
    y_binary = (df['stroke_1Y'] == 1).astype(int)
    
    auc_score = roc_auc_score(y_binary, df['score_risk'])
    auc_model = roc_auc_score(y_binary, df['model_risk'])
    
    print(f"AUC (CHADS-VAsC): {auc_score:.3f}")
    print(f"AUC (UK Model): {auc_model:.3f}")
    print(f"Difference: {auc_model - auc_score:+.3f}")
    
    # 6. Bootstrap confidence intervals
    print("Computing bootstrap confidence intervals...")
    _, ci_lower, ci_upper = bootstrap_auc_difference(y_binary, df['score_risk'], df['model_risk'])
    
    # 7. Collect results
    results = {
        'n_patients': len(df),
        'n_strokes': int(y_binary.sum()),
        'stroke_rate': float(y_binary.mean() * 100),
        'mean_age': float(df['age'].mean()),
        'female_pct': float((df['gender'] != 1).mean() * 100),
        'auc_score': auc_score,
        'auc_model': auc_model,
        'auc_diff_ci': (ci_lower, ci_upper),
        'feature_names': feature_names,
        'coefficients': logistic_model.coef_[0].tolist(),
        'intercept': float(logistic_model.intercept_[0]),
    }
    
    # 8. Save results
    results_dir = get_results_dir()
    
    # Save ROC plot
    plot_path = results_dir / 'model_comparison_roc.png'
    create_comparison_plot(df, plot_path, filter_stats)
    print(f"ROC plot saved to: {plot_path}")
    
    # Save results table
    table_path = results_dir / 'model_comparison_results.md'
    save_results_table(results, table_path, filter_stats)
    print(f"Results table saved to: {table_path}")
    
    print("Model comparison analysis completed!")
    return results


def main():
    """Run the model comparison analysis."""
    try:
        results = perform_model_comparison()
        
        print("\n" + "="*50)
        print("FINAL RESULTS")
        print("="*50)
        print(f"AUC (CHADS-VAsC): {results['auc_score']:.3f}")
        print(f"AUC (UK Model): {results['auc_model']:.3f}")
        print(f"Difference: {results['auc_model'] - results['auc_score']:+.3f}")
        print(f"95% CI: [{results['auc_diff_ci'][0]:.3f}, {results['auc_diff_ci'][1]:.3f}]")
        
        
    except Exception as e:
        print(f"Error in model comparison: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()