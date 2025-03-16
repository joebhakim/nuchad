"""
Survival model interpretation and visualization utilities.

This module provides tools for interpreting and visualizing the results of survival models,
making them more clinically useful and interpretable.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Try to import scikit-survival
try:
    from sksurv.nonparametric import kaplan_meier_estimator
    from sksurv.metrics import concordance_index_censored, integrated_brier_score
    from sksurv.compare import compare_survival
    SKSURV_AVAILABLE = True
except ImportError:
    print("Warning: scikit-survival not installed. Some functions may not be available.")
    SKSURV_AVAILABLE = False


def plot_kaplan_meier(y, strata=None, strata_names=None, figsize=(10, 6)):
    """
    Plot Kaplan-Meier survival curves.
    
    Parameters:
    -----------
    y : structured array
        Survival data with 'event' and 'time' fields
    strata : array-like, optional
        Array of group labels for stratification
    strata_names : list, optional
        Names for the strata
    figsize : tuple, default=(10, 6)
        Figure size
    
    Returns:
    --------
    fig : Figure
        Matplotlib figure
    """
    if not SKSURV_AVAILABLE:
        raise ImportError("scikit-survival is required for Kaplan-Meier plots")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if strata is None:
        # Overall survival curve
        time, survival_prob = kaplan_meier_estimator(y["event"], y["time"])
        ax.step(time, survival_prob, where="post", label="Overall")
    else:
        # Stratified survival curves
        for i, group in enumerate(np.unique(strata)):
            mask = strata == group
            time, survival_prob = kaplan_meier_estimator(
                y["event"][mask], y["time"][mask]
            )
            
            label = f"Group {group}" if strata_names is None else strata_names[i]
            ax.step(time, survival_prob, where="post", label=label)
    
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival Probability")
    ax.set_title("Kaplan-Meier Survival Estimate")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    
    return fig


def plot_feature_partial_effects(model, feature_names, X, feature_idx, n_points=50, figsize=(12, 8)):
    """
    Plot partial effects of a feature on survival.
    
    Parameters:
    -----------
    model : object
        Fitted survival model with predict_survival_function method
    feature_names : list
        Names of features
    X : array-like
        Feature matrix
    feature_idx : int
        Index of the feature to analyze
    n_points : int, default=50
        Number of points to use for the feature grid
    figsize : tuple, default=(12, 8)
        Figure size
    
    Returns:
    --------
    fig : Figure
        Matplotlib figure
    """
    if not hasattr(model, 'predict_survival_function'):
        raise ValueError("Model must have predict_survival_function method")
    
    # Create a reference instance (using the mean values)
    X_ref = np.tile(X.mean(axis=0), (n_points, 1))
    
    # Create a grid of values for the feature of interest
    feature_values = np.linspace(
        np.percentile(X[:, feature_idx], 5),
        np.percentile(X[:, feature_idx], 95),
        n_points
    )
    
    # Set the grid values for the feature of interest
    X_ref[:, feature_idx] = feature_values
    
    # Predict survival functions
    surv_funcs = model.predict_survival_function(X_ref)
    
    # Plot feature effect on survival
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a colormap for the curves
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(feature_values), max(feature_values))
    
    for i, surv_func in enumerate(surv_funcs):
        if hasattr(surv_func, 'x') and hasattr(surv_func, 'y'):
            # For StepFunction objects
            color = cmap(norm(feature_values[i]))
            ax.step(surv_func.x, surv_func.y, where="post", color=color, alpha=0.6)
        else:
            # For array returns - need time points from model
            if hasattr(model, 'unique_times_'):
                times = model.unique_times_
                color = cmap(norm(feature_values[i]))
                ax.step(times, surv_func, where="post", color=color, alpha=0.6)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(feature_names[feature_idx])
    
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival Probability")
    ax.set_title(f"Effect of {feature_names[feature_idx]} on Survival Probability")
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_risk_stratification(model, X, y, n_groups=3, figsize=(10, 6)):
    """
    Plot survival curves stratified by predicted risk.
    
    Parameters:
    -----------
    model : object
        Fitted survival model with predict method
    X : array-like
        Feature matrix
    y : structured array
        Survival data with 'event' and 'time' fields
    n_groups : int, default=3
        Number of risk groups
    figsize : tuple, default=(10, 6)
        Figure size
    
    Returns:
    --------
    fig : Figure
        Matplotlib figure
    """
    if not SKSURV_AVAILABLE:
        raise ImportError("scikit-survival is required for risk stratification plots")
    
    # Predict risk scores
    risk_scores = model.predict(X)
    
    # Create risk groups
    risk_percentiles = np.percentile(risk_scores, np.linspace(0, 100, n_groups + 1))
    risk_groups = np.zeros(len(risk_scores), dtype=int)
    
    for i in range(n_groups):
        if i < n_groups - 1:
            mask = (risk_scores >= risk_percentiles[i]) & (risk_scores < risk_percentiles[i+1])
        else:
            mask = (risk_scores >= risk_percentiles[i])
        risk_groups[mask] = i
    
    # Group names
    group_names = ["Low Risk", "Medium Risk", "High Risk"] if n_groups == 3 else [f"Group {i+1}" for i in range(n_groups)]
    
    # Calculate and test difference between curves
    if n_groups > 1 and hasattr(y, 'dtype') and y.dtype.names is not None:
        p_value = compare_survival(y, risk_groups)[1]
        print(f"Log-rank test p-value: {p_value:.6f}")
    
    # Plot Kaplan-Meier curves by risk group
    fig = plot_kaplan_meier(y, strata=risk_groups, strata_names=group_names, figsize=figsize)
    fig.axes[0].set_title(f"Survival by Predicted Risk Group (n={n_groups})")
    
    return fig


def create_patient_risk_table(model, X, feature_names, patient_ids=None, n_patients=5):
    """
    Create a table of patient risk factors and predicted risk.
    
    Parameters:
    -----------
    model : object
        Fitted survival model with predict method
    X : array-like
        Feature matrix
    feature_names : list
        Names of features
    patient_ids : list, optional
        IDs for the patients (defaults to indices)
    n_patients : int, default=5
        Number of patients to include in the table
    
    Returns:
    --------
    risk_table : pandas.DataFrame
        Table with patient risk factors and predicted risk
    """
    if patient_ids is None:
        patient_ids = list(range(len(X)))
    
    # Predict risk scores
    risk_scores = model.predict(X)
    
    # Select sample of patients for the table
    if n_patients < len(X):
        # Use stratified sampling to get patients across the risk spectrum
        risk_percentiles = np.percentile(risk_scores, np.linspace(0, 100, n_patients + 2)[1:-1])
        selected_indices = []
        
        for percentile in risk_percentiles:
            # Find patient with risk closest to this percentile
            closest_idx = np.abs(risk_scores - percentile).argmin()
            if closest_idx not in selected_indices:
                selected_indices.append(closest_idx)
        
        # If we don't have enough patients, add more
        while len(selected_indices) < n_patients:
            idx = np.random.randint(0, len(X))
            if idx not in selected_indices:
                selected_indices.append(idx)
    else:
        selected_indices = list(range(min(n_patients, len(X))))
    
    # Create DataFrame with selected patients and their features
    data = {
        'Patient ID': [patient_ids[i] for i in selected_indices],
        'Predicted Risk': [risk_scores[i] for i in selected_indices]
    }
    
    # Add key features
    for i, name in enumerate(feature_names):
        data[name] = [X[idx, i] for idx in selected_indices]
    
    # Sort by predicted risk
    risk_table = pd.DataFrame(data).sort_values('Predicted Risk', ascending=False)
    
    return risk_table


def run_clinical_insights(model, X, y, feature_names, figsize=(12, 10)):
    """
    Generate a comprehensive clinical insights report.
    
    Parameters:
    -----------
    model : object
        Fitted survival model 
    X : array-like
        Feature matrix
    y : structured array
        Survival data with 'event' and 'time' fields
    feature_names : list
        Names of features
    figsize : tuple, default=(12, 10)
        Figure size for the combined plot
    
    Returns:
    --------
    fig : Figure
        Matplotlib figure with clinical insights
    """
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # Create grid layout
    gs = fig.add_gridspec(2, 2)
    
    # 1. Risk stratification plot
    ax1 = fig.add_subplot(gs[0, 0])
    risk_fig = plot_risk_stratification(model, X, y)
    for ax in risk_fig.axes:
        for line in ax.lines:
            ax1.add_line(line.deepcopy())
    ax1.set_xlim(risk_fig.axes[0].get_xlim())
    ax1.set_ylim(risk_fig.axes[0].get_ylim())
    ax1.set_title(risk_fig.axes[0].get_title())
    ax1.set_xlabel(risk_fig.axes[0].get_xlabel())
    ax1.set_ylabel(risk_fig.axes[0].get_ylabel())
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.close(risk_fig)
    
    # 2. Feature importance plot
    ax2 = fig.add_subplot(gs[0, 1])
    
    if hasattr(model, 'feature_importances_'):
        # For tree-based models
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_indices = indices[:10]
        
        ax2.barh(range(len(top_indices)), importances[top_indices])
        ax2.set_yticks(range(len(top_indices)))
        ax2.set_yticklabels([feature_names[i] for i in top_indices])
        ax2.set_title("Feature Importance")
    elif hasattr(model, 'coef_'):
        # For linear models like Cox PH
        coefs = model.coef_
        indices = np.argsort(np.abs(coefs))[::-1]
        top_indices = indices[:10]
        
        ax2.barh(range(len(top_indices)), coefs[top_indices])
        ax2.set_yticks(range(len(top_indices)))
        ax2.set_yticklabels([feature_names[i] for i in top_indices])
        ax2.set_title("Feature Coefficients")
    else:
        ax2.text(0.5, 0.5, "Feature importance not available for this model",
                ha='center', va='center', transform=ax2.transAxes)
    
    # 3. Partial effect plot for the most important feature
    ax3 = fig.add_subplot(gs[1, :])
    
    if hasattr(model, 'feature_importances_'):
        top_feature_idx = np.argsort(model.feature_importances_)[::-1][0]
    elif hasattr(model, 'coef_'):
        top_feature_idx = np.argsort(np.abs(model.coef_))[::-1][0]
    else:
        top_feature_idx = 0  # Default to first feature
    
    try:
        effect_fig = plot_feature_partial_effects(model, feature_names, X, top_feature_idx)
        
        # Transfer content
        for ax in effect_fig.axes:
            for line in ax.lines:
                ax3.add_line(line.deepcopy())
        
        # Transfer colorbar
        cbar = effect_fig.axes[1]
        cax = fig.add_axes([0.92, 0.1, 0.02, 0.3])
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=cbar.norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cax)
        cax.set_ylabel(feature_names[top_feature_idx])
        
        ax3.set_title(effect_fig.axes[0].get_title())
        ax3.set_xlabel(effect_fig.axes[0].get_xlabel())
        ax3.set_ylabel(effect_fig.axes[0].get_ylabel())
        ax3.grid(True, alpha=0.3)
        plt.close(effect_fig)
    except Exception as e:
        ax3.text(0.5, 0.5, f"Partial effect plot not available: {str(e)}",
                ha='center', va='center', transform=ax3.transAxes)
    
    # Adjust layout
    fig.tight_layout()
    
    return fig


if __name__ == "__main__":
    print("This module provides utility functions for survival analysis visualization.")
    print("Import and use functions from another script rather than running directly.") 