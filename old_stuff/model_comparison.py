"""
Model comparison framework for stroke prediction in AF patients.

This module implements tools to compare different survival models:
1. Clinical scores (CHADSVASC)
2. Machine learning models (Random Survival Forests, Cox PH, etc.)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from eda import calculate_chadsvasc
from random_survival_forest import RandomSurvivalForestModel, StrokeDataPreprocessor, SKSURV_AVAILABLE

# Import scikit-survival for Cox PH and other models
try:
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored, integrated_brier_score
    from sksurv.util import Surv
    from sksurv.compare import compare_survival
except ImportError:
    print("Warning: scikit-survival not fully installed. Some models may not be available.")


class CHADSVASCModel:
    """
    Wrapper for CHADSVASC score to make it compatible with the model comparison framework.
    """
    
    def __init__(self):
        """Initialize the CHADSVASC model."""
        self.name = "CHADSVASC"
        self.feature_names = None
        self.feature_data = None
    
    def fit(self, X, y, feature_data=None):
        """
        "Fit" the CHADSVASC model (no actual fitting needed).
        
        Parameters:
        -----------
        X : array-like
            Not used for CHADSVASC
        y : structured array
            Not used for CHADSVASC
        feature_data : pandas.DataFrame
            Original dataframe with all features needed for CHADSVASC calculation
            
        Returns:
        --------
        self : object
            Returns self
        """
        if feature_data is None:
            raise ValueError("Feature data is required for CHADSVASC model")
        self.feature_data = feature_data
        return self
    
    def predict(self, X, feature_data=None):
        """
        Predict risk using CHADSVASC score.
        
        Parameters:
        -----------
        X : array-like
            Not used for CHADSVASC
        feature_data : pandas.DataFrame
            Original dataframe with all features needed for CHADSVASC calculation
            
        Returns:
        --------
        risk : array
            CHADSVASC scores
        """
        if feature_data is None and self.feature_data is None:
            raise ValueError("Feature data is required for CHADSVASC prediction")
        
        data = feature_data if feature_data is not None else self.feature_data
        
        # Ensure data is not None before applying function
        if data is None:
            raise ValueError("Feature data is None")
            
        # Calculate CHADSVASC scores
        scores = data.apply(calculate_chadsvasc, axis=1).values
        
        # Higher scores mean higher risk, so we return as is
        return scores


class CoxPHModel:
    """
    Wrapper for Cox Proportional Hazards model.
    """
    
    def __init__(self, alpha=0.01):
        """
        Initialize the Cox PH model.
        
        Parameters:
        -----------
        alpha : float, default=0.01
            Regularization strength
        """
        if not SKSURV_AVAILABLE:
            raise ImportError("scikit-survival is required for CoxPH model")
        
        self.name = "Cox PH"
        # Use a default alpha value that works with the model
        # The scikit-survival API may have changed from what was expected
        self.model = CoxPHSurvivalAnalysis()
        self.feature_names = None
    
    def fit(self, X, y, feature_names=None):
        """
        Fit the Cox PH model.
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : structured array
            Survival target
        feature_names : list, optional
            Feature names
            
        Returns:
        --------
        self : object
            Returns self
        """
        self.model.fit(X, y)
        self.feature_names = feature_names
        return self
    
    def predict(self, X):
        """
        Predict risk scores.
        
        Parameters:
        -----------
        X : array-like
            Test features
            
        Returns:
        --------
        risk : array
            Risk scores
        """
        return self.model.predict(X)
    
    def predict_survival_function(self, X):
        """
        Predict survival function.
        
        Parameters:
        -----------
        X : array-like
            Test features
            
        Returns:
        --------
        survival : array
            Survival function values
        """
        return self.model.predict_survival_function(X)


class ModelComparison:
    """
    Framework for comparing different survival models.
    """
    
    def __init__(self, data_loader_func=None):
        """
        Initialize the model comparison framework.
        
        Parameters:
        -----------
        data_loader_func : callable, optional
            Function to load the data
        """
        self.data_loader_func = data_loader_func
        self.models = {}
        self.results = {}
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.train_df = None
        self.test_df = None
    
    def load_data(self, data_loader_func=None):
        """
        Load and preprocess the data.
        
        Parameters:
        -----------
        data_loader_func : callable, optional
            Function to load the data
            
        Returns:
        --------
        self : object
            Returns self
        """
        if data_loader_func is not None:
            self.data_loader_func = data_loader_func
            
        if self.data_loader_func is None:
            # Use default data loader (from eda.py)
            from eda import get_df
            self.data = get_df()
        else:
            self.data = self.data_loader_func()
        
        # Preprocess the data
        preprocessor = StrokeDataPreprocessor()
        X, y, self.feature_names = preprocessor.fit_transform(self.data)
        
        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test, train_idx, test_idx = train_test_split(
            X, y, np.arange(len(y)), test_size=0.2, random_state=42
        )
        
        # Keep the original dataframes for models that need raw features
        self.train_df = self.data.iloc[train_idx].reset_index(drop=True)
        self.test_df = self.data.iloc[test_idx].reset_index(drop=True)
        
        return self
    
    def add_model(self, name, model):
        """
        Add a model to the comparison.
        
        Parameters:
        -----------
        name : str
            Name of the model
        model : object
            Model object with fit and predict methods
            
        Returns:
        --------
        self : object
            Returns self
        """
        self.models[name] = model
        return self
    
    def fit_all_models(self):
        """
        Fit all added models.
        
        Returns:
        --------
        self : object
            Returns self
        """
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_data first.")
            
        for name, model in self.models.items():
            print(f"Fitting {name}...")
            
            # Different handling for CHADSVASC
            if name == "CHADSVASC":
                model.fit(self.X_train, self.y_train, feature_data=self.train_df)
            else:
                model.fit(self.X_train, self.y_train, feature_names=self.feature_names)
        
        return self
    
    def evaluate_all_models(self):
        """
        Evaluate all fitted models.
        
        Returns:
        --------
        results : dict
            Dictionary with evaluation metrics
        """
        if self.X_test is None:
            raise ValueError("Data not loaded. Call load_data first.")
            
        if self.y_test is None:
            raise ValueError("Target data is missing. Check data loading.")
            
        self.results = {}
        
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            
            # Different handling for CHADSVASC
            if name == "CHADSVASC":
                risk_scores = model.predict(self.X_test, feature_data=self.test_df)
            else:
                risk_scores = model.predict(self.X_test)
            
            # Validate inputs to concordance_index_censored
            if self.y_test is None or 'event' not in self.y_test.dtype.names or 'time' not in self.y_test.dtype.names:
                print(f"Warning: Cannot calculate c-index for {name}, target data is invalid")
                self.results[name] = {'c_index': float('nan')}
                continue
                
            # Calculate C-index
            c_index = concordance_index_censored(
                self.y_test['event'], self.y_test['time'], risk_scores
            )[0]
            
            self.results[name] = {
                'c_index': c_index
            }
            
            print(f"{name} C-index: {c_index:.4f}")
        
        return self.results
    
    def plot_c_indices(self, figsize=(10, 6)):
        """
        Plot C-indices for all models.
        
        Parameters:
        -----------
        figsize : tuple, default=(10, 6)
            Figure size
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure
        """
        if not self.results:
            raise ValueError("No results available. Call evaluate_all_models first.")
            
        # Extract C-indices
        names = list(self.results.keys())
        c_indices = [self.results[name]['c_index'] for name in names]
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(names, c_indices)
        ax.set_ylabel('C-index (discrimination)')
        ax.set_title('Model Comparison: C-index')
        ax.set_ylim(0.5, 1.0)  # C-index ranges from 0.5 (random) to 1.0 (perfect)
        
        # Add values on top of bars
        for i, v in enumerate(c_indices):
            ax.text(i, v + 0.01, f"{v:.3f}", ha='center')
        
        return fig


def run_model_comparison():
    """
    Run model comparison for stroke prediction.
    
    Returns:
    --------
    comparison : ModelComparison
        Model comparison object with results
    """
    print("Starting model comparison...")
    
    # Initialize comparison framework
    comparison = ModelComparison()
    
    try:
        # Load data
        print("Loading data...")
        comparison.load_data()
        
        # Add models
        print("Adding models...")
        comparison.add_model("CHADSVASC", CHADSVASCModel())
        
        if SKSURV_AVAILABLE:
            try:
                # Add Cox PH model
                comparison.add_model("Cox PH", CoxPHModel())
                
                # Add Random Survival Forest model
                rsf = RandomSurvivalForestModel(
                    n_estimators=100,
                    min_samples_split=10,
                    min_samples_leaf=15
                )
                comparison.add_model("Random Survival Forest", rsf)
            except Exception as e:
                print(f"Warning: Could not add one or more models: {e}")
        
        # Fit models
        comparison.fit_all_models()
        
        # Evaluate models
        results = comparison.evaluate_all_models()
        
        # Plot comparison
        try:
            fig = comparison.plot_c_indices()
            fig.savefig("model_comparison.png")
        except Exception as e:
            print(f"Warning: Could not create comparison plot: {e}")
    
    except Exception as e:
        print(f"Error in model comparison: {e}")
        
    return comparison


if __name__ == "__main__":
    print("Running model comparison...")
    comparison = run_model_comparison()
    print("Comparison complete.") 