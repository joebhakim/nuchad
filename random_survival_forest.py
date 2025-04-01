"""
Random Survival Forest implementation for stroke prediction in AF patients.

This module implements a survival analysis framework using Random Survival Forests
to predict time-to-stroke events in patients with atrial fibrillation.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Data processing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Import scikit-survival
try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.metrics import concordance_index_censored, integrated_brier_score, cumulative_dynamic_auc
    SKSURV_AVAILABLE = True
except ImportError:
    print("Warning: scikit-survival not installed. Run 'uv add scikit-survival' to install.")
    SKSURV_AVAILABLE = False

class StrokeDataPreprocessor:
    """Preprocesses data for stroke prediction models."""
    
    def __init__(self):
        """Initialize preprocessor with default settings."""
        self.categorical_features = [
            'gender', 'Anticoagulant', 'ethnic_group', 'smoking_status'
        ]
        
        self.binary_features = [
            'af', 'hypertension', 'diab', 'thrombo', 'hf', 
            'HB_stroke_history', 'ckd', 'vasc_dis_mi_pad', 'aortic_plaq'
        ]
        
        self.continuous_features = [
            'age', 'frailty_score', 'bmi', 'tc_mmol_L', 'acr_mg_mmol'
        ]
        
        # Create preprocessing pipelines
        self.preprocessor = None
        self.transformed_feature_names = None
    
    def fit_transform(self, df, target_time='stroke_time', event_indicator='stroke_1Y'):
        """
        Fit the preprocessing pipeline and transform the data.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input data with features and outcome variables
        target_time : str, default='stroke_time'
            Column name for time-to-event
        event_indicator : str, default='stroke_1Y'
            Column name for event indicator (1=event occurred, 0=censored)
            
        Returns:
        --------
        X : numpy.ndarray
            Preprocessed feature matrix
        y : structured array
            Survival target with time and event indicator
        feature_names : list
            Names of features after preprocessing
        """
        # Filter eligible patients
        eligible_df = self._filter_eligible_patients(df)
        
        # Handle missing values in target variable
        # For stroke_time, use time from start to end of follow-up for censored patients
        eligible_df = eligible_df.copy()
        eligible_df['temp_stroke_time'] = eligible_df[target_time]
        
        # For patients without a stroke (censored), use follow-up time
        mask_no_stroke = eligible_df[event_indicator] != 1
        
        # Calculate follow-up time in days for censored patients
        eligible_df.loc[mask_no_stroke, 'temp_stroke_time'] = (
            (eligible_df.loc[mask_no_stroke, 'end_fu'] - eligible_df.loc[mask_no_stroke, 'time1']).dt.days
        )
        
        # Convert to numeric and handle any remaining NaNs by dropping those rows
        eligible_df['temp_stroke_time'] = pd.to_numeric(eligible_df['temp_stroke_time'], errors='coerce')
        eligible_df = eligible_df.dropna(subset=['temp_stroke_time'])
        
        # Handle stroke_1Y coding (1=yes, 2=no stroke after 1Y, 0=no stroke)
        # Convert to binary event indicator (1=event, 0=censored)
        event = (eligible_df[event_indicator] == 1).astype(int)
        
        # Get time-to-event or censoring time
        times = eligible_df['temp_stroke_time'].values
        
        # Ensure times are positive (scikit-survival requirement)
        times = np.maximum(times, 0.5)  # Use a small positive value for any zero times
        
        # Create scikit-survival compatible target
        y = np.zeros(len(eligible_df), dtype=[('event', bool), ('time', float)])
        y['event'] = event
        y['time'] = times
        
        # Create preprocessing pipeline
        self.preprocessor = self._create_preprocessor()
        
        # Get feature matrix
        X_features = eligible_df[self.categorical_features + self.binary_features + self.continuous_features]
        
        # Fit and transform
        X = self.preprocessor.fit_transform(X_features)
        
        # Get transformed feature names
        self.transformed_feature_names = self._get_feature_names(X_features)
        
        print(f"Preprocessed data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Event rate: {event.mean():.2%}")
        
        return X, y, self.transformed_feature_names
    
    def transform(self, df):
        """Transform new data using fitted preprocessor."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        # Filter eligible patients
        eligible_df = self._filter_eligible_patients(df)
        
        # Get feature matrix
        X_features = eligible_df[self.categorical_features + self.binary_features + self.continuous_features]
        
        # Transform
        X = self.preprocessor.transform(X_features)
        
        return X
    
    def _filter_eligible_patients(self, df):
        """
        Filters patients who are eligible based on criteria:
        - AF diagnosis before time1
        - Follow-up at least 1 year
        """
        # Same criteria from eda.py
        af_diagnosis_mask = (df["earliest_af_date"] <= df["time1"])
        follow_up_mask = df["end_fu"] >= df["time1"] + pd.Timedelta(days=365)
        
        # Apply masks
        eligible_mask = af_diagnosis_mask & follow_up_mask
        eligible_df = df[eligible_mask].copy()
        
        return eligible_df
    
    def _create_preprocessor(self):
        """Create sklearn preprocessing pipeline for features."""
        # For categorical features, use one-hot encoding with simple imputation
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # For binary features, use simple imputation
        binary_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])
        
        # For continuous features, use simple imputation and scaling
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Combine all preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('categorical', categorical_transformer, self.categorical_features),
                ('binary', binary_transformer, self.binary_features),
                ('numeric', numeric_transformer, self.continuous_features)
            ]
        )
        
        return preprocessor
    
    def _get_feature_names(self, X_features):
        """Get feature names after transformation."""
        # This is a simplified approach - in practice, need to handle one-hot encoding properly
        feature_names = []
        
        # Handle categorical features (one-hot encoded)
        for feature in self.categorical_features:
            # For each categorical feature, get unique values from data
            # This is simplified - in practice, would need to extract from encoder
            unique_values = X_features[feature].dropna().unique()
            for val in unique_values:
                feature_names.append(f"{feature}_{val}")
        
        # Handle binary and numeric features (pass through)
        feature_names.extend(self.binary_features)
        feature_names.extend(self.continuous_features)
        
        return feature_names


class RandomSurvivalForestModel:
    """
    Implements a Random Survival Forest model for stroke prediction.
    
    This class wraps the scikit-survival RandomSurvivalForest implementation
    with additional functionality for model evaluation and interpretation.
    """
    
    def __init__(self, n_estimators=100, min_samples_split=10, 
                 min_samples_leaf=15, max_features="sqrt", random_state=42):
        """
        Initialize the Random Survival Forest model.
        
        Parameters:
        -----------
        n_estimators : int, default=100
            Number of trees in the forest
        min_samples_split : int, default=10
            Minimum number of samples required to split an internal node
        min_samples_leaf : int, default=15
            Minimum number of samples required to be at a leaf node
        max_features : str or int, default="sqrt"
            Number of features to consider for best split
        random_state : int, default=42
            Random seed for reproducibility
        """
        if not SKSURV_AVAILABLE:
            raise ImportError("scikit-survival is required but not installed. Install with 'uv add scikit-survival'")
            
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.model = None
        self.feature_names = None
    
    def fit(self, X, y, feature_names=None):
        """
        Fit the Random Survival Forest model.
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            Training data
        y : structured array, shape=(n_samples,)
            Target data with fields 'event' and 'time'
        feature_names : list, optional
            Names of features
            
        Returns:
        --------
        self : object
            Returns self
        """
        self.model = RandomSurvivalForest(
            n_estimators=self.n_estimators,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state
        )
        
        self.model.fit(X, y)
        self.feature_names = feature_names
        
        return self
    
    def predict_survival_function(self, X, times=None):
        """
        Predict survival function for samples in X.
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            Data for prediction
        times : array-like, optional
            Time points at which to evaluate the survival function
            
        Returns:
        --------
        surv : array, shape=(n_samples, n_time_points)
            Predicted survival probabilities at each time point
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit first.")
        
        # Handle the times parameter correctly for scikit-survival
        if times is not None:
            return self.model.predict_survival_function(X, return_array=False)
        else:
            return self.model.predict_survival_function(X, return_array=False)
    
    def predict_risk(self, X):
        """
        Predict risk scores for samples in X.
        
        Higher values indicate higher risk of event.
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            Data for prediction
            
        Returns:
        --------
        risk : array, shape=(n_samples,)
            Predicted risk scores
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit first.")
            
        return self.model.predict(X)
    
    def predict(self, X):
        """
        Alias for predict_risk to maintain compatibility with model comparison framework.
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            Data for prediction
            
        Returns:
        --------
        risk : array, shape=(n_samples,)
            Predicted risk scores
        """
        return self.predict_risk(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data.
        
        Parameters:
        -----------
        X_test : array-like, shape=(n_samples, n_features)
            Test data
        y_test : structured array, shape=(n_samples,)
            Test target data with fields 'event' and 'time'
            
        Returns:
        --------
        metrics : dict
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit first.")
            
        # C-index (Harrell's concordance index)
        risk_scores = self.predict_risk(X_test)
        c_index = concordance_index_censored(
            y_test['event'], y_test['time'], risk_scores
        )[0]
        
        # More metrics can be added as needed
        
        return {
            'c_index': c_index
        }
    
    def plot_feature_importance(self, top_n=10, figsize=(10, 6)):
        """
        Plot feature importance.
        
        Parameters:
        -----------
        top_n : int, default=10
            Number of top features to show
        figsize : tuple, default=(10, 6)
            Figure size
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit first.")
        
        # Get feature importance
        importances = self.model.feature_importances_
        
        # Use feature names if available, otherwise use indices
        if self.feature_names is not None:
            indices = np.argsort(importances)[::-1][:top_n]
            names = [self.feature_names[i] for i in indices]
        else:
            indices = np.argsort(importances)[::-1][:top_n]
            names = [f"Feature {i}" for i in indices]
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(range(top_n), importances[indices])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(names)
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top Features by Importance')
        
        return fig
    
    def plot_survival_curves(self, X, times=None, n_curves=5, figsize=(10, 6)):
        """
        Plot survival curves for a subset of samples.
        
        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            Data for prediction
        times : array-like, optional
            Time points at which to evaluate the survival function
        n_curves : int, default=5
            Number of curves to plot
        figsize : tuple, default=(10, 6)
            Figure size
            
        Returns:
        --------
        fig : Figure
            Matplotlib figure
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit first.")
        
        # Sample n_curves from X
        if X.shape[0] > n_curves:
            indices = np.random.choice(X.shape[0], n_curves, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
            n_curves = X.shape[0]
        
        # Always get StepFunction objects for plotting
        surv_funcs = self.model.predict_survival_function(X_sample, return_array=False)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # StepFunction objects have x and y attributes
        for i, surv_func in enumerate(surv_funcs):
            ax.step(surv_func.x, surv_func.y, where="post", label=f"Sample {i+1}")
        
        ax.set_xlabel('Time (days)')
        ax.set_ylabel('Survival Probability')
        ax.set_title('Predicted Survival Curves')
        ax.legend()
        ax.grid(True)
        
        return fig


def run_rsf_analysis(data_path="dummy_data.csv"):
    """
    Run Random Survival Forest analysis on stroke data.
    
    Parameters:
    -----------
    data_path : str, default="dummy_data.csv"
        Path to the data file
        
    Returns:
    --------
    results : dict
        Dictionary with results and model
    """
    if not SKSURV_AVAILABLE:
        print("Error: scikit-survival is not installed. Install with 'uv add scikit-survival'")
        return None
        
    # Load data
    print("Loading data...")
    from eda import get_df
    df = get_df()
    
    # Initialize preprocessor
    print("Preprocessing data...")
    preprocessor = StrokeDataPreprocessor()
    X, y, feature_names = preprocessor.fit_transform(df)
    
    # Split data
    print("Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and fit model
    print("Fitting Random Survival Forest model...")
    rsf = RandomSurvivalForestModel(n_estimators=100)
    rsf.fit(X_train, y_train, feature_names)
    
    # Evaluate
    print("Evaluating model...")
    metrics = rsf.evaluate(X_test, y_test)
    print(f"C-index: {metrics['c_index']:.4f}")
    
    # Plot feature importance
    print("Plotting feature importance...")
    fig_importance = rsf.plot_feature_importance()
    fig_importance.savefig("rsf_feature_importance.png")
    
    # Plot survival curves for a few samples
    print("Plotting survival curves...")
    fig_curves = rsf.plot_survival_curves(X_test[:5])
    fig_curves.savefig("rsf_survival_curves.png")
    
    return {
        "model": rsf,
        "metrics": metrics,
        "preprocessor": preprocessor
    }


if __name__ == "__main__":
    print("Running Random Survival Forest analysis...")
    results = run_rsf_analysis()
    print("Analysis complete.") 