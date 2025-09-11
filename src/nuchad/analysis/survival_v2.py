"""
Enhanced Survival Analysis Functions (v2)

This module provides improved survival analysis functions that handle the stroke encoding
differences between old and new datasets. It includes explicit event/control/censoring 
classification with validation and diagnostic capabilities.

Key improvements:
- Dataset-aware stroke encoding logic
- Explicit event/control/censoring classification
- Validation of mutual exclusivity
- Diagnostic correlation analysis
- Modular architecture for better testing
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from scipy.stats import pearsonr
from nuchad.utils import calculate_chadsvasc


def detect_dataset_type(df: pd.DataFrame) -> str:
    """
    Detect whether this is the old or new dataset based on stroke encoding patterns.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        'old' or 'new' indicating dataset type
    """
    if 'stroke_1Y' not in df.columns:
        return 'unknown'
    
    # Check stroke_1Y value distribution
    stroke_values = df['stroke_1Y'].dropna().unique()
    
    # Old dataset: primarily values 1 and 2
    # New dataset: values 1, 2, 3, 4 with many missing
    has_values_3_4 = any(val in [3.0, 4.0] for val in stroke_values if not pd.isna(val))
    missing_rate = df['stroke_1Y'].isna().sum() / len(df)
    
    if has_values_3_4 or missing_rate > 0.5:
        return 'new'
    else:
        return 'old'


def classify_events_and_controls(df: pd.DataFrame, 
                               diagnostic_correlate: str = 'age',
                               time_window_days: int = 365) -> Tuple[pd.DataFrame, Dict]:
    """
    Classify patients as events or controls with explicit validation.
    
    Args:
        df: DataFrame with patient data
        diagnostic_correlate: Column to check correlation with (default: 'age')
        time_window_days: Time window for event definition (default: 365)
    
    Returns:
        df_classified: DataFrame with 'event' and 'control' boolean columns
        validation_report: Dict with classification stats and correlations
    """
    df_classified = df.copy()
    dataset_type = detect_dataset_type(df)
    
    # Initialize columns
    df_classified['event'] = False
    df_classified['control'] = False
    
    # Dataset-specific classification logic
    if dataset_type == 'old':
        # Old dataset logic: stroke_1Y = 1 (event), stroke_1Y = 2 (control)
        
        # Events: stroke_1Y = 1 AND has stroke date AND stroke after AF diagnosis AND within time window
        event_mask = (
            (df_classified['stroke_1Y'] == 1) &
            df_classified['earliest_stroke_date'].notna() &
            (df_classified['earliest_stroke_date'] > df_classified['time1']) &
            ((df_classified['earliest_stroke_date'] - df_classified['time1']).dt.days <= time_window_days)
        )
        df_classified.loc[event_mask, 'event'] = True
        
        # Controls: stroke_1Y = 2 (explicit controls in old encoding)
        control_mask = (df_classified['stroke_1Y'] == 2)
        df_classified.loc[control_mask, 'control'] = True
        
    elif dataset_type == 'new':
        # New dataset logic: stroke_1Y = 1 (event within 1Y), Missing (control)
        
        # Events: stroke_1Y = 1 AND has stroke date AND stroke after AF diagnosis AND within time window
        event_mask = (
            (df_classified['stroke_1Y'] == 1.0) &
            df_classified['earliest_stroke_date'].notna() &
            (df_classified['earliest_stroke_date'] > df_classified['time1']) &
            ((df_classified['earliest_stroke_date'] - df_classified['time1']).dt.days <= time_window_days)
        )
        df_classified.loc[event_mask, 'event'] = True
        
        # Controls: Missing stroke_1Y (controls in new encoding)
        control_mask = df_classified['stroke_1Y'].isna()
        df_classified.loc[control_mask, 'control'] = True
        
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Validation: Check mutual exclusivity
    event_count = df_classified['event'].sum()
    control_count = df_classified['control'].sum()
    both_count = (df_classified['event'] & df_classified['control']).sum()
    neither_count = (~df_classified['event'] & ~df_classified['control']).sum()
    
    # Calculate CHADS-VASc scores for correlation analysis
    df_classified['chadsvasc'] = df_classified.apply(calculate_chadsvasc, axis=1)
    
    # Create risk groups
    df_classified['risk_group'] = pd.cut(
        df_classified['chadsvasc'], 
        bins=[-0.5, 1.5, 3.5, 10], 
        labels=['Low (0-1)', 'Moderate (2-3)', 'High (4+)']
    )
    
    # Calculate diagnostic correlation
    correlation_results = {}
    if diagnostic_correlate in df_classified.columns:
        # Correlation between event rate and diagnostic variable
        valid_data = df_classified[df_classified[diagnostic_correlate].notna()]
        if len(valid_data) > 0:
            try:
                correlation, p_value = pearsonr(valid_data['event'].astype(int), 
                                              valid_data[diagnostic_correlate])
                correlation_results[diagnostic_correlate] = {
                    'correlation': correlation,
                    'p_value': p_value
                }
            except Exception as e:
                correlation_results[diagnostic_correlate] = {
                    'error': str(e)
                }
    
    # Event rates by risk group
    risk_group_stats = {}
    for group in ['Low (0-1)', 'Moderate (2-3)', 'High (4+)']:
        group_data = df_classified[df_classified['risk_group'] == group]
        if len(group_data) > 0:
            event_rate = group_data['event'].mean()
            control_rate = group_data['control'].mean()
            risk_group_stats[group] = {
                'total': len(group_data),
                'events': group_data['event'].sum(),
                'controls': group_data['control'].sum(),
                'event_rate': event_rate,
                'control_rate': control_rate
            }
    
    # Validation report
    validation_report = {
        'dataset_type': dataset_type,
        'total_patients': len(df_classified),
        'events': int(event_count),
        'controls': int(control_count),
        'both_event_and_control': int(both_count),
        'neither_event_nor_control': int(neither_count),
        'event_rate_overall': event_count / len(df_classified),
        'control_rate_overall': control_count / len(df_classified),
        'mutual_exclusivity_violations': int(both_count),
        'unclassified_patients': int(neither_count),
        'correlation_analysis': correlation_results,
        'risk_group_statistics': risk_group_stats,
        'time_window_days': time_window_days
    }
    
    return df_classified, validation_report


def classify_censoring(df_with_events_controls: pd.DataFrame, 
                      censoring_reasons: Optional[Dict] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Classify censoring for patients who are neither events nor controls.
    
    Args:
        df_with_events_controls: DataFrame with 'event' and 'control' columns
        censoring_reasons: Optional dict specifying censoring logic
        
    Returns:
        df_with_censoring: DataFrame with additional 'censored' column
        censoring_report: Dict with censoring statistics and diagnostics
    """
    df_with_censoring = df_with_events_controls.copy()
    
    # Patients who are neither events nor controls are censored
    censored_mask = ~df_with_censoring['event'] & ~df_with_censoring['control']
    df_with_censoring['censored'] = censored_mask
    
    # Analyze censoring patterns
    censored_count = censored_mask.sum()
    
    # Censoring by risk group
    censoring_by_risk = {}
    if 'risk_group' in df_with_censoring.columns:
        for group in ['Low (0-1)', 'Moderate (2-3)', 'High (4+)']:
            group_data = df_with_censoring[df_with_censoring['risk_group'] == group]
            if len(group_data) > 0:
                censored_in_group = group_data['censored'].sum()
                censoring_by_risk[group] = {
                    'total': len(group_data),
                    'censored': int(censored_in_group),
                    'censoring_rate': censored_in_group / len(group_data)
                }
    
    # Check for different types of censoring (if data allows)
    censoring_types = {}
    
    # Administrative censoring (reached end of follow-up without event)
    if 'end_fu' in df_with_censoring.columns and 'earliest_stroke_date' in df_with_censoring.columns:
        censored_data = df_with_censoring[df_with_censoring['censored']]
        
        # Patients who reached end of follow-up
        admin_censored = censored_data['end_fu'].notna()
        censoring_types['administrative'] = int(admin_censored.sum())
        
        # Patients lost to follow-up (unclear - would need more data to determine)
        censoring_types['unknown'] = int((~admin_censored).sum())
    
    # Validate three-way classification
    three_way_sum = (df_with_censoring['event'] + 
                     df_with_censoring['control'] + 
                     df_with_censoring['censored']).astype(int)
    
    validation_errors = (three_way_sum != 1).sum()
    
    censoring_report = {
        'total_patients': len(df_with_censoring),
        'censored_patients': int(censored_count),
        'censoring_rate_overall': censored_count / len(df_with_censoring),
        'censoring_by_risk_group': censoring_by_risk,
        'censoring_types': censoring_types,
        'three_way_validation_errors': int(validation_errors),
        'three_way_classification_valid': validation_errors == 0
    }
    
    return df_with_censoring, censoring_report


def prepare_survival_data_v3(df: pd.DataFrame,
                            diagnostic_correlate: str = 'age',
                            time_window_days: int = 365) -> Tuple[pd.DataFrame, Dict]:
    """
    Prepare survival data with comprehensive event/control/censoring classification.
    
    Args:
        df: Input DataFrame
        diagnostic_correlate: Column to check correlation with
        time_window_days: Time window for event definition
        
    Returns:
        survival_df: DataFrame ready for survival analysis
        comprehensive_report: Combined validation and diagnostic report
    """
    print(f"Preparing survival data with {time_window_days}-day event window...")
    
    # Step 1: Classify events and controls
    df_with_events, events_report = classify_events_and_controls(
        df, diagnostic_correlate, time_window_days
    )
    
    # Step 2: Classify censoring
    df_with_all_labels, censoring_report = classify_censoring(df_with_events)
    
    # Step 3: Calculate survival times
    survival_df = df_with_all_labels.copy()
    
    # Initialize survival time
    survival_df['survival_time'] = (survival_df['end_fu'] - survival_df['time1']).dt.days
    
    # For events: use time to stroke
    event_mask = survival_df['event']
    survival_df.loc[event_mask, 'survival_time'] = (
        survival_df.loc[event_mask, 'earliest_stroke_date'] - 
        survival_df.loc[event_mask, 'time1']
    ).dt.days
    
    # Validate survival times
    negative_times = (survival_df['survival_time'] < 0).sum()
    zero_times = (survival_df['survival_time'] == 0).sum()
    
    if negative_times > 0:
        print(f"Warning: {negative_times} patients have negative survival time")
        
    if zero_times > 0:
        print(f"Warning: {zero_times} patients have 0 survival time")
    
    # Create comprehensive report
    comprehensive_report = {
        'events_and_controls_report': events_report,
        'censoring_report': censoring_report,
        'survival_time_validation': {
            'negative_times': int(negative_times),
            'zero_times': int(zero_times),
            'median_survival_time': float(survival_df['survival_time'].median()),
            'mean_survival_time': float(survival_df['survival_time'].mean())
        },
        'final_classification_summary': {
            'total_patients': len(survival_df),
            'events': int(survival_df['event'].sum()),
            'controls': int(survival_df['control'].sum()),
            'censored': int(survival_df['censored'].sum()),
            'analyzable_patients': int((survival_df['event'] | survival_df['control']).sum())
        }
    }
    
    return survival_df, comprehensive_report


def print_classification_summary(report: Dict, title: str = "Classification Summary"):
    """Print a formatted summary of classification results."""
    print("\n" + "="*60)
    print(f"{title.upper()}")
    print("="*60)
    
    # Basic counts
    events_report = report.get('events_and_controls_report', {})
    censoring_report = report.get('censoring_report', {})
    
    print(f"Dataset Type: {events_report.get('dataset_type', 'Unknown')}")
    print(f"Total Patients: {events_report.get('total_patients', 0):,}")
    
    print(f"\nClassification Results:")
    print(f"  Events: {events_report.get('events', 0):,} ({events_report.get('event_rate_overall', 0)*100:.1f}%)")
    print(f"  Controls: {events_report.get('controls', 0):,} ({events_report.get('control_rate_overall', 0)*100:.1f}%)")
    print(f"  Censored: {censoring_report.get('censored_patients', 0):,} ({censoring_report.get('censoring_rate_overall', 0)*100:.1f}%)")
    
    # Validation results
    violations = events_report.get('mutual_exclusivity_violations', 0)
    three_way_valid = censoring_report.get('three_way_classification_valid', False)
    
    print(f"\nValidation:")
    print(f"  Mutual exclusivity violations: {violations}")
    print(f"  Three-way classification valid: {'✓' if three_way_valid else '✗'}")
    
    # Risk group analysis
    risk_stats = events_report.get('risk_group_statistics', {})
    if risk_stats:
        print(f"\nEvent Rates by CHADS-VASc Risk Group:")
        for group in ['Low (0-1)', 'Moderate (2-3)', 'High (4+)']:
            stats = risk_stats.get(group, {})
            if stats:
                event_rate = stats.get('event_rate', 0) * 100
                print(f"  {group}: {event_rate:.1f}%")
    
    # Correlation analysis
    corr_results = events_report.get('correlation_analysis', {})
    if corr_results:
        print(f"\nDiagnostic Correlations:")
        for var, results in corr_results.items():
            if 'correlation' in results:
                corr = results['correlation']
                p_val = results['p_value']
                print(f"  {var}: r = {corr:.3f}, p = {p_val:.3f}")