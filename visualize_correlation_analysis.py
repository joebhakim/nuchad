#!/usr/bin/env python3
"""
Comprehensive visualization of event correlation analysis.

Creates visualizations for all the correlation experiments including:
1. Age vs event correlations with swarmplots
2. CHADS-VASc vs event correlations with swarmplots  
3. Treatment regime differences between datasets
4. Summary correlation heatmap
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from pathlib import Path

from nuchad.utils import get_df, calculate_chadsvasc

# Set style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


def prepare_analysis_data():
    """Load and prepare data for all analyses."""
    print("Loading and preparing analysis data...")
    
    # Load datasets
    old_df = get_df("random_nuchad.csv")
    new_df = get_df("random_nuchad_250623.csv")
    
    datasets = {'OLD': old_df, 'NEW': new_df}
    prepared_data = {}
    
    for name, df in datasets.items():
        # Calculate CHADS-VASc
        df_prep = df.copy()
        df_prep['chadsvasc'] = df_prep.apply(calculate_chadsvasc, axis=1)
        df_prep['risk_group'] = pd.cut(
            df_prep['chadsvasc'], 
            bins=[-0.5, 1.5, 3.5, 10], 
            labels=['Low (0-1)', 'Moderate (2-3)', 'High (4+)']
        )
        df_prep['dataset'] = name
        
        # Create different event definitions
        event_definitions = {}
        
        if name == 'OLD':
            event_definitions['stroke_1Y_eq_1'] = (df_prep['stroke_1Y'] == 1)
            event_definitions['has_stroke_date'] = df_prep['earliest_stroke_date'].notna()
            event_definitions['stroke_1Y_1_and_date'] = (
                (df_prep['stroke_1Y'] == 1) & df_prep['earliest_stroke_date'].notna()
            )
            # Treatment info
            df_prep['has_anticoag'] = df_prep['Anticoagulant'].notna() & (df_prep['Anticoagulant'] != 0)
            df_prep['treatment_simple'] = df_prep['has_anticoag'].map({True: 'OAC', False: 'No OAC'})
            
        else:  # NEW
            event_definitions['stroke_1Y_eq_1'] = (df_prep['stroke_1Y'] == 1.0)
            event_definitions['has_stroke_date'] = df_prep['earliest_stroke_date'].notna()
            event_definitions['stroke_1Y_1_and_date'] = (
                (df_prep['stroke_1Y'] == 1.0) & df_prep['earliest_stroke_date'].notna()
            )
            event_definitions['stroke_1Y_any'] = df_prep['stroke_1Y'].notna()
            event_definitions['stroke_1Y_1_3'] = df_prep['stroke_1Y'].isin([1.0, 3.0])
            
            # Treatment info - more complex for new dataset
            df_prep['has_oac'] = (
                df_prep['first_OAC_date'].notna() &
                (df_prep['first_OAC_date'] >= df_prep['time1']) &
                (df_prep['first_OAC_date'] <= df_prep['end_fu'])
            )
            df_prep['has_antiplatelet'] = (
                df_prep['first_antiplatelet_date'].notna() &
                (df_prep['first_antiplatelet_date'] >= df_prep['time1']) &
                (df_prep['first_antiplatelet_date'] <= df_prep['end_fu'])
            )
            
            # Simplified treatment categories
            df_prep['treatment_simple'] = 'No treatment'
            df_prep.loc[df_prep['has_oac'] & ~df_prep['has_antiplatelet'], 'treatment_simple'] = 'OAC only'
            df_prep.loc[~df_prep['has_oac'] & df_prep['has_antiplatelet'], 'treatment_simple'] = 'Antiplatelet only'
            df_prep.loc[df_prep['has_oac'] & df_prep['has_antiplatelet'], 'treatment_simple'] = 'Both'
        
        # Store event definitions in dataframe
        for event_name, event_mask in event_definitions.items():
            df_prep[f'event_{event_name}'] = event_mask
        
        prepared_data[name] = df_prep
    
    return prepared_data


def calculate_correlations(prepared_data):
    """Calculate all correlations for summary tables."""
    correlation_results = []
    
    for dataset_name, df in prepared_data.items():
        # Get event columns
        event_cols = [col for col in df.columns if col.startswith('event_')]
        
        for event_col in event_cols:
            event_name = event_col.replace('event_', '')
            
            # Age correlation
            valid_data = df[df['age'].notna() & df[event_col].notna()]
            if len(valid_data) > 0 and valid_data[event_col].sum() > 0 and valid_data[event_col].sum() < len(valid_data):
                try:
                    age_corr, age_p = pearsonr(valid_data[event_col].astype(int), valid_data['age'])
                except:
                    age_corr, age_p = np.nan, np.nan
            else:
                age_corr, age_p = np.nan, np.nan
            
            # CHADS-VASc correlation
            valid_chads = df[df['chadsvasc'].notna() & df[event_col].notna()]
            if len(valid_chads) > 0 and valid_chads[event_col].sum() > 0 and valid_chads[event_col].sum() < len(valid_chads):
                try:
                    chads_corr, chads_p = pearsonr(valid_chads[event_col].astype(int), valid_chads['chadsvasc'])
                except:
                    chads_corr, chads_p = np.nan, np.nan
            else:
                chads_corr, chads_p = np.nan, np.nan
            
            # Risk group event rates
            risk_rates = []
            for group in ['Low (0-1)', 'Moderate (2-3)', 'High (4+)']:
                group_data = df[df['risk_group'] == group]
                if len(group_data) > 0:
                    rate = group_data[event_col].mean() * 100
                    risk_rates.append(rate)
                else:
                    risk_rates.append(0)
            
            gradient = max(risk_rates) - min(risk_rates)
            
            correlation_results.append({
                'dataset': dataset_name,
                'event_definition': event_name,
                'age_correlation': age_corr,
                'age_p_value': age_p,
                'chadsvasc_correlation': chads_corr,
                'chadsvasc_p_value': chads_p,
                'risk_gradient': gradient,
                'low_rate': risk_rates[0],
                'moderate_rate': risk_rates[1],
                'high_rate': risk_rates[2],
                'n_events': int(df[event_col].sum()),
                'total_n': len(df)
            })
    
    return pd.DataFrame(correlation_results)


def create_age_swarmplots(prepared_data, correlation_df):
    """Create swarmplots showing age vs event for different definitions."""
    # Get main event definitions to plot
    main_events = ['stroke_1Y_eq_1', 'has_stroke_date', 'stroke_1Y_1_and_date']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Age vs Event Status by Definition and Dataset', fontsize=16, fontweight='bold')
    
    for col, event_def in enumerate(main_events):
        for row, (dataset_name, df) in enumerate(prepared_data.items()):
            ax = axes[row, col]
            
            event_col = f'event_{event_def}'
            if event_col in df.columns:
                # Prepare plot data
                plot_data = df[df['age'].notna() & df[event_col].notna()].copy()
                plot_data['Event'] = plot_data[event_col].map({True: 'Yes', False: 'No'})
                
                # Create swarmplot
                sns.swarmplot(data=plot_data, x='Event', y='age', ax=ax, alpha=0.6, size=2)
                
                # Get correlation stats
                corr_row = correlation_df[
                    (correlation_df['dataset'] == dataset_name) & 
                    (correlation_df['event_definition'] == event_def)
                ]
                
                if not corr_row.empty:
                    r = corr_row['age_correlation'].iloc[0]
                    p = corr_row['age_p_value'].iloc[0]
                    
                    if not pd.isna(r) and not pd.isna(p):
                        significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                        ax.set_title(f'{dataset_name}: {event_def.replace("_", " ").title()}\nr = {r:.3f}, p = {p:.4f} {significance}')
                    else:
                        ax.set_title(f'{dataset_name}: {event_def.replace("_", " ").title()}\nr = N/A')
                else:
                    ax.set_title(f'{dataset_name}: {event_def.replace("_", " ").title()}')
            else:
                ax.set_title(f'{dataset_name}: {event_def} (Not Available)')
                ax.text(0.5, 0.5, 'Not Available', ha='center', va='center', transform=ax.transAxes)
            
            ax.set_xlabel('Event Status')
            ax.set_ylabel('Age (years)')
    
    plt.tight_layout()
    return fig


def create_chadsvasc_swarmplots(prepared_data, correlation_df):
    """Create swarmplots showing CHADS-VASc vs event for different definitions."""
    main_events = ['stroke_1Y_eq_1', 'has_stroke_date', 'stroke_1Y_1_and_date']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('CHADS-VASc Score vs Event Status by Definition and Dataset', fontsize=16, fontweight='bold')
    
    for col, event_def in enumerate(main_events):
        for row, (dataset_name, df) in enumerate(prepared_data.items()):
            ax = axes[row, col]
            
            event_col = f'event_{event_def}'
            if event_col in df.columns:
                # Prepare plot data
                plot_data = df[df['chadsvasc'].notna() & df[event_col].notna()].copy()
                plot_data['Event'] = plot_data[event_col].map({True: 'Yes', False: 'No'})
                
                # Create swarmplot
                sns.swarmplot(data=plot_data, x='Event', y='chadsvasc', ax=ax, alpha=0.6, size=2)
                
                # Get correlation stats
                corr_row = correlation_df[
                    (correlation_df['dataset'] == dataset_name) & 
                    (correlation_df['event_definition'] == event_def)
                ]
                
                if not corr_row.empty:
                    r = corr_row['chadsvasc_correlation'].iloc[0]
                    p = corr_row['chadsvasc_p_value'].iloc[0]
                    
                    if not pd.isna(r) and not pd.isna(p):
                        significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                        ax.set_title(f'{dataset_name}: {event_def.replace("_", " ").title()}\nr = {r:.3f}, p = {p:.4f} {significance}')
                    else:
                        ax.set_title(f'{dataset_name}: {event_def.replace("_", " ").title()}\nr = N/A')
                else:
                    ax.set_title(f'{dataset_name}: {event_def.replace("_", " ").title()}')
            else:
                ax.set_title(f'{dataset_name}: {event_def} (Not Available)')
                ax.text(0.5, 0.5, 'Not Available', ha='center', va='center', transform=ax.transAxes)
            
            ax.set_xlabel('Event Status')
            ax.set_ylabel('CHADS-VASc Score')
    
    plt.tight_layout()
    return fig


def create_risk_group_barplots(prepared_data, correlation_df):
    """Create bar plots showing event rates by risk group."""
    main_events = ['stroke_1Y_eq_1', 'has_stroke_date']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Event Rates by CHADS-VASc Risk Group', fontsize=16, fontweight='bold')
    
    for col, event_def in enumerate(main_events):
        for row, (dataset_name, df) in enumerate(prepared_data.items()):
            ax = axes[row, col]
            
            event_col = f'event_{event_def}'
            if event_col in df.columns:
                # Calculate rates by risk group
                rates_data = []
                for group in ['Low (0-1)', 'Moderate (2-3)', 'High (4+)']:
                    group_data = df[df['risk_group'] == group]
                    if len(group_data) > 0:
                        rate = group_data[event_col].mean() * 100
                        rates_data.append({'Risk Group': group, 'Event Rate (%)': rate})
                
                if rates_data:
                    rates_df = pd.DataFrame(rates_data)
                    bars = ax.bar(rates_df['Risk Group'], rates_df['Event Rate (%)'])
                    
                    # Color bars
                    colors = ['lightblue', 'orange', 'lightcoral']
                    for bar, color in zip(bars, colors):
                        bar.set_color(color)
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}%', ha='center', va='bottom')
                    
                    # Get gradient for title
                    corr_row = correlation_df[
                        (correlation_df['dataset'] == dataset_name) & 
                        (correlation_df['event_definition'] == event_def)
                    ]
                    
                    if not corr_row.empty:
                        gradient = corr_row['risk_gradient'].iloc[0]
                        ax.set_title(f'{dataset_name}: {event_def.replace("_", " ").title()}\nGradient = {gradient:.1f}%')
                    else:
                        ax.set_title(f'{dataset_name}: {event_def.replace("_", " ").title()}')
                else:
                    ax.set_title(f'{dataset_name}: {event_def} (No Data)')
                    
            ax.set_xlabel('CHADS-VASc Risk Group')
            ax.set_ylabel('Event Rate (%)')
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig


def create_treatment_visualization(prepared_data):
    """Visualize treatment differences between datasets."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Treatment Regime Differences Between Datasets', fontsize=16, fontweight='bold')
    
    # Treatment distribution pie charts
    for idx, (dataset_name, df) in enumerate(prepared_data.items()):
        ax = ax1 if idx == 0 else ax2
        
        treatment_counts = df['treatment_simple'].value_counts()
        
        wedges, texts, autotexts = ax.pie(treatment_counts.values, labels=treatment_counts.index, 
                                         autopct='%1.1f%%', startangle=90)
        ax.set_title(f'{dataset_name} Dataset Treatment Distribution')
    
    # Treatment vs age distribution
    combined_data = []
    for dataset_name, df in prepared_data.items():
        df_subset = df[df['age'].notna()].copy()
        df_subset['Dataset'] = dataset_name
        combined_data.append(df_subset[['age', 'treatment_simple', 'Dataset']])
    
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # Box plot of age by treatment and dataset
    sns.boxplot(data=combined_df, x='treatment_simple', y='age', hue='Dataset', ax=ax3)
    ax3.set_title('Age Distribution by Treatment and Dataset')
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_xlabel('Treatment Type')
    ax3.set_ylabel('Age (years)')
    
    # Sample sizes by treatment
    treatment_counts_both = combined_df.groupby(['Dataset', 'treatment_simple']).size().reset_index(name='Count')
    treatment_pivot = treatment_counts_both.pivot(index='treatment_simple', columns='Dataset', values='Count').fillna(0)
    
    treatment_pivot.plot(kind='bar', ax=ax4)
    ax4.set_title('Sample Sizes by Treatment Type')
    ax4.set_xlabel('Treatment Type')
    ax4.set_ylabel('Number of Patients')
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend(title='Dataset')
    
    plt.tight_layout()
    return fig


def create_correlation_heatmap(correlation_df):
    """Create heatmap summarizing all correlations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Correlation Summary: Age and CHADS-VASc vs Events', fontsize=16, fontweight='bold')
    
    # Age correlations heatmap
    age_pivot = correlation_df.pivot(index='event_definition', columns='dataset', values='age_correlation')
    sns.heatmap(age_pivot, annot=True, fmt='.3f', cmap='RdBu_r', center=0, ax=ax1,
                cbar_kws={'label': 'Age Correlation (r)'})
    ax1.set_title('Age vs Event Correlations')
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Event Definition')
    
    # CHADS-VASc correlations heatmap
    chads_pivot = correlation_df.pivot(index='event_definition', columns='dataset', values='chadsvasc_correlation')
    sns.heatmap(chads_pivot, annot=True, fmt='.3f', cmap='RdBu_r', center=0, ax=ax2,
                cbar_kws={'label': 'CHADS-VASc Correlation (r)'})
    ax2.set_title('CHADS-VASc vs Event Correlations')
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Event Definition')
    
    plt.tight_layout()
    return fig


def save_correlation_table(correlation_df):
    """Save detailed correlation table."""
    # Format the table nicely
    formatted_df = correlation_df.copy()
    
    # Round numeric columns
    numeric_cols = ['age_correlation', 'age_p_value', 'chadsvasc_correlation', 
                   'chadsvasc_p_value', 'risk_gradient', 'low_rate', 'moderate_rate', 'high_rate']
    
    for col in numeric_cols:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].round(4)
    
    # Add significance markers
    formatted_df['age_sig'] = formatted_df['age_p_value'].apply(
        lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else 'ns' if not pd.isna(x) else 'N/A'
    )
    
    formatted_df['chads_sig'] = formatted_df['chadsvasc_p_value'].apply(
        lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else 'ns' if not pd.isna(x) else 'N/A'
    )
    
    # Reorder columns
    column_order = ['dataset', 'event_definition', 'n_events', 'total_n',
                   'age_correlation', 'age_p_value', 'age_sig',
                   'chadsvasc_correlation', 'chadsvasc_p_value', 'chads_sig',
                   'risk_gradient', 'low_rate', 'moderate_rate', 'high_rate']
    
    formatted_df = formatted_df[column_order]
    
    # Save to CSV
    output_path = Path('results') / 'correlation_analysis_summary.csv'
    output_path.parent.mkdir(exist_ok=True)
    formatted_df.to_csv(output_path, index=False)
    
    print(f"Correlation summary table saved to: {output_path}")
    return formatted_df


def main():
    """Run comprehensive correlation visualization analysis."""
    print("="*80)
    print("COMPREHENSIVE CORRELATION VISUALIZATION ANALYSIS")
    print("="*80)
    
    # Prepare data
    prepared_data = prepare_analysis_data()
    
    # Calculate correlations
    correlation_df = calculate_correlations(prepared_data)
    
    # Create output directory
    output_dir = Path('results') / 'correlation_visualizations'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create visualizations
    print("Creating age vs event swarmplots...")
    fig1 = create_age_swarmplots(prepared_data, correlation_df)
    fig1.savefig(output_dir / 'age_vs_event_swarmplots.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    print("Creating CHADS-VASc vs event swarmplots...")
    fig2 = create_chadsvasc_swarmplots(prepared_data, correlation_df)
    fig2.savefig(output_dir / 'chadsvasc_vs_event_swarmplots.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print("Creating risk group bar plots...")
    fig3 = create_risk_group_barplots(prepared_data, correlation_df)
    fig3.savefig(output_dir / 'risk_group_event_rates.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    print("Creating treatment visualizations...")
    fig4 = create_treatment_visualization(prepared_data)
    fig4.savefig(output_dir / 'treatment_differences.png', dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    print("Creating correlation heatmap...")
    fig5 = create_correlation_heatmap(correlation_df)
    fig5.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(fig5)
    
    # Save correlation table
    print("Saving correlation summary table...")
    formatted_df = save_correlation_table(correlation_df)
    
    # Print summary
    print(f"\n" + "="*60)
    print("VISUALIZATION SUMMARY")
    print("="*60)
    print(f"Created 5 comprehensive visualizations in: {output_dir}")
    print(f"1. age_vs_event_swarmplots.png - Age distributions by event status")
    print(f"2. chadsvasc_vs_event_swarmplots.png - CHADS-VASc distributions by event status")
    print(f"3. risk_group_event_rates.png - Event rates by risk group")
    print(f"4. treatment_differences.png - Treatment regime comparisons")
    print(f"5. correlation_heatmap.png - Summary correlation matrix")
    print(f"6. correlation_analysis_summary.csv - Detailed correlation table")
    
    # Key findings summary
    print(f"\nKEY FINDINGS:")
    old_working = correlation_df[
        (correlation_df['dataset'] == 'OLD') & 
        (correlation_df['age_correlation'] > 0.02) &
        (correlation_df['age_p_value'] < 0.05)
    ]
    new_working = correlation_df[
        (correlation_df['dataset'] == 'NEW') & 
        (correlation_df['age_correlation'] > 0.02) &
        (correlation_df['age_p_value'] < 0.05)
    ]
    
    print(f"OLD dataset: {len(old_working)}/{len(correlation_df[correlation_df['dataset']=='OLD'])} event definitions show significant age correlation")
    print(f"NEW dataset: {len(new_working)}/{len(correlation_df[correlation_df['dataset']=='NEW'])} event definitions show significant age correlation")
    
    if len(new_working) == 0:
        print("🚨 CONFIRMED: New dataset shows NO significant age correlations across all event definitions")
    
    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()