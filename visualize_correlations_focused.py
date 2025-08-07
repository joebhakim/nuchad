#!/usr/bin/env python3
"""
Focused visualization of event correlation analysis with efficient handling of large datasets.
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


def calculate_key_correlations():
    """Calculate correlations efficiently without storing large datasets."""
    print("Loading datasets and calculating correlations...")
    
    # Load datasets
    old_df = get_df("random_nuchad.csv") 
    new_df = get_df("random_nuchad_250623.csv")
    
    results = []
    
    for dataset_name, df in [('OLD', old_df), ('NEW', new_df)]:
        # Calculate CHADS-VASc
        df['chadsvasc'] = df.apply(calculate_chadsvasc, axis=1)
        df['risk_group'] = pd.cut(
            df['chadsvasc'], 
            bins=[-0.5, 1.5, 3.5, 10], 
            labels=['Low (0-1)', 'Moderate (2-3)', 'High (4+)']
        )
        
        # Define events based on dataset
        if dataset_name == 'OLD':
            events = {
                'stroke_1Y=1': df['stroke_1Y'] == 1,
                'has_stroke_date': df['earliest_stroke_date'].notna(),
                'stroke_1Y=1_AND_date': (df['stroke_1Y'] == 1) & df['earliest_stroke_date'].notna()
            }
        else:  # NEW
            events = {
                'stroke_1Y=1': df['stroke_1Y'] == 1.0,
                'has_stroke_date': df['earliest_stroke_date'].notna(),
                'stroke_1Y=1_AND_date': (df['stroke_1Y'] == 1.0) & df['earliest_stroke_date'].notna(),
                'any_stroke_1Y': df['stroke_1Y'].notna()
            }
        
        for event_name, event_mask in events.items():
            # Age correlation
            age_valid = df[df['age'].notna()]
            event_valid = event_mask[age_valid.index]
            if len(age_valid) > 0 and event_valid.sum() > 0 and event_valid.sum() < len(event_valid):
                try:
                    age_corr, age_p = pearsonr(event_valid.astype(int), age_valid['age'])
                except:
                    age_corr, age_p = np.nan, np.nan
            else:
                age_corr, age_p = np.nan, np.nan
            
            # CHADS-VASc correlation
            chads_valid = df[df['chadsvasc'].notna()]
            event_valid_chads = event_mask[chads_valid.index]
            if len(chads_valid) > 0 and event_valid_chads.sum() > 0 and event_valid_chads.sum() < len(event_valid_chads):
                try:
                    chads_corr, chads_p = pearsonr(event_valid_chads.astype(int), chads_valid['chadsvasc'])
                except:
                    chads_corr, chads_p = np.nan, np.nan
            else:
                chads_corr, chads_p = np.nan, np.nan
            
            # Risk group rates
            risk_rates = []
            for group in ['Low (0-1)', 'Moderate (2-3)', 'High (4+)']:
                group_data = df[df['risk_group'] == group]
                if len(group_data) > 0:
                    rate = event_mask[group_data.index].mean() * 100
                    risk_rates.append(rate)
                else:
                    risk_rates.append(0)
            
            gradient = max(risk_rates) - min(risk_rates)
            
            results.append({
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
                'n_events': int(event_mask.sum()),
                'total_n': len(df)
            })
    
    return pd.DataFrame(results)


def create_correlation_heatmaps(correlation_df):
    """Create correlation heatmaps."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Event Correlation Analysis Summary', fontsize=16, fontweight='bold')
    
    # Age correlations
    age_pivot = correlation_df.pivot(index='event_definition', columns='dataset', values='age_correlation')
    im1 = axes[0,0].imshow(age_pivot.values, cmap='RdBu_r', aspect='auto', vmin=-0.1, vmax=0.1)
    axes[0,0].set_title('Age vs Event Correlations')
    axes[0,0].set_xticks(range(len(age_pivot.columns)))
    axes[0,0].set_xticklabels(age_pivot.columns)
    axes[0,0].set_yticks(range(len(age_pivot.index)))
    axes[0,0].set_yticklabels(age_pivot.index, rotation=0)
    
    # Add correlation values as text
    for i in range(len(age_pivot.index)):
        for j in range(len(age_pivot.columns)):
            val = age_pivot.values[i,j]
            if not np.isnan(val):
                axes[0,0].text(j, i, f'{val:.3f}', ha='center', va='center',
                              color='white' if abs(val) > 0.05 else 'black')
    
    plt.colorbar(im1, ax=axes[0,0], label='Correlation (r)')
    
    # CHADS-VASc correlations
    chads_pivot = correlation_df.pivot(index='event_definition', columns='dataset', values='chadsvasc_correlation')
    im2 = axes[0,1].imshow(chads_pivot.values, cmap='RdBu_r', aspect='auto', vmin=-0.1, vmax=0.1)
    axes[0,1].set_title('CHADS-VASc vs Event Correlations')
    axes[0,1].set_xticks(range(len(chads_pivot.columns)))
    axes[0,1].set_xticklabels(chads_pivot.columns)
    axes[0,1].set_yticks(range(len(chads_pivot.index)))
    axes[0,1].set_yticklabels(chads_pivot.index, rotation=0)
    
    # Add correlation values as text
    for i in range(len(chads_pivot.index)):
        for j in range(len(chads_pivot.columns)):
            val = chads_pivot.values[i,j]
            if not np.isnan(val):
                axes[0,1].text(j, i, f'{val:.3f}', ha='center', va='center',
                              color='white' if abs(val) > 0.05 else 'black')
    
    plt.colorbar(im2, ax=axes[0,1], label='Correlation (r)')
    
    # Risk gradients
    gradient_pivot = correlation_df.pivot(index='event_definition', columns='dataset', values='risk_gradient')
    im3 = axes[1,0].imshow(gradient_pivot.values, cmap='Reds', aspect='auto', vmin=0, vmax=15)
    axes[1,0].set_title('Risk Group Gradients (%)')
    axes[1,0].set_xticks(range(len(gradient_pivot.columns)))
    axes[1,0].set_xticklabels(gradient_pivot.columns)
    axes[1,0].set_yticks(range(len(gradient_pivot.index)))
    axes[1,0].set_yticklabels(gradient_pivot.index, rotation=0)
    
    # Add gradient values as text
    for i in range(len(gradient_pivot.index)):
        for j in range(len(gradient_pivot.columns)):
            val = gradient_pivot.values[i,j]
            if not np.isnan(val):
                axes[1,0].text(j, i, f'{val:.1f}%', ha='center', va='center',
                              color='white' if val > 7 else 'black')
    
    plt.colorbar(im3, ax=axes[1,0], label='Gradient (%)')
    
    # Event counts
    events_pivot = correlation_df.pivot(index='event_definition', columns='dataset', values='n_events')
    im4 = axes[1,1].imshow(events_pivot.values, cmap='Blues', aspect='auto')
    axes[1,1].set_title('Number of Events')
    axes[1,1].set_xticks(range(len(events_pivot.columns)))
    axes[1,1].set_xticklabels(events_pivot.columns)
    axes[1,1].set_yticks(range(len(events_pivot.index)))
    axes[1,1].set_yticklabels(events_pivot.index, rotation=0)
    
    # Add event counts as text
    for i in range(len(events_pivot.index)):
        for j in range(len(events_pivot.columns)):
            val = events_pivot.values[i,j]
            if not np.isnan(val):
                axes[1,1].text(j, i, f'{int(val):,}', ha='center', va='center',
                              color='white' if val > events_pivot.values.max()/2 else 'black')
    
    plt.colorbar(im4, ax=axes[1,1], label='Count')
    
    plt.tight_layout()
    return fig


def create_risk_group_comparison(correlation_df):
    """Create risk group event rate comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Event Rates by CHADS-VASc Risk Group', fontsize=16, fontweight='bold')
    
    # Key event definitions to show
    key_events = ['stroke_1Y=1', 'has_stroke_date']
    
    for idx, event_def in enumerate(key_events):
        event_data = correlation_df[correlation_df['event_definition'] == event_def]
        
        if not event_data.empty:
            # Bar plot for this event definition
            ax = axes[idx, 0]
            
            datasets = event_data['dataset'].values
            low_rates = event_data['low_rate'].values
            mod_rates = event_data['moderate_rate'].values
            high_rates = event_data['high_rate'].values
            
            x = np.arange(len(datasets))
            width = 0.25
            
            ax.bar(x - width, low_rates, width, label='Low (0-1)', color='lightblue', alpha=0.8)
            ax.bar(x, mod_rates, width, label='Moderate (2-3)', color='orange', alpha=0.8)
            ax.bar(x + width, high_rates, width, label='High (4+)', color='lightcoral', alpha=0.8)
            
            ax.set_xlabel('Dataset')
            ax.set_ylabel('Event Rate (%)')
            ax.set_title(f'{event_def.replace("_", " ").title()}')
            ax.set_xticks(x)
            ax.set_xticklabels(datasets)
            ax.legend()
            
            # Add gradient annotation
            for i, dataset in enumerate(datasets):
                gradient = event_data[event_data['dataset'] == dataset]['risk_gradient'].iloc[0]
                ax.text(i, max(high_rates[i], mod_rates[i], low_rates[i]) + 0.5,
                       f'Gradient: {gradient:.1f}%', ha='center', fontsize=9)
            
            # Line plot showing risk progression
            ax2 = axes[idx, 1]
            for i, dataset in enumerate(datasets):
                rates = [low_rates[i], mod_rates[i], high_rates[i]]
                ax2.plot(['Low', 'Moderate', 'High'], rates, 'o-', label=dataset, linewidth=2, markersize=8)
            
            ax2.set_xlabel('CHADS-VASc Risk Group')
            ax2.set_ylabel('Event Rate (%)')
            ax2.set_title(f'{event_def.replace("_", " ").title()} - Risk Progression')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_treatment_comparison():
    """Create treatment comparison visualization."""
    print("Creating treatment comparison...")
    
    # Load datasets for treatment analysis
    old_df = get_df("random_nuchad.csv")
    new_df = get_df("random_nuchad_250623.csv")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Treatment Regime Differences Between Datasets', fontsize=16, fontweight='bold')
    
    # OLD dataset treatment (simple)
    old_has_anticoag = old_df['Anticoagulant'].notna() & (old_df['Anticoagulant'] != 0)
    old_treatment_counts = old_has_anticoag.value_counts()
    old_treatment_labels = ['No OAC', 'OAC'] if False in old_treatment_counts.index else ['OAC']
    
    axes[0,0].pie(old_treatment_counts.values, labels=old_treatment_labels, 
                  autopct='%1.1f%%', startangle=90)
    axes[0,0].set_title('OLD Dataset Treatment Distribution')
    
    # NEW dataset treatment (complex)
    new_has_oac = (
        new_df['first_OAC_date'].notna() &
        (new_df['first_OAC_date'] >= new_df['time1']) &
        (new_df['first_OAC_date'] <= new_df['end_fu'])
    )
    new_has_antiplatelet = (
        new_df['first_antiplatelet_date'].notna() &
        (new_df['first_antiplatelet_date'] >= new_df['time1']) &
        (new_df['first_antiplatelet_date'] <= new_df['end_fu'])
    )
    
    # Create treatment categories for new dataset
    new_treatment = pd.Series('No treatment', index=new_df.index)
    new_treatment[new_has_oac & ~new_has_antiplatelet] = 'OAC only'
    new_treatment[~new_has_oac & new_has_antiplatelet] = 'Antiplatelet only'  
    new_treatment[new_has_oac & new_has_antiplatelet] = 'Both treatments'
    
    new_treatment_counts = new_treatment.value_counts()
    axes[0,1].pie(new_treatment_counts.values, labels=new_treatment_counts.index,
                  autopct='%1.1f%%', startangle=90)
    axes[0,1].set_title('NEW Dataset Treatment Distribution')
    
    # Treatment counts comparison
    axes[1,0].bar(['OLD Dataset'], [len(old_df)], color='lightblue', alpha=0.7, label='Total Patients')
    axes[1,0].bar(['OLD Dataset'], [old_has_anticoag.sum()], color='darkblue', alpha=0.7, label='On OAC')
    axes[1,0].set_ylabel('Number of Patients')
    axes[1,0].set_title('OLD Dataset Treatment Numbers')
    axes[1,0].legend()
    
    # Add text annotations
    axes[1,0].text(0, len(old_df)/2, f'Total: {len(old_df):,}', ha='center', va='center', fontweight='bold')
    if old_has_anticoag.sum() > 0:
        axes[1,0].text(0, old_has_anticoag.sum()/2, f'OAC: {old_has_anticoag.sum():,}', ha='center', va='center', color='white', fontweight='bold')
    
    # NEW dataset treatment numbers
    treatment_names = new_treatment_counts.index.tolist()
    treatment_counts = new_treatment_counts.values.tolist()
    colors = ['lightcoral', 'lightgreen', 'lightskyblue', 'wheat']
    
    bars = axes[1,1].bar(range(len(treatment_names)), treatment_counts, color=colors[:len(treatment_names)])
    axes[1,1].set_xticks(range(len(treatment_names)))
    axes[1,1].set_xticklabels(treatment_names, rotation=45, ha='right')
    axes[1,1].set_ylabel('Number of Patients')
    axes[1,1].set_title('NEW Dataset Treatment Numbers')
    
    # Add count labels on bars
    for bar, count in zip(bars, treatment_counts):
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + bar.get_width()/2., height,
                      f'{count:,}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig


def create_summary_table(correlation_df):
    """Create formatted summary table."""
    # Add significance indicators
    correlation_df['age_sig'] = correlation_df['age_p_value'].apply(
        lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else 'ns' if not pd.isna(x) else 'N/A'
    )
    
    # Create formatted display
    summary_table = correlation_df[[
        'dataset', 'event_definition', 'n_events', 'total_n',
        'age_correlation', 'age_p_value', 'age_sig', 
        'risk_gradient', 'low_rate', 'moderate_rate', 'high_rate'
    ]].copy()
    
    # Round values
    summary_table['age_correlation'] = summary_table['age_correlation'].round(3)
    summary_table['age_p_value'] = summary_table['age_p_value'].round(4)
    summary_table['risk_gradient'] = summary_table['risk_gradient'].round(1)
    
    for col in ['low_rate', 'moderate_rate', 'high_rate']:
        summary_table[col] = summary_table[col].round(1)
    
    return summary_table


def main():
    """Run focused correlation visualization."""
    print("="*80)
    print("FOCUSED CORRELATION VISUALIZATION ANALYSIS")
    print("="*80)
    
    # Calculate correlations
    correlation_df = calculate_key_correlations()
    
    # Create output directory
    output_dir = Path('results') / 'correlation_visualizations'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("Creating correlation heatmaps...")
    fig1 = create_correlation_heatmaps(correlation_df)
    fig1.savefig(output_dir / 'correlation_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    print("Creating risk group comparisons...")
    fig2 = create_risk_group_comparison(correlation_df)
    fig2.savefig(output_dir / 'risk_group_comparisons.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    print("Creating treatment comparisons...")
    fig3 = create_treatment_comparison()
    fig3.savefig(output_dir / 'treatment_comparisons.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # Save summary table
    summary_table = create_summary_table(correlation_df)
    summary_table.to_csv(output_dir / 'correlation_summary.csv', index=False)
    
    print(f"\n" + "="*60)
    print("VISUALIZATION SUMMARY")
    print("="*60)
    print(f"Created visualizations in: {output_dir}")
    print("1. correlation_heatmaps.png - All correlation matrices")
    print("2. risk_group_comparisons.png - Risk stratification comparison")
    print("3. treatment_comparisons.png - Treatment regime differences")
    print("4. correlation_summary.csv - Detailed results table")
    
    # Print key findings
    print(f"\nKEY FINDINGS:")
    old_corrs = correlation_df[correlation_df['dataset'] == 'OLD']['age_correlation']
    new_corrs = correlation_df[correlation_df['dataset'] == 'NEW']['age_correlation']
    
    old_significant = ((correlation_df['dataset'] == 'OLD') & 
                      (correlation_df['age_correlation'] > 0.02) & 
                      (correlation_df['age_p_value'] < 0.05)).sum()
    
    new_significant = ((correlation_df['dataset'] == 'NEW') & 
                      (correlation_df['age_correlation'] > 0.02) & 
                      (correlation_df['age_p_value'] < 0.05)).sum()
    
    print(f"OLD dataset age correlations: {old_corrs.min():.3f} to {old_corrs.max():.3f}")
    print(f"NEW dataset age correlations: {new_corrs.min():.3f} to {new_corrs.max():.3f}")
    print(f"OLD dataset significant correlations: {old_significant}/{len(old_corrs)}")
    print(f"NEW dataset significant correlations: {new_significant}/{len(new_corrs)}")
    
    if new_significant == 0:
        print("\n🚨 CONFIRMED: New dataset shows NO significant correlations")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()