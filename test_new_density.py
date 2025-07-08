#!/usr/bin/env python3
"""Test density reweighting analysis on the new dataset."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from nuchad.analysis import eda, density_ratio_reweighting
from nuchad.data_processing import eligibility_filters as data_utils
from nuchad.utils import get_project_root

def test_new_density_analysis():
    """Test density reweighting analysis on the new dataset."""
    
    print("Testing density reweighting analysis on new dataset (random_nuchad_250623.csv)...")
    
    # Create results directory for new dataset
    results_dir = get_project_root() / 'results_250623'
    results_dir.mkdir(exist_ok=True)
    print(f"Results will be saved to: {results_dir}")
    
    try:
        # Load new dataset
        print("\n1. Loading new dataset...")
        df = eda.get_df("random_nuchad_250623.csv")
        print(f"   Loaded dataset shape: {df.shape}")
        print(f"   Has time1: {'time1' in df.columns}")
        print(f"   Has time2: {'time2' in df.columns}")
        print(f"   Has Anticoag3m_type: {'Anticoag3m_type' in df.columns}")
        
        # Apply eligibility filtering
        print("\n2. Applying eligibility filtering...")
        eligible_df, filter_stats = data_utils.filter_eligible_patients(df)
        print(f"   Eligible patients: {len(eligible_df)} out of {filter_stats['total']}")
        
        if len(eligible_df) == 0:
            print("   ERROR: No eligible patients found!")
            print("   Filter stats:", filter_stats)
            return False
        
        # Check data quality
        print("\n3. Checking data quality...")
        print(f"   Follow-up years available: {'Follow_Up_Years' in eligible_df.columns}")
        if 'Follow_Up_Years' not in eligible_df.columns:
            eligible_df['Follow_Up_Years'] = (eligible_df['end_fu'] - eligible_df['time1']).dt.days / 365.25
            print(f"   Created Follow_Up_Years column")
        
        print(f"   Follow-up stats: min={eligible_df['Follow_Up_Years'].min():.2f}, max={eligible_df['Follow_Up_Years'].max():.2f}, mean={eligible_df['Follow_Up_Years'].mean():.2f}")
        
        # Calculate CHADS-VASc if needed
        if 'CHADS-Vasc' not in eligible_df.columns:
            eligible_df['CHADS-Vasc'] = eligible_df.apply(density_ratio_reweighting.calculate_chadsvasc, axis=1)
            print(f"   Calculated CHADS-VASc scores")
        
        print(f"   CHADS-VASc distribution: {eligible_df['CHADS-Vasc'].value_counts().sort_index()}")
        
        # Show anticoagulant distribution
        if 'Anticoag3m_type' in eligible_df.columns:
            print(f"   Anticoagulant distribution: {eligible_df['Anticoag3m_type'].value_counts()}")
        
        # Run density ratio reweighting analysis
        print("\n4. Running density ratio reweighting analysis...")
        
        # We need to update the perform_reweighting_analysis function to save to custom results dir
        # For now, let's run the components manually
        
        # Prepare data for weighting
        df_prep = density_ratio_reweighting.prepare_data_for_weighting(eligible_df)
        
        # Make sure CHADS-Vasc score is preserved
        if 'CHADS-Vasc' not in df_prep.columns and 'CHADS-Vasc' in eligible_df.columns:
            df_prep['CHADS-Vasc'] = eligible_df['CHADS-Vasc']
        
        # Compute weights
        df_weighted = density_ratio_reweighting.density_ratio_weighting(df_prep)
        
        # Make sure we have the Follow_Up_Years column
        if 'Follow_Up_Years' not in df_weighted.columns and 'Follow_Up_Years' in eligible_df.columns:
            df_weighted['Follow_Up_Years'] = eligible_df['Follow_Up_Years']
        
        # Make sure CHADS-Vasc score is preserved
        if 'CHADS-Vasc' not in df_weighted.columns and 'CHADS-Vasc' in eligible_df.columns:
            df_weighted['CHADS-Vasc'] = eligible_df['CHADS-Vasc']
        
        print(f"   Weighted dataset shape: {df_weighted.shape}")
        print(f"   Weight stats: min={df_weighted['weight'].min():.3f}, max={df_weighted['weight'].max():.3f}, mean={df_weighted['weight'].mean():.3f}")
        
        # Evaluate CHADS-VASc performance
        print("\n5. Evaluating CHADS-VASc performance...")
        results = density_ratio_reweighting.evaluate_chadsvasc(df_weighted)
        print(f"   Original AUC: {results['original_auc']:.3f}")
        print(f"   Weighted AUC: {results['weighted_auc']:.3f}")
        
        # Create metadata
        from datetime import datetime
        import os
        
        data_file_name = "random_nuchad_250623.csv"
        data_file_creation_date = "unknown"
        
        # Get file creation date
        data_path = Path("data") / data_file_name
        if data_path.exists():
            stat = os.stat(data_path)
            data_file_creation_date = datetime.fromtimestamp(stat.st_mtime).strftime("%Y %b %d %H:%M")
        
        metadata = {
            "data_file_name": data_file_name,
            "data_file_creation_date": data_file_creation_date,
            "analysis_run_date": datetime.now().strftime("%Y %b %d %H:%M"),
            "num_patients": len(df_weighted),
            "analysis_type": "density_ratio_reweighting",
            "effective_sample_size": float((df_weighted['weight'].sum() ** 2) / (df_weighted['weight'] ** 2).sum()),
            "results_directory": str(results_dir),
            "original_auc": results['original_auc'],
            "weighted_auc": results['weighted_auc'],
            "dataset_version": "250623"
        }
        
        # Save results
        print("\n6. Saving results...")
        
        # Save plots with metadata
        save_path = results_dir / 'density_ratio_weighted_rates.png'
        density_ratio_reweighting.plot_results(results, save_path, metadata)
        print(f"   Saved weighted rates plot to: {save_path}")
        
        weight_plot_path = results_dir / 'density_ratio_weight_distribution.png'
        density_ratio_reweighting.plot_weight_distribution(df_weighted, weight_plot_path, metadata)
        print(f"   Saved weight distribution plot to: {weight_plot_path}")
        
        # Save markdown results
        results_markdown_path = results_dir / 'density_ratio_results.md'
        density_ratio_reweighting.save_results_to_markdown(results, df_weighted, results_markdown_path, metadata)
        print(f"   Saved results to: {results_markdown_path}")
        
        print("\n✅ Analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_new_density_analysis()
    sys.exit(0 if success else 1)