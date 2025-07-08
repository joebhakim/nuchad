#!/usr/bin/env python3
"""
Run All Key Analyses for CHADS-VAsC Transportability Study

This script calls the existing analysis functions to generate the three key figures:
1. observed_vs_original_stroke_rates.png (from eda.py)
2. density_ratio_weighted_rates.png (from density_ratio_reweighting.py) 
3. model_comparison_roc.png (from model_comparison.py)

Usage:
    python run_all_analyses.py

This maintains modularity by calling existing functions rather than reimplementing logic.
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from nuchad.analysis import density_ratio_reweighting, eda_old, model_comparison
from nuchad.data_processing.eligibility_filters import filter_eligible_patients
from nuchad.utils import get_results_dir


def main():
    """Run all three key analyses by calling existing functions."""
    print("="*60)
    print("  CHADS-VAsC Transportability Analysis")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    results_dir = get_results_dir()
    
    # Load and filter data once
    print("\n1. Loading and filtering data...")
    df = eda_old.get_df()
    eligible_df, filter_stats = filter_eligible_patients(df)
    print(f"   Eligible patients: {len(eligible_df):,} out of {filter_stats['total']:,}")
    
    try:
        # Analysis 1: CHADS-VAsC Validation
        print("\n2. Running CHADS-VAsC validation...")
        results_df = eda_old.validate_chadsvasc(eligible_df, "time1", "end_fu", "stroke_1Y")
        results_df.to_markdown(results_dir / "results_df.md", numalign="left", stralign="left")
        eda_old.plot_already_results()
        print("   ✓ Generated: observed_vs_original_stroke_rates.png")
        
        # Analysis 2: Density Ratio Reweighting  
        print("\n3. Running density ratio reweighting...")
        df_weighted = density_ratio_reweighting.perform_reweighting_analysis(eligible_df)
        print("   ✓ Generated: density_ratio_weighted_rates.png")
        print("   ✓ Generated: density_ratio_weight_distribution.png")
        
        # Analysis 3: Model Comparison
        print("\n4. Running model comparison...")
        comparison_results = model_comparison.perform_model_comparison(eligible_df)
        auc_diff = comparison_results['auc_model'] - comparison_results['auc_score']
        print("   ✓ Generated: model_comparison_roc.png")
        print(f"   📊 ΔAUC = {auc_diff:+.3f} (UK model vs CHADS-VAsC)")
        
        # Summary
        elapsed_time = time.time() - start_time
        print("\n" + "="*60)
        print("✅ All analyses completed successfully!")
        print(f"⏱ Runtime: {elapsed_time:.1f} seconds")
        print(f"📁 Results saved to: {results_dir}")
        print("\n📊 Key figures generated:")
        print("   1. observed_vs_original_stroke_rates.png")
        print("   2. density_ratio_weighted_rates.png") 
        print("   3. model_comparison_roc.png")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)