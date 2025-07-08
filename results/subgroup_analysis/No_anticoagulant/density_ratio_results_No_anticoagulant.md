# CHADS-VASc Density Ratio Reweighting Results

## Analysis Metadata

```json
{
  "data_file_name": "random_nuchad_250623.csv",
  "data_file_creation_date": "2025 Jul 08 12:11",
  "analysis_run_date": "2025 Jul 08 12:52",
  "num_patients": 64376,
  "analysis_type": "density_ratio_reweighting_subgroup",
  "subgroup": "No anticoagulant",
  "subgroup_column": "Anticoag3m_type",
  "effective_sample_size": 16388.65780906954,
  "results_directory": "/home/joe/skunk_here/nuchad/results/subgroup_analysis/No_anticoagulant",
  "original_auc": 0.49414763414441,
  "weighted_auc": 0.5064485007323903,
  "dataset_version": "250623"
}
```

## Weight Statistics

- Number of patients: 64376
- Weight range: 0.01 - 38.45
- Weight mean: 1.00
- Effective sample size: 16388.7 (25.5% of original)

## Performance Metrics

- Original AUC: 0.494
- Weighted AUC: 0.506

## Stroke Rates by CHADS-VASc Score (per 100 person-years)

| CHADS-VASc | Lip et al. Rate | Observed Rate | Weighted Rate |
|------------|-----------------|---------------|---------------|
| 0 | 0.2 | 0.41 | 0.40 |
| 1 | 0.6 | 0.48 | 0.48 |
| 2 | 2.2 | 0.48 | 0.45 |
| 3 | 3.2 | 0.45 | 0.49 |
| 4 | 4.8 | 0.44 | 0.48 |
| 5 | 7.2 | 0.45 | 0.42 |
| 6 | 9.7 | 0.54 | 0.44 |
| 7 | 11.2 | 1.91 | 2.42 |

## Figures

![Stroke Rates](density_ratio_weighted_rates.png)
