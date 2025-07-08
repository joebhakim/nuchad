# CHADS-VASc Density Ratio Reweighting Results

## Analysis Metadata

```json
{
  "data_file_name": "random_nuchad_250623.csv",
  "data_file_creation_date": "2025 Jul 08 12:11",
  "analysis_run_date": "2025 Jul 08 12:52",
  "num_patients": 23247,
  "analysis_type": "density_ratio_reweighting_subgroup",
  "subgroup": "VKA",
  "subgroup_column": "Anticoag3m_type",
  "effective_sample_size": 6469.124371435456,
  "results_directory": "/home/joe/skunk_here/nuchad/results/subgroup_analysis/VKA",
  "original_auc": 0.5076263099918952,
  "weighted_auc": 0.5360087789593836,
  "dataset_version": "250623"
}
```

## Weight Statistics

- Number of patients: 23247
- Weight range: 0.01 - 35.74
- Weight mean: 1.00
- Effective sample size: 6469.1 (27.8% of original)

## Performance Metrics

- Original AUC: 0.508
- Weighted AUC: 0.536

## Stroke Rates by CHADS-VASc Score (per 100 person-years)

| CHADS-VASc | Lip et al. Rate | Observed Rate | Weighted Rate |
|------------|-----------------|---------------|---------------|
| 0 | 0.2 | 0.38 | 0.37 |
| 1 | 0.6 | 0.35 | 0.33 |
| 2 | 2.2 | 0.45 | 0.45 |
| 3 | 3.2 | 0.51 | 0.46 |
| 4 | 4.8 | 0.43 | 0.46 |
| 5 | 7.2 | 0.48 | 0.59 |
| 6 | 9.7 | 0.35 | 0.17 |
| 7 | 11.2 | 0.00 | 0.00 |

## Figures

![Stroke Rates](density_ratio_weighted_rates.png)
