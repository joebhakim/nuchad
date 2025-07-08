# CHADS-VASc Density Ratio Reweighting Results

## Analysis Metadata

```json
{
  "data_file_name": "random_nuchad_250623.csv",
  "data_file_creation_date": "2025 Jul 08 12:11",
  "analysis_run_date": "2025 Jul 08 12:52",
  "num_patients": 4492,
  "analysis_type": "density_ratio_reweighting_subgroup",
  "subgroup": "DOAC",
  "subgroup_column": "Anticoag3m_type",
  "effective_sample_size": 1247.5390802799975,
  "results_directory": "/home/joe/skunk_here/nuchad/results/subgroup_analysis/DOAC",
  "original_auc": 0.5587500540143462,
  "weighted_auc": 0.4828294669723934,
  "dataset_version": "250623"
}
```

## Weight Statistics

- Number of patients: 4492
- Weight range: 0.01 - 18.59
- Weight mean: 1.00
- Effective sample size: 1247.5 (27.8% of original)

## Performance Metrics

- Original AUC: 0.559
- Weighted AUC: 0.483

## Stroke Rates by CHADS-VASc Score (per 100 person-years)

| CHADS-VASc | Lip et al. Rate | Observed Rate | Weighted Rate |
|------------|-----------------|---------------|---------------|
| 0 | 0.2 | 0.62 | 0.60 |
| 1 | 0.6 | 0.29 | 0.20 |
| 2 | 2.2 | 0.24 | 0.17 |
| 3 | 3.2 | 0.38 | 0.26 |
| 4 | 4.8 | 0.38 | 0.41 |
| 5 | 7.2 | 0.65 | 0.38 |
| 6 | 9.7 | 0.61 | 1.15 |
| 7 | 11.2 | 0.00 | 0.00 |

## Figures

![Stroke Rates](density_ratio_weighted_rates.png)
