# Filtering Configuration Schema

## JSON Configuration Format

Each filtering configuration is a JSON file with the following structure:

```json
{
  "name": "Configuration Name",
  "description": "Detailed description of the filtering logic",
  "dataset_compatibility": ["original", "new", "both"],
  "filters": {
    "require_af": boolean,
    "require_follow_up": boolean, 
    "require_stroke": boolean,
    "af_before_time1": boolean,
    "min_follow_up_days": integer,
    "stroke_window_days": integer,
    "stroke_outcome_filter": {
      "enabled": boolean,
      "include_values": [list of stroke_1Y values to include],
      "exclude_pre_af_strokes": boolean
    }
  },
  "metadata": {
    "created_date": "YYYY-MM-DD",
    "author": "Creator name",
    "use_case": "Primary intended use case",
    "notes": "Additional notes"
  }
}
```

## Field Descriptions

### Basic Filters
- `require_af`: Whether to require AF diagnosis
- `require_follow_up`: Whether to require minimum follow-up period
- `require_stroke`: Whether to require stroke within window
- `af_before_time1`: AF must be diagnosed before/at time1
- `min_follow_up_days`: Minimum follow-up period in days
- `stroke_window_days`: Window after time1 for stroke requirement

### Stroke Outcome Filter (New)
- `enabled`: Whether to apply stroke outcome filtering
- `include_values`: Which stroke_1Y values to include (e.g., [1, 2, 3])
- `exclude_pre_af_strokes`: Whether to exclude stroke_1Y=4 (pre-AF strokes)

### Metadata
- `name`: Human-readable configuration name
- `description`: Detailed description of filtering logic
- `dataset_compatibility`: Which datasets this config works with
- `created_date`: When the configuration was created
- `author`: Who created the configuration
- `use_case`: Primary intended use case
- `notes`: Additional notes or warnings

## Naming Convention

Configuration files should be named descriptively:
- `AF_FU365_stroke1Y_original.json` - AF + 1yr followup + stroke within 1Y for original dataset
- `AF_FU90_nostroke_both.json` - AF + 90 day followup + no stroke requirement for both datasets  
- `strokeenriched_new.json` - Stroke-enriched cohort for new dataset
- `population_based_original.json` - Population-based sample for original dataset