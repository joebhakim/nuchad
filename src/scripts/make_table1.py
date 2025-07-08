#!/usr/bin/env python3
"""Generate Table 1 - Patient characteristics."""

import sys
from nuchad.analysis import eda_old, table1
from nuchad.data_processing import eligibility_filters as data_utils
from nuchad.utils import get_results_dir

def main():
    """Generate Table 1 with patient characteristics."""
    print("Generating Table 1 - Patient characteristics...")
    
    # Load and filter data
    df = eda_old.get_df()
    eligible_df, _ = data_utils.filter_eligible_patients(df)
    
    # Generate table
    table = table1.create_table1(eligible_df)
    results_dir = get_results_dir()
    
    # Save as markdown file
    with open(results_dir / 'table1.md', 'w') as f:
        f.write("# Table 1: Baseline Characteristics\n\n")
        f.write(table.to_markdown(index=False))
    
    print(f"Table 1 has been generated and saved to {results_dir / 'table1.md'}")
    return 0

if __name__ == "__main__":
    print("THIS IS SUPPOSED TO BE RUN AS A SCRIPT, NOT DIRECTLY, USE SOMETHING LIKE UV RUN")
    sys.exit(main())