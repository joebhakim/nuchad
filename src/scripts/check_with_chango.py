#!/usr/bin/env python3
"""Demonstrate lack of separation for 1-year stroke outcome.

This minimal script computes the outcome "stroke from AF diagnosis (time1)
to 1-year follow-up (yes/no)" using dates, then reports:
- Overall event rate
- Event rates by CHADS-VASc score
- Correlation with age and with CHADS-VASc

By default it uses the newer dataset (random_nuchad_250623.csv) and applies
the standard eligibility filters. Results are printed and saved to results/.
"""

import sys
import argparse
import numpy as np
import pandas as pd

from nuchad.utils import get_df, calculate_chadsvasc, get_results_dir
from nuchad.data_processing import eligibility_filters as data_utils


def main() -> int:
    parser = argparse.ArgumentParser(description='Demonstrate no separation for 1Y stroke outcome')
    parser.add_argument('--dataset', choices=['new', 'old'], default='new',
                        help='Dataset to use (default: new)')
    parser.add_argument('--no-filter', action='store_true',
                        help='Skip eligibility filtering')
    args = parser.parse_args()

    data_file = 'random_nuchad_250623.csv' if args.dataset == 'new' else 'random_nuchad.csv'

    print(f"Loading dataset: {data_file}")
    df = get_df(data_file)

    if not args.no_filter:
        print("Applying default eligibility filters...")
        df, _ = data_utils.filter_eligible_patients(df)
        print(f"Filtered to {len(df):,} patients")
    else:
        print(f"Using all {len(df):,} patients (no filtering)")

    # For new data
    # Anticoag columns: Anticoag3m_type,first_OAC_date,Antiplatelet3m,first_antiplatelet_date
    # get rates of anticoag in each column

    if args.dataset == 'new':
        print('Total rows: ', len(df))
        print('Anticoag3m_type: ', df['Anticoag3m_type'].value_counts())
        print('first_OAC_date: ', df['first_OAC_date'].value_counts())
        print('Antiplatelet3m: ', df['Antiplatelet3m'].value_counts())
        print('first_antiplatelet_date: ', df['first_antiplatelet_date'].value_counts())

        # Filter df by No anticoagulant
        df = df[df['Anticoag3m_type'] == 'No anticoagulant']
        if 'CHADS-Vasc' not in df.columns:
            df['CHADS-Vasc'] = df.apply(calculate_chadsvasc, axis=1)

        for chads_vasc in df['CHADS-Vasc'].unique():
            print(f'CHADS-Vasc: {chads_vasc}')
            stroke_rate = df[df['CHADS-Vasc'] == chads_vasc][df['stroke_1Y']==1].shape[0]/df[df['CHADS-Vasc'] == chads_vasc].shape[0]
            print(f'Event rate: {stroke_rate}')
        

        # Get event rate, using stroke_1Y, by each sratum of chads_vasc
        
        

    else:
        print('Total rows: ', len(df))
        print('Anticoagulant: ', df['Anticoagulant'].value_counts())


    

    #print(df.head())

    # Outcome by dates
    return 0


if __name__ == "__main__":
    sys.exit(main())
