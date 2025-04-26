# load dependencies
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

# import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy import stats

from scipy.stats import poisson

# Create directories if they don't exist
os.makedirs('../data', exist_ok=True)
os.makedirs('../results', exist_ok=True)

# load data
# df = pd.read_csv("./data/changauto4225.csv")


def get_df():
    # Updated path to use data directory
    df = pd.read_csv("../data/random_nuchad.csv").rename(columns={"patid": "patient_id"}).set_index("patient_id")
    df = df.drop(columns=["Unnamed: 0"])

    # load data
    # df = pd.read_csv("./data/changauto4225.csv")

    # --- Convert time1 and time2 to datetime objects ---

    # convert time1 and time2 to datetime objects

    # Change to format; 01-Jan-2020
    df["time1"] = pd.to_datetime(df["time1"], format="%Y-%m-%d", errors="raise")
    df["time2"] = pd.to_datetime(df["time2"], format="%Y-%m-%d", errors="raise")

    df["earliest_af_date"] = pd.to_datetime(df["earliest_af_date"], format="%d%b%Y", errors="raise")
    df["earliest_stroke_date"] = pd.to_datetime(df["earliest_stroke_date"], format="%d%b%Y", errors="raise")
    df["end_fu"] = pd.to_datetime(df["end_fu"], format="%d%b%Y", errors="raise")


    return df


## Validating chadsvasc


def calculate_chadsvasc(row):
    """Calculates the CHADS-VASc score for a single patient (row of a DataFrame)."""
    score = 0
    # Congestive heart failure
    score += int(row["hf"])
    # Hypertension
    score += int(row["hypertension"])
    # Age >= 75
    score += 2 * int(row["age"] >= 75)
    score += int(65 <= row["age"] < 75)
    # Diabetes mellitus
    score += int(row["diab"])
    # Stroke/TIA/Thromboembolism
    score += 2 * int(row["thrombo"] or row["HB_stroke_history"])
    # Vascular disease
    score += int(row["vasc_dis_mi_pad"])
    # Sex (Female)
    score += int(row["gender"] != 1)
    return score


def calculate_stroke_rate(group, total_patients, total_years, event_col="stroke_1Y"):
    """Calculates the adjusted stroke rate per 100 patient-years."""
    if total_patients == 0 or pd.isna(total_years):  # Handle potential NaNs
        return 0
    
    # Stroke_1Y is 1=Yes, 2=No

    num_strokes = group[group[event_col]==1].shape[0]

    return (num_strokes / total_years) * 100


def confidence_interval(rate, total_years):
    """Calculates the 95% confidence interval for the stroke rate."""
    if rate == 0 or total_years == 0 or pd.isna(total_years):  # Handle potential NaNs
        return (0, 0)

    z = 1.96  # For 95% CI
    lambda_val = (rate / 100) * total_years
    lower_bound = poisson.ppf(0.025, lambda_val) / total_years * 100
    upper_bound = poisson.ppf(0.975, lambda_val) / total_years * 100
    return (lower_bound, upper_bound)


def validate_chadsvasc(df, time1_col, time2_col, stroke_event_col="stroke_1Y"):
    """
    Validates the CHADS-VASc score, handling datetime conversion.
    """
    df["CHADS-Vasc"] = df.apply(calculate_chadsvasc, axis=1)

    # For each chadsvasc level, calculate the rate of


    # Calculate follow-up time in years, time2_col should be end_fu not time2
    df["Follow_Up_Years"] = (
        (df[time2_col] - df[time1_col]) / np.timedelta64(1, "D") / 365.25
    )
    

    df['anticoag_binary'] = df['Anticoagulant'].apply(lambda x: 0 if x == 'No anticoagulant' else 1)
    # Calculate the rate of Anticoagulant use for each chadsvasc level
    anticoagulant_rates = df.groupby("CHADS-Vasc").agg(
        {
            "anticoag_binary": "mean",
        }
    )

    grouped = df.groupby("CHADS-Vasc")

    print(grouped.head().to_markdown(index=False, numalign="left", stralign="left"))

    # Calculate the rate of Anticoagulant use for each chadsvasc level

    results = []
    for score, group in grouped:
        total_patients = len(group)
        total_years = group["Follow_Up_Years"].sum()
        observed_rate = calculate_stroke_rate(
            group, total_patients, total_years, event_col=stroke_event_col
        )
        ci = confidence_interval(observed_rate, total_years)
        results.append(
            {
                "CHADS-Vasc": score,
                "Number of Patients": total_patients,
                "Total Patient Years": total_years,
                "Observed Stroke Rate": observed_rate,
                "95% CI Lower": ci[0],
                "95% CI Upper": ci[1],
            }
        )

    results_df = pd.DataFrame(results)

    # TODO: RE: original rates
    # FIND THE CITATION WHERE THESE COME FROM!? THANKS!
    # CANGO: please find this AND units
    original_rates = {
        0: 0.0,
        1: 1.3,
        2: 2.2,
        3: 3.2,
        4: 4.0,
        5: 6.7,
        6: 9.8,
        7: 9.6,
        8: 6.7,
        9: 15.2,
    }

    # TEmporarily, https://www.mdcalc.com/calc/801/cha2ds2-vasc-score-atrial-fibrillation-stroke-risk#evidence

    original_rates = {
        0: 0.2,
        1: 0.6,
        2: 2.2,
        3: 3.2,
        4: 4.8,
        5: 7.2,
        6: 9.7,
        7: 11.2,
        8: 10.8,
        9: 12.2,
    }

    # TODO: RE: original rates
    # CANGO: please find this AND units
    original_rates_ci_lower = {score: np.nan for score in original_rates}
    original_rates_ci_upper = {score: np.nan for score in original_rates}

    results_df["Original Stroke Rate"] = results_df["CHADS-Vasc"].map(original_rates)
    #results_df["Original CI Lower"] = results_df["CHADS-Vasc"].map(
    #    original_rates_ci_lower
    #)
    #results_df["Original CI Upper"] = results_df["CHADS-Vasc"].map(
    #    original_rates_ci_upper
    #)

    results_df["Original Rate Within CI"] = (
        results_df["Original Stroke Rate"] >= results_df["95% CI Lower"]
    ) & (results_df["Original Stroke Rate"] <= results_df["95% CI Upper"])

    results_df = results_df[
        [
            "CHADS-Vasc",
            "Number of Patients",
            "Total Patient Years",
            "Observed Stroke Rate",
            "95% CI Lower",
            "95% CI Upper",
            "Original Stroke Rate",
            # "Original CI Lower",
            # "Original CI Upper",
            "Original Rate Within CI",
        ]
    ]
    return results_df


def filter_eligible_patients(df):
    """
    Filters patients who are eligible for the study based on the following criteria:
    - AF diagnosis must have been no more than 1 year before the start of the observation window
    - Stroke diagnosis must have been no more than 1 year before the start of the observation window
    - Patient must have a follow-up period of at least 1 year
    """
    # Filter patients who have an AF diagnosis before time1, doesnt matter how LONG before time1
    af_diagnosis_mask = (df["earliest_af_date"] <= df["time1"])

    # Filter patients who have a stroke diagnosis between time1 (inclusive) and one year after time1
    stroke_diagnosis_mask = (df["earliest_stroke_date"] >= df["time1"]) & (
        df["earliest_stroke_date"] <= df["time1"] + pd.Timedelta(days=365)
    )

    # Filter patients who have a follow-up period of at least 1 year, observed with time2
    # TODO: find out why time2 is not the same as end_fu, TODO RE: ITS IN THE DOC
    # TODO: these should be treated as censored, not omitted
    follow_up_mask = df["end_fu"] >= df["time1"] + pd.Timedelta(days=365)

    # Do diagostics on these masks: how many did we start with, how many do each prune? How many are pruned in this order:
    # All patients
    # All patients with AF
    # All patients with stroke
    # All patients with follow-up
    # All patients with AF and stroke
    # All patients with AF and stroke and follow-up

    print(f"Total patients: {len(df)}")
    print(f"Patients with AF (earliest_af_date <= time1): {len(df[af_diagnosis_mask])}")
    print(f"Patients with stroke (earliest_stroke_date >= time1 and <= time1 + 365): {len(df[stroke_diagnosis_mask])}")
    print(f"Patients with follow-up (end_fu >= time1 + 365): {len(df[follow_up_mask])}")
    print(f"Patients with AF and stroke: {len(df[af_diagnosis_mask & stroke_diagnosis_mask])}")
    print(f"Patients with AF and stroke and follow-up: {len(df[af_diagnosis_mask & stroke_diagnosis_mask & follow_up_mask])}")

    # Check: percentiles of earliest_af_date - time1
    # earliest_af_date_minus_time1 = df['earliest_af_date'] - df['time1']
    # print(earliest_af_date_minus_time1.describe()) There are NONE, were good

    
    # Eligibility: with AF, with enough followup, NOT NECESSARILY WITH STROKE
    #eligible_mask = af_diagnosis_mask & stroke_diagnosis_mask  & follow_up_mask
    eligible_mask = af_diagnosis_mask & follow_up_mask

    # Apply the mask to the DataFrame
    eligible_patients_df = df[eligible_mask]

    return eligible_patients_df


def more_checking(df):

    df_check = df.copy()

    df_check["earliest_stroke_date"] = pd.to_datetime(
        df_check["earliest_stroke_date"], format="%d-%b-%y", errors="raise"
    )

    earliest_stroke_date_minus_time1 = (
        df_check["earliest_stroke_date"] - df_check["time1"]
    )
    print(earliest_stroke_date_minus_time1.describe())


def plot_already_results():
    observed_rates = {
        0: 0.115974,
        1: 0.180523,
        2: 0.397818,
        3: 0.678945,
        4: 1.12856,
        5: 2.86256,
        6: 7.32759,
        7: 10.1742,
        8: 11.1649,
        9: 14.8506,
    }

    observed_rates_ci_lower = {
        0: 0.0802899,
        1: 0.14995,
        2: 0.358451,
        3: 0.629987,
        4: 1.05917,
        5: 2.68667,
        6: 6.8414,
        7: 8.97498,
        8: 8.37367,
        9: 5.56897,
    }

    observed_rates_ci_upper = {
        0: 0.154632,
        1: 0.212551,
        2: 0.438221,
        3: 0.728827,
        4: 1.1991,
        5: 3.0413,
        6: 7.82247,
        7: 11.4122,
        8: 14.1422,
        9: 25.9885,
    }
    

    original_rates = {
        0: 0.2,
        1: 0.6,
        2: 2.2,
        3: 3.2,
        4: 4.8,
        5: 7.2,
        6: 9.7,
        7: 11.2,
        8: 10.8,
        9: 12.2,
    }

    # Plot observed rates vs original rates
    plt.figure(figsize=(10, 6))
    plt.plot(list(observed_rates.keys()), list(observed_rates.values()), label='Observed Rates', marker='o')
    plt.plot(list(original_rates.keys()), list(original_rates.values()), label='Original Rates', marker='x')
    plt.xlabel('CHADS-VASc Score')
    plt.ylabel('Stroke Rate')
    plt.title('Observed vs Original Stroke Rates')

    # Add confidence intervals
    for i in range(len(observed_rates)):
        plt.fill_between(
            [i, i],
            observed_rates_ci_lower[i],
            observed_rates_ci_upper[i],
            color='gray',
            alpha=0.8
        )


    plt.legend()
    plt.tight_layout()
    # Save to results directory
    plt.savefig('../results/observed_vs_original_stroke_rates.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    df = get_df()

    #more_checking(df)

    eligible_patients_df = filter_eligible_patients(df)

    results_df = validate_chadsvasc(eligible_patients_df, "time1", "end_fu", "stroke_1Y")

    # save results_df to markdown in results directory
    results_df.to_markdown("../results/results_df.md", numalign="left", stralign="left")

    print(results_df.head().to_markdown(index=False, numalign="left", stralign="left"))
    
    # Generate and save the plot
    plot_already_results() 