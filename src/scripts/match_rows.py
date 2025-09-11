#!/usr/bin/env python3
"""Row-level matching across old and new datasets to probe scrambling.

This utility attempts to identify likely-matching patient rows between
the original dataset (random_nuchad.csv) and the newer dataset
(random_nuchad_250623.csv) using robust date-based fingerprints, then
compares covariates to highlight potential scrambling (predictors not
aligning while outcomes/dates do).

Run:
    python -m scripts.match_rows [--limit 50]

Outputs:
    results/row_match_report.md  (summary + top discrepant matches)
    results/row_match_pairs.csv  (matched pairs with key fields)
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

def get_results_dir() -> Path:
    """Resolve and ensure the results/ directory at project root.

    This avoids importing the full nuchad package (and heavy deps) to keep
    this utility lightweight.
    """
    p = Path.cwd() / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p


# Columns present in both datasets that we can compare
BIN_COLS = [
    "af",
    "hypertension",
    "diab",
    "thrombo",
    "hf",
    "HB_stroke_history",
    "Stroke_TIA_hx",  # new dataset alt
    "ckd",
    "vasc_dis_mi_pad",
    "aortic_plaq",
]

CONT_COLS = [
    "age",
    "frailty_score",
    "bmi",
    "tc_mmol_L",
    "acr_mg_mmol",
]

DATE_COLS = [
    "earliest_af_date",
    "earliest_stroke_date",
    "end_fu",
]


def robust_parse_date(s: pd.Series) -> pd.Series:
    """Parse dates trying multiple formats, falling back to generic parse."""
    if s is None:
        return pd.Series(pd.NaT, index=[])
    # Preserve original for multiple attempts
    src = s.astype(str)
    # Common formats across both datasets
    out = pd.to_datetime(src, format="%Y-%m-%d", errors="coerce")
    mask = out.isna()
    if mask.any():
        out.loc[mask] = pd.to_datetime(src.loc[mask], format="%d%b%Y", errors="coerce")
    mask = out.isna()
    if mask.any():
        out.loc[mask] = pd.to_datetime(src.loc[mask], format="%d-%b-%y", errors="coerce")
    mask = out.isna()
    if mask.any():
        out.loc[mask] = pd.to_datetime(src.loc[mask], errors="coerce")
    return out


def load_old(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize index/id
    if "patid" in df.columns:
        df = df.rename(columns={"patid": "patient_id"})
    # Parse dates
    for c in ["time1", "time2"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], format="%Y-%m-%d", errors="coerce")
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = robust_parse_date(df[c])
    # Clean extras
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def load_new(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in DATE_COLS + ["first_OAC_date", "first_antiplatelet_date", "earliest_tia_date"]:
        if c in df.columns:
            df[c] = robust_parse_date(df[c])
    return df


def make_key(df: pd.DataFrame) -> pd.Series:
    """Create a robust fingerprint key based on dates + demographics.

    Key components:
      - earliest_af_date (YYYY-MM-DD or NA)
      - earliest_stroke_date (YYYY-MM-DD or NA)
      - end_fu (YYYY-MM-DD or NA)
      - gender (1/2 if available else 9)
      - age bucket (int age or rounded by 1 year if float)
    """
    af = df["earliest_af_date"].dt.strftime("%Y-%m-%d").fillna("NA") if "earliest_af_date" in df.columns else "NA"
    st = df["earliest_stroke_date"].dt.strftime("%Y-%m-%d").fillna("NA") if "earliest_stroke_date" in df.columns else "NA"
    fu = df["end_fu"].dt.strftime("%Y-%m-%d").fillna("NA") if "end_fu" in df.columns else "NA"
    gender = df["gender"].fillna(9).astype(int) if "gender" in df.columns else pd.Series(9, index=df.index)
    # Age to nearest integer, robust to missing/non-numeric
    age = pd.to_numeric(df.get("age", pd.Series(np.nan, index=df.index)), errors="coerce").round().fillna(-1).astype(int)
    return af.astype(str) + "|" + st.astype(str) + "|" + fu.astype(str) + "|g" + gender.astype(str) + "|a" + age.astype(str)


def compare_pairs(df_old: pd.DataFrame, df_new: pd.DataFrame, idx_old: pd.Index, idx_new: pd.Index) -> pd.DataFrame:
    """Build a DataFrame summarizing differences for matched pairs."""
    left = df_old.loc[idx_old].reset_index(drop=True).copy()
    right = df_new.loc[idx_new].reset_index(drop=True).copy()

    # Harmonize stroke history naming
    if "HB_stroke_history" not in right.columns and "Stroke_TIA_hx" in right.columns:
        right["HB_stroke_history"] = right["Stroke_TIA_hx"]

    rows = []
    n = len(left)
    for i in range(n):
        lo = left.iloc[i]
        rn = right.iloc[i]
        # Binary mismatch score
        bin_mismatches = 0
        bin_checked = 0
        for col in BIN_COLS:
            if col in left.columns or col in right.columns:
                lv = lo.get(col, np.nan)
                rv = rn.get(col, np.nan)
                if pd.notna(lv) and pd.notna(rv):
                    bin_checked += 1
                    if int(lv) != int(rv):
                        bin_mismatches += 1
        # Continuous diffs (absolute)
        cont_diffs = {}
        for col in CONT_COLS:
            lv = pd.to_numeric(lo.get(col, np.nan), errors="coerce")
            rv = pd.to_numeric(rn.get(col, np.nan), errors="coerce")
            if pd.notna(lv) and pd.notna(rv):
                cont_diffs[col] = float(abs(lv - rv))
        # Age/gender direct
        age_old = pd.to_numeric(lo.get("age", np.nan), errors="coerce")
        age_new = pd.to_numeric(rn.get("age", np.nan), errors="coerce")
        gender_old = lo.get("gender", np.nan)
        gender_new = rn.get("gender", np.nan)
        # Compose summary
        rows.append({
            "key": lo.get("__key__", ""),
            "age_old": age_old,
            "age_new": age_new,
            "gender_old": gender_old,
            "gender_new": gender_new,
            "bin_mismatch_count": bin_mismatches,
            "bin_checked": bin_checked,
            **{f"diff_{k}": v for k, v in cont_diffs.items()},
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Find row-level matches between datasets and compare covariates.")
    parser.add_argument("--old", default=str(Path("data") / "random_nuchad.csv"), help="Path to old dataset CSV")
    parser.add_argument("--new", default=str(Path("data") / "random_nuchad_250623.csv"), help="Path to new dataset CSV")
    parser.add_argument("--limit", type=int, default=50, help="Rows to include in top discrepant sample")
    args = parser.parse_args()

    old_path = Path(args.old)
    new_path = Path(args.new)
    if not old_path.exists() or not new_path.exists():
        print(f"Missing datasets. Old exists: {old_path.exists()} | New exists: {new_path.exists()}")
        return 1

    print("Loading datasets and parsing dates (robust)...")
    df_old = load_old(old_path)
    df_new = load_new(new_path)

    # Create keys
    print("Creating robust matching keys...")
    key_old = make_key(df_old)
    key_new = make_key(df_new)
    df_old = df_old.copy(); df_old["__key__"] = key_old
    df_new = df_new.copy(); df_new["__key__"] = key_new

    # Only consider keys that are unique in each dataset for high-confidence matches
    counts_old = key_old.value_counts()
    counts_new = key_new.value_counts()
    uniq_keys = set(counts_old[counts_old == 1].index) & set(counts_new[counts_new == 1].index)

    print(f"Rows (old): {len(df_old):,} | (new): {len(df_new):,}")
    print(f"Unique keys (old): {int((counts_old == 1).sum()):,} | (new): {int((counts_new == 1).sum()):,}")
    print(f"High-confidence matches (1:1 on key): {len(uniq_keys):,}")

    # Build indices for matched pairs
    idx_old = df_old.index[df_old["__key__"].isin(uniq_keys)]
    idx_new_map = {k: i for i, k in df_new["__key__"].items()}
    idx_new = [idx_new_map[k] for k in df_old.loc[idx_old, "__key__"].tolist()]

    # Compare pairs
    print("Comparing matched pairs across covariates...")
    comp = compare_pairs(df_old, df_new, idx_old, idx_new)

    # Score discrepancy: sum of binary mismatches + normalized continuous diffs
    cont_cols_present = [c for c in CONT_COLS if f"diff_{c}" in comp.columns]
    comp["discrepancy_score"] = comp["bin_mismatch_count"].fillna(0)
    for c in cont_cols_present:
        # Scale by robust IQR of diffs (avoid division by zero)
        diffs = comp[f"diff_{c}"].replace([np.inf, -np.inf], np.nan).dropna()
        scale = float(np.nanpercentile(diffs, 75) - np.nanpercentile(diffs, 25)) or 1.0
        comp["discrepancy_score"] += comp[f"diff_{c}"].fillna(0) / scale

    # Save detailed pairs (key metrics only) for further inspection
    results_dir = get_results_dir()
    pairs_path = results_dir / "row_match_pairs.csv"
    comp_sorted = comp.sort_values("discrepancy_score", ascending=False)
    comp_sorted.to_csv(pairs_path, index=False)

    # Summary report
    report_path = results_dir / "row_match_report.md"
    examples_path = results_dir / "row_match_examples.md"
    with open(report_path, "w") as f:
        f.write("# Row-Level Matching Report\n\n")
        f.write("This report attempts to match rows between the old and new datasets using a robust date+demographics fingerprint, then compares covariates to highlight potential scrambling.\n\n")
        f.write("## Matching Summary\n\n")
        f.write(f"- Rows (old): {len(df_old):,}\n")
        f.write(f"- Rows (new): {len(df_new):,}\n")
        f.write(f"- High-confidence 1:1 matches: {len(uniq_keys):,}\n\n")

        # Basic concordance stats on age and gender
        age_match = (comp_sorted["age_old"].round().fillna(-1) == comp_sorted["age_new"].round().fillna(-1)).mean()
        gender_match = (comp_sorted["gender_old"].fillna(-1) == comp_sorted["gender_new"].fillna(-1)).mean()
        f.write("## Concordance (Matched Pairs)\n\n")
        f.write(f"- Age exact match (rounded years): {age_match*100:.1f}%\n")
        f.write(f"- Gender exact match: {gender_match*100:.1f}%\n")
        if "bin_checked" in comp_sorted.columns and (comp_sorted["bin_checked"] > 0).any():
            avg_bin_checked = comp_sorted["bin_checked"].replace(0, np.nan).mean()
            avg_bin_mismatch = (comp_sorted["bin_mismatch_count"].replace(0, np.nan) / comp_sorted["bin_checked"].replace(0, np.nan)).mean()
            if pd.notna(avg_bin_mismatch):
                f.write(f"- Mean binary mismatch rate (across available indicators): {avg_bin_mismatch*100:.1f}%\n")
        f.write("\n")

        # Show top discrepant examples
        f.write("## Top Discrepant Matches (sample)\n\n")
        sample = comp_sorted.head(args.limit).copy()
        show_cols = [
            "key", "age_old", "age_new", "gender_old", "gender_new",
            "bin_mismatch_count", "bin_checked", "discrepancy_score",
        ] + [c for c in comp_sorted.columns if c.startswith("diff_")]
        f.write(sample[show_cols].to_markdown(index=False))
        f.write("\n\n")

        f.write("## Notes\n\n")
        f.write("- Keys are constructed from earliest_af_date, earliest_stroke_date, end_fu, gender, and age (rounded).\n")
        f.write("- Only keys unique within each dataset are used to form high-confidence pairs.\n")
        f.write("- Large disagreement on covariates given identical dates suggests scrambling of non-outcome fields.\n")

    print(f"Saved: {report_path}")
    print(f"Saved: {pairs_path}")

    # Also save side-by-side examples for manual inspection
    try:
        EXAMPLE_VARS = [
            "earliest_af_date", "earliest_stroke_date", "end_fu",
            "age", "gender", "af", "hypertension", "diab",
            "thrombo", "hf",
            # stroke history column differs by dataset name
            "HB_stroke_history", "Stroke_TIA_hx",
            "ckd", "vasc_dis_mi_pad", "aortic_plaq",
            "stroke_1Y",
        ]
        with open(examples_path, "w") as fx:
            fx.write("# Matched Row Examples (Top Discrepancies)\n\n")
            top = comp_sorted.head(min(10, len(comp_sorted)))
            for j, row in enumerate(top.itertuples(index=False), 1):
                key = getattr(row, "key")
                # Find the paired rows again
                lo = df_old[df_old["__key__"] == key].iloc[0]
                rn = df_new[df_new["__key__"] == key].iloc[0]
                fx.write(f"## Pair {j}: key={key}\n\n")
                fx.write("| Variable | Old | New |\n|---|---:|---:|\n")
                for var in EXAMPLE_VARS:
                    if var in ("HB_stroke_history", "Stroke_TIA_hx"):
                        # Show both history fields if present
                        old_val = lo.get("HB_stroke_history", np.nan)
                        new_val = rn.get("Stroke_TIA_hx", rn.get("HB_stroke_history", np.nan))
                    else:
                        old_val = lo.get(var, np.nan)
                        new_val = rn.get(var, np.nan)
                    # Format dates
                    if var in DATE_COLS and pd.notna(old_val):
                        old_val = pd.to_datetime(old_val).strftime("%Y-%m-%d")
                    if var in DATE_COLS and pd.notna(new_val):
                        new_val = pd.to_datetime(new_val).strftime("%Y-%m-%d")
                    fx.write(f"| {var} | {old_val} | {new_val} |\n")
                fx.write("\n")
        print(f"Saved: {examples_path}")
    except Exception as e:
        print(f"Warning: could not write examples file: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
