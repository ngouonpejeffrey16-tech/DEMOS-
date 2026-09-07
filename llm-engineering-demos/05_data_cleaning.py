"""Demo 5 — Data cleaning with pandas: from messy logs to analysis-ready data.

Real-world data is messy. This demo generates a deliberately dirty dataset
of truck maintenance logs, then applies a typical cleaning pipeline:
  1. Inspect and profile the raw data
  2. Standardize inconsistent text and categories
  3. Parse mixed date formats explicitly (no silent guessing)
  4. Remove duplicates AFTER normalization (so '2026-05-12' == '12/05/2026')
  5. Flag outliers BEFORE imputation (so they don't pollute the statistics)
  6. Handle missing values with an explicit, column-specific strategy
No API key needed — runs fully offline.
"""

import numpy as np
import pandas as pd

# --- 1. A deliberately messy dataset (as exported from 3 different tools) --
RAW = pd.DataFrame(
    {
        "truck_id": ["FH16-001", "fh16-001", "FM-207", "FMX-113", "FM-207",
                     "FH16-002", None, "FMX-113", "FH16-001", "FM-207"],
        "intervention": ["Oil change", "oil change", "BRAKE PADS", "Oil Change",
                         "brake pads", "battery", "oil change", "Battery ",
                         "Oil change", "brake pads"],
        "date": ["2026-05-12", "12/05/2026", "2026-06-01", "01-06-2026",
                 "2026-06-15", "not_recorded", "2026-07-02", "2026/07/10",
                 "2026-05-12", "2026-08-01"],
        "cost_eur": ["450", "450", "1200", "480", "n/a", "310", "465",
                     "290", "450", "999999"],
        "km_at_service": [238000, 238000, 187500, np.nan, 189000, 402000,
                          61000, 405500, 238000, 191000],
    }
)

KNOWN_DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"]


def parse_date(value: str) -> pd.Timestamp:
    """Try each known format explicitly. Unknown formats -> NaT (auditable),
    never a silent wrong guess (e.g. day/month swapped)."""
    for fmt in KNOWN_DATE_FORMATS:
        try:
            return pd.to_datetime(value, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.NaT


def profile(df: pd.DataFrame, label: str) -> None:
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    print(df.to_string())
    print(f"\nShape: {df.shape} | Missing values per column:")
    print(df.isna().sum().to_string())


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 2. Standardize text first (else 'Oil change' != 'oil change')
    df["truck_id"] = df["truck_id"].str.upper().str.strip()
    df["intervention"] = df["intervention"].str.lower().str.strip()

    # 3. Explicit date parsing and numeric conversion
    df["date"] = df["date"].apply(parse_date)
    df["cost_eur"] = pd.to_numeric(df["cost_eur"], errors="coerce")

    # 4. Deduplicate AFTER normalization, so equivalent rows actually match
    before = len(df)
    df = df.drop_duplicates(subset=["truck_id", "intervention", "date"])
    print(f"\n[clean] Removed {before - len(df)} duplicate rows")

    # 5. Flag outliers BEFORE imputation so they don't skew the median.
    #    Threshold: domain knowledge (no routine intervention costs > 5000 EUR).
    df["cost_outlier"] = df["cost_eur"] > 5000

    # 6. Missing-value strategy — explicit and column-specific:
    #    - truck_id missing -> row unusable, drop it
    #    - cost missing     -> impute with median of NON-outlier costs
    #                          for that intervention type
    df = df.dropna(subset=["truck_id"])
    clean_costs = df.loc[~df["cost_outlier"]]
    medians = clean_costs.groupby("intervention")["cost_eur"].median()
    missing_cost = df["cost_eur"].isna()
    df.loc[missing_cost, "cost_eur"] = df.loc[missing_cost, "intervention"].map(medians)

    return df.sort_values(["truck_id", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    profile(RAW, "RAW DATA (as received)")
    cleaned = clean(RAW)
    profile(cleaned, "CLEANED DATA")

    print("\nCost summary by intervention type (outliers excluded):")
    print(
        cleaned[~cleaned["cost_outlier"]]
        .groupby("intervention")["cost_eur"]
        .agg(["count", "mean", "min", "max"])
        .round(0)
        .to_string()
    )
