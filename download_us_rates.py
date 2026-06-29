"""Download daily US Treasury constant-maturity yields (2y + 10y) from FRED to a parquet.

DGS2  = 2-Year Treasury Constant Maturity Rate  (daily, percent)
DGS10 = 10-Year Treasury Constant Maturity Rate (daily, percent)

Uses the FRED API (api.stlouisfed.org), which needs a free API key. Set it via the
FRED_API_KEY environment variable, or paste it into the FRED_API_KEY fallback below
(same key handling as Download_macro_event_dates.ipynb). Get a free key at
https://fred.stlouisfed.org/docs/api/api_key.html

Output: us_rates_daily.parquet
    index   : Date (tz-naive datetime64[ns], midnight)
    columns : rate_2y (float, %), rate_10y (float, %)
"""

import os
import time
import pandas as pd
import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (research data download)"}
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

# --- API key: env var first, else paste into the fallback string -------------------
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")  # or: FRED_API_KEY = "your_key_here"

# Covering window for the project sample (RV/GVZ/IV span ~2009-09 .. 2026-06).
OBS_START = "2009-09-01"
OBS_END = "2026-06-30"

SERIES = {"DGS2": "rate_2y", "DGS10": "rate_10y"}


def _get(url, params, timeout=60, retries=4):
    """GET with retries (network can be flaky)."""
    last = None
    for k in range(retries):
        try:
            return requests.get(url, params=params, headers=HEADERS, timeout=timeout)
        except requests.exceptions.RequestException as e:
            last = e
            print(f"  retry {k + 1}/{retries} after error: {e}")
            time.sleep(3)
    raise last


def download_fred_series(series_id, out_col, start=OBS_START, end=OBS_END):
    """Fetch a FRED daily series -> tz-naive Date-indexed float Series (NaNs dropped).

    FRED returns '.' for holidays/missing observations -> coerced to NaN and dropped.
    """
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    resp = _get(FRED_URL, params)
    resp.raise_for_status()
    obs = resp.json().get("observations", [])
    df = pd.DataFrame(obs)[["date", "value"]].rename(columns={"date": "Date", "value": out_col})
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()      # tz-naive midnight
    df[out_col] = pd.to_numeric(df[out_col], errors="coerce")   # '.' -> NaN
    df = df.set_index("Date").sort_index().dropna()

    print(f"{series_id} -> {out_col}: {df.shape[0]} rows, "
          f"{df.index.min().date()} -> {df.index.max().date()}, NaNs=0")
    return df


def main():
    assert FRED_API_KEY, (
        "No FRED_API_KEY set. Get a free key at "
        "https://fred.stlouisfed.org/docs/api/api_key.html and either "
        "`export FRED_API_KEY=...` or paste it into the FRED_API_KEY fallback in this file."
    )

    parts = [download_fred_series(sid, col) for sid, col in SERIES.items()]
    rates = pd.concat(parts, axis=1, join="inner").sort_index()
    assert rates.notna().all().all(), "unexpected NaNs after inner join"

    outpath = "us_rates_daily.parquet"
    rates.to_parquet(outpath, engine="pyarrow")
    print(f"\nSaved {outpath}: {rates.shape} cols={list(rates.columns)}, "
          f"{rates.index.min().date()} -> {rates.index.max().date()}")

    # Same-time-period alignment: report coverage of the modelling sample.
    sample = pd.read_parquet("merged_RV_GVZ_with_macro_event.parquet")
    covered = sample.index.intersection(rates.index)
    print(f"Coverage of merged_RV_GVZ_with_macro_event ({sample.index.min().date()} .. "
          f"{sample.index.max().date()}): {len(covered)} of {len(sample)} trading days")


if __name__ == "__main__":
    main()
