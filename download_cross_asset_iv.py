"""Download cross-asset implied-vol indices (VIX + OVX) from CBOE and save to parquet.

VIX  = CBOE S&P 500 Volatility Index        (equity implied vol)
OVX  = CBOE Crude Oil ETF Volatility Index  (crude oil implied vol)

Both come from CBOE's public CDN, mirroring the existing GVZ download in
Download_price_Data.ipynb (Cell 2). The downstream goal is to test, with the log-HAR
model, whether cross-asset IV (VIX + OVX) forecasts gold realized vol as well as the
gold-specific implied vol (GVZ), or whether GVZ is genuinely better.

Output: cross_asset_iv.parquet
    index   : Date (tz-naive datetime64[ns], midnight)
    columns : VIX_close (float64), OVX_close (float64)
"""

import pandas as pd
import requests
from io import StringIO

HEADERS = {"User-Agent": "Mozilla/5.0 (research data download)"}
CBOE = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"


def download_cboe_index(symbol, out_col):
    """Fetch a CBOE daily index history CSV -> tz-naive Date-indexed close series.

    CBOE history CSVs are not uniform: VIX is OHLC (DATE, OPEN, HIGH, LOW, CLOSE) while
    OVX is a single-value file (DATE, OVX). Pick a column named "close" when present,
    otherwise the last column (the value column for single-series files) -- this avoids
    grabbing OPEN from the OHLC files and works for both layouts.
    """
    resp = requests.get(CBOE.format(symbol), headers=HEADERS, timeout=30)
    resp.raise_for_status()
    raw = pd.read_csv(StringIO(resp.text))

    date_col = raw.columns[0]
    close_col = next((c for c in raw.columns if c.strip().lower() == "close"), raw.columns[-1])
    df = raw[[date_col, close_col]].rename(columns={date_col: "Date", close_col: out_col})
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()      # tz-naive midnight
    df = df.set_index("Date").sort_index()
    df[out_col] = df[out_col].astype(float)

    print(f"{symbol}: {df.shape[0]} rows, {df.index.min().date()} -> {df.index.max().date()}, "
          f"NaNs={int(df[out_col].isna().sum())}")
    return df


def main():
    vix = download_cboe_index("VIX", "VIX_close")   # equity IV
    ovx = download_cboe_index("OVX", "OVX_close")   # crude oil IV (CBOE Crude Oil ETF VIX)

    # One combined parquet (inner join on common dates), GVZ_daily.parquet style
    iv = pd.concat([vix, ovx], axis=1, join="inner").sort_index()
    assert iv.notna().all().all(), "unexpected NaNs after inner join"

    outpath = "cross_asset_iv.parquet"
    iv.to_parquet(outpath, engine="pyarrow")
    print(f"\nSaved {outpath}: {iv.shape} cols={list(iv.columns)}, "
          f"{iv.index.min().date()} -> {iv.index.max().date()}")


if __name__ == "__main__":
    main()
