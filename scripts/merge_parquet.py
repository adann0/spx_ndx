#!/usr/bin/env python3
"""
merge_parquet.py - Merge raw parquet sources into unified files.

For assets with multiple sources (e.g. gold from datahub monthly + Yahoo daily),
produces a single merged parquet with the best available data:
  - Backfill from low-freq source (datahub monthly)
  - Overwrite with high-freq source (Yahoo daily) where available

Input:  datas/datahub_*.parquet, datas/*.parquet (Yahoo/FRED)
Output: datas/merged_*.parquet
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("datas")

print("=== merge_parquet.py ===")


def merge_gold():
    """Merge datahub gold (monthly, since 1833) with Yahoo gold (daily, since 2000)."""
    dh = pd.read_parquet(DATA_DIR / "datahub_gold_monthly.parquet")
    dh["date"] = pd.to_datetime(dh["date"])
    dh = dh.set_index("date").sort_index()
    dh = dh.rename(columns={"value": "close"})

    yh = pd.read_parquet(DATA_DIR / "gold.parquet")
    yh.index = pd.to_datetime(yh.index)
    yh = yh.sort_index()

    # Before Yahoo: use datahub monthly, forward-filled to daily
    # After Yahoo starts: use Yahoo daily
    yahoo_start = yh.index.min()

    # Datahub portion: monthly -> daily via ffill
    dh_daily = dh[["close"]].resample("D").ffill()
    dh_daily = dh_daily[dh_daily.index < yahoo_start]

    # Concat: datahub backfill + Yahoo daily
    merged = pd.concat([dh_daily, yh[["close"]]])
    merged = merged.sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]

    out = pd.DataFrame(index=merged.index)
    out["close"] = merged["close"]
    # Fill OHLC from close for backfill period (only close available)
    out["open"] = out["close"]
    out["high"] = out["close"]
    out["low"] = out["close"]
    # Use Yahoo OHLC where available
    for col in ["open", "high", "low"]:
        if col in yh.columns:
            out.loc[yh.index, col] = yh[col]
    if "volume" in yh.columns:
        out["volume"] = 0
        out.loc[yh.index, "volume"] = yh["volume"]

    out.index.name = "date"
    out.to_parquet(DATA_DIR / "merged_gold.parquet")
    print(f"  gold: datahub {dh.index.min().date()}->{yahoo_start.date()} + Yahoo {yahoo_start.date()}->{yh.index.max().date()}")
    print(f"    -> merged_gold.parquet ({len(out)} rows)")


def merge_copper():
    """Merge datahub copper (monthly, 1980-2017) with Yahoo copper (daily, since 2000)."""
    dh = pd.read_parquet(DATA_DIR / "datahub_copper_monthly.parquet")
    dh["date"] = pd.to_datetime(dh["date"])
    dh = dh.set_index("date").sort_index()
    dh = dh.rename(columns={"value": "close"})

    yh = pd.read_parquet(DATA_DIR / "copper.parquet")
    yh.index = pd.to_datetime(yh.index)
    yh = yh.sort_index()

    yahoo_start = yh.index.min()

    # Datahub portion: monthly -> daily via ffill
    dh_daily = dh[["close"]].resample("D").ffill()
    dh_daily = dh_daily[dh_daily.index < yahoo_start]

    # Concat: datahub backfill + Yahoo daily
    merged = pd.concat([dh_daily, yh[["close"]]])
    merged = merged.sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]

    out = pd.DataFrame(index=merged.index)
    out["close"] = merged["close"]
    out["open"] = out["close"]
    out["high"] = out["close"]
    out["low"] = out["close"]
    for col in ["open", "high", "low"]:
        if col in yh.columns:
            out.loc[yh.index, col] = yh[col]
    if "volume" in yh.columns:
        out["volume"] = 0
        out.loc[yh.index, "volume"] = yh["volume"]

    out.index.name = "date"
    out.to_parquet(DATA_DIR / "merged_copper.parquet")
    print(f"  copper: datahub {dh.index.min().date()}->{yahoo_start.date()} + Yahoo {yahoo_start.date()}->{yh.index.max().date()}")
    print(f"    -> merged_copper.parquet ({len(out)} rows)")


merge_gold()
merge_copper()
print("\nDone.")
