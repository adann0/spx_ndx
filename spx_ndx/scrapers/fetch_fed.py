#!/usr/bin/env python3
"""
fetch_fed.py
Downloads data from the Federal Reserve Board website.
Saves to datas/ as parquet files.
"""
import requests
import pandas as pd
from io import StringIO
from pathlib import Path

OUTPUT_DIR = Path("datas")
OUTPUT_DIR.mkdir(exist_ok=True)

SOURCES = {
    "fed_ebp": {
        "url": "https://www.federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv",
        "desc": "Excess Bond Premium (Gilchrist-Zakrajsek)",
    },
}

def fetch_fed(name, url):
    response = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    df = pd.read_csv(StringIO(response.text))
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.to_parquet(OUTPUT_DIR / f"{name}.parquet", index=False)
    print(f"  -> {name}.parquet  ({len(df)} rows, {df['date'].min().date()} > {df['date'].max().date()})")

print("=" * 50)
print("spx-ndx - fetch_fed.py")
print("=" * 50)

for name, cfg in SOURCES.items():
    print(f"  Fetching {cfg['desc']}...")
    fetch_fed(name, cfg["url"])

print("\nDone - Fed data saved to datas/")
