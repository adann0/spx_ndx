#!/usr/bin/env python3
"""
fetch_datahub.py
Downloads commodity data from datahub.io.
Saves to datas/ as parquet files.
"""
import pandas as pd
import requests
from pathlib import Path

OUTPUT_DIR = Path("datas")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 50)
print("spx-ndx - fetch_datahub.py")
print("=" * 50)

# Gold monthly (since 1833)
print("  Fetching gold monthly from datahub.io...")
url = "https://datahub.io/core/gold-prices/r/monthly.csv"
gold = pd.read_csv(url)
gold.columns = ["date", "value"]
gold["date"] = pd.to_datetime(gold["date"])
gold = gold.sort_values("date").reset_index(drop=True)
gold.to_parquet(OUTPUT_DIR / "datahub_gold_monthly.parquet", index=False)
print(f"    -> datahub_gold_monthly.parquet  ({len(gold)} rows, {gold['date'].min().date()} to {gold['date'].max().date()})")

# Copper monthly from IMF commodity prices v1 (1980-2017, USD/metric ton -> USD/lb)
print("  Fetching copper monthly from datahub.io (IMF commodity prices v1)...")
import csv
from io import StringIO
url_v1 = "https://datahub.io/core/commodity-prices/_r/-/data/commodity-prices.csv"
r = requests.get(url_v1, timeout=30)
reader = csv.reader(StringIO(r.text))
header = next(reader)
copper_idx = header.index("Copper")
dates, vals = [], []
for row in reader:
    dates.append(row[0])
    v = row[copper_idx]
    vals.append(float(v) if v else None)
copper = pd.DataFrame({"date": pd.to_datetime(dates), "value": vals}).dropna()
LBS_PER_MT = 2204.62
copper["value"] = copper["value"] / LBS_PER_MT  # USD/mt -> USD/lb
copper = copper.sort_values("date").reset_index(drop=True)
copper.to_parquet(OUTPUT_DIR / "datahub_copper_monthly.parquet", index=False)
print(f"    -> datahub_copper_monthly.parquet  ({len(copper)} rows, {copper['date'].min().date()} to {copper['date'].max().date()})")

print("\nDone - Datahub data saved to datas/")
