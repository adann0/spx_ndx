#!/usr/bin/env python3
"""
fetch_multpl.py
Scrapes historical data from multpl.com.
Re-downloads everything on every run.
Saves to datas/ as parquet files.
"""
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

OUTPUT_DIR = Path("datas")
OUTPUT_DIR.mkdir(exist_ok=True)

SITE_MAP = {
    "https://www.multpl.com/s-p-500-pe-ratio/table/by-month":              "spx_pe_ratio.parquet",
    "https://www.multpl.com/s-p-500-price-to-book/table/by-quarter":       "spx_price_to_book_value.parquet",
    "https://www.multpl.com/s-p-500-price-to-sales/table/by-quarter":      "spx_price_to_sales_ratio.parquet",
    "https://www.multpl.com/shiller-pe/table/by-month":                    "spx_shiller_pe_ratio.parquet",
    "https://www.multpl.com/s-p-500-sales/table/by-quarter":               "spx_sales_per_share.parquet",
    "https://www.multpl.com/s-p-500-earnings-yield/table/by-month":        "spx_earnings_yield.parquet",
    "https://www.multpl.com/s-p-500-earnings/table/by-month":              "spx_earning.parquet",
    "https://www.multpl.com/s-p-500-historical-prices/table/by-month":     "spx_historical_prices.parquet",
    "https://www.multpl.com/s-p-500-dividend-yield/table/by-month":        "spx_dividend_yield.parquet",
}

def fetch(url, file):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")
    rows = table.find_all("tr")
    results = []
    for row in rows[1:]:
        cols = row.find_all("td")
        if len(cols) < 2:
            continue
        try:
            date_str = cols[0].get_text(strip=True)
            val_str = cols[1].get_text(strip=True).replace(",", "").replace("\u2020", "").replace("%", "")
            value = float(val_str)
            date = pd.to_datetime(date_str, format="%b %d, %Y") + pd.offsets.MonthEnd(0)
            results.append({"date": date, "value": value})
        except Exception:
            continue
    df = pd.DataFrame(results)
    df.to_parquet(OUTPUT_DIR / file, index=False)
    print(f"  -> {file}  ({len(df)} rows)")
    time.sleep(1.5)

print("=" * 50)
print("spx-ndx - fetch_multpl.py")
print("=" * 50)

for url, file in SITE_MAP.items():
    print(f"  Scraping {file}...")
    fetch(url, file)

print("\nDone - multpl.com data saved to datas/")