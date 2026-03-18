#!/usr/bin/env python3
"""
fetch_fred.py
Downloads FRED series via public XLS endpoint - no API key required.
Re-downloads everything on every run.
Saves to datas/ as parquet files.
"""
from curl_cffi import requests
import pandas as pd
from io import BytesIO
from pathlib import Path

OUTPUT_DIR = Path("datas")
OUTPUT_DIR.mkdir(exist_ok=True)

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.xls?id="
HEADERS = {"User-Agent": "Mozilla/5.0"}

SERIES = {
    # Monthly
    "GS10":            "fred_rate_10y.parquet",
    "T10Y2Y":          "fred_yield_curve.parquet",
    "T10YIE":          "fred_breakeven_10y.parquet",
    "REAINTRATREARAT10Y": "fred_real_rate_10y.parquet",
    "GDP":             "fred_gdp.parquet",
    "CPIAUCSL":        "fred_cpi.parquet",
    "UNRATE":          "fred_unemployment.parquet",
    "M2SL":            "fred_m2.parquet",
    "BAMLH0A0HYM2":    "fred_credit_spread.parquet",
    "A464RC1A027NBEA": "fred_corp_margins.parquet",
    "BAA":              "fred_baa_yield.parquet",
    "AAA":              "fred_aaa_yield.parquet",
    "ICSA":            "fred_initial_claims.parquet",
    "UMCSENT":         "fred_consumer_sentiment.parquet",
    "INDPRO":          "fred_industrial_production.parquet",
    "RSAFS":           "fred_retail_sales.parquet",
    "DTWEXBGS":        "fred_dollar_index.parquet",
    "DCOILWTICO":      "fred_wti_oil.parquet",
    "PERMIT":          "fred_building_permits.parquet",
    "FEDFUNDS":        "fred_fed_funds_rate.parquet",
    "GFDEGDQ188S":     "fred_fed_debt_gdp.parquet",
    # Daily
    "DGS2":            "fred_rate_2y_daily.parquet",
    "DGS10":           "fred_rate_10y_daily.parquet",
    "BAMLH0A0HYM2EY":  "fred_hy_oas_daily.parquet",
    "VIXCLS":          "fred_vix_daily.parquet",
    # Financial Conditions (weekly, from 1971)
    "NFCI":            "fred_nfci.parquet",
    "ANFCI":           "fred_anfci.parquet",
    "NFCIRISK":        "fred_nfci_risk.parquet",
    "NFCILEVERAGE":    "fred_nfci_leverage.parquet",
    "NFCICREDIT":      "fred_nfci_credit.parquet",
    # Credit growth (monthly, from 1947)
    "TOTBKCR":         "fred_total_bank_credit.parquet",
    "BUSLOANS":        "fred_business_loans.parquet",
    # Housing (monthly, from 1987)
    "CSUSHPINSA":      "fred_case_shiller.parquet",
    # Liquidity (weekly/daily, from 2002-2003)
    "WALCL":           "fred_fed_bs.parquet",
    "WTREGEN":         "fred_tga.parquet",
    "RRPONTSYD":       "fred_rrp.parquet",
    "WRESBAL":         "fred_reserves.parquet",
    # Dollar (daily, from 1973 - pre-1993)
    "DTWEXM":          "fred_dollar_major.parquet",
}

RETRIES = 6
BACKOFF = [10, 20, 30, 45, 60, 90]

def fetch_fred(series_id, file):
    url = FRED_BASE + series_id
    for attempt in range(RETRIES):
        try:
            response = requests.get(url, timeout=120, headers=HEADERS, impersonate="chrome")
            response.raise_for_status()
            xls = pd.ExcelFile(BytesIO(response.content))
            df = pd.read_excel(xls, sheet_name=xls.sheet_names[-1])
            df.columns = ["date", "value"]
            df["date"]   = pd.to_datetime(df["date"])
            df = df[df["value"] != "."].copy()
            df["value"]  = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["value"])
            df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
            df.to_parquet(OUTPUT_DIR / file, index=False)
            print(f"  -> {file}  ({len(df)} rows, {df['date'].min().date()} > {df['date'].max().date()})")
            return
        except (requests.exceptions.RequestException, requests.exceptions.ReadTimeout, Exception) as e:
            if attempt < RETRIES - 1:
                wait = BACKOFF[attempt]
                print(f"  RETRY {attempt+1}/{RETRIES} for {series_id}: {e} (wait {wait}s)")
                import time; time.sleep(wait)
            else:
                raise

print("=" * 50)
print("spx-ndx - fetch_fred.py")
print("=" * 50)

for series_id, file in SERIES.items():
    print(f"  Fetching {series_id}...")
    fetch_fred(series_id, file)

print("\nDone - FRED data saved to datas/")
