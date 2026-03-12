#!/usr/bin/env python3
"""
fetch_yahoo.py
Re-downloads all Yahoo Finance data from scratch on every run.
Saves to datas/ as parquet files.
"""
import os
import pandas as pd
import yfinance as yf

os.makedirs("datas", exist_ok=True)

TICKERS = {
    "gspc":  "^GSPC",   # S&P 500
    "ndx":   "^NDX",    # Nasdaq-100
    "vix":   "^VIX",    # VIX
    "irx":   "^IRX",    # 3M T-Bill
    "w5000": "^W5000",  # Wilshire 5000
    "spy":   "SPY",     # S&P 500 ETF
    "qqq":   "QQQ",     # Nasdaq-100 ETF
    "urth":  "URTH",    # MSCI World ETF
    "gold":  "GC=F",    # Gold futures (depuis ~2000 sur Yahoo)
    "msci-world": "^990100-USD-STRD", # MSCI WORLD
    "vix9d": "^VIX9D",  # VIX 9-day (short-term vol)
    "vix3m": "^VIX3M",  # VIX 3-month (term structure)
    "hyg":   "HYG",     # High Yield Corporate Bond ETF
    "lqd":   "LQD",     # Investment Grade Corporate Bond ETF
    "tlt":   "TLT",     # 20+ Year Treasury Bond ETF
    "dxy":   "DX-Y.NYB", # US Dollar Index
    "rut":   "^RUT",     # Russell 2000 (small caps)
    "copper": "HG=F",    # Copper futures (Dr. Copper)
    "move":  "^MOVE",    # MOVE Index (bond volatility)
    "ief":   "IEF",      # 7-10 Year Treasury Bond ETF
    "kbe":   "KBE",      # SPDR S&P Bank ETF (bank stocks)
}

for name, ticker in TICKERS.items():
    print(f"  Fetching {ticker}...")
    df = yf.download(ticker, period="max", auto_adjust=True, progress=False)
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                  for c in df.columns]
    df.index.name = "date"
    path = f"datas/{name}.parquet"
    df.to_parquet(path)
    print(f"    -> {path}  ({len(df)} rows, {df.index[0].date()} to {df.index[-1].date()})")

print("\nDone - Yahoo Finance data saved to datas/")
