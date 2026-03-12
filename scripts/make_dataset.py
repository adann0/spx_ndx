#!/usr/bin/env python3

"""
make_dataset.py - Build SPX datasets (daily + weekly + monthly).

Daily backbone with all sources merged. Weekly/monthly resampled from daily.
Output: datas/dataset_{daily,weekly,monthly}.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Config
DATA_DIR = Path("datas")
OUT_PATH = DATA_DIR / "dataset_daily.parquet"
START_DATE = "1990-01-01"
HORIZONS_DAYS = {21: "1m", 63: "3m", 126: "6m"}
DRAWDOWN_THRESHOLDS = [0.10, 0.15, 0.20, 0.30]
CUTOFF = pd.Timestamp("1993-01-01")


# PART 1 - Raw data merge (daily backbone)

def load_daily_ohlcv(filename):
    """Load a daily OHLCV parquet -> DataFrame with date index."""
    df = pd.read_parquet(DATA_DIR / filename)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def load_dated(filename, col_name):
    """Load a parquet with (date, value) columns -> daily Series (ffill)."""
    df = pd.read_parquet(DATA_DIR / filename)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").set_index("date").sort_index()
    return df["value"].rename(col_name)


def load_daily_dated(filename, col_name):
    """Load a FRED daily parquet (date, value) -> Series."""
    df = pd.read_parquet(DATA_DIR / filename)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df["value"].rename(col_name)


print("=== PART 1: Loading & merging raw data ===")

# Backbone: daily SPX
spx = load_daily_ohlcv("gspc.parquet")
spx = spx[spx.index >= START_DATE]
idx = spx.index  # daily trading dates

dataset = pd.DataFrame(index=idx)
dataset["spx_open"] = spx["open"]
dataset["spx_close"] = spx["close"]
dataset["spx_high"] = spx["high"]
dataset["spx_low"] = spx["low"]
dataset["spx_volume"] = spx["volume"]

# Daily Yahoo sources (reindex to SPX trading days, ffill)
# merged_gold = datahub backfill + Yahoo daily (built by merge_parquet.py)
daily_yahoo = {
    "vix.parquet": "vix_close",
    "irx.parquet": "tbill_rate",
    "rut.parquet": "rut_close",
    "merged_gold.parquet": "gold_close",
    "merged_copper.parquet": "copper_close",
    "tlt.parquet": "tlt_close",
    "dxy.parquet": "dxy_close",
    "hyg.parquet": "hyg_close",
    "lqd.parquet": "lqd_close",
    "move.parquet": "move_close",
    "ndx.parquet": "ndx_close",
    "qqq.parquet": "qqq_close",
    "spy.parquet": "spy_close",
    "w5000.parquet": "w5000_close",
    "msci-world.parquet": "msci_close",
    "urth.parquet": "urth_close",
    "vix3m.parquet": "vix3m_close",
    "vix9d.parquet": "vix9d_close",
    "ief.parquet": "ief_close",
    "kbe.parquet": "kbe_close",
}
for filename, col_name in daily_yahoo.items():
    try:
        df = load_daily_ohlcv(filename)
        dataset[col_name] = df["close"].reindex(idx, method="ffill")
    except Exception as e:
        print(f"  WARNING: {filename} failed: {e}")

# Daily FRED sources
daily_fred = {
    "fred_rate_10y_daily.parquet": "rate_10y",
    "fred_rate_2y_daily.parquet": "rate_2y",
    "fred_hy_oas_daily.parquet": "hy_oas",
    "fred_yield_curve.parquet": "yield_curve",
    "fred_breakeven_10y.parquet": "breakeven_10y",
    "fred_credit_spread.parquet": "fred_credit_spread",
    "fred_dollar_index.parquet": "fred_dollar_index",
    "fred_vix_daily.parquet": "fred_vix_daily",
    # Financial Conditions (weekly, from 1971)
    "fred_nfci.parquet": "nfci",
    "fred_anfci.parquet": "anfci",
    "fred_nfci_risk.parquet": "nfci_risk",
    "fred_nfci_leverage.parquet": "nfci_leverage",
    "fred_nfci_credit.parquet": "nfci_credit",
    # Liquidity (weekly/daily, from 2002-2003)
    "fred_fed_bs.parquet": "fed_bs",
    "fred_tga.parquet": "tga",
    "fred_rrp.parquet": "rrp",
    "fred_reserves.parquet": "fed_reserves",
    # Dollar major currencies (daily, from 1973)
    "fred_dollar_major.parquet": "dollar_major",
    # Credit growth (monthly, from 1947)
    "fred_total_bank_credit.parquet": "total_bank_credit",
    "fred_business_loans.parquet": "business_loans",
    # Housing (monthly, from 1987)
    "fred_case_shiller.parquet": "case_shiller",
}
for filename, col_name in daily_fred.items():
    try:
        s = load_daily_dated(filename, col_name)
        dataset[col_name] = s.reindex(idx, method="ffill")
    except Exception as e:
        print(f"  WARNING: {filename} failed: {e}")

# Monthly/weekly/quarterly macro (ffill to daily)
monthly_sources = {
    "fred_wti_oil.parquet": "wti_oil",
    "spx_shiller_pe_ratio.parquet": "cape",
    "spx_pe_ratio.parquet": "pe_ratio",
    "spx_dividend_yield.parquet": "dividend_yield",
    "spx_earnings_yield.parquet": "earnings_yield",
    "fred_cpi.parquet": "cpi",
    "fred_unemployment.parquet": "unemployment",
    "fred_m2.parquet": "m2",
    "fred_real_rate_10y.parquet": "real_rate_10y",
    "fred_baa_yield.parquet": "baa_yield",
    "fred_aaa_yield.parquet": "aaa_yield",
    "fred_fed_funds_rate.parquet": "fed_funds_rate",
    "fred_consumer_sentiment.parquet": "consumer_sentiment",
    "fred_industrial_production.parquet": "industrial_production",
    "fred_building_permits.parquet": "building_permits",
    "fred_initial_claims.parquet": "initial_claims",
    "fred_retail_sales.parquet": "retail_sales",
    "fred_corp_margins.parquet": "corp_margins",
    "spx_earning.parquet": "spx_earnings",
    "spx_price_to_book_value.parquet": "price_to_book",
    "spx_price_to_sales_ratio.parquet": "price_to_sales",
    "spx_sales_per_share.parquet": "sales_per_share",
}
for filename, col_name in monthly_sources.items():
    try:
        s = load_dated(filename, col_name)
        dataset[col_name] = s.reindex(idx, method="ffill")
    except Exception as e:
        print(f"  WARNING: {filename} failed: {e}")

# GDP growth (quarterly -> daily ffill)
_gdp_raw = pd.read_parquet(DATA_DIR / "fred_gdp.parquet")
_gdp_raw["date"] = pd.to_datetime(_gdp_raw["date"])
_gdp_raw = _gdp_raw.set_index("date").sort_index()
gdp_growth = _gdp_raw["value"].pct_change().rename("gdp_growth")
dataset["gdp_growth"] = gdp_growth.reindex(idx, method="ffill")

# Excess Bond Premium (Fed Board, monthly, from 1973)
try:
    _ebp = pd.read_parquet(DATA_DIR / "fed_ebp.parquet").set_index("date").sort_index()
    dataset["ebp"] = _ebp["ebp"].reindex(idx, method="ffill")
    dataset["gz_spread"] = _ebp["gz_spread"].reindex(idx, method="ffill")
except Exception as e:
    print(f"  WARNING: fed_ebp.parquet failed: {e}")

# Derived raw columns
dataset["credit_spread"] = dataset["baa_yield"] - dataset["aaa_yield"]

# Forward-fill, but don't drop rows based on late-starting columns
dataset = dataset.ffill()

dataset_raw = dataset.copy()  # keep pre-cutoff version for full export
print(f"  Raw dataset: {dataset.shape[0]} rows, {dataset.shape[1]} columns")


# PART 2 - Feature engineering

print("=== PART 2: Feature engineering ===")

spx_close = dataset["spx_close"]
daily_ret = spx_close.pct_change()

f = pd.DataFrame(index=dataset.index)

# SPX raw
f["spx_volume"] = dataset["spx_volume"]
f["spx_high"] = dataset["spx_high"]
f["spx_low"] = dataset["spx_low"]

# Raw levels - include everything available
raw_level_cols = [
    "cape", "pe_ratio", "earnings_yield", "dividend_yield",
    "vix_close", "yield_curve", "credit_spread", "rate_10y",
    "real_rate_10y", "fed_funds_rate", "unemployment",
    "consumer_sentiment", "initial_claims", "wti_oil",
    "building_permits", "industrial_production",
    # New FRED
    "breakeven_10y", "fred_credit_spread", "retail_sales", "corp_margins",
    # New Multpl
    "spx_earnings", "price_to_book", "price_to_sales", "sales_per_share",
    # Financial Conditions
    "nfci", "anfci", "nfci_risk", "nfci_leverage", "nfci_credit",
    # Liquidity
    "fed_bs", "tga", "rrp", "fed_reserves",
    # Dollar major
    "dollar_major",
]
for col in raw_level_cols:
    if col in dataset.columns:
        f[col] = dataset[col]

# Macro with publication lag (1 day for daily, was 1 month for monthly)
f["gdp_growth"] = dataset["gdp_growth"].shift(1)

# Valuation derived
f["cape_implied_return"] = (1 / f["cape"]) * 100

# Spreads
f["ey_10y_spread"] = f["earnings_yield"] - f["rate_10y"]
f["excess_earnings_yield"] = f["earnings_yield"] - f["real_rate_10y"]
cpi_yoy = dataset["cpi"].pct_change(252, fill_method=None) * 100  # ~1 year in trading days
f["real_fed_funds"] = f["fed_funds_rate"] - cpi_yoy

# Daily-native features (the real advantage over monthly)
ema200 = spx_close.ewm(span=200, adjust=False).mean()
f["spx_ema200_ratio"] = spx_close / ema200 - 1
f["realized_vol"] = daily_ret.rolling(21).std() * np.sqrt(252)
f["vol_spread"] = f["vix_close"] - f["realized_vol"] * 100

# SMA (daily periods: 3m≈63, 5m≈105, 10m≈210, 12m≈252, 15m≈315)
for months, days in [(3, 63), (5, 105), (10, 210), (12, 252), (15, 315)]:
    f[f"sma_{months}m"] = spx_close.rolling(days).mean()

# RSI (14-day)
_delta = spx_close.diff()
_gain = _delta.where(_delta > 0, 0).rolling(14).mean()
_loss = (-_delta.where(_delta < 0, 0)).rolling(14).mean()
_rs = _gain / _loss
f["rsi_14"] = 100 - (100 / (1 + _rs))

# MACD histogram (12/26/9)
_ema12 = spx_close.ewm(span=12).mean()
_ema26 = spx_close.ewm(span=26).mean()
_macd_line = _ema12 - _ema26
_macd_signal = _macd_line.ewm(span=9).mean()
f["macd_line"] = _macd_line
f["macd_hist"] = _macd_line - _macd_signal

# Daily rates/spreads
for col in ["rate_2y", "hy_oas"]:
    if col in dataset.columns:
        f[col] = dataset[col]
if "rate_2y" in dataset.columns:
    f["yield_curve_2_10"] = dataset["rate_10y"] - dataset["rate_2y"]

# VIX term structure
if "vix3m_close" in dataset.columns:
    f["vix3m"] = dataset["vix3m_close"]
    f["vix_term_structure"] = dataset["vix_close"] - dataset["vix3m_close"]
if "vix9d_close" in dataset.columns:
    f["vix9d"] = dataset["vix9d_close"]

# Cross-asset raw closes
for src, name in [("gold_close", "gold"), ("rut_close", "rut"),
                  ("tlt_close", "tlt"), ("copper_close", "copper"),
                  ("dxy_close", "dxy"), ("hyg_close", "hyg"),
                  ("lqd_close", "lqd"), ("move_close", "move"),
                  ("ndx_close", "ndx"), ("spy_close", "spy"),
                  ("w5000_close", "w5000"), ("msci_close", "msci"),
                  ("qqq_close", "qqq"), ("urth_close", "urth"),
                  ("ief_close", "ief")]:
    if src in dataset.columns:
        f[f"{name}_close"] = dataset[src]

# --- New derived features (from spx_ndx plots) ---

# Drawdown from ATH
spx_ath = spx_close.expanding().max()
f["spx_drawdown_from_ath"] = (spx_close - spx_ath) / spx_ath * 100

# Volume Profile (rolling POC, VAH, VAL on daily OHLCV)
print("  Computing volume profiles...")
spx_vp = spx[spx["volume"] > 0].reindex(idx).dropna(subset=["volume"])
for lookback, label in [(63, "3m"), (126, "6m")]:
    highs = spx_vp["high"].values
    lows = spx_vp["low"].values
    volumes = spx_vp["volume"].values
    n_bins = 30
    poc_arr = np.full(len(spx_vp), np.nan)
    vah_arr = np.full(len(spx_vp), np.nan)
    val_arr = np.full(len(spx_vp), np.nan)

    for i in range(lookback, len(spx_vp)):
        h_win = highs[i - lookback:i]
        l_win = lows[i - lookback:i]
        v_win = volumes[i - lookback:i]
        lo, hi = l_win.min(), h_win.max()
        if hi <= lo:
            continue
        edges = np.linspace(lo, hi, n_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        vp = np.zeros(n_bins)
        for j in range(lookback):
            span = h_win[j] - l_win[j]
            if span <= 0:
                continue
            overlap = np.minimum(edges[1:], h_win[j]) - np.maximum(edges[:-1], l_win[j])
            overlap = np.maximum(overlap, 0)
            vp += (overlap / span) * v_win[j]
        poc_i = vp.argmax()
        poc_arr[i] = centers[poc_i]
        # Value Area (70%)
        lo_i = hi_i = poc_i
        cum = vp[poc_i]
        target = vp.sum() * 0.70
        while cum < target and (lo_i > 0 or hi_i < n_bins - 1):
            add_lo = vp[lo_i - 1] if lo_i > 0 else 0
            add_hi = vp[hi_i + 1] if hi_i < n_bins - 1 else 0
            if add_lo >= add_hi and lo_i > 0:
                lo_i -= 1
                cum += vp[lo_i]
            elif hi_i < n_bins - 1:
                hi_i += 1
                cum += vp[hi_i]
            else:
                break
        vah_arr[i] = centers[hi_i]
        val_arr[i] = centers[lo_i]

    poc_s = pd.Series(poc_arr, index=spx_vp.index).reindex(idx, method="ffill")
    vah_s = pd.Series(vah_arr, index=spx_vp.index).reindex(idx, method="ffill")
    val_s = pd.Series(val_arr, index=spx_vp.index).reindex(idx, method="ffill")
    f[f"vp_poc_{label}"] = poc_s
    f[f"vp_vah_{label}"] = vah_s
    f[f"vp_val_{label}"] = val_s
    print(f"    VP {label}: POC/VAH/VAL computed ({lookback}d lookback)")

# Ichimoku Cloud (9/26/52 daily periods)
_ichi_high = dataset["spx_high"]
_ichi_low = dataset["spx_low"]
_tenkan = (_ichi_high.rolling(9).max() + _ichi_low.rolling(9).min()) / 2
_kijun = (_ichi_high.rolling(26).max() + _ichi_low.rolling(26).min()) / 2
_senkou_a = ((_tenkan + _kijun) / 2).shift(26)
_senkou_b = ((_ichi_high.rolling(52).max() + _ichi_low.rolling(52).min()) / 2).shift(26)
f["ichimoku_tenkan"] = _tenkan
f["ichimoku_kijun"] = _kijun
_cloud_top = np.maximum(_senkou_a, _senkou_b)
_cloud_bot = np.minimum(_senkou_a, _senkou_b)
f["ichimoku_cloud_top"] = _cloud_top
f["ichimoku_cloud_bot"] = _cloud_bot

# KST (Know Sure Thing) - multi-timeframe momentum
_roc1 = spx_close.pct_change(10).rolling(10).mean()
_roc2 = spx_close.pct_change(15).rolling(10).mean()
_roc3 = spx_close.pct_change(20).rolling(10).mean()
_roc4 = spx_close.pct_change(30).rolling(15).mean()
_kst = _roc1 + 2 * _roc2 + 3 * _roc3 + 4 * _roc4
f["kst"] = _kst
f["kst_signal"] = _kst.rolling(9).mean()

# Parabolic SAR (AF start=0.02, step=0.02, max=0.20)
_sar = np.full(len(spx_close), np.nan)
_sar_af = 0.02
_sar_af_step = 0.02
_sar_af_max = 0.20
_h = dataset["spx_high"].values
_l = dataset["spx_low"].values
_c = spx_close.values
# Init: first valid bar
_first = 0
while _first < len(_c) and np.isnan(_c[_first]):
    _first += 1
if _first < len(_c) - 1:
    _sar_val = _l[_first]
    _sar_bull = True
    _sar_ep = _h[_first]
    _sar_af_cur = _sar_af
    for i in range(_first + 1, len(_c)):
        if np.isnan(_h[i]) or np.isnan(_l[i]):
            continue
        _sar_val = _sar_val + _sar_af_cur * (_sar_ep - _sar_val)
        if _sar_bull:
            if _l[i] < _sar_val:
                _sar_bull = False
                _sar_val = _sar_ep
                _sar_ep = _l[i]
                _sar_af_cur = _sar_af
            else:
                if _h[i] > _sar_ep:
                    _sar_ep = _h[i]
                    _sar_af_cur = min(_sar_af_cur + _sar_af_step, _sar_af_max)
        else:
            if _h[i] > _sar_val:
                _sar_bull = True
                _sar_val = _sar_ep
                _sar_ep = _h[i]
                _sar_af_cur = _sar_af
            else:
                if _l[i] < _sar_ep:
                    _sar_ep = _l[i]
                    _sar_af_cur = min(_sar_af_cur + _sar_af_step, _sar_af_max)
        _sar[i] = _sar_val
f["parabolic_sar"] = _sar

# CPI YoY inflation
f["cpi_yoy"] = cpi_yoy  # already computed above (~252 trading days)

# Excess CAPE Yield (ECY) = CAPE yield - real rate
f["ecy"] = f["cape_implied_return"] - f["real_rate_10y"]

# Buffett Indicator: Wilshire 5000 / GDP
if "w5000_close" in dataset.columns:
    _gdp_level = pd.read_parquet(DATA_DIR / "fred_gdp.parquet")
    _gdp_level["date"] = pd.to_datetime(_gdp_level["date"])
    _gdp_level = _gdp_level.set_index("date").sort_index()["value"]
    _gdp_daily = _gdp_level.reindex(dataset.index, method="ffill")
    f["buffett_indicator"] = dataset["w5000_close"] / _gdp_daily * 100

# M2 / GDP
if "m2" in dataset.columns:
    if "_gdp_daily" not in dir():
        _gdp_level = pd.read_parquet(DATA_DIR / "fred_gdp.parquet")
        _gdp_level["date"] = pd.to_datetime(_gdp_level["date"])
        _gdp_level = _gdp_level.set_index("date").sort_index()["value"]
        _gdp_daily = _gdp_level.reindex(dataset.index, method="ffill")
    f["m2_gdp_ratio"] = dataset["m2"] / _gdp_daily * 100

# CAPE Z-score & P/E Z-score (expanding to avoid lookahead)
f["cape_zscore"] = (f["cape"] - f["cape"].expanding().mean()) / f["cape"].expanding().std()
f["pe_zscore"] = (f["pe_ratio"] - f["pe_ratio"].expanding().mean()) / f["pe_ratio"].expanding().std()

# Composite valuation = (CAPE_z + PE_z) / 2
f["composite_valuation"] = (f["cape_zscore"] + f["pe_zscore"]) / 2

# CAPE-P/E spread (z-score difference)
f["cape_pe_spread"] = f["cape_zscore"] - f["pe_zscore"]

# Defragment before cross-asset derived columns
f = f.copy()

# Cross-asset ratios
if "gold_close" in dataset.columns:
    f["gold_spx_ratio"] = dataset["gold_close"] / spx_close
if "rut_close" in dataset.columns:
    f["rut_spx_ratio"] = dataset["rut_close"] / spx_close
if "tlt_close" in dataset.columns:
    f["tlt_spx_ratio"] = dataset["tlt_close"] / spx_close
if "copper_close" in dataset.columns and "gold_close" in dataset.columns:
    f["copper_gold_ratio"] = dataset["copper_close"] / dataset["gold_close"]
if "ndx_close" in dataset.columns:
    f["ndx_spx_ratio"] = dataset["ndx_close"] / spx_close
if "hyg_close" in dataset.columns and "lqd_close" in dataset.columns:
    f["hyg_lqd_ratio"] = dataset["hyg_close"] / dataset["lqd_close"]
if "ief_close" in dataset.columns and "tlt_close" in dataset.columns:
    f["ief_tlt_ratio"] = dataset["ief_close"] / dataset["tlt_close"]
if "kbe_close" in dataset.columns:
    f["kbe_close"] = dataset["kbe_close"]
    f["kbe_spx_ratio"] = dataset["kbe_close"] / spx_close

# Cross-asset momentum (12-month = 252 trading days)
if "gold_close" in dataset.columns:
    f["gold_mom_12m"] = dataset["gold_close"].pct_change(252, fill_method=None)
    f["gold_spx_ratio_zscore"] = (f["gold_spx_ratio"] - f["gold_spx_ratio"].expanding().mean()) / f["gold_spx_ratio"].expanding().std()
if "copper_close" in dataset.columns:
    f["copper_mom_12m"] = dataset["copper_close"].pct_change(252, fill_method=None)
if "copper_close" in dataset.columns and "gold_close" in dataset.columns:
    f["copper_gold_zscore"] = (f["copper_gold_ratio"] - f["copper_gold_ratio"].expanding().mean()) / f["copper_gold_ratio"].expanding().std()
    f["copper_gold_mom_12m"] = f["copper_gold_ratio"].pct_change(252, fill_method=None)
if "tlt_close" in dataset.columns:
    f["tlt_spx_zscore"] = (f["tlt_spx_ratio"] - f["tlt_spx_ratio"].expanding().mean()) / f["tlt_spx_ratio"].expanding().std()
if "ndx_close" in dataset.columns:
    f["ndx_spx_mom"] = f["ndx_spx_ratio"].pct_change(252, fill_method=None)
if "rut_close" in dataset.columns:
    f["rut_spx_mom"] = f["rut_spx_ratio"].pct_change(252, fill_method=None)
if "wti_oil" in dataset.columns:
    f["oil_mom_12m"] = dataset["wti_oil"].pct_change(252, fill_method=None)
if "m2" in dataset.columns:
    f["m2_yoy"] = dataset["m2"].pct_change(252, fill_method=None)

# Credit growth YoY (leading indicator - Greenwood et al.)
if "total_bank_credit" in dataset.columns:
    f["credit_growth_yoy"] = dataset["total_bank_credit"].pct_change(252, fill_method=None) * 100
if "business_loans" in dataset.columns:
    f["business_loans_yoy"] = dataset["business_loans"].pct_change(252, fill_method=None) * 100

# Housing YoY (leading indicator - ECB WP1486, Greenwood)
if "case_shiller" in dataset.columns:
    f["housing_yoy"] = dataset["case_shiller"].pct_change(252, fill_method=None) * 100

# Bank stocks momentum (Greenwood R-zone)
if "kbe_close" in dataset.columns:
    f["kbe_spx_mom"] = f["kbe_spx_ratio"].pct_change(252, fill_method=None)

# Building permits YoY
if "building_permits" in dataset.columns:
    f["building_permits_yoy"] = dataset["building_permits"].pct_change(252, fill_method=None) * 100

# Initial claims YoY (inverted - rising = bad)
if "initial_claims" in dataset.columns:
    f["initial_claims_yoy"] = dataset["initial_claims"].pct_change(252, fill_method=None) * 100

# Industrial production YoY
if "industrial_production" in dataset.columns:
    f["indpro_yoy"] = dataset["industrial_production"].pct_change(252, fill_method=None) * 100

# Retail sales YoY
if "retail_sales" in dataset.columns:
    f["retail_sales_yoy"] = dataset["retail_sales"].pct_change(252, fill_method=None) * 100

# --- Derived indicators for consensus signals ---

# Defragment after bulk column assignments
f = f.copy()

# Z-scores (expanding to avoid lookahead)
f["rsi_zscore"] = (f["rsi_14"] - f["rsi_14"].expanding().mean()) / f["rsi_14"].expanding().std()
f["vix_zscore"] = (f["vix_close"] - f["vix_close"].expanding().mean()) / f["vix_close"].expanding().std()
f["yield_curve_zscore"] = (f["yield_curve"] - f["yield_curve"].expanding().mean()) / f["yield_curve"].expanding().std()
f["yc_12m_min"] = f["yield_curve"].rolling(12).min()  # was inverted in last 12 months?
f["credit_spread_zscore"] = (f["credit_spread"] - f["credit_spread"].expanding().mean()) / f["credit_spread"].expanding().std()

# RSI momentum (change over 1 period - rising/falling)
f["rsi_momentum"] = f["rsi_14"].diff()

# Bollinger Bands (20-day, 2 std) - position within band as ratio [-1, +1]
_bb_mid = spx_close.rolling(20).mean()
_bb_std = spx_close.rolling(20).std()
_bb_upper = _bb_mid + 2 * _bb_std
_bb_lower = _bb_mid - 2 * _bb_std
f["bb_position"] = (spx_close - _bb_mid) / (_bb_std * 2)  # 0 = middle, +1 = upper, -1 = lower
f["bb_width"] = (_bb_upper - _bb_lower) / _bb_mid  # bandwidth (volatility proxy)

# Ichimoku: price vs cloud (1 = above, -1 = below, 0 = inside)
_ichi_above = (spx_close > f["ichimoku_cloud_top"]).astype(float)
_ichi_below = (spx_close < f["ichimoku_cloud_bot"]).astype(float)
f["ichimoku_cloud_position"] = _ichi_above - _ichi_below

# Ichimoku: tenkan-kijun cross (tenkan > kijun = bullish)
f["ichimoku_tk_diff"] = f["ichimoku_tenkan"] - f["ichimoku_kijun"]

# KST: difference from signal (>0 = bullish)
f["kst_diff"] = f["kst"] - f["kst_signal"]

# Parabolic SAR: price above SAR = bullish
f["sar_bullish"] = (spx_close > f["parabolic_sar"]).astype(float)

# Drawdown regime (current DD from ATH, already have spx_drawdown_from_ath)
# Nothing to add, column exists

# Realized vol z-score
f["realized_vol_zscore"] = (f["realized_vol"] - f["realized_vol"].expanding().mean()) / f["realized_vol"].expanding().std()

# Buffett indicator z-score
if "buffett_indicator" in f.columns:
    f["buffett_zscore"] = (f["buffett_indicator"] - f["buffett_indicator"].expanding().mean()) / f["buffett_indicator"].expanding().std()

# ECY z-score (excess CAPE yield)
if "ecy" in f.columns:
    f["ecy_zscore"] = (f["ecy"] - f["ecy"].expanding().mean()) / f["ecy"].expanding().std()
    f["ecy_detrend_5y"] = f["ecy"] - f["ecy"].rolling(60).mean()

# Net liquidity proxy = Fed BS - TGA - RRP
if all(c in dataset.columns for c in ["fed_bs", "tga", "rrp"]):
    f["net_liquidity"] = dataset["fed_bs"] - dataset["tga"] - dataset["rrp"]
    f["net_liquidity_yoy"] = f["net_liquidity"].pct_change(252, fill_method=None)
    f["net_liquidity_zscore"] = (f["net_liquidity"] - f["net_liquidity"].expanding().mean()) / f["net_liquidity"].expanding().std()

# NFCI z-score (already normalized but expanding z-score captures drift)
if "nfci" in dataset.columns:
    f["nfci_zscore"] = (f["nfci"] - f["nfci"].expanding().mean()) / f["nfci"].expanding().std()

# ANFCI z-score
if "anfci" in dataset.columns:
    f["anfci_zscore"] = (f["anfci"] - f["anfci"].expanding().mean()) / f["anfci"].expanding().std()

# Dollar major z-score
if "dollar_major" in dataset.columns:
    f["dollar_major_zscore"] = (f["dollar_major"] - f["dollar_major"].expanding().mean()) / f["dollar_major"].expanding().std()

# MOVE z-score (bond volatility)
if "move_close" in dataset.columns:
    f["move_zscore"] = (f["move_close"] - f["move_close"].expanding().mean()) / f["move_close"].expanding().std()

# Fed funds rate z-score
if "fed_funds_rate" in f.columns:
    f["fed_funds_zscore"] = (f["fed_funds_rate"] - f["fed_funds_rate"].expanding().mean()) / f["fed_funds_rate"].expanding().std()

# Fed funds peak ratio (current / 12m rolling max)
if "fed_funds_rate" in f.columns:
    f["ff_peak_ratio"] = f["fed_funds_rate"] / f["fed_funds_rate"].rolling(12).max()

# Unemployment 6m momentum (clipped ±3% for COVID robustness)
if "unemployment" in f.columns:
    f["unemp_mom_6m"] = f["unemployment"].diff(126)  # ~6 months in trading days

# Consumer sentiment 12m change
if "consumer_sentiment" in f.columns:
    f["sentiment_mom_12m"] = f["consumer_sentiment"].diff(12)

# Initial claims z-score
if "initial_claims" in f.columns:
    f["initial_claims_zscore"] = (f["initial_claims"] - f["initial_claims"].expanding().mean()) / f["initial_claims"].expanding().std()

# Housing z-score
if "case_shiller" in dataset.columns:
    f["housing_zscore"] = (dataset["case_shiller"] - dataset["case_shiller"].expanding().mean()) / dataset["case_shiller"].expanding().std()

# Credit growth z-score
if "total_bank_credit" in dataset.columns:
    _cg = dataset["total_bank_credit"].pct_change(252, fill_method=None) * 100
    f["credit_growth_zscore"] = (_cg - _cg.expanding().mean()) / _cg.expanding().std()

# Bank/SPX z-score
if "kbe_close" in dataset.columns:
    f["kbe_spx_zscore"] = (f["kbe_spx_ratio"] - f["kbe_spx_ratio"].expanding().mean()) / f["kbe_spx_ratio"].expanding().std()

print(f"  Added derived indicators for consensus signals")

# Identify late-starting columns (for cutoff version)
late_cols = []
for col in f.columns:
    fv = f[col].first_valid_index()
    if fv is None:
        late_cols.append((col, fv))
    elif fv > CUTOFF:
        late_cols.append((col, fv))
if late_cols:
    label = f"after {CUTOFF.date()}"
    print(f"  {len(late_cols)} columns start {label}:")
    for col, fv in late_cols:
        print(f"    {col}: first valid = {fv.date() if fv else 'NONE'}")

# f_full = all features (with NaN for late starters)
# f      = cutoff version (late columns dropped, rows trimmed)
f_full = f.copy()

f = f.drop(columns=[c for c, _ in late_cols])
feature_mask = f.notna().all(axis=1)
first_complete = feature_mask.idxmax() if feature_mask.any() else None
if first_complete is not None:
    f = f[f.index >= first_complete]
    dataset = dataset.reindex(f.index)
    spx_close = dataset["spx_close"]

print(f"  Features (cutoff):  {len(f.columns)} columns")
print(f"  Features (full):    {len(f_full.columns)} columns")


# PART 3 - Targets

print("=== PART 3: Targets ===")

tbill_daily_full = (1 + dataset_raw["tbill_rate"] / 100) ** (1 / 252) - 1


def _tbill_cum_return_daily(tbill_src, index, h):
    """Compounded T-Bill return over h trading days (forward-looking)."""
    tbill_arr = tbill_src.reindex(index).values
    n = len(index)
    result = np.full(n, np.nan)
    for i in range(n - h):
        cum = 1.0
        valid = True
        for j in range(h):
            r = tbill_arr[i + j]
            if np.isnan(r):
                valid = False
                break
            cum *= (1 + r)
        if valid:
            result[i] = cum - 1
    return pd.Series(result, index=index)


def _max_drawdown_fwd(spx_col, h):
    """Max drawdown over next h periods: min(future prices) / current - 1."""
    n = len(spx_col)
    vals = spx_col.values
    result = np.full(n, np.nan)
    for i in range(n - h):
        cur = vals[i]
        if np.isnan(cur) or cur == 0:
            continue
        future_min = np.nanmin(vals[i + 1: i + h + 1])
        result[i] = future_min / cur - 1
    return pd.Series(result, index=spx_col.index)


def add_daily_targets(df, spx_col, tbill_src):
    """Build SPX vs T-Bills + drawdown target columns; return as dict."""
    new_cols = {}
    for h, suffix in HORIZONS_DAYS.items():
        # SPX vs T-Bills
        spx_fwd = spx_col.shift(-h) / spx_col - 1
        tbill_cum = _tbill_cum_return_daily(tbill_src, df.index, h)
        t = (spx_fwd > tbill_cum).astype(float)
        t[spx_fwd.isna() | tbill_cum.isna()] = np.nan
        label = f"target_spx_vs_tbill_{suffix}"
        new_cols[label] = t
        valid = t.dropna()
        print(f"    {label}: {valid.sum():.0f}/{len(valid)} positive ({valid.mean()*100:.1f}%)")

        # Price up (SPX in h days > SPX now)
        t_up = (spx_fwd > 0).astype(float)
        t_up[spx_fwd.isna()] = np.nan
        label_up = f"target_price_up_{suffix}"
        new_cols[label_up] = t_up
        valid_up = t_up.dropna()
        print(f"    {label_up}: {valid_up.sum():.0f}/{len(valid_up)} positive ({valid_up.mean()*100:.1f}%)")

        # Drawdown targets
        dd = _max_drawdown_fwd(spx_col, h)
        for thresh in DRAWDOWN_THRESHOLDS:
            pct = int(thresh * 100)
            t_dd = (dd > -thresh).astype(float)  # 1 = no drawdown exceeding threshold
            t_dd[dd.isna()] = np.nan
            label_dd = f"target_no_dd_{pct}pct_{suffix}"
            new_cols[label_dd] = t_dd
            valid_dd = t_dd.dropna()
            print(f"    {label_dd}: {valid_dd.sum():.0f}/{len(valid_dd)} positive ({valid_dd.mean()*100:.1f}%)")
    return new_cols


def add_meta_cols(spx_close_s, spx_open_s, tbill_s):
    """Return price/rate columns for backtest as dict."""
    return {"spx_close": spx_close_s, "spx_open": spx_open_s, "tbill_rate": tbill_s}


# --- Cutoff version ---
print("  [cutoff]")
_tgt = add_daily_targets(f, spx_close, tbill_daily_full)
_meta = add_meta_cols(spx_close, dataset["spx_open"], dataset["tbill_rate"])
f = pd.concat([f, pd.DataFrame({**_tgt, **_meta}, index=f.index)], axis=1)

# --- Full version ---
print("  [full]")
spx_close_full = dataset_raw["spx_close"].reindex(f_full.index)
_tgt_full = add_daily_targets(f_full, spx_close_full, tbill_daily_full)
_meta_full = add_meta_cols(spx_close_full, dataset_raw["spx_open"].reindex(f_full.index),
                           dataset_raw["tbill_rate"].reindex(f_full.index))
f_full = pd.concat([f_full, pd.DataFrame({**_tgt_full, **_meta_full}, index=f_full.index)], axis=1)


# Resample helper

def resample_and_save(daily_df, freq, freq_label, horizons, horizon_suffix, out_path, full=False):
    """Resample daily dataset to monthly/weekly, add targets, save."""
    feat_cols = [c for c in daily_df.columns
                 if not c.startswith("target_") and c not in ("spx_close", "tbill_rate", "spx_open")]

    rule = "ME" if freq == "monthly" else "W-FRI"

    rf = daily_df[feat_cols].resample(rule).last()

    # SPX OHLCV: proper aggregation
    if "spx_high" in daily_df.columns:
        rf["spx_high"] = daily_df["spx_high"].resample(rule).max()
    if "spx_low" in daily_df.columns:
        rf["spx_low"] = daily_df["spx_low"].resample(rule).min()
    if "spx_volume" in daily_df.columns:
        rf["spx_volume"] = daily_df["spx_volume"].resample(rule).sum()

    # Meta
    rf["spx_close"] = daily_df["spx_close"].resample(rule).last()
    rf["spx_open"] = daily_df["spx_open"].resample(rule).first()
    rf["tbill_rate"] = daily_df["tbill_rate"].resample(rule).last()

    # Drop incomplete last period (e.g. mid-month data resample'd as if month-end)
    last_daily = daily_df.index[-1]
    last_period_end = rf.index[-1]
    if last_daily < last_period_end:
        rf = rf.iloc[:-1]

    # Targets
    r_spx = rf["spx_close"]
    periods_per_year = 12 if freq == "monthly" else 52
    r_tbill = (1 + rf["tbill_rate"] / 100) ** (1 / periods_per_year) - 1

    for h in horizons:
        # SPX vs T-Bills
        spx_fwd = r_spx.shift(-h) / r_spx - 1
        tbill_cum = pd.Series(np.nan, index=rf.index)
        tbill_arr = r_tbill.values
        for i in range(len(rf.index) - h):
            cum = 1.0
            valid = True
            for j in range(h):
                r = tbill_arr[i + j]
                if np.isnan(r):
                    valid = False
                    break
                cum *= (1 + r)
            if valid:
                tbill_cum.iloc[i] = cum - 1
        t = (spx_fwd > tbill_cum).astype(float)
        t[spx_fwd.isna() | tbill_cum.isna()] = np.nan
        label = f"target_spx_vs_tbill_{h}{horizon_suffix}"
        rf[label] = t
        valid = t.dropna()
        print(f"    {label}: {valid.sum():.0f}/{len(valid)} positive ({valid.mean()*100:.1f}%)")

        # Price up
        t_up = (spx_fwd > 0).astype(float)
        t_up[spx_fwd.isna()] = np.nan
        label_up = f"target_price_up_{h}{horizon_suffix}"
        rf[label_up] = t_up
        valid_up = t_up.dropna()
        print(f"    {label_up}: {valid_up.sum():.0f}/{len(valid_up)} positive ({valid_up.mean()*100:.1f}%)")

        # Drawdown targets
        dd = _max_drawdown_fwd(r_spx, h)
        for thresh in DRAWDOWN_THRESHOLDS:
            pct = int(thresh * 100)
            t_dd = (dd > -thresh).astype(float)
            t_dd[dd.isna()] = np.nan
            label_dd = f"target_no_dd_{pct}pct_{h}{horizon_suffix}"
            rf[label_dd] = t_dd
            valid_dd = t_dd.dropna()
            print(f"    {label_dd}: {valid_dd.sum():.0f}/{len(valid_dd)} positive ({valid_dd.mean()*100:.1f}%)")

    r_feat = [c for c in rf.columns
              if not c.startswith("target_") and c not in ("spx_close", "tbill_rate", "spx_open")]
    r_tgt = [c for c in rf.columns if c.startswith("target_")]

    if full:
        # Full: only drop rows where ALL targets are NaN (keep feature NaN)
        r_tgt = [c for c in rf.columns if c.startswith("target_")]
        rf_out = rf.dropna(subset=r_tgt, how="all")
    else:
        # Cutoff: drop rows with NaN in features, but allow target NaN at the end
        rf_out = rf.dropna(subset=r_feat)

    print(f"\n{'='*60}")
    print(f"{freq_label}:  {rf_out.shape[0]} rows x {rf_out.shape[1]} columns")
    print(f"Features: {len(r_feat)}")
    print(f"Targets:  {r_tgt}")
    print(f"Range:    {rf_out.index.min()} -> {rf_out.index.max()}")
    print(f"{'='*60}")

    rf_out.to_parquet(out_path)
    print(f"Saved to {out_path}")

    return rf_out


# Summary & save daily
feature_cols = [c for c in f.columns
                if not c.startswith("target_") and c not in ("spx_close", "tbill_rate", "spx_open")]
target_cols = [c for c in f.columns if c.startswith("target_")]

print(f"\n{'='*60}")
print(f"Daily (cutoff):  {f.shape[0]} rows x {f.shape[1]} columns")
print(f"Features: {len(feature_cols)}")
print(f"Targets:  {target_cols}")
print(f"Range:    {f.index.min()} -> {f.index.max()}")
print(f"{'='*60}")

f = f.dropna(subset=feature_cols)
f.to_parquet(OUT_PATH)
print(f"Saved to {OUT_PATH}")

# Full daily (keep feature NaN, only drop target NaN)
full_feature_cols = [c for c in f_full.columns
                     if not c.startswith("target_") and c not in ("spx_close", "tbill_rate", "spx_open")]
full_target_cols = [c for c in f_full.columns if c.startswith("target_")]
FULL_DAILY_OUT = DATA_DIR / "dataset_daily_full.parquet"
f_full_out = f_full.dropna(subset=full_target_cols, how="all")
f_full_out.to_parquet(FULL_DAILY_OUT)

print(f"\n{'='*60}")
print(f"Daily (full):  {f_full_out.shape[0]} rows x {f_full_out.shape[1]} columns")
print(f"Features: {len(full_feature_cols)}")
print(f"Range:    {f_full_out.index.min()} -> {f_full_out.index.max()}")
print(f"{'='*60}")
print(f"Saved to {FULL_DAILY_OUT}")


# PART 4 - Resample to monthly & weekly (cutoff + full)

print("\n=== PART 4: Monthly resample ===")
print("  [cutoff]")
resample_and_save(f, "monthly", "Monthly (cutoff)", [1, 3, 6], "m", DATA_DIR / "dataset_monthly.parquet")
print("\n  [full]")
resample_and_save(f_full, "monthly", "Monthly (full)", [1, 3, 6], "m", DATA_DIR / "dataset_monthly_full.parquet", full=True)

print("\n=== PART 5: Weekly resample ===")
print("  [cutoff]")
resample_and_save(f, "weekly", "Weekly (cutoff)", [4, 13, 26], "w", DATA_DIR / "dataset_weekly.parquet")
print("\n  [full]")
resample_and_save(f_full, "weekly", "Weekly (full)", [4, 13, 26], "w", DATA_DIR / "dataset_weekly_full.parquet", full=True)


# Coverage check: verify all parquet files in datas/ are used

print("\n=== Data inventory ===")

# Map parquet file -> role in the pipeline
ALL_USED_FILES = {}
ALL_USED_FILES["gspc.parquet"] = "backbone"
for fn in daily_yahoo:
    ALL_USED_FILES[fn] = "daily_yahoo"
for fn in daily_fred:
    ALL_USED_FILES[fn] = "daily_fred"
for fn in monthly_sources:
    ALL_USED_FILES[fn] = "monthly"
ALL_USED_FILES["fred_gdp.parquet"] = "monthly"
ALL_USED_FILES["datahub_gold_monthly.parquet"] = "merge_input"
ALL_USED_FILES["datahub_copper_monthly.parquet"] = "merge_input"
ALL_USED_FILES["gold.parquet"] = "merge_input"
ALL_USED_FILES["copper.parquet"] = "merge_input"
ALL_USED_FILES["merged_gold.parquet"] = "daily_yahoo"
ALL_USED_FILES["merged_copper.parquet"] = "daily_yahoo"

# Collect info for all parquet files
all_parquets = sorted(p.name for p in DATA_DIR.glob("*.parquet") if "dataset" not in p.name)

# Map parquet -> feature column name
file_to_feature = {}
file_to_feature["gspc.parquet"] = "spx_close"
for fn, col in {**daily_yahoo, **daily_fred, **monthly_sources}.items():
    file_to_feature[fn] = col
file_to_feature["fred_gdp.parquet"] = "gdp_growth"

# Dropped columns (from cutoff filter above)
dropped_col_names = {c for c, _ in late_cols}

# Print table
print(f"{'file':<42} {'start':>12} {'end':>12} {'feature':<28} {'status'}")
print("-" * 110)

n_kept, n_dropped, n_merge, n_unused = 0, 0, 0, 0
for pf in all_parquets:
    role = ALL_USED_FILES.get(pf)

    try:
        tmp = pd.read_parquet(DATA_DIR / pf)
        if "date" in tmp.columns:
            dates = pd.to_datetime(tmp["date"])
            d_start, d_end = dates.min().date(), dates.max().date()
        elif tmp.index.name == "date" or tmp.index.dtype == "datetime64[ns]":
            tmp.index = pd.to_datetime(tmp.index)
            d_start, d_end = tmp.index.min().date(), tmp.index.max().date()
        else:
            d_start, d_end = "?", "?"
    except Exception:
        d_start, d_end = "?", "?"

    feat = file_to_feature.get(pf, "")

    if role is None:
        status = "UNUSED"
        n_unused += 1
    elif role == "merge_input":
        status = "merge_input"
        n_merge += 1
    elif feat and feat in dropped_col_names:
        status = f"full only (cutoff {CUTOFF.date()})"
        n_dropped += 1
    else:
        status = "cutoff + full"
        n_kept += 1

    print(f"  {pf:<40} {str(d_start):>12} {str(d_end):>12}   {feat:<26} {status}")

print(f"\nSummary: {n_kept} in both, {n_dropped} full only (after cutoff), {n_merge} merge inputs, {n_unused} unused")
print(f"Cutoff: {CUTOFF.date()}")
print("Done.")
