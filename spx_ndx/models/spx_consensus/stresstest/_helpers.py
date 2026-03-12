"""Shared helpers for the stresstest module."""

import json

import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for NumPy types."""

    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_stresstest_data(parquet_path):
    """Load signal parquet and return (df, signal, strategy_returns, buy_hold_returns, cash_returns, periods_per_year, label).

    The parquet must have columns: signal, strategy_returns, buy_hold_returns, cash_returns.
    Index must be a DatetimeIndex.
    """
    df = pd.read_parquet(parquet_path)
    signal = df["signal"].values
    strategy_returns = df["strategy_returns"].values
    buy_hold_returns = df["buy_hold_returns"].values
    cash_returns = df["cash_returns"].values

    n_periods = len(signal)
    days = (df.index[-1] - df.index[0]).days
    periods_per_year = round(n_periods / (days / 365.25))

    label = (
        parquet_path.split("/")[-1]
        .replace("wf_signals_", "")
        .replace("wf_signals", "consensus")
        .replace(".parquet", "")
    )

    return df, signal, strategy_returns, buy_hold_returns, cash_returns, periods_per_year, label


def build_baseline_signals(dataset_path, df_index):
    """Build simple baseline signals aligned to OOS dates.

    Returns dict of {name: array} for correlation analysis.
    """
    dataset = pd.read_parquet(dataset_path)
    dataset.index = pd.to_datetime(dataset.index)
    spx = dataset["spx_close"]

    def _make_signal(series, threshold, direction):
        if direction == "above":
            signal_series = (series > threshold).astype(float)
        else:
            signal_series = (series < threshold).astype(float)
        out = signal_series.shift(1).reindex(df_index).ffill()
        if out.isna().any():
            return None
        return out.values

    signals = {}

    # Trend
    sma10 = spx.rolling(10).mean()
    result = _make_signal(spx - sma10, 0, "above")
    if result is not None:
        signals["SMA 10m"] = result

    momentum_12m = spx.pct_change(12)
    result = _make_signal(momentum_12m, 0, "above")
    if result is not None:
        signals["12M mom"] = result

    if "spx_ema200_ratio" in dataset.columns:
        result = _make_signal(dataset["spx_ema200_ratio"], 0, "above")
        if result is not None:
            signals["EMA200"] = result

    # Volatility
    if "vix_close" in dataset.columns:
        result = _make_signal(dataset["vix_close"], 30, "below")
        if result is not None:
            signals["VIX<30"] = result

    # Momentum oscillator
    if "rsi_14" in dataset.columns:
        result = _make_signal(dataset["rsi_14"], 70, "below")
        if result is not None:
            signals["RSI<70"] = result

    # Macro
    if "cpi_yoy" in dataset.columns:
        result = _make_signal(dataset["cpi_yoy"], 4, "below")
        if result is not None:
            signals["CPI<4"] = result

    # Valuation
    if "cape_zscore" in dataset.columns:
        result = _make_signal(dataset["cape_zscore"], 1.5, "below")
        if result is not None:
            signals["CAPE z<1.5"] = result

    if "composite_valuation" in dataset.columns:
        result = _make_signal(dataset["composite_valuation"], 1, "below")
        if result is not None:
            signals["Compo<1"] = result

    return signals
