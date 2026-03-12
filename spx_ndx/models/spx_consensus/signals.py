"""Signal generation - raw indicators, thresholding, and signal computation."""

import numpy as np


# Thresholding

def threshold_signal(series, threshold, direction="below"):
    """Signal = 1 when series crosses threshold.

    direction: "below" (1 when < threshold) or "above" (1 when >).
    Returns Series of 0/1, NaN preserved.
    """
    if direction == "below":
        signal = (series < threshold).astype(float)
    else:
        signal = (series > threshold).astype(float)
    signal[series.isna()] = np.nan
    return signal


def band_signal(series, lower, upper):
    """Signal = 1 when lower < series < upper. NaN preserved."""
    signal = ((series > lower) & (series < upper)).astype(float)
    signal[series.isna()] = np.nan
    return signal


# Signal registry - declarative handler per template
# Each handler: (dataset, closes, value) -> Series or None

def _sma(dataset, closes, value):
    column_name = f"sma_{value}m"
    return threshold_signal(closes - dataset[column_name], 0, "above") if column_name in dataset.columns else None

def _vp_val(dataset, closes, value):
    column_name = f"vp_val_{value}m"
    return threshold_signal(closes - dataset[column_name], 0, "above") if column_name in dataset.columns else None

def _col_threshold(column_name, direction, transform=None):
    """Factory for simple column threshold handlers."""
    def handler(dataset, closes, value):
        if column_name not in dataset.columns:
            return None
        transformed = transform(value) if transform else value
        return threshold_signal(dataset[column_name], transformed, direction)
    return handler

def _col_band(column_name, lower):
    def handler(dataset, closes, value):
        return band_signal(dataset[column_name], lower, value) if column_name in dataset.columns else None
    return handler


HANDLERS = {
    "SMA Xm":      _sma,
    "VIX<X":        _col_threshold("vix_close", "below"),
    "RSI<X":        _col_threshold("rsi_14", "below"),
    "RSI>X":        _col_threshold("rsi_14", "above"),
    "RSI 30-X":     _col_band("rsi_14", 30),
    "MACDhist>X":   _col_threshold("macd_hist", "above"),
    "MACDline>X":   _col_threshold("macd_line", "above"),
    "CAPE z<X":     _col_threshold("cape_zscore", "below"),
    "PE z<X":       _col_threshold("pe_zscore", "below"),
    "Compo<X":      _col_threshold("composite_valuation", "below"),
    "CAPE-PE<X":    _col_threshold("cape_pe_spread", "below"),
    "CPI<X":        _col_threshold("cpi_yoy", "below"),
    "EMA200>X":     _col_threshold("spx_ema200_ratio", "above", transform=lambda value: value / 100),
    "VP>VAL Xm":    _vp_val,
    "Gold/SPX z<X": _col_threshold("gold_spx_ratio_zscore", "below"),
    "YC z>X":       _col_threshold("yield_curve_zscore", "above"),
    "YCmin>X":      _col_threshold("yc_12m_min", "above"),
    "Credit z<X":   _col_threshold("credit_spread_zscore", "below"),
    "EBP<X":        _col_threshold("ebp", "below"),
    "CuAu z>X":     _col_threshold("copper_gold_zscore", "above"),
    "RVol z<X":     _col_threshold("realized_vol_zscore", "below"),
    "BB>X":         _col_threshold("bb_position", "above"),
    "Ichi>X":       _col_threshold("ichimoku_cloud_position", "above"),
    "RSI z<X":      _col_threshold("rsi_zscore", "below"),
    "BBw<X":        _col_threshold("bb_width", "below"),
    "SAR>X":        _col_threshold("sar_bullish", "above"),
    "KST>X":        _col_threshold("kst_diff", "above"),
    "VolSpread<X":  _col_threshold("vol_spread", "below"),
    "Buffett z<X":  _col_threshold("buffett_zscore", "below"),
    "ECY z>X":      _col_threshold("ecy_zscore", "above"),
    "ECYdt>X":      _col_threshold("ecy_detrend_5y", "above"),
    "EY-10Y>X":     _col_threshold("ey_10y_spread", "above"),
    "Unemp<X":      _col_threshold("unemployment", "below"),
    "UnempMom<X":   _col_threshold("unemp_mom_6m", "below"),
    "SentMom>X":    _col_threshold("sentiment_mom_12m", "above"),
    "M2<X":         _col_threshold("m2_yoy", "below"),
    "NFCI<X":       _col_threshold("nfci", "below"),
    "ANFCI z<X":    _col_threshold("anfci_zscore", "below"),
    "USD z<X":      _col_threshold("dollar_major_zscore", "below"),
    "CuAu mom>X":   _col_threshold("copper_gold_mom_12m", "above"),
    "Oil mom>X":    _col_threshold("oil_mom_12m", "above"),
    "RUT/SPX mom>X": _col_threshold("rut_spx_mom", "above"),
    "NDX/SPX mom>X": _col_threshold("ndx_spx_mom", "above"),
    "VIX z<X":      _col_threshold("vix_zscore", "below"),
    "DD>X":         _col_threshold("spx_drawdown_from_ath", "above"),
    # --- Literature-validated crisis predictors ---
    "CreditGr z<X": _col_threshold("credit_growth_zscore", "below"),
    "Housing z<X":  _col_threshold("housing_zscore", "below"),
    "BankMom>X":    _col_threshold("kbe_spx_mom", "above"),
    "Permits>X":    _col_threshold("building_permits_yoy", "above"),
    "Claims z<X":   _col_threshold("initial_claims_zscore", "below"),
    "IndPro>X":     _col_threshold("indpro_yoy", "above"),
    "RetailS>X":    _col_threshold("retail_sales_yoy", "above"),
    "FFpeak>X":     _col_threshold("ff_peak_ratio", "above"),
    "NetLiq z>X":   _col_threshold("net_liquidity_zscore", "above"),
    "MOVE z<X":     _col_threshold("move_zscore", "below"),
}


# Signal computation from indicators

def _make_name(template, value):
    """Replace last 'X' in template with the parameter value."""
    if value is None:
        return template
    last_valid_index = template.rfind("X")
    return template[:last_valid_index] + str(value) + template[last_valid_index+1:]


def compute_signals(dataset, closes, indicators):
    """Derive binary signals from a dataset and indicator config.

    Args:
        dataset: DataFrame with columns like vix_close, rsi_14, etc.
        closes: Series of SPX closing prices.
        indicators: Dict mapping template strings to parameter lists.

    Returns:
        Dict of {signal_name: Series of 0/1}.
    """
    signals_dict = {}
    for template, params in indicators.items():
        handler = HANDLERS.get(template)
        if handler is None:
            continue
        for value in ([None] if params is None else params):
            result = handler(dataset, closes, value)
            if result is not None:
                signals_dict[_make_name(template, value)] = result
    return signals_dict


def build_sig_matrix(signals, signal_names, index):
    """Build a (T x n_sig) matrix of shifted/filled signals.

    Each signal is ffilled, NaN->1, shifted by 1 (use previous month's value),
    then reindexed to the target index.
    """
    return np.column_stack([
        signals[name].ffill().fillna(1).shift(1).fillna(1).reindex(index).values
        for name in signal_names
    ]).astype(np.float64)
