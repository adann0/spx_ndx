"""Numba-accelerated grid search for trader/group selection."""

from itertools import combinations

import numba
import numpy as np


_SORT_BY_DEFAULT = ("stability", "cagr")


def make_configs(n_items, min_size, max_size):
    """Generate all (min_votes, combo) configurations as a 2D int array.

    Each row: [min_votes, idx_0, idx_1, ..., -1 padding].
    """
    rows = []
    for size in range(min_size, max_size + 1):
        for combo in combinations(range(n_items), size):
            for min_votes in range(1, size + 1):
                row = [min_votes] + list(combo) + [-1] * (max_size - size)
                rows.append(row)
    if not rows:
        return np.empty((0, 1 + max_size), dtype=np.int64)
    return np.array(rows, dtype=np.int64)


@numba.njit(parallel=True, cache=True, fastmath=True)
def _eval_all(matrix, returns, cash, configs, periods_per_year):
    """Evaluate all configs in parallel. Returns (rtrs, cagrs, stabs) arrays."""
    ppy_sqrt = periods_per_year ** 0.5
    T = matrix.shape[0]
    nc = configs.shape[0]
    mc = configs.shape[1] - 1
    rtrs = np.empty(nc, dtype=np.float64)
    cagrs = np.empty(nc, dtype=np.float64)
    stabs = np.empty(nc, dtype=np.float64)
    ss_xx = T * (T - 1) * (2 * T - 1) / 6.0 - T * ((T - 1) / 2.0) ** 2
    for ci in numba.prange(nc):
        mv = configs[ci, 0]
        sz = 0
        for j in range(mc):
            if configs[ci, 1 + j] >= 0:
                sz += 1
            else:
                break
        s_sum = 0.0
        s_sq = 0.0
        cum = 1.0
        sum_y = 0.0
        sum_y2 = 0.0
        sum_xy = 0.0
        for t in range(T):
            votes = 0.0
            for k in range(sz):
                votes += matrix[t, configs[ci, 1 + k]]
            sig = 1.0 if votes >= mv else 0.0
            r = sig * returns[t] + (1.0 - sig) * cash[t]
            s_sum += r
            s_sq += r * r
            cum *= (1.0 + r)
            if cum > 0.0:
                log_cum = np.log(cum)
            else:
                log_cum = -50.0
            sum_y += log_cum
            sum_y2 += log_cum * log_cum
            sum_xy += t * log_cum
        mean = s_sum / T
        var = s_sq / T - mean * mean
        if T > 1:
            var = var * T / (T - 1)
        if var < 1e-24:
            rtrs[ci] = 0.0
        else:
            rtrs[ci] = mean / var ** 0.5 * ppy_sqrt
        n_years = T / periods_per_year
        if cum > 0.0 and n_years > 0.0:
            cagrs[ci] = cum ** (1.0 / n_years) - 1.0
        else:
            cagrs[ci] = -1.0
        mean_y = sum_y / T
        ss_yy = sum_y2 - T * mean_y * mean_y
        ss_xy_val = sum_xy - T * ((T - 1) / 2.0) * mean_y
        if ss_xx < 1e-24 or ss_yy < 1e-24:
            stabs[ci] = 1.0
        else:
            r_val = ss_xy_val / (ss_xx ** 0.5 * ss_yy ** 0.5)
            stabs[ci] = r_val * r_val
    return rtrs, cagrs, stabs


def grid_search(matrix, returns, cash, configs, top_n, periods_per_year,
                min_rtr, min_cagr, cached_metrics=None, sort_by=None):
    """Filter and rank configs by sort_by keys. Returns top_n results.

    Returns:
        results: list of (rtr, cagr, stability, min_votes, combo) tuples
        n_total: total configs evaluated
        n_pass: configs passing filters
    """
    if sort_by is None:
        sort_by = _SORT_BY_DEFAULT
    if len(configs) == 0:
        return [], 0, 0
    if cached_metrics is not None:
        rtrs, cagrs, stabs = cached_metrics
    else:
        rtrs, cagrs, stabs = _eval_all(matrix, returns, cash, configs, periods_per_year)
    n_total = len(configs)
    mask = (rtrs >= min_rtr) & (cagrs >= min_cagr)
    n_pass = int(mask.sum())
    if n_pass == 0:
        return [], n_total, 0
    passing_indices = np.where(mask)[0]
    metrics = {"rtr": rtrs[passing_indices], "cagr": cagrs[passing_indices], "stability": stabs[passing_indices]}
    primary = metrics[sort_by[0]]
    secondary = metrics[sort_by[1]]
    order = np.lexsort((secondary, primary))
    if len(passing_indices) > top_n:
        top_idx = order[-top_n:][::-1]
    else:
        top_idx = order[::-1]
    max_size = configs.shape[1] - 1
    results = []
    for top_index in top_idx:
        config_index = passing_indices[top_index]
        min_votes = int(configs[config_index, 0])
        size = 0
        for slot in range(max_size):
            if configs[config_index, 1 + slot] >= 0:
                size += 1
            else:
                break
        combo = tuple(int(configs[config_index, 1 + slot]) for slot in range(size))
        results.append((float(rtrs[config_index]), float(cagrs[config_index]), float(stabs[config_index]), min_votes, combo))
    return results, n_total, n_pass


def warmup():
    """JIT warmup - call once at import time."""
    dummy_cfg = make_configs(3, 2, 2)
    _eval_all(
        np.ones((2, 3), dtype=np.float64),
        np.ones(2, dtype=np.float64),
        np.ones(2, dtype=np.float64),
        dummy_cfg, 12.0,
    )
