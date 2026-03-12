#!/usr/bin/env python3

"""
check_dataset.py - Validate a dataset parquet against the framework contract.

Rules:
    - Index: DatetimeIndex, sorted, no duplicates.
    - Features (non target_* columns): zero NaN.
    - Targets (target_* columns): trailing NaN only (no NaN at start or middle).
    - Utility columns (spx_close, spx_open, tbill_rate): excluded from checks.

Usage:
    python scripts/check_dataset.py datas/dataset_monthly.parquet
"""

import sys
import pandas as pd

def check_dataset(path):
    """Run all checks on a parquet file. Returns (errors, warnings)."""
    errors = []
    warnings = []

    # --- Load ---
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return [f"Cannot read parquet: {e}"], []

    n_rows, n_cols = df.shape

    # --- Index checks ---
    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append(f"Index is {type(df.index).__name__}, expected DatetimeIndex")
    else:
        if not df.index.is_monotonic_increasing:
            errors.append("Index is not sorted (must be monotonic increasing)")
        dupes = df.index.duplicated().sum()
        if dupes > 0:
            errors.append(f"Index has {dupes} duplicate dates")

    # --- Column classification ---
    target_cols = [c for c in df.columns if c.startswith("target_")]
    feature_cols = [c for c in df.columns if not c.startswith("target_")]

    if not target_cols:
        errors.append("No target columns found (expected target_* columns)")
    if not feature_cols:
        errors.append("No feature columns found")

    # --- Feature checks: zero NaN ---
    for col in feature_cols:
        n_null = df[col].isna().sum()
        if n_null > 0:
            errors.append(f"Feature '{col}' has {n_null} NaN ({n_null/n_rows*100:.1f}%)")

    # --- Target checks: trailing NaN only ---
    for col in target_cols:
        nulls = df[col].isna()
        if not nulls.any():
            continue  # no NaN at all - OK

        # Find first NaN position
        first_nan_idx = nulls.values.argmax()  # first True

        # All values after first NaN must also be NaN
        tail = nulls.values[first_nan_idx:]
        non_null_after = (~tail).sum()
        if non_null_after > 0:
            errors.append(
                f"Target '{col}' has non-trailing NaN "
                f"(first NaN at row {first_nan_idx}, "
                f"but {non_null_after} non-NaN values follow)")

        # Count trailing NaN
        n_trailing = nulls.sum()
        if n_trailing == n_rows:
            errors.append(f"Target '{col}' is entirely NaN")

    # --- Warnings ---
    for col in feature_cols:
        if df[col].nunique() <= 1:
            warnings.append(f"Feature '{col}' is constant (nunique={df[col].nunique()})")

    return errors, warnings


def print_report(path, df, errors, warnings):
    """Print a human-readable report."""
    target_cols = [c for c in df.columns if c.startswith("target_")]
    feature_cols = [c for c in df.columns if not c.startswith("target_")]

    print(f"\n{'='*60}")
    print(f"Dataset check: {path}")
    print(f"{'='*60}")
    print(f"  Rows:     {len(df)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Targets:  {len(target_cols)} ({', '.join(target_cols)})")

    if isinstance(df.index, pd.DatetimeIndex):
        print(f"  Range:    {df.index.min().date()} -> {df.index.max().date()}")

    for col in target_cols:
        valid = df[col].dropna()
        n_trailing = df[col].isna().sum()
        print(f"  {col}: class balance {valid.mean()*100:.1f}% "
              f"(1: {valid.sum():.0f}, 0: {len(valid)-valid.sum():.0f}), "
              f"{n_trailing} trailing NaN")

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"  [FAIL] {e}")
    else:
        print(f"\nNo errors.")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  [WARN] {w}")

    print()
    print("RESULT: FAILED" if errors else "RESULT: OK")
    print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("USAGE: check_dataset.py <parquet>")
        sys.exit(1)

    path = sys.argv[1]
    errors, warnings = check_dataset(path)

    try:
        df = pd.read_parquet(path)
        print_report(path, df, errors, warnings)
    except Exception:
        for e in errors:
            print(f"[FAIL] {e}")

    sys.exit(1 if errors else 0)
