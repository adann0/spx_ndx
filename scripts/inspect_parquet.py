"""
explore_parquet.py - Explore all .parquet files in datas/
Run from repo root: python3 explore_parquet.py
"""
import sys
import pandas as pd
from pathlib import Path

DATAS = Path("datas")

if not DATAS.exists():
    print(f"[FAIL] Directory '{DATAS}' not found. Run from repo root.")
    sys.exit(1)

parquets = sorted(DATAS.glob("*.parquet"))
if not parquets:
    print(f"[FAIL] No .parquet files found in '{DATAS}'")
    sys.exit(1)

print(f"Found {len(parquets)} parquet files in '{DATAS}'\n")
print("=" * 80)

for path in parquets:
    try:
        df = pd.read_parquet(path)
        print(f"\n {path.name}")
        print(f"   Shape      : {df.shape[0]} rows × {df.shape[1]} cols")
        print(f"   Index type : {type(df.index).__name__} - dtype: {df.index.dtype}")
        if len(df) > 0:
            print(f"   Index range: {df.index[0]} -> {df.index[-1]}")
        print(f"   Columns    :")
        for col in df.columns:
            dtype = df[col].dtype
            n_null = df[col].isna().sum()
            pct_null = n_null / len(df) * 100 if len(df) > 0 else 0
            # Sample values
            sample = df[col].dropna().head(3).tolist()
            sample_str = ", ".join([str(round(v, 4)) if isinstance(v, float) else str(v) for v in sample])
            last_val = df[col].dropna().iloc[-1] if df[col].dropna().shape[0] > 0 else "N/A"
            print(f"     - {col:<30} dtype={str(dtype):<10} nulls={pct_null:.1f}%  last={last_val}  samples=[{sample_str}]")
    except Exception as e:
        print(f"\n[FAIL] {path.name} - Error: {e}")

print("\n" + "=" * 80)
print("Done.")
