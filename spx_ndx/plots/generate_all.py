"""
generate_all.py - Master script, runs all individual plot scripts in sequence.

Usage (from repo root):
    python3 spx_ndx/plots/generate_all.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Scripts ordered by README category: prices -> spx_fundamentals ->
# market_internals -> analysis -> macro -> rates_credits
SCRIPTS = [
    # 1. Prices
    "spx_ndx/plots/prices/spx_ema200.py",
    "spx_ndx/plots/prices/ndx_ema200.py",
    "spx_ndx/plots/prices/msci_ndx_spx.py",
    "spx_ndx/plots/prices/spy_volume_profile.py",
    "spx_ndx/plots/prices/qqq_volume_profile.py",
    # 2. SPX Fundamentals
    "spx_ndx/plots/spx_fundamentals/buffett_indicator.py",
    "spx_ndx/plots/spx_fundamentals/spx_cape.py",
    "spx_ndx/plots/spx_fundamentals/spx_pe.py",
    "spx_ndx/plots/spx_fundamentals/ps_ratio.py",
    "spx_ndx/plots/spx_fundamentals/dividend_yield.py",
    "spx_ndx/plots/spx_fundamentals/eps.py",
    "spx_ndx/plots/spx_fundamentals/sales_per_share.py",
    "spx_ndx/plots/spx_fundamentals/corp_margins.py",
    # 3. Market Internals
    "spx_ndx/plots/market_internals/vix.py",
    "spx_ndx/plots/market_internals/spx_drawdown.py",
    "spx_ndx/plots/market_internals/ndx_spx_ratio.py",
    # 4. Analysis
    "spx_ndx/plots/analysis/spx_cape_overlay.py",
    "spx_ndx/plots/analysis/implied_return.py",
    "spx_ndx/plots/analysis/spx_cape_pe_spread.py",
    "spx_ndx/plots/analysis/composite_valuation.py",
    "spx_ndx/plots/analysis/cape_scatter_10y.py",
    # 5. Macro
    "spx_ndx/plots/macro/m2_gdp.py",
    "spx_ndx/plots/macro/cpi_inflation.py",
    "spx_ndx/plots/macro/unemployment.py",
    "spx_ndx/plots/macro/federal_debt.py",
    # 6. Rates and Credits
    "spx_ndx/plots/rates_credits/yield_curve.py",
    "spx_ndx/plots/rates_credits/real_rates.py",
    "spx_ndx/plots/rates_credits/credit_spreads.py",
    "spx_ndx/plots/rates_credits/ecy.py",
    "spx_ndx/plots/rates_credits/fed_model.py",
]


def run(script):
    path = ROOT / script
    t0 = time.time()
    result = subprocess.run([sys.executable, str(path)], cwd=str(ROOT))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  [FAIL]  {script}  (exit {result.returncode})")
        sys.exit(result.returncode)
    print(f"  [OK]  {script}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    t_start = time.time()

    for script in SCRIPTS:
        print(f"\n{'='*60}")
        print(f"  {script}")
        print(f"{'='*60}")
        run(script)

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  [OK]  All {len(SCRIPTS)} plots generated in {total:.1f}s")
    print(f"{'='*60}\n")
