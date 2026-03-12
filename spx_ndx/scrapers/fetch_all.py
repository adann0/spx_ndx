#!/usr/bin/env python3
"""
fetch_all.py
Master fetch script - runs all data fetchers in sequence.
Called by: make fetch

Usage:
    python3 spx_ndx/scrapers/fetch_all.py
"""
import subprocess
import sys

SCRIPTS = [
    "spx_ndx/scrapers/fetch_yahoo.py",
    "spx_ndx/scrapers/fetch_fred.py",
    "spx_ndx/scrapers/fetch_fed.py",
    "spx_ndx/scrapers/fetch_multpl.py",
    "spx_ndx/scrapers/fetch_datahub.py",
]

print("=" * 50)
print("spx-ndx - Data Fetch")
print("=" * 50)

for script in SCRIPTS:
    print(f"\n>>> {script}")
    print("-" * 40)
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"ERROR: {script} failed")
        sys.exit(result.returncode)

print("\n" + "=" * 50)
print("All data fetched successfully.")
print("=" * 50)
