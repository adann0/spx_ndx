"""CLI: generate all consensus stress-test plots from a JSON metrics file.

Usage:
    python -m spx_ndx.models.spx_consensus.plot
    python -m spx_ndx.models.spx_consensus.plot output/spx_consensus_stress_metrics.json
"""

import argparse
import json

import pandas as pd

from .generate import generate_all


def main():
    parser = argparse.ArgumentParser(
        description="Generate consensus stress-test plots."
    )
    parser.add_argument(
        "json",
        nargs="?",
        default="output/spx_consensus_stress_metrics.json",
        help="Path to stress-test metrics JSON (default: %(default)s)",
    )
    parser.add_argument(
        "--explain",
        default="output/spx_consensus_explainability.json",
        help="Path to explain JSON (default: %(default)s)",
    )
    args = parser.parse_args()

    with open(args.json) as f:
        metrics = json.load(f)

    meta = metrics["meta"]
    label = meta["label"]
    df = pd.read_parquet(meta["parquet"])

    print(f"Plotting from {args.json} ({label})")

    saved = generate_all(metrics, df, label, explain_path=args.explain)

    print(f"\nSaved {len(saved)} plots:")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
