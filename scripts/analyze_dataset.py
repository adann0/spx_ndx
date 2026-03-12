#!/usr/bin/env python3

"""Analyse exploratoire d'un dataset parquet."""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "datas/dataset_monthly.parquet"
df = pd.read_parquet(path)
print(f"Dataset: {path}")

# --- Info generales ---
print("=" * 60)
print("INFORMATIONS GENERALES")
print("=" * 60)
print(f"Shape : {df.shape[0]} lignes x {df.shape[1]} colonnes")
print(f"Index : {df.index.name} ({df.index.min()} -> {df.index.max()})")
print(f"\nValeurs manquantes :\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print(f"\nStatistiques descriptives :\n{df.describe().round(2)}")

# --- Correlation matrix ---
features = [c for c in df.columns if not c.startswith("target") and c not in ("spx_close", "spx_open")]
targets = [c for c in df.columns if c.startswith("target_")]

corr_with_targets = df[features + targets].corr()[targets].drop(targets).sort_values(targets[0], ascending=False)
print("\n" + "=" * 60)
print("CORRELATIONS FEATURES -> TARGETS")
print("=" * 60)
print(corr_with_targets.round(3))

# --- Plots ---
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle("Dataset Analysis", fontsize=16, fontweight="bold")

# 1. SPX price
ax = axes[0, 0]
ax.plot(df.index, df["spx_close"], color="steelblue", linewidth=0.8)
ax.set_title("S&P 500 Close")
ax.set_ylabel("Price")
ax.grid(alpha=0.3)

# 2. Target distributions
ax = axes[0, 1]
for t in targets:
    ax.hist(df[t].dropna(), bins=50, alpha=0.5, label=t)
ax.set_title("Target Distributions")
ax.legend()
ax.grid(alpha=0.3)

# 3. Top correlations with target_3m (bar chart)
ax = axes[1, 0]
main_target = targets[0]
top_corr = corr_with_targets[main_target].abs().sort_values(ascending=False).head(10)
colors = ["green" if corr_with_targets.loc[f, main_target] > 0 else "red" for f in top_corr.index]
ax.barh(top_corr.index, top_corr.values, color=colors)
ax.set_title(f"Top 10 |Correlation| with {main_target}")
ax.invert_yaxis()
ax.grid(alpha=0.3, axis="x")

# 4. Missing values over time
ax = axes[1, 1]
missing_pct = df[features].isnull().rolling(12).mean()
if missing_pct.sum().sum() > 0:
    cols_with_na = missing_pct.columns[missing_pct.sum() > 0]
    for c in cols_with_na:
        ax.plot(missing_pct.index, missing_pct[c], label=c, linewidth=0.8)
    ax.legend(fontsize=7)
else:
    ax.text(0.5, 0.5, "No missing values", ha="center", va="center", transform=ax.transAxes)
ax.set_title("Missing Values (12-month rolling %)")
ax.grid(alpha=0.3)

# 5. VIX vs realized vol
ax = axes[2, 0]
ax.scatter(df["realized_vol"], df["vix_close"], alpha=0.4, s=10, color="purple")
ax.plot([df["realized_vol"].min(), df["realized_vol"].max()],
        [df["realized_vol"].min(), df["realized_vol"].max()],
        "k--", alpha=0.5, label="y=x")
ax.set_xlabel("Realized Vol")
ax.set_ylabel("VIX Close")
ax.set_title("VIX vs Realized Volatility")
ax.legend()
ax.grid(alpha=0.3)

# 6. Feature correlation heatmap (top features)
ax = axes[2, 1]
top_features = corr_with_targets[main_target].abs().sort_values(ascending=False).head(8).index.tolist()
corr_matrix = df[top_features].corr()
im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(top_features)))
ax.set_yticks(range(len(top_features)))
ax.set_xticklabels(top_features, rotation=45, ha="right", fontsize=7)
ax.set_yticklabels(top_features, fontsize=7)
ax.set_title("Top Features Correlation")
fig.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout()
plt.savefig("dataset_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nPlot saved -> dataset_analysis.png")
