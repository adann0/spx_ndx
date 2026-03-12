"""Pipeline configuration loaded from YAML."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class PipelineConfig:
    frequency: str = "monthly"
    train_start_year: int = 1993
    first_test_year: int = 1999
    test_years: int = 2

    trader_min_rtr: float = 0.6
    vote_threshold: float = 0.6

    min_signals_per_trader: int = 2
    max_signals_per_trader: int = 5
    top_traders: int = 100

    min_traders_per_group: int = 2
    max_traders_per_group: int = 2
    group_min_rtr: float = 0.01
    group_min_cagr: float = 0.01
    top_groups: int = 50
    group_aggregation: str = "equal"
    sort_by: tuple = ("stability", "cagr")

    cagr_thresholds: tuple = (0.07, 0.08, 0.09, 0.10, 0.11)

    indicators: dict = field(default_factory=dict)
    adaptive_val_years: float | str = 0.5
    adaptive_grid: dict = field(default_factory=dict)


def load_config(path: str | Path) -> PipelineConfig:
    """Load PipelineConfig from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    indicators = raw.pop("indicators", {})
    adaptive = raw.pop("adaptive", {})

    # cagr_thresholds / sort_by: list -> tuple
    if "cagr_thresholds" in raw:
        raw["cagr_thresholds"] = tuple(raw["cagr_thresholds"])
    if "sort_by" in raw:
        raw["sort_by"] = tuple(raw["sort_by"])

    config = PipelineConfig(
        **{k: v for k, v in raw.items() if k in PipelineConfig.__dataclass_fields__},
        indicators=indicators,
        adaptive_val_years=adaptive.get("val_years", 0.5),
        adaptive_grid=adaptive.get("grid", {}),
    )
    return config


# Periods per year by frequency
FREQ_PERIODS = {"monthly": 12, "weekly": 52, "daily": 252}
