import pytest
from spx_ndx.models.spx_consensus.config import PipelineConfig, load_config

class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.frequency == "monthly"
        assert cfg.train_start_year == 1993
        assert cfg.first_test_year == 1999
        assert cfg.test_years == 2
        assert len(cfg.cagr_thresholds) == 5

    def test_custom_values(self):
        cfg = PipelineConfig(train_start_year=2000, top_traders=50)
        assert cfg.train_start_year == 2000
        assert cfg.top_traders == 50

class TestLoadConfig:
    def test_load_yaml(self, tmp_path):
        yaml_content = '''
frequency: monthly
train_start_year: 1993
first_test_year: 1999
test_years: 2
trader_min_rtr: 0.6
vote_threshold: 0.6
min_signals_per_trader: 2
max_signals_per_trader: 5
top_traders: 100
min_traders_per_group: 2
max_traders_per_group: 2
group_min_rtr: 0.01
group_min_cagr: 0.01
top_groups: 50
group_aggregation: equal
cagr_thresholds: [0.07, 0.08, 0.09]
indicators:
  SMA Xm: [3, 5, 10]
  VIX<X: [25, 30]
adaptive:
  val_years: 0.5
  grid:
    vote_threshold: [0.5, 0.6]
'''
        p = tmp_path / "test.yaml"
        p.write_text(yaml_content)
        cfg = load_config(p)
        assert cfg.frequency == "monthly"
        assert cfg.cagr_thresholds == (0.07, 0.08, 0.09)
        assert len(cfg.indicators) == 2
        assert cfg.indicators["SMA Xm"] == [3, 5, 10]
        assert cfg.adaptive_val_years == 0.5
        assert cfg.adaptive_grid == {"vote_threshold": [0.5, 0.6]}
