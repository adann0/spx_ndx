"""spy_volume_profile.py - SPY Volume Profile -> output/spy_volume_profile.png"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from spx_ndx import utils as u

u.plot_volume_profile(u.load_ohlcv("spy"), "SPY", "spy_volume_profile.png")
