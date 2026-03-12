"""ndx_ema200.py - NDX Price & EMA 200 -> output/ndx_ema200.png"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from spx_ndx import utils as u

u.plot_ema200(u.load_yahoo("ndx"), "Nasdaq-100", u.BLUE,
              "ndx_ema200.png", "NDX - Price & EMA 200")
