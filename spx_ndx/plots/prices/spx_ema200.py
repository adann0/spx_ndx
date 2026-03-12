"""spx_ema200.py - SPX Price & EMA 200 -> output/spx_ema200.png"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from spx_ndx import utils as u

u.plot_ema200(u.load_yahoo("gspc"), "S&P 500", u.RED,
              "spx_ema200.png", "SPX - Price & EMA 200")
