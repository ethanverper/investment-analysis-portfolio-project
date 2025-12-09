# config.py
# Central place for file paths and any global constants I might reuse.

from pathlib import Path

# Base data directory
DATA_DIR = Path("data")

# Input files
PRICES_FILE = DATA_DIR / "S&P 400.xlsx"
FF_FILE = DATA_DIR / "F-F_Research_Data_Factors.csv"