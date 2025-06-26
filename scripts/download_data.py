import yfinance as yf
import pandas as pd
from datetime import datetime

#symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "SPY", "QQQ"]
#start_date = "2020-05-21"
#end_date = "2025-05-21"

# download_data.py is now deprecated. Use build_dataset.py for all data collection and feature engineering.
# To build the full dataset for all tickers, indices, and forex pairs, run:
#
#   python scripts/build_dataset.py
#
# This will read tickers from tickers.json and indices/forex from index.json, and save RL-ready datasets in the data/ directory.
#
# The script is robust to missing data, handles updates, and adds technical indicators for all assets.
#
# Remove this file if not needed.

try:
    data = yf.download(symbols, start=start_date, end=end_date, interval="1d")
    print("Data downloaded successfully")
    print(data.head())
    data.to_csv("data/stock_data.csv")  
except Exception as e:
    print(f"Error downloading data: {e}")