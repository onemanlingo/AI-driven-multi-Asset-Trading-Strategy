import yfinance as yf
import pandas as pd
from datetime import datetime

symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "SPY", "QQQ"]
start_date = "2020-05-21"
end_date = "2025-05-21"

try:
    data = yf.download(symbols, start=start_date, end=end_date, interval="1d")
    print("Data downloaded successfully")
    print(data.head())
    data.to_csv("data/stock_data.csv")  
except Exception as e:
    print(f"Error downloading data: {e}")