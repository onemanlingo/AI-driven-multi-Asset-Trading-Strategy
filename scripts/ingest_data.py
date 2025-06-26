import os
import json
import yfinance as yf
import psycopg2
import pandas as pd
from datetime import datetime

# Load settings
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), '../settings.json')
with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
    settings = json.load(f)

symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "SPY", "QQQ"]
start_date = settings.get("START_DATE", "2020-05-21")
end_date = settings.get("END_DATE", "2025-05-21")
if end_date == "today":
    end_date = datetime.today().strftime('%Y-%m-%d')

# Download data
data = yf.download(symbols, start=start_date, end=end_date, interval="1d")

# Connect to PostgreSQL
try:
    conn = psycopg2.connect(dbname=settings["DB_NAME"], user=settings["DB_USER"], password=settings["DB_PASS"], host=settings["DB_HOST"])
    cursor = conn.cursor()

    # Ingest data 
    for symbol in symbols:
        df = data.xs(symbol, axis=1, level=1).reset_index()
        df["symbol"] = symbol
        df = df[["Date", "symbol", "Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["date", "symbol", "open", "high", "low", "close", "volume"]
        values = df[["date", "symbol", "open", "high", "low", "close", "volume"]].values.tolist()
        cursor.executemany(
            """
            INSERT INTO ohlc (date, symbol, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (date, symbol) DO NOTHING
            """,
            [(pd.Timestamp(v[0]), v[1], v[2], v[3], v[4], v[5], int(v[6])) for v in values]
        )
        print(f"Ingested data for {symbol}")
    conn.commit()
    print("Data ingestion complete")
except Exception as e:
    print(f"Error ingesting data: {e}")
finally:
    cursor.close()
    conn.close()