import yfinance as yf
import psycopg2
import pandas as pd
from datetime import datetime


symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "SPY", "QQQ"]
start_date = "2020-05-21"
end_date = "2025-05-21"


data = yf.download(symbols, start=start_date, end=end_date, interval="1d")

# Connect to PostgreSQL
try:
    conn = psycopg2.connect(dbname="stocks_db", user="postgres", password="postgresvarun", host="localhost")
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