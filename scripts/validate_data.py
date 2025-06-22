import psycopg2
import pandas as pd

try:
    conn = psycopg2.connect(dbname="stocks_db", user="postgres", password="postgresvarun", host="localhost")
    df = pd.read_sql("SELECT symbol, COUNT(*) as count FROM ohlc GROUP BY symbol", conn)
    print("Rows per symbol:\n", df)
    df_dates = pd.read_sql("SELECT MIN(date), MAX(date) FROM ohlc", conn)
    print("Date range:\n", df_dates)
except Exception as e:
    print(f"Error querying data: {e}")
finally:
    conn.close()