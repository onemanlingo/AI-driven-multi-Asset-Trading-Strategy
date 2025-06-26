import os
import json
import psycopg2
import pandas as pd
from datetime import datetime

# Load settings
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), '../settings.json')
with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
    settings = json.load(f)

try:
    conn = psycopg2.connect(dbname=settings["DB_NAME"], user=settings["DB_USER"], password=settings["DB_PASS"], host=settings["DB_HOST"])
    df = pd.read_sql("SELECT symbol, COUNT(*) as count FROM ohlc GROUP BY symbol", conn)
    print("Rows per symbol:\n", df)
    df_dates = pd.read_sql("SELECT MIN(date), MAX(date) FROM ohlc", conn)
    print("Date range:\n", df_dates)
except Exception as e:
    print(f"Error querying data: {e}")
finally:
    conn.close()