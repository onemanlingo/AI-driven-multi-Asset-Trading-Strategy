import os
import json
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# Load settings
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), '../settings.json')
with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
    settings = json.load(f)

engine = create_engine(f"postgresql+psycopg2://{settings['DB_USER']}:{settings['DB_PASS']}@{settings['DB_HOST']}/{settings['DB_NAME']}")
df = pd.read_sql("SELECT date, symbol, close FROM ohlc ORDER BY date", engine)
returns = df.pivot(index="date", columns="symbol", values="close").pct_change()
portfolio_returns = returns.mean(axis=1).dropna()
var_95 = np.percentile(portfolio_returns, 5)
print(f"95% VaR: {var_95*100:.2f}%")
with open("results.txt", "a") as f:
    f.write(f"95% VaR: {var_95*100:.2f}%\n")