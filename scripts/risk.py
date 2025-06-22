import pandas as pd
import numpy as np
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://postgres:postgresvarun@localhost/stocks_db")
df = pd.read_sql("SELECT date, symbol, close FROM ohlc ORDER BY date", engine)
returns = df.pivot(index="date", columns="symbol", values="close").pct_change()
portfolio_returns = returns.mean(axis=1).dropna()
var_95 = np.percentile(portfolio_returns, 5)
print(f"95% VaR: {var_95*100:.2f}%")
with open("results.txt", "a") as f:
    f.write(f"95% VaR: {var_95*100:.2f}%\n")