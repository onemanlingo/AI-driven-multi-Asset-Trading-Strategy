from dash import Dash, dcc, html
import plotly.graph_objects as go
from sqlalchemy import create_engine
import pandas as pd
import backtrader as bt
from strategy import SMACrossRSI

app = Dash(__name__)
engine = create_engine("postgresql+psycopg2://postgres:postgresvarun@localhost/stocks_db")
df = pd.read_sql("SELECT date, close FROM ohlc WHERE symbol='AAPL' ORDER BY date", engine)
df_spy = pd.read_sql("SELECT date, close FROM ohlc WHERE symbol='SPY' ORDER BY date", engine)

# Simulate strategy data 
cerebro = bt.Cerebro()
symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "SPY", "QQQ"]
for symbol in symbols:
    df_data = pd.read_sql("SELECT date, open, high, low, close, volume FROM ohlc WHERE symbol=%s ORDER BY date", engine, params=(symbol,))
    df_data["date"] = pd.to_datetime(df_data["date"])
    data_feed = bt.feeds.PandasData(dataname=df_data.set_index("date"))
    cerebro.adddata(data_feed, name=symbol)
cerebro.addstrategy(SMACrossRSI)
results = cerebro.run()
orders = results[0].orders
num_shares_history = results[0].num_shares_history

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["date"], y=df["close"], name="AAPL Price"))
fig.add_trace(go.Scatter(x=df_spy["date"], y=df_spy["close"], name="SPY Price"))
fig.add_trace(go.Scatter(x=df["date"], y=num_shares_history, name="Trade Size", yaxis="y2"))
layout = dict(yaxis2=dict(overlaying="y", side="right"))
fig.update_layout(layout)
app.layout = html.Div([dcc.Graph(figure=fig)])
app.run(debug=True)