import os  # Import for file and path operations
import json  # Import for reading JSON config files
import pandas as pd  # Import for data analysis and manipulation
import numpy as np  # Import for numerical operations
from datetime import datetime  # Import for handling date and time
from sqlalchemy import create_engine  # Import for database connections
import pickle  # Import for serializing and saving Python objects

from sklearn.ensemble import RandomForestClassifier  # Import for machine learning classification
import backtrader as bt  # Import for backtesting trading strategies

import heapq  # Import for priority queue (heap queue) algorithms

# --------------- Load settings from the JSON config file ---------------
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), '../settings.json')  # Locate the settings file
with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
    settings = json.load(f)  # Load the settings as a dictionary

# --------------- Create a SQLAlchemy engine for connecting to the PostgreSQL database ---------------
engine = create_engine(
    f"postgresql+psycopg2://{settings['DB_USER']}:{settings['DB_PASS']}@{settings['DB_HOST']}/{settings['DB_NAME']}"
)

# --------------- Utility function: Calculate Simple Moving Average (SMA) using a sliding window ---------------
def sliding_window_sma(prices, window=10):
    if len(prices) < window:
        return []  # Not enough data, return empty list
    sma = []
    window_sum = sum(prices[:window])  # Sum for the first window
    sma.append(window_sum / window)  # First SMA value
    for i in range(window, len(prices)):
        window_sum += prices[i] - prices[i - window]  # Add new price, remove oldest
        sma.append(window_sum / window)  # Append new SMA value
    return sma

# --------------- Utility function: Compute Relative Strength Index (RSI) ---------------
def compute_rsi(data, periods=14):
    delta = data.diff()  # Calculate price differences
    gain = delta.where(delta > 0, 0).rolling(window=periods).mean()  # Average gains
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()  # Average losses
    rs = gain / loss  # Relative strength
    rsi = 100 - (100 / (1 + rs))  # RSI formula
    return rsi

# --------------- Train a RandomForest classifier to predict buy/sell/hold signals ---------------
def train_rf_model(df):
    features = pd.DataFrame()
    # Calculate features for the model
    features["sma_ratio"] = df["close"].rolling(10).mean() / df["close"].rolling(50).mean()
    features["rsi"] = compute_rsi(df["close"], 14)
    features["returns_5d"] = df["close"].pct_change(5)
    features["volume"] = df["volume"]
    # Target: 1 if tomorrow's close is higher than today's, else 0
    target = (df["close"].shift(-1) > df["close"]).astype(int)
    train_data = features.dropna()  # Remove rows with NaN values
    train_target = target[train_data.index]  # Align target and features
    train_size = int(0.8 * len(train_data))  # 80% for training, 20% for validation
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(train_data.iloc[:train_size], train_target.iloc[:train_size])  # Train the model
    return model

# --------------- Select top-k assets with strongest buy signals, using a heap/priority queue ---------------
def select_top_signals(predictions, confidences, k=3):
    # Build a heap with negative confidence for max-heap behavior, only for predicted buys
    heap = [(-conf, symbol) for symbol, conf in confidences.items() if symbol in predictions and predictions[symbol] == 1]
    heapq.heapify(heap)
    return [heapq.heappop(heap)[1] for _ in range(min(k, len(heap)))]  # Return top k symbols

# --------------- Dynamic Risk Management Parameters ---------------
MIN_SHARES = 1  # Minimum trade size
MAX_SHARES = 50  # Maximum trade size
INCREMENT_SHARES = 2  # Increment/decrement for trade size
BASE_STOP_LOSS_PCT = 0.05  # Base stop loss (5%)
INCREMENT_STOP_LOSS_PCT = 0.005  # Increment/decrement for stop loss

# --------------- Adjust trade size and stop loss based on recent performance ---------------
def adjust_risk(pnls, num_shares, stop_loss_pct, last_risk_change_index):
    if len(pnls) > 40:  # Require at least 40 PnL points
        monthly_pnl = pnls[-1] - pnls[-20]  # PnL over the last 20 periods
        if len(pnls) - last_risk_change_index > 20:  # Adjust only every 20 periods
            if monthly_pnl > 0:
                num_shares = min(num_shares + INCREMENT_SHARES, MAX_SHARES)
                stop_loss_pct = min(stop_loss_pct + INCREMENT_STOP_LOSS_PCT, 0.1)
                print(f"Increasing trade size to {num_shares}, stop-loss to {stop_loss_pct*100}%")
            elif monthly_pnl < 0:
                num_shares = max(num_shares - INCREMENT_SHARES, MIN_SHARES)
                stop_loss_pct = max(stop_loss_pct - INCREMENT_STOP_LOSS_PCT, 0.03)
                print(f"Decreasing trade size to {num_shares}, stop-loss to {stop_loss_pct*100}%")
            return num_shares, stop_loss_pct, len(pnls)  # Update last risk change index
    return num_shares, stop_loss_pct, last_risk_change_index  # No change

# --------------- Backtrader strategy: combines SMA crossover, RSI, and ML signals ---------------
class SMACrossRSI(bt.Strategy):
    params = (('sma1', 10), ('sma2', 50), ('rsi_period', 14))  # Default parameters

    def __init__(self):
        self.num_shares = MIN_SHARES  # Start with minimum trade size
        self.stop_loss_pct = BASE_STOP_LOSS_PCT  # Start with base stop loss
        self.last_risk_change_index = 0  # Last time risk params were changed
        self.pnls = []  # Track profit/loss over time
        self.models = {}  # ML models for each asset
        self.sma1 = {}  # Short SMA for each asset
        self.sma2 = {}  # Long SMA for each asset
        self.rsi = {}  # RSI indicator for each asset
        self.orders = []  # Track order history
        self.num_shares_history = []  # Track trade sizes over time
        # For each data feed (asset)
        for d in self.datas:
            # Load historical data for current symbol from database
            df = pd.read_sql("SELECT date, close, volume FROM ohlc WHERE symbol=%s ORDER BY date", engine, params=(d._name,))
            self.models[d._name] = train_rf_model(df)  # Train a model on this asset's history
            self.sma1[d._name] = sliding_window_sma(df["close"].values, self.p.sma1)  # Precompute short SMA
            self.sma2[d._name] = sliding_window_sma(df["close"].values, self.p.sma2)  # Precompute long SMA
            self.rsi[d._name] = bt.indicators.RSI(d.close, period=self.p.rsi_period)  # Set up RSI indicator

    def next(self):
        # Called on each bar (new data point)
        self.pnls.append(self.broker.getvalue() - self.broker.startingcash)  # Update PnL
        # Adjust risk parameters if needed
        self.num_shares, self.stop_loss_pct, self.last_risk_change_index = adjust_risk(
            self.pnls, self.num_shares, self.stop_loss_pct, self.last_risk_change_index
        )
        self.num_shares_history.append(self.num_shares)  # Save trade size

        predictions = {}  # ML buy/sell predictions
        confidences = {}  # ML prediction confidences

        for d in self.datas:
            # Pull recent data for features (last 100 bars)
            df = pd.read_sql("SELECT date, close, volume FROM ohlc WHERE symbol=%s ORDER BY date DESC LIMIT 100", engine, params=(d._name,))
            features = pd.DataFrame()
            features["sma_ratio"] = df["close"].rolling(10).mean() / df["close"].rolling(50).mean()
            features["rsi"] = compute_rsi(df["close"], 14)
            features["returns_5d"] = df["close"].pct_change(5)
            features["volume"] = df["volume"]
            X = features.dropna().iloc[-1:]  # Use latest valid row
            if not X.empty:
                pred = self.models[d._name].predict(X)[0]  # Predict class (buy/sell/hold)
                conf = self.models[d._name].predict_proba(X)[0][pred]  # Confidence in prediction
                predictions[d._name] = pred
                confidences[d._name] = conf

        # Pick top 3 assets with strongest buy signals
        top_symbols = select_top_signals(predictions, confidences, k=3)

        for d in self.datas:
            pos = self.getposition(d).size  # Current position size in this asset
            # Compute price movement as a percent
            price_move = abs(d.close[0] / d.close[-1] - 1) if d.close[-1] != 0 else 0
            # Buy condition: in top buys, ML says buy, short SMA > long SMA, RSI oversold, price is moving
            if (d._name in top_symbols and
                predictions.get(d._name) == 1 and
                self.sma1[d._name][-1] > self.sma2[d._name][-1] and
                self.rsi[d._name][0] < 40 and
                price_move > 0.005):  # 0.5% move
                self.buy(data=d, size=self.num_shares)
                self.orders.append(1)
                print(f"Buy {d._name} at {d.close[0]}")
            # Sell condition: SMA cross under, RSI overbought, or hit stop loss
            elif pos > 0 and (
                self.sma1[d._name][-1] < self.sma2[d._name][-1] or
                self.rsi[d._name][0] > 60 or
                (d.close[0] / d.close[-1] - 1) < -self.stop_loss_pct):
                self.sell(data=d, size=pos)
                self.orders.append(-1)
                print(f"Sell {d._name} at {d.close[0]}")
            else:
                self.orders.append(0)  # No action

# --------------- Script entrypoint: run backtest when executed directly ---------------
if __name__ == "__main__":
    cerebro = bt.Cerebro()  # Backtrader engine
    symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "SPY", "QQQ"]  # List of symbols to backtest
    for symbol in symbols:
        # Pull complete historical OHLCV data for each symbol
        df = pd.read_sql(
            "SELECT date, open, high, low, close, volume FROM ohlc WHERE symbol=%s ORDER BY date",
            engine, params=(symbol,)
        )
        df["date"] = pd.to_datetime(df["date"])  # Ensure date is datetime type
        data_feed = bt.feeds.PandasData(dataname=df.set_index("date"))  # Create Backtrader data feed
        cerebro.adddata(data_feed, name=symbol)  # Add data feed to engine

    cerebro.addstrategy(SMACrossRSI)  # Add the strategy
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")  # Add Sharpe ratio analysis
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")  # Add drawdown analysis
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")  # Add returns analysis
    cerebro.broker.setcash(100000)  # Set initial cash
    results = cerebro.run()  # Run the backtest
    sharpe = results[0].analyzers.sharpe.get_analysis().get("sharperatio", None)
    drawdown = results[0].analyzers.drawdown.get_analysis().get("max", {}).get("drawdown", None)
    returns = results[0].analyzers.returns.get_analysis().get("rnorm100", None)
    print(f"Sharpe Ratio: {sharpe}")
    print(f"Max Drawdown: {drawdown}%")
    print(f"Annualized Return: {returns}%")
    # Save results to files
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)
    with open("results.txt", "w") as f:
        f.write(f"Sharpe Ratio: {sharpe}\nMax Drawdown: {drawdown}%\nAnnualized Return: {returns}%\n")