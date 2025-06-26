"""
build_dataset.py
Unified, robust, and efficient pipeline for OHLCV download, TA feature engineering, and RL dataset preparation for all tickers, indices, and forex pairs.
"""
import os
import json
import yfinance as yf
import pandas as pd
import ta
import numpy as np
from datetime import datetime, timedelta

# Load settings from settings.json
SETTINGS_FILE = os.path.join(os.path.dirname(__file__), '../settings.json')
with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
    settings = json.load(f)

DATA_DIR = settings.get('DATA_DIR', 'data')
TICKERS_FILE = 'tickers.json'
INDEX_FILE = 'index.json'
START_DATE = settings.get('START_DATE', '2008-01-01')
END_DATE = settings.get('END_DATE', 'today')
if END_DATE == 'today':
    END_DATE = datetime.today().strftime('%Y-%m-%d')

os.makedirs(DATA_DIR, exist_ok=True)

def load_tickers():
    with open(TICKERS_FILE, 'r', encoding='utf-8') as f:
        tickers = json.load(f)['tickers']
    with open(INDEX_FILE, 'r', encoding='utf-8') as f:
        indices = json.load(f)['index']
    return tickers + indices

def get_last_date(ticker):
    csv_path = os.path.join(DATA_DIR, f"{ticker.replace('=','_').replace('^','_')}.csv")
    if not os.path.exists(csv_path):
        return START_DATE
    try:
        df = pd.read_csv(csv_path)
        if 'date' in df.columns:
            last_date = pd.to_datetime(df['date'], errors='coerce').max()
        else:
            last_date = pd.to_datetime(df.iloc[:,0], errors='coerce').max()
        if pd.isnull(last_date):
            return START_DATE
        return (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    except Exception:
        return START_DATE

def download_ohlcv(ticker):
    start = get_last_date(ticker)
    end = datetime.today().strftime('%Y-%m-%d')
    if start >= end:
        print(f"{ticker}: Already up to date.")
        return None
    print(f"Downloading {ticker} from {start} to {end}")
    df = yf.download(ticker, start=start, end=end)
    # Defensive: Unwrap tuple until DataFrame or error out
    max_unwrap = 5
    unwrap_count = 0
    while isinstance(df, tuple) and unwrap_count < max_unwrap:
        if len(df) > 0:
            df = df[0]
        else:
            print(f"[ERROR] Empty tuple returned from yfinance for {ticker}.")
            return None
        unwrap_count += 1
    # If columns are a MultiIndex (tuples), flatten them
    if hasattr(df, 'columns') and hasattr(df.columns, 'to_flat_index'):
        if any(isinstance(c, tuple) for c in df.columns):
            df.columns = ['_'.join([str(x) for x in c if x and x != '']) for c in df.columns.to_flat_index()]
    if not isinstance(df, pd.DataFrame):
        print(f"[ERROR] Unexpected data type from yfinance for {ticker}: {type(df)}")
        return None
    if df.empty:
        print(f"No data for {ticker}.")
        return None
    # Try to select columns by name, fallback to first 5 columns if needed
    try:
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception:
        df = df.iloc[:, :5]
        print(f"[WARN] Could not select OHLCV by name, used first 5 columns.")
    # Fix: If columns are like 'Close_^OMXSPI', rename to 'close', etc.
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if 'close' in cl: col_map[c] = 'close'
        elif 'open' in cl: col_map[c] = 'open'
        elif 'high' in cl: col_map[c] = 'high'
        elif 'low' in cl: col_map[c] = 'low'
        elif 'volume' in cl: col_map[c] = 'volume'
    df = df.rename(columns=col_map)
    df.columns = [c.lower() for c in df.columns]
    df['date'] = df.index.strftime('%Y-%m-%d')
    df = df.reset_index(drop=True)
    return df

def add_ta_features(df):
    # Ensure price columns are numeric
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # RSI
    df['rsi_6'] = ta.momentum.RSIIndicator(df['close'], window=6).rsi()
    df['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['rsi_30'] = ta.momentum.RSIIndicator(df['close'], window=30).rsi()
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    # Moving Averages
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['ma_50'] = df['close'].rolling(window=50).mean()
    df['ma_200'] = df['close'].rolling(window=200).mean()
    # EMA
    df['ema_10'] = ta.trend.EMAIndicator(df['close'], window=10).ema_indicator()
    df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    # OBV
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    # VWAP
    df['vwap'] = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()
    # ATR
    if 'atr' not in df.columns:
        if len(df) >= 14:
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        else:
            df['atr'] = 0
    # Custom feature to be more pessimistic side when simulating trades
    df['buy_price'] = df['low'] + 0.85 * (df['high'] - df['low'])
    df['sell_price'] = df['low'] + 0.15 * (df['high'] - df['low'])
    df = df.fillna(0)
    return df

def main():
    tickers = load_tickers()
    for ticker in tickers:
        print(f'\n=== Processing {ticker} ===')
        # Download and update OHLCV
        new_data = download_ohlcv(ticker)
        csv_path = os.path.join(DATA_DIR, f'{ticker.replace("=","_").replace("^","_")}.csv')
        if new_data is not None:
            if os.path.exists(csv_path):
                old = pd.read_csv(csv_path)
                old.columns = [c.lower() for c in old.columns]
                # Ensure 'date' column exists in old data
                if 'date' not in old.columns:
                    # Try to infer from index or first column
                    if old.shape[1] > 0:
                        old.insert(0, 'date', old.iloc[:,0])
                        print(f"[WARN] 'date' column missing in {csv_path}, inferred from first column.")
                    else:
                        print(f"[ERROR] 'date' column missing and cannot be inferred in {csv_path}, skipping.")
                        continue
                merged = pd.concat([old, new_data], ignore_index=True)
                merged = merged.drop_duplicates(subset=['date'])
            else:
                merged = new_data
            merged = merged.sort_values('date').reset_index(drop=True)
            merged.to_csv(csv_path, index=False)
        else:
            if not os.path.exists(csv_path):
                print(f'[WARN] No data for {ticker}, skipping.')
                continue
            merged = pd.read_csv(csv_path)
            merged.columns = [c.lower() for c in merged.columns]
            # Ensure 'date' column exists in merged data
            if 'date' not in merged.columns:
                if merged.shape[1] > 0:
                    merged.insert(0, 'date', merged.iloc[:,0])
                    print(f"[WARN] 'date' column missing in {csv_path}, inferred from first column.")
                else:
                    print(f"[ERROR] 'date' column missing and cannot be inferred in {csv_path}, skipping.")
                    continue
        # Add TA features
        merged = add_ta_features(merged)
        # Remove news merge: only use OHLCV+TA
        final = merged
        final = final.sort_values('date').reset_index(drop=True)
        final = final.fillna(0)
        final = final.loc[:,~final.columns.duplicated()]
        # Only keep the requested columns in the output
        output_cols = [
            'date','open','high','low','close','volume','rsi_6',
            'rsi_14','rsi_30','macd','macd_signal','macd_diff',
            'ma_5','ma_20','ma_50','ma_200','ema_10','ema_50','stoch_k','stoch_d',
            'obv','vwap','atr','buy_price','sell_price'
        ]
        # Add missing TA columns if not present (compute on the fly)
        for col in ['obv','vwap','atr']:
            if col not in final.columns:
                final = add_ta_features(final)
                break  # add_ta_features adds all at once
        missing = [c for c in output_cols if c not in final.columns]
        if missing:
            print(f"[WARN] {ticker}: missing columns in output: {missing}")
        final = final[[c for c in output_cols if c in final.columns]]
        # Save only the RL-ready CSV
        out_path = os.path.join(DATA_DIR, f'{ticker.replace("=","_").replace("^","_")}_rl_full_features.csv')
        final.to_csv(out_path, index=False)
        print(f'[OK] {ticker}: RL-ready dataset saved: {out_path}')
        print(f'[INFO] Columns: {list(final.columns)}')
        print(final.head(2))
        print(final.tail(2))

if __name__ == '__main__':
    main()
