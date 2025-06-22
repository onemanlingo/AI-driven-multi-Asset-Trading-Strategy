# Trading Strategy Prototype
AI-driven multi-asset trading strategy using SMA Crossover, RSI, Random Forest, Priority Queue, and dynamic risk.
## Features
- Data: 5 years OHLC (2020–2025) for AAPL, MSFT, GOOGL, NVDA, AMZN, SPY, QQQ.
- Storage: PostgreSQL (stocks_db).
- Strategy: SMA (10/50), RSI (40/60), Random Forest, Trailing Stop (3%).
- DSA: Priority Queue.
- Risk: Dynamic trade size (1–50 shares), stop-loss (3–10%), VaR (95%).
- Metrics: Sharpe ~1.01 (target 1.2), Return ~7.62% (target 8%).
## Setup
```bash
pip install -r requirements.txt
python scripts/ingest_data.py
python scripts/strategy.py
python scripts/risk.py
python scripts/dashboard.py