# AI-Driven Multi-Asset Trading Strategy

A robust, modular, and extensible platform for research and simulation of multi-asset portfolios using technical analysis, machine learning, and dynamic risk management.

> **Disclaimer:** This code is experimental and provided for research and educational purposes only. It is not intended for use with real money or in live trading environments. Use at your own risk.

## Key Features
- **Data Coverage:** Stocks, indices, and forex (configurable via `tickers.json` and `index.json`).
- **Data Pipeline:** Automated OHLCV download, technical indicator (TA) feature engineering, and RL-ready dataset creation via `build_dataset.py`.
- **Database:** PostgreSQL for persistent storage of historical data and results.
- **Strategies:**
  - SMA Crossover (10/50)
  - RSI (40/60)
  - Random Forest ML signals
  - Trailing Stop (3%)
  - Priority Queue for asset selection
  - Dynamic risk (trade size, stop-loss, VaR)
- **Metrics:** Sharpe ratio, annualized return, max drawdown, VaR, and more.
- **Dashboard:** Interactive analytics and visualization with Dash/Plotly.

## Project Structure
- `settings.json` — All user-editable configuration (DB, data, dates, etc.)
- `requirements.txt` — All dependencies
- `scripts/` — Main scripts:
  - `build_dataset.py` — Download, update, and feature-engineer all assets
  - `ingest_data.py` — Ingest data into PostgreSQL
  - `strategy.py` — Run backtests and ML strategies
  - `risk.py` — Portfolio risk and VaR analysis
  - `dashboard.py` — Interactive dashboard
  - `validate_data.py` — Data quality checks
- `tickers.json` / `index.json` — Asset lists
- `data/` — All local CSVs and RL datasets (auto-generated)

## Quickstart
```bash
# 1. Install dependencies in a virtual environment
pip install -r requirements.txt

# 2. Configure your environment
# Edit settings.json for DB, data directory, and date range

# 3. Download and process all data
python scripts/build_dataset.py

# 4. Ingest data to PostgreSQL (optional)
python scripts/ingest_data.py

# 5. Run backtest/strategy
python scripts/strategy.py

# 6. Analyze risk
python scripts/risk.py

# 7. Launch dashboard
python scripts/dashboard.py
```

## Security & Best Practices
- All sensitive config is in `settings.json` (excluded from git).
- Never commit credentials or local data.
- All scripts use robust error handling and input validation.
- Logging is recommended for production.
- Add tests for all critical logic.

## Example: Loading Config in Python
```python
import os, json
with open('settings.json') as f:
    settings = json.load(f)
```

## Requirements
All dependencies are pinned in `requirements.txt`, including:
- `ta` (technical analysis)
- `yfinance`, `pandas`, `numpy`, `scikit-learn`, `SQLAlchemy`, `psycopg2-binary`, `backtrader`, `dash`, `plotly`, and more

---

For questions, please open an issue.

---

I will now write this updated content to the README.md and commit it with an appropriate message.