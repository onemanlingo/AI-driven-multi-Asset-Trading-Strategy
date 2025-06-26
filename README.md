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
```

## Robustness & Security Improvements

- **Environment Variables**: Database credentials and sensitive information should be set using environment variables or a configuration file. Example (Windows PowerShell):
  ```powershell
  $env:DB_USER = "postgres"
  $env:DB_PASS = "yourpassword"
  $env:DB_HOST = "localhost"
  $env:DB_NAME = "stocks_db"
  ```
  Update your scripts to read these variables using `os.environ`.

- **Error Handling**: All scripts now include improved error handling. Database connections and file operations use context managers (`with` statements) to ensure resources are always closed, even on error.

- **Logging**: Print statements have been replaced or supplemented with logging for better traceability. You can further configure logging as needed.

- **Directory Checks**: Scripts that write files (e.g., `download_data.py`) now check for the existence of the target directory and create it if missing.

- **Input Validation**: Dataframes and database query results are validated before use to prevent runtime errors.

- **Testing**: Add unit tests for critical functions (not included in this repo, but recommended for production use).

- **Sensitive Data in Files**: All sensitive configuration (database credentials, etc.) is now stored in `settings.json`, which is excluded from version control via `.gitignore`. Never commit this file.

- **User Configuration**: All user-settable options (database, data directory, date range, etc.) are now in `settings.json`. Edit this file to change your environment or data settings.

- **Requirements**: Ensure your environment is up to date by running:
  ```bash
  pip install -r requirements.txt
  ```

- **.gitignore**: The repository now ignores sensitive files (`settings.json`, `.env`), data outputs, and unnecessary files for security and cleanliness.

## Example: Secure Database Connection
```python
import os
import psycopg2
conn = psycopg2.connect(
    dbname=os.environ["DB_NAME"],
    user=os.environ["DB_USER"],
    password=os.environ["DB_PASS"],
    host=os.environ["DB_HOST"]
)
```

## General Recommendations
- Never commit credentials to version control.
- Use logging instead of print for production.
- Validate all external data sources.
- Add tests for all critical logic.