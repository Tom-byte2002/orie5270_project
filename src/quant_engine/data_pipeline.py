import yfinance as yf
import sqlite3
import pandas as pd
import multiprocessing as mp
from pathlib import Path

# DB connection configuration
DB_PATH = Path("data_cache.db")
DEFAULT_PERIOD = "2y"

def fetch_single_stock(ticker):
    """
    Worker function for parallel data fetching.
    Retrieves historical OHLCV data for a given ticker.
    """
    print(f"Fetching {ticker}...")
    try:
        data = yf.download(ticker, period=DEFAULT_PERIOD, progress=False)
        if data.empty:
            return None
            
        data = data.reset_index()
        data['Ticker'] = ticker
        
        # Standardize columns for database insertion
        cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
        return data[cols]
        
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

class DataManager:
    """
    Manages data ingestion pipeline and local SQLite caching.
    Ensures data consistency and provides offline access for backtesting.
    """
    
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize SQLite schema. Enforces composite primary key."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_prices (
                Date TEXT,
                Ticker TEXT,
                Open REAL,
                High REAL,
                Low REAL,
                Close REAL,
                Volume INTEGER,
                PRIMARY KEY (Date, Ticker)
            )
        ''')
        conn.commit()
        conn.close()

    def fetch_and_store(self, tickers):
        """
        Execute parallel API requests and bulk insert into local database.
        Utilizes multiprocessing pool to bypass GIL for I/O operations.
        """
        num_cores = min(len(tickers), mp.cpu_count())
        print(f"Starting parallel fetch with {num_cores} workers...")
        
        with mp.Pool(processes=num_cores) as pool:
            results = pool.map(fetch_single_stock, tickers)

        valid_data = [df for df in results if df is not None]

        if not valid_data:
            print("Warning: No valid data retrieved.")
            return

        final_df = pd.concat(valid_data, ignore_index=True)

        print(f"Persisting {len(final_df)} records to SQLite cache...")
        conn = sqlite3.connect(self.db_path)
        # Replace existing data to maintain a fresh cache
        final_df.to_sql('daily_prices', conn, if_exists='replace', index=False)
        conn.close()
        
        print("Data pipeline executed successfully.")

if __name__ == '__main__':
    # Quick test execution
    test_universe = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'JPM']
    manager = DataManager()
    manager.fetch_and_store(test_universe)