import yfinance as yf
import sqlite3
import pandas as pd
import multiprocessing as mp
from pathlib import Path

DB_PATH = Path("data_cache.db")
DEFAULT_PERIOD = "5y"

def fetch_single_stock(ticker):
    """
    Worker function for parallel data fetching.
    Retrieves historical OHLCV data for a given ticker and normalizes its column structure.
    """
    print(f"Fetching {ticker}...")
    try:
        # multi_level_index=False prevents yfinance from creating MultiIndex columns
        data = yf.download(ticker, period=DEFAULT_PERIOD, progress=False, multi_level_index=False)
        if data.empty:
            return None
            
        data = data.reset_index()
        
        # Flatten MultiIndex if it still exists due to pandas/yfinance version variance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        # Ensure 'Date' column name is properly capitalized
        data.columns = [str(col).capitalize() if str(col).lower() == 'date' else col for col in data.columns]
        
        # Append ticker symbol
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

    def fetch_and_store(self, tickers, use_mp=True):
        """
        Execute parallel API requests and bulk insert into local database.
        Utilizes multiprocessing pool to bypass GIL for I/O operations.
        """
        if use_mp:
            num_cores = min(len(tickers), mp.cpu_count())
            print(f"Starting parallel fetch with {num_cores} workers...")
            with mp.Pool(processes=num_cores) as pool:
                results = pool.map(fetch_single_stock, tickers)
        else:
            print("Running sequentially (Single Process Mode)...")
            results = [fetch_single_stock(t) for t in tickers]

        valid_data = [df for df in results if df is not None]

        if not valid_data:
            print("Warning: No valid data retrieved.")
            return

        final_df = pd.concat(valid_data, ignore_index=True)

        print(f"Persisting {len(final_df)} records to SQLite cache...")
        conn = sqlite3.connect(self.db_path)
        
        # if_exists='replace' deletes the old table and ensures new correct columns are created
        final_df.to_sql('daily_prices', conn, if_exists='replace', index=False)
        conn.close()
        
        print("Data pipeline executed successfully.")

if __name__ == '__main__':
    # Quick test execution
    test_universe = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'JPM']
    manager = DataManager()
    manager.fetch_and_store(test_universe)