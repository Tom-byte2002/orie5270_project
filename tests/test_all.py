import pytest
import pandas as pd
import sqlite3
from unittest.mock import patch
from src.quant_engine.data_pipeline import DataManager, fetch_single_stock

# --- Fixtures and Mocks ---

@pytest.fixture
def temp_db(tmp_path):
    """Provides an isolated SQLite database path to prevent mutating the production cache."""
    return tmp_path / "test_cache.db"

def mock_yf_download(*args, **kwargs):
    """
    Mocks yfinance API responses to ensure deterministic testing without network I/O.
    """
    ticker = args[0]
    if ticker == 'INVALID':
        return pd.DataFrame() 
    
    # Construct synthetic OHLCV data
    data = {
        'Open': [150.0, 151.0],
        'High': [155.0, 152.0],
        'Low': [149.0, 150.0],
        'Close': [152.0, 151.5],
        'Volume': [10000, 12000]
    }
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(['2023-01-01', '2023-01-02'])
    df.index.name = 'Date'
    return df

# --- Test Cases ---

def test_db_initialization(temp_db):
    """Verifies DDL execution and schema creation."""
    manager = DataManager(db_path=temp_db)
    
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    # Verify table existence in sqlite_master
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='daily_prices'")
    assert cursor.fetchone() is not None
    conn.close()

@patch('src.quant_engine.data_pipeline.yf.download', side_effect=mock_yf_download)
def test_fetch_single_stock_success(mock_download):
    """Validates data parsing and standardization for a valid ticker."""
    df = fetch_single_stock('AAPL')
    
    assert df is not None
    assert len(df) == 2
    assert df['Ticker'].iloc[0] == 'AAPL'

@patch('src.quant_engine.data_pipeline.yf.download', side_effect=mock_yf_download)
def test_fetch_single_stock_empty(mock_download):
    """Tests error handling for delisted or invalid tickers."""
    df = fetch_single_stock('INVALID')
    assert df is None

@patch('src.quant_engine.data_pipeline.yf.download', side_effect=mock_yf_download)
def test_fetch_and_store(mock_download, temp_db):
    """Integration test for the data pipeline (Single Process Mode for testing)."""
    manager = DataManager(db_path=temp_db)
    
    manager.fetch_and_store(['AAPL', 'MSFT'], use_mp=False) 
    
    # Verify data persistence
    conn = sqlite3.connect(temp_db)
    df_db = pd.read_sql("SELECT * FROM daily_prices", conn)
    conn.close()
    
    # Expecting 2 tickers * 2 days = 4 records
    assert len(df_db) == 4 
    assert set(df_db['Ticker'].unique()) == {'AAPL', 'MSFT'}

@patch('src.quant_engine.data_pipeline.yf.download', side_effect=Exception("Network Timeout"))
def test_fetch_single_stock_exception(mock_download):
    """Tests exception handling during network failure."""
    df = fetch_single_stock('AAPL')
    assert df is None

@patch('src.quant_engine.data_pipeline.yf.download', side_effect=mock_yf_download)
def test_fetch_and_store_empty(mock_download, temp_db):
    """Tests pipeline behavior when all tickers return invalid/empty data."""
    manager = DataManager(db_path=temp_db)
    # Should handle empty lists gracefully without crashing
    manager.fetch_and_store(['INVALID'], use_mp=False)