import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

DB_PATH = Path("data_cache.db")

class FeatureEngineer:
    """
    Constructs machine learning features from raw OHLCV data.
    Implements cross-sectional standardization and handles missing values.
    """
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.scaler = StandardScaler()

    def load_data(self) -> pd.DataFrame:
        """Extracts panel data from SQLite database."""
        conn = sqlite3.connect(self.db_path)
        # Sort by Ticker and Date to ensure proper time-series calculations
        query = "SELECT * FROM daily_prices ORDER BY Ticker, Date"
        df = pd.read_sql(query, conn)
        conn.close()
        
        df['Date'] = pd.to_datetime(df['Date'])
        return df

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates features and keeps unscaled returns for backtesting."""
        print("Engineering financial features...")
        
        # Target variable for ML forecasting
        df['Target_Return'] = df.groupby('Ticker')['Close'].pct_change().shift(-1)
        
        # Features computation
        df['Return_1d'] = df.groupby('Ticker')['Close'].pct_change()
        
        df['True_Return'] = df['Return_1d']
        
        df['SMA_10'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(10).mean())
        df['SMA_30'] = df.groupby('Ticker')['Close'].transform(lambda x: x.rolling(30).mean())
        df['Volatility_20'] = df.groupby('Ticker')['Return_1d'].transform(lambda x: x.rolling(20).std())
        df['Momentum_10'] = df.groupby('Ticker')['Close'].pct_change(periods=10)
        
        df = df.dropna().reset_index(drop=True)
        return df

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies StandardScaler to cross-sectional features to prevent 
        magnitude bias in downstream ML models.
        """
        feature_cols = ['Return_1d', 'SMA_10', 'SMA_30', 'Volatility_20', 'Momentum_10']
        
        print("Applying StandardScaler to feature space...")
        # Fit and transform the features
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        
        return df
        
    def process(self) -> pd.DataFrame:
        """Executes the full feature engineering pipeline."""
        df_raw = self.load_data()
        if df_raw.empty:
            raise ValueError("Database is empty. Run data_pipeline.py first.")
            
        df_features = self.generate_features(df_raw)
        df_scaled = self.scale_features(df_features)
        
        print(f"Feature engineering complete. Dataset shape: {df_scaled.shape}")
        return df_scaled

if __name__ == "__main__":
    # Quick execution test
    engineer = FeatureEngineer()
    processed_data = engineer.process()
    print(processed_data.head())