import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

class StockPredictor:
    """
    Handles training and inference for stock return forecasting.
    Uses Ridge regression to mitigate multi-collinearity in financial features.
    """
    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)
        self.feature_cols = ['Return_1d', 'SMA_10', 'SMA_30', 'Volatility_20', 'Momentum_10']

    def train(self, train_df: pd.DataFrame):
        """
        Fits the model using historical features and next-day target returns.
        """
        X = train_df[self.feature_cols]
        y = train_df['Target_Return']
        
        self.model.fit(X, y)
        
        preds = self.model.predict(X)
        mse = mean_squared_error(y, preds)
        print(f"Model trained. Training MSE: {mse:.6f}")

    def predict_next_returns(self, current_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generates predictions for the next period's returns.
        Returns a DataFrame mapping Tickers to their expected returns.
        """
        X = current_data[self.feature_cols]
        preds = self.model.predict(X)
        
        result = current_data[['Ticker']].copy()
        result['Predicted_Return'] = preds
        return result

if __name__ == "__main__":
    from quant_engine.features import FeatureEngineer
    
    try:
        # Load and process the real data from SQLite
        engineer = FeatureEngineer()
        data = engineer.process()
        
        predictor = StockPredictor()
        
        # Simple 80/20 train/test split for validation
        split_idx = int(len(data) * 0.8)
        train_set = data.iloc[:split_idx]
        test_set = data.iloc[split_idx:]
        
        predictor.train(train_set)
        predictions = predictor.predict_next_returns(test_set)
        
        print("\n=== Sample Predictions ===")
        print(predictions.head())
        
    except Exception as e:
        print(f"Inference failed: {e}")