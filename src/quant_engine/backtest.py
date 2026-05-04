import pandas as pd
import numpy as np
from datetime import datetime
from quant_engine.features import FeatureEngineer
from quant_engine.ml_predictor import StockPredictor
from quant_engine.optimizer import PortfolioOptimizer

class BacktestEngine:
    """
    Simulates walk-forward portfolio optimization.
    Executes ML-driven asset allocation while factoring in transaction costs.
    """
    def __init__(self, initial_capital=100000.0, tx_cost=0.001):
        self.capital = initial_capital
        self.tx_cost = tx_cost  # 0.1% transaction fee

    def run_backtest(self, df_processed: pd.DataFrame) -> pd.DataFrame:
        """
        Rolls through the data chronologically. Trains the model using past data, 
        optimizes asset weights, and calculates portfolio returns.
        """
        # Sort by date to prevent look-ahead bias
        dates = sorted(df_processed['Date'].unique())
        
        # We need enough initial data to train our first model (e.g., first 60% of dates)
        train_split = int(len(dates) * 0.6)
        train_dates = dates[:train_split]
        test_dates = dates[train_split:]
        
        print(f"Total trading days: {len(dates)}. Testing across: {len(test_dates)} periods.")
        
        predictor = StockPredictor()
        optimizer = PortfolioOptimizer()
        
        portfolio_values = []
        current_value = self.capital
        
        # Previous period's weights (starts at cash)
        prev_weights = {}
        
        # Chronological out-of-sample backtest loop
        for i, current_date in enumerate(test_dates):
            # 1. Look-back to train the model up to yesterday's data
            train_data = df_processed[df_processed['Date'] < current_date]
            if train_data.empty:
                continue
                
            predictor.train(train_data)
            
            # 2. Get today's cross-sectional data to forecast tomorrow's return
            today_data = df_processed[df_processed['Date'] == current_date]
            if today_data.empty:
                continue
                
            predictions = predictor.predict_next_returns(today_data)
            
            # 3. Optimize portfolio weights
            optimal_weights = optimizer.optimize_weights(predictions, train_data)
            
            # 4. Calculate portfolio performance for today's market returns
            daily_returns = today_data.set_index('Ticker')['True_Return'].to_dict()
            
            # Calculate portfolio return before transaction fees
            port_return = sum(optimal_weights[t] * daily_returns.get(t, 0.0) for t in optimal_weights)
            
            # Calculate turnover (how much we traded) to penalize transaction costs
            turnover = sum(abs(optimal_weights[t] - prev_weights.get(t, 0.0)) for t in optimal_weights)
            
            # Apply transaction fees
            cost_penalty = turnover * self.tx_cost
            net_return = port_return - cost_penalty
            
            # Update capital
            current_value *= (1 + net_return)
            portfolio_values.append({'Date': current_date, 'Portfolio_Value': current_value})
            
            # Update cache for turnover tracking
            prev_weights = optimal_weights
            
        print("Backtest processing complete.")
        return pd.DataFrame(portfolio_values)

if __name__ == "__main__":
    try:
        engineer = FeatureEngineer()
        data = engineer.process()
        
        engine = BacktestEngine()
        equity_curve = engine.run_backtest(data)
        
        print("\n=== Sample Equity Curve ===")
        print(equity_curve.head())
        print(f"Final Portfolio Value: {equity_curve['Portfolio_Value'].iloc[-1]:.2f}")
    except Exception as e:
        print(f"Backtest execution failed: {e}")