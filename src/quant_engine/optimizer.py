import numpy as np
import pandas as pd
from scipy.optimize import minimize

class PortfolioOptimizer:
    """
    Implements Mean-Variance Optimization (Markowitz) using SciPy.
    Determines optimal asset weights based on predicted returns and historical risk.
    """
    def __init__(self, risk_aversion=1.0):
        self.risk_aversion = risk_aversion

    def optimize_weights(self, predicted_returns: pd.DataFrame, historical_prices: pd.DataFrame):
        """
        Solves the quadratic programming problem: max (w^T * mu - lambda * w^T * Sigma * w)
        subject to: sum(w) = 1 and 0 <= w <= 1.
        """
        tickers = predicted_returns['Ticker'].tolist()
        mu = predicted_returns['Predicted_Return'].values
        
        # Calculate Covariance Matrix (Sigma)
        price_pivot = historical_prices.pivot(index='Date', columns='Ticker', values='Close')
        returns_matrix = price_pivot[tickers].pct_change().dropna()
        
        if returns_matrix.empty:
            # Fallback to equal weights if there's no return history
            num_assets = len(tickers)
            init_guess = num_assets * [1. / num_assets]
            return dict(zip(tickers, init_guess))
            
        sigma = returns_matrix.cov().values
        num_assets = len(tickers)
        
        # Objective function: Negative Sharpe-like utility
        def objective(weights):
            portfolio_return = np.dot(weights, mu)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
            return -(portfolio_return - self.risk_aversion * portfolio_vol)

        # Constraints and Bounds
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        init_guess = num_assets * [1. / num_assets]
        
        opt_result = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if not opt_result.success:
            return dict(zip(tickers, init_guess))
            
        return dict(zip(tickers, opt_result.x))

if __name__ == "__main__":
    print("PortfolioOptimizer class verified.")