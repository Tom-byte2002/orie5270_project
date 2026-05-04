from quant_engine.data_pipeline import DataManager
from quant_engine.features import FeatureEngineer
from quant_engine.backtest import BacktestEngine

def run_custom_backtest(tickers, initial_capital=100000.0, tx_cost=0.001):
    """
    Runner function that allows users to pick their own tickers and initial capital.
    """
    print("=" * 50)
    print(f"Starting Custom Backtest Engine")
    print(f"Portfolio Universe: {tickers}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print("=" * 50)

    # 1. Pipeline: Fetch any tickers the user desires
    print("\n[Step 1/3] Fetching data for the selected universe...")
    manager = DataManager()
    manager.fetch_and_store(tickers, use_mp=True)

    # 2. Features: Extract and engineer signals from SQLite
    print("\n[Step 2/3] Generating ML features and scaling...")
    engineer = FeatureEngineer()
    data = engineer.process()

    # 3. Backtest: Run the Walk-Forward optimization with custom capital
    print("\n[Step 3/3] Running out-of-sample backtest...")
    engine = BacktestEngine(initial_capital=initial_capital, tx_cost=tx_cost)
    equity_curve = engine.run_backtest(data)

    # Summary Report
    print("\n" + "=" * 50)
    print("✨ BACKTEST SUMMARY REPORT ✨")
    print("=" * 50)
    start_val = equity_curve['Portfolio_Value'].iloc[0]
    end_val = equity_curve['Portfolio_Value'].iloc[-1]
    net_profit = end_val - start_val
    return_pct = (net_profit / start_val) * 100

    print(f"Initial Investment : ${start_val:,.2f}")
    print(f"Ending Value       : ${end_val:,.2f}")
    print(f"Net Profit / Loss  : ${net_profit:,.2f} ({return_pct:+.2f}%)")
    print("=" * 50)

if __name__ == "__main__":
    # =========================================================================
    # Configuration Console: Customize ticker universe and initial capital
    # =========================================================================
    
    # Scenario 1: Mega-Cap Technology Stocks
    my_tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMZN', 'META', 'TSLA']
    
    # Scenario 2: Consumer Staples & Defensive Stocks
    # my_tickers = ['PG', 'KO', 'PEP', 'WMT', 'JNJ']
    
    # Set initial portfolio allocation capital in USD
    my_capital = 500000.0
    
    run_custom_backtest(tickers=my_tickers, initial_capital=my_capital)