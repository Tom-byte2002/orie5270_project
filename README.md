# ORIE 5270 Project: ML-Driven Portfolio Optimization Engine

This repository contains a quantitative investment framework that utilizes machine learning and convex optimization to construct, allocate, and backtest a custom stock portfolio. The project is fully packaged and compliant with standard software engineering practices.

## Project Purpose

The primary objective of this project is to implement an out-of-sample walk-forward backtesting simulation. The engine follows a 4-step quantitative lifecycle:
1. **Data Ingestion:** Parallelized retrieval of real market data from Yahoo Finance into a structured SQLite cache.
2. **Feature Engineering:** Creation of rolling alpha signals (e.g., momentum, moving averages) and cross-sectional normalization.
3. **Alpha Generation:** Training a localized Ridge Regression model to forecast next-day cross-sectional returns.
4. **Portfolio Optimization:** Solving a Markowitz Mean-Variance optimization problem using SciPy SLSQP to dynamically allocate asset weights under constraints, factoring in transaction costs.

---

## Dataset Description & Advanced Customization

The project relies exclusively on real-world equity market data retrieved via the `yfinance` API. It is completely modular and **fully supports user customization for assets, capital, and timeframe**:

* **Custom Tickers:** You can freely select your portfolio universe in `main.py` by modifying the `my_tickers` list.
* **Custom Investment Capital:** You can change the starting investment amount in `main.py` by adjusting `my_capital`.
* **Custom Backtest Timeframe:** You can adjust the historical data period (e.g., from `2y` to `5y` or `1y`) by changing the `DEFAULT_PERIOD` variable inside `src/quant_engine/data_pipeline.py`.

---

## Package Installation

This project is configured as a standard Python package via `pyproject.toml`. 

To install the package in editable mode along with its development dependencies, execute the following commands in your terminal:

1. Clone the repository:
   git clone https://github.com/Tom-byte2002/orie5270_project.git
   cd orie5270_project

2. Create and activate virtual environment:
   python -m venv .venv
   source .venv/Scripts/activate  # On Windows Git Bash
   # source .venv/bin/activate    # On macOS/Linux

3. Install in editable mode:
   pip install -e .

---

## Usage and Execution Instructions

### 1. Run the Pipeline with Custom Configurations
To run the main pipeline with your own portfolio configurations, open `main.py` and modify the parameters:

# Change the tickers to any stocks from Yahoo Finance
my_tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMZN', 'META', 'TSLA']

# Change the initial investment capital
my_capital = 500000.0

To adjust the data range, modify `src/quant_engine/data_pipeline.py`:

# Change "2y" to "5y", "1y", "max", etc.
DEFAULT_PERIOD = "2y"

Once saved, run the project from the root directory:
python main.py

### 2. Run the Test Suite
To verify the technical integrity of the modules and check test coverage (which exceeds 80%), run:
pytest

---

## Project Structure

orie5270_project/
├── src/
│   └── quant_engine/
│       ├── __init__.py
│       ├── data_pipeline.py    # Multi-threaded fetching and SQLite caching
│       ├── features.py         # Rolling window calculations & feature scaling
│       ├── ml_predictor.py     # Alpha generation via Ridge regression
│       ├── optimizer.py        # Portfolio optimization using SciPy
│       └── backtest.py         # Simulation engine with transaction cost tracking
├── tests/
│   └── test_all.py             # Unit tests with >80% coverage
├── pyproject.toml              # Project configuration and package setup
├── main.py                     # Entry point for custom configurations
└── README.md                   # Complete project documentation