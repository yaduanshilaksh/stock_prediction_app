# AI-Powered Stock Market Prediction System

An end-to-end Machine Learning pipeline and Streamlit dashboard that analyzes historical stock market data, computes technical indicators, and forecasts future price targets and price directions.

## Features
- **Data Ingestion**: Direct pipeline using `yfinance` to fetch OHLCV stock data.
- **Feature Engineering**: Calculates SMA, EMA, MACD, RSI, Volatility, and Returns.
- **Predictive Modeling**: Compares Linear Regression, Random Forest, and XGBoost models across regression and classification setups.
- **Interactive Visualizations**: Dynamic candlestick charts and visual prediction comparisons using Plotly and Streamlit.

## Setup Instructions

1. **Clone repository & set up environment**:
   ```bash
   git clone <your-repo-url>
   cd stock_prediction_app
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate