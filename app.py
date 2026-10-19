import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

from data_loader import fetch_stock_data
from feature_engineer import add_technical_indicators
from model_pipeline import train_and_evaluate_regressors, train_and_evaluate_classifiers

# Configuration
st.set_page_config(page_title="AI Stock Prediction Dashboard", layout="wide", page_icon="📈")

st.title("📈 AI-Powered Stock Market Prediction System")
st.markdown("Automated historical technical analysis, feature extraction, and ML modeling.")

# Sidebar Parameters
st.sidebar.header("Data Settings")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL").upper()
default_start = datetime.today() - timedelta(days=365 * 3)
start_date = st.sidebar.date_input("Start Date", value=default_start)
end_date = st.sidebar.date_input("End Date", value=datetime.today())

st.sidebar.header("Model Parameters")
train_split = st.sidebar.slider("Training Data Ratio", min_value=0.5, max_value=0.9, value=0.8, step=0.05)

# Pipeline Execution
if st.sidebar.button("Run Prediction Pipeline"):
    with st.spinner(f"Downloading data for {ticker}..."):
        raw_df = fetch_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    if raw_df is None or raw_df.empty:
        st.error(f"Failed to fetch data for ticker '{ticker}'. Please check the symbol and date range.")
    else:
        # Technical Indicator Generation
        processed_df = add_technical_indicators(raw_df)

        st.subheader(f"Historical Price & Technical Indicators for {ticker}")
        
        # Plotly Candlestick Chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=processed_df.index,
            open=processed_df['Open'],
            high=processed_df['High'],
            low=processed_df['Low'],
            close=processed_df['Close'],
            name='Candlestick'
        ))
        fig.add_trace(go.Scatter(x=processed_df.index, y=processed_df['SMA_10'], name='SMA 10', line=dict(color='orange', width=1)))
        fig.add_trace(go.Scatter(x=processed_df.index, y=processed_df['SMA_50'], name='SMA 50', line=dict(color='blue', width=1)))
        fig.update_layout(title=f"{ticker} Price Chart with Moving Averages", xaxis_rangeslider_visible=False, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # Model Training & Predictions
        reg_results, preds_df = train_and_evaluate_regressors(processed_df, train_ratio=train_split)
        clf_results = train_and_evaluate_classifiers(processed_df, train_ratio=train_split)

        st.markdown("---")
        st.subheader("Model Performance Evaluation")

        col1, col2 = st.columns(2)

        with col1:
            st.write("#### 🎯 Regression Models (Next-Day Price Prediction)")
            reg_summary = []
            for m_name, metrics in reg_results.items():
                reg_summary.append({
                    "Model": m_name,
                    "RMSE ($)": round(metrics["RMSE"], 3),
                    "R² Score": round(metrics["R2"], 3)
                })
            st.dataframe(reg_summary, use_container_width=True)

        with col2:
            st.write("#### 🧭 Classification Models (Direction: Up / Down)")
            clf_summary = []
            for m_name, metrics in clf_results.items():
                clf_summary.append({
                    "Model": m_name,
                    "Accuracy": f"{round(metrics['Accuracy'] * 100, 2)}%",
                    "Precision": f"{round(metrics['Precision'] * 100, 2)}%"
                })
            st.dataframe(clf_summary, use_container_width=True)

        # Plot Predictions vs Actuals
        st.subheader("Forecast vs Actual Price Comparison")
        pred_fig = go.Figure()
        pred_fig.add_trace(go.Scatter(x=preds_df.index, y=preds_df['Actual'], name='Actual Price', line=dict(color='white', width=2)))
        pred_fig.add_trace(go.Scatter(x=preds_df.index, y=preds_df['Linear Regression_Pred'], name='Linear Regression', line=dict(dash='dash')))
        pred_fig.add_trace(go.Scatter(x=preds_df.index, y=preds_df['Random Forest_Pred'], name='Random Forest', line=dict(dash='dot')))
        pred_fig.add_trace(go.Scatter(x=preds_df.index, y=preds_df['XGBoost_Pred'], name='XGBoost', line=dict(color='green')))
        pred_fig.update_layout(title="Out-of-Sample Predictions vs Actual Close Prices", template="plotly_dark")
        st.plotly_chart(pred_fig, use_container_width=True)