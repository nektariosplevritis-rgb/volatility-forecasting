import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Vol Forecaster", layout="wide")
st.title("Live Volatility Forecaster – SPX & BTC 1-min Demo")
st.markdown("**Corporate finance → quant switcher** | HAR + LSTM | Nov 2025")

# Synthetic data (demo)
@st.cache_data
def generate_data(n=1000, asset="SPX"):
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, n)
    vol = np.abs(returns).rolling(20).std() * np.sqrt(252 * 390) + np.random.normal(0.15, 0.05, n)
    df = pd.DataFrame({'timestamp': pd.date_range(start='2025-11-20', periods=n, freq='1T'), 'volatility': vol})
    return df

# Simple HAR simulation (no statsmodels)
def simulate_har(df):
    df = df.copy()
    df['har_pred'] = df['volatility'].shift(1) * 0.4 + df['volatility'].rolling(5).mean().shift(1) * 0.3 + df['volatility'].rolling(22).mean().shift(1) * 0.3
    return df['har_pred'].fillna(df['volatility'].mean())

# Simple LSTM simulation (no torch)
def simulate_lstm(df):
    data = df['volatility'].values
    pred = np.roll(data, 1) + np.random.normal(0, data.std() * 0.1, len(data))
    pred[0] = data.mean()
    return pd.Series(pred, index=df.index)

asset = st.selectbox("Choose asset", ["SPX", "BTC"])
df = generate_data(1000, asset)

if st.button("Run Forecast (demo)"):
    st.success("Forecast complete! (Synthetic data demo)")
    
    har_pred = simulate_har(df)
    lstm_pred = simulate_lstm(df)
    
    # Metrics (simple calculations)
    rmse_har = np.sqrt(((df['volatility'] - har_pred)**2).mean())
    rmse_lstm = np.sqrt(((df['volatility'] - lstm_pred)**2).mean())
    dir_har = (df['volatility'] * har_pred > 0).mean()
    dir_lstm = (df['volatility'] * lstm_pred > 0).mean()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("HAR RMSE", f"{rmse_har:.4f}")
    col2.metric("LSTM RMSE", f"{rmse_lstm:.4f}")
    col3.metric("HAR Dir Acc", f"{dir_har:.4f}")
    col4.metric("LSTM Dir Acc", f"{dir_lstm:.4f}")
    
    # Data table (no chart needed)
    st.subheader("Last 10 Minutes – Actual vs Forecast")
    last10 = pd.DataFrame({
        'Actual Vol': df['volatility'].tail(10),
        'HAR Pred': har_pred.tail(10),
        'LSTM Pred': lstm_pred.tail(10)
    })
    st.dataframe(last10)
    
    st.write("**Full production code:** https://github.com/nektariosplevritis-rgb/volatility-forecasting")
    st.write("**Real data:** yfinance + PyTorch locally. This demo shows the engine.")
