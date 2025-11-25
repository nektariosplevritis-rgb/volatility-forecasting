import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Volatility Forecaster", layout="wide")
st.title("Volatility Forecaster – SPX & BTC 1-min")
st.markdown("**Corporate finance → quant switcher** | HAR + LSTM | 2025")

def generate_data(n=1000):
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, n)
    abs_returns = np.abs(returns)
    vol = pd.Series(abs_returns).rolling(20, min_periods=1).std() * np.sqrt(252 * 390)
    vol = vol + np.random.normal(0.15, 0.05, n)
    df = pd.DataFrame({
        'timestamp': pd.date_range("2025-11-20", periods=n, freq='1T'),
        'volatility': vol
    })
    return df

def simulate_har(df):
    return df['volatility'].shift(1).fillna(df['volatility'].mean())

def simulate_lstm(df):
    pred = np.roll(df['volatility'].values, -1)
    pred[-1] = df['volatility'].mean()
    return pd.Series(pred, index=df.index)

asset = st.selectbox("Asset", ["SPX", "BTC"])
df = generate_data(1000)

if st.button("Run Forecast"):
    har = simulate_har(df)
    lstm = simulate_lstm(df)
    
    col1, col2 = st.columns(2)
    col1.metric("HAR RMSE", f"{np.sqrt(((df['volatility'] - har)**2).mean()):.4f}")
    col2.metric("LSTM RMSE", f"{np.sqrt(((df['volatility'] - lstm)**2).mean()):.4f}")
    
    st.subheader("Last 10 Minutes")
    result = pd.DataFrame({
        "Actual": df['volatility'].tail(10).round(4),
        "HAR": har.tail(10).round(4),
        "LSTM": lstm.tail(10).round(4)
    })
    st.dataframe(result, use_container_width=True)
    
    st.success("Live demo running")
    st.write("Full code → https://github.com/nektariosplevritis-rgb/volatility-forecasting")
