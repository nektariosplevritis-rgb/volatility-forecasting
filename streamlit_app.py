import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.data_loader import get_spx_data, get_btc_data
from src.har_model import har_forecast
from src.lstm_model import lstm_forecast
from src.evaluate import evaluate_forecast

st.title("Volatility Forecaster – SPX & BTC 1-min")
st.write("Corporate finance → quant switcher | Live HAR vs LSTM")

asset = st.selectbox("Asset", ["SPX", "BTC"])
if asset == "SPX":
    df = get_spx_data()
else:
    df = get_btc_data()

if st.button("Run Forecast"):
    with st.spinner("Training models..."):
        har_pred = har_forecast(df)
        lstm_pred = lstm_forecast(df)
        metrics = evaluate_forecast(df['volatility'], har_pred, lstm_pred)
    
    st.success("Forecast complete!")
    for k, v in metrics.items():
        st.metric(k, f"{v:.4f}")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df['volatility'].tail(200), name="Actual", line=dict(width=3)))
    fig.add_trace(go.Scatter(y=har_pred.tail(200), name="HAR Forecast"))
    fig.add_trace(go.Scatter(y=lstm_pred.tail(200), name="LSTM Forecast"))
    fig.update_layout(title=f"{asset} Realized Volatility – Last 200 minutes")
    st.plotly_chart(fig, use_container_width=True)
