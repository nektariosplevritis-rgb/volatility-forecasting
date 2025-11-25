import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from src.data_loader import get_data
from src.har_model import har_forecast
from src.lstm_model import lstm_forecast

st.title("Volatility Forecaster – SPX & BTC 1-min")
st.write("Corporate finance → quant switcher | Live HAR + LSTM")

asset = st.selectbox("Asset", ["SPX", "BTCUSDT"])
df = get_data(asset)

if st.button("Run Forecast"):
    har_pred = har_forecast(df)
    lstm_pred = lstm_forecast(df)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df['volatility'].tail(100), name="Actual"))
    fig.add_trace(go.Scatter(y=har_pred[-100:], name="HAR"))
    fig.add_trace(go.Scatter(y=lstm_pred[-100:], name="LSTM"))
    st.plotly_chart(fig)
