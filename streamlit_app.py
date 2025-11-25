import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
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
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("HAR RMSE", f"{metrics['HAR RMSE']:.4f}")
    col2.metric("LSTM RMSE", f"{metrics['LSTM RMSE']:.4f}")
    col3.metric("HAR Directional", f"{metrics['HAR Directional']:.4f}")
    col4.metric("LSTM Directional", f"{metrics['LSTM Directional']:.4f}")
    
    # Plotly chart (now fixed with version pin)
    fig = px.line(df.tail(200), y='volatility', title=f"{asset} Realized Volatility – Last 200 minutes")
    fig.add_scatter(y=har_pred.tail(200).values, name="HAR Forecast", line=dict(color='red'))
    fig.add_scatter(y=lstm_pred.tail(200).values, name="LSTM Forecast", line=dict(color='green'))
    st.plotly_chart(fig, use_container_width=True)
    
    # Backup Matplotlib chart if Plotly fails
    fig_mat, ax = plt.subplots()
    ax.plot(df.tail(200).index, df['volatility'].tail(200), label='Actual', linewidth=2)
    ax.plot(df.tail(200).index, har_pred.tail(200).values, label='HAR', color='red')
    ax.plot(df.tail(200).index, lstm_pred.tail(200).values, label='LSTM', color='green')
    ax.legend()
    ax.set_title(f"{asset} Volatility Forecast")
    st.pyplot(fig_mat)
