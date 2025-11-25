import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(page_title="Vol Forecaster", layout="wide")
st.title("Live Volatility Forecaster – SPX & BTC 1-min")
st.markdown("**Corporate finance → quant switcher** | HAR + LSTM | Nov 2025")

# Simple data loader
@st.cache_data(ttl=300)
def load_data(asset):
    if asset == "SPX":
        df = yf.download("^GSPC", period="5d", interval="1m")
    else:
        df = yf.download("BTC-USD", period="5d", interval="1m")
    df['return'] = df['Close'].pct_change()
    df['vol'] = df['return'].rolling(20).std() * (252*390)**0.5
    return df.dropna()

# HAR model (simple)
def har_forecast(df):
    df = df.copy()
    df['vol_lag1'] = df['vol'].shift(1)
    df['vol_lag5'] = df['vol'].rolling(5).mean().shift(1)
    df['vol_lag22'] = df['vol'].rolling(22).mean().shift(1)
    df = df.dropna()
    X = sm.add_constant(df[['vol_lag1', 'vol_lag5', 'vol_lag22']])
    model = sm.OLS(df['vol'], X).fit()
    df['har_pred'] = model.predict(X)
    return df['har_pred']

# Super-simple LSTM (no external deps)
class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 50, batch_first=True)
        self.fc = nn.Linear(50, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

def lstm_forecast(df, seq_len=60):
    data = df['vol'].values[-500:]  # last 500 points
    if len(data) < seq_len + 10:
        return pd.Series([data.mean()] * len(data), index=df.index[-len(data):])
    
    scaled = (data - data.min()) / (data.max() - data.min() + 1e-8)
    X = []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i+seq_len])
    X = torch.FloatTensor(X).unsqueeze(-1)
    
    model = SimpleLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for _ in range(15):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out.squeeze(), torch.FloatTensor(scaled[seq_len:]))
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        pred = model(X).squeeze().numpy()
        pred = pred * (data.max() - data.min()) + data.min()
    
    return pd.Series(pred, index=df.index[-len(pred):])

asset = st.selectbox("Choose asset", ["SPX", "BTC"])
df = load_data(asset)

if st.button("Run Forecast (takes ~15 sec)"):
    with st.spinner("Running HAR + LSTM..."):
        har_pred = har_forecast(df)
        lstm_pred = lstm_forecast(df)
    
    st.success("Done!")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index[-200:], df['vol'].tail(200), label="Actual", linewidth=2)
    ax.plot(df.index[-200:], har_pred.tail(200), label="HAR Forecast", color="red")
    ax.plot(lstm_pred.tail(200).index, lstm_pred.tail(200), label="LSTM Forecast", color="green")
    ax.legend()
    ax.set_title(f"{asset} 1-min Realized Volatility – Last 200 minutes")
    st.pyplot(fig)
    
    st.write("Repo: https://github.com/nektariosplevritis-rgb/volatility-forecasting")
