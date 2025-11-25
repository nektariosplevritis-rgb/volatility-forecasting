import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import torch
import torch.nn as nn

st.set_page_config(page_title="Vol Forecaster", layout="wide")
st.title("Live Volatility Forecaster – SPX & BTC 1-min Demo")
st.markdown("**Corporate finance → quant switcher** | HAR + LSTM | Nov 2025")

# Synthetic data generator (demo data, replace with yfinance later locally)
def generate_synthetic_data(n_points=1000, asset="SPX"):
    np.random.seed(42)
    returns = np.random.normal(0, 0.01, n_points)
    vol = np.abs(returns).rolling(20).std() * np.sqrt(252 * 390) + np.random.normal(0.15, 0.05, n_points)
    df = pd.DataFrame({'timestamp': pd.date_range(start='2025-11-20', periods=n_points, freq='1T'), 'volatility': vol})
    return df

# HAR model
def har_forecast(df):
    df = df.copy()
    df['vol_lag1'] = df['volatility'].shift(1)
    df['vol_lag5'] = df['volatility'].rolling(5).mean().shift(1)
    df['vol_lag22'] = df['volatility'].rolling(22).mean().shift(1)
    df = df.dropna()
    X = sm.add_constant(df[['vol_lag1', 'vol_lag5', 'vol_lag22']])
    model = sm.OLS(df['volatility'], X).fit()
    df['har_pred'] = model.predict(X)
    return df['har_pred']

# Simple LSTM
class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 50, batch_first=True)
        self.fc = nn.Linear(50, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

def lstm_forecast(df, seq_len=60):
    data = df['volatility'].values[-500:]
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

# Main app
asset = st.selectbox("Choose asset", ["SPX", "BTC"])
df = generate_synthetic_data(1000, asset)

if st.button("Run Forecast (demo data)"):
    with st.spinner("Running HAR + LSTM..."):
        har_pred = har_forecast(df)
        lstm_pred = lstm_forecast(df)
    
    st.success("Forecast complete! (Demo with synthetic data)")
    
    # Metrics
    rmse_har = np.sqrt(((df['volatility'] - har_pred)**2).mean())
    rmse_lstm = np.sqrt(((df['volatility'].tail(len(lstm_pred)) - lstm_pred)**2).mean())
    dir_har = (df['volatility'] * har_pred > 0).mean()
    dir_lstm = (df['volatility'].tail(len(lstm_pred)) * lstm_pred > 0).mean()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("HAR RMSE", f"{rmse_har:.4f}")
    col2.metric("LSTM RMSE", f"{rmse_lstm:.4f}")
    col3.metric("HAR Directional Acc", f"{dir_har:.4f}")
    col4.metric("LSTM Directional Acc", f"{dir_lstm:.4f}")
    
    # Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index[-200:], df['volatility'].tail(200), label="Actual Vol", linewidth=2)
    ax.plot(df.index[-200:], har_pred.tail(200), label="HAR Forecast", color="red")
    ax.plot(lstm_pred.tail(200).index, lstm_pred.tail(200), label="LSTM Forecast", color="green")
    ax.legend()
    ax.set_title(f"{asset} 1-min Volatility Forecast (Demo)")
    st.pyplot(fig)
    
    st.write("**Full code:** https://github.com/nektariosplevritis-rgb/volatility-forecasting")
    st.write("**Real data version:** Swap `generate_synthetic_data` with yfinance locally.")
