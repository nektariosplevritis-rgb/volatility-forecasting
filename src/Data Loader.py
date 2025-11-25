import yfinance as yf
import pandas as pd
from binance.client import Client

def get_spx_data():
    spx = yf.download("^GSPC", period="5d", interval="1m")
    spx['returns'] = spx['Close'].pct_change()
    spx['volatility'] = spx['returns'].rolling(20).std() * (252*390)**0.5
    return spx.dropna()

def get_btc_data():
    client = Client()
    klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1MINUTE, "5 days ago UTC")
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
    df['close'] = pd.to_numeric(df['close'])
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std() * (365*1440)**0.5
    return df.dropna()
