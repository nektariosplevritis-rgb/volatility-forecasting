import pandas as pd
import statsmodels.api as sm

def har_forecast(df):
    df = df.copy()
    df['vol_1'] = df['volatility'].shift(1)
    df['vol_5'] = df['volatility'].rolling(5).mean().shift(1)
    df['vol_22'] = df['volatility'].rolling(22).mean().shift(1)
    df = df.dropna()
    
    X = df[['vol_1', 'vol_5', 'vol_22']]
    X = sm.add_constant(X)
    y = df['volatility']
    
    model = sm.OLS(y, X).fit()
    pred = model.predict(X)
    return pd.Series(pred, index=df.index)
