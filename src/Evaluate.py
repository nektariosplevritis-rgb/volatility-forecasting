import pandas as pd

def evaluate_forecast(actual, har_pred, lstm_pred):
    results = {
        "HAR RMSE": ((actual - har_pred)**2).mean()**0.5,
        "LSTM RMSE": ((actual[-len(lstm_pred):] - lstm_pred)**2).mean()**0.5,
        "HAR Directional": ((actual * har_pred > 0).mean()),
        "LSTM Directional": ((actual[-len(lstm_pred):] * lstm_pred > 0).mean())
    }
    return results
