import pandas as pd
import numpy as np

def calculate_rsi(data, period):
    # Check if data is valid
    if data is None or data.empty:
        raise ValueError("Data is empty. Cannot calculate RSI.")

    # Ensure 'close' column exists
    if "close" not in data.columns:
        raise KeyError(
            f"Column 'close' not found in DataFrame. Available columns: {list(data.columns)}"
        )

    delta = data["close"].diff(1)

    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = (
        pd.Series(gain, index=data.index)
        .rolling(window=period, min_periods=1)
        .mean()
    )
    avg_loss = (
        pd.Series(loss, index=data.index)
        .rolling(window=period, min_periods=1)
        .mean()
    )

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    data["RSI"] = rsi

    return data

def calculate_bollinger_bands(data, period, std_dev):
    data["BB_Middle"] = data["close"].rolling(window=period).mean()
    data["BB_Upper"] = data["BB_Middle"] + (
        std_dev * data["close"].rolling(window=period).std()
    )
    data["BB_Lower"] = data["BB_Middle"] - (
        std_dev * data["close"].rolling(window=period).std()
    )
    return data

def calculate_sma(data, period):
    column_name = f"SMA{period}"
    data[column_name] = data["close"].rolling(window=period).mean()
    return data

def calculate_sharpe_ratio(returns):
    if not returns:
        return 0.0

    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)

    if std_return == 0:
        return 0.0

    return mean_return / std_return

def calculate_max_drawdown(capital_map):
    if not capital_map:
        return 0.0
    
    peak = float('-inf')
    max_drawdown = 0.0

    for capital in capital_map.values():
        if capital > peak:
            peak = capital
        drawdown = (peak - capital) / peak * 100
        max_drawdown = max(max_drawdown, drawdown)

    return max_drawdown

