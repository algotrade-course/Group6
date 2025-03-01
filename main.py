from backtesting.backtesting import *
from data.service import *
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp

# Initialize backtesting
backtesting = Backtesting()
backtesting.initiate_data()
# backtesting.print_data()

# Load CSV file
df = pd.read_csv('daily_data.csv')

# Ensure date column is in datetime format
df['date'] = pd.to_datetime(df['date'])

# Sort by date
df = df.sort_values(by='date')

# ✅ RSI Calculation Function
def calculate_rsi(data, period=14):
    delta = data["close"].diff(1)

    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    return rsi

# ✅ Bollinger Bands Calculation
def calculate_bollinger_bands(data, period=20, std_dev=2):
    rolling_mean = data["close"].rolling(window=period).mean()
    rolling_std = data["close"].rolling(window=period).std()

    data["BB_Middle"] = rolling_mean
    data["BB_Upper"] = rolling_mean + (std_dev * rolling_std)
    data["BB_Lower"] = rolling_mean - (std_dev * rolling_std)

    return data

# ✅ Moving Averages Calculation
def calculate_sma(data, period):
    return data["close"].rolling(window=period).mean()

# Apply RSI, Bollinger Bands & Moving Averages
df["RSI"] = calculate_rsi(df)
df = calculate_bollinger_bands(df)
df["SMA50"] = calculate_sma(df, 50)
df["SMA200"] = calculate_sma(df, 200)

# ✅ Create subplots: Candlestick chart + RSI
fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.2, subplot_titles=('Candlestick Chart with SMA & Bollinger Bands', 'RSI Indicator'),
                       row_heights=[0.7, 0.3])

# 📈 Candlestick chart with Bollinger Bands
fig.add_trace(go.Candlestick(
    x=df['date'],
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close'],
    increasing_line_color='green',
    decreasing_line_color='red'
), row=1, col=1)

# 📊 Bollinger Bands
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df["BB_Upper"],
    mode='lines',
    name="Bollinger Upper",
    line=dict(color='purple', width=1, dash='dot')
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=df['date'],
    y=df["BB_Lower"],
    mode='lines',
    name="Bollinger Lower",
    line=dict(color='purple', width=1, dash='dot')
), row=1, col=1)

# 📊 SMA50 (Short-term Moving Average)
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df["SMA50"],
    mode='lines',
    name="SMA 50",
    line=dict(color='blue', width=1.5)
), row=1, col=1)

# 📊 SMA200 (Long-term Moving Average)
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df["SMA200"],
    mode='lines',
    name="SMA 200",
    line=dict(color='orange', width=1.5)
), row=1, col=1)

# 📉 RSI Chart
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['RSI'],
    mode='lines',
    name='RSI',
    line=dict(color='blue', width=2)
), row=2, col=1)

# 🔥 Add RSI Levels (Overbought 70, Oversold 30)
fig.add_hline(y=90, line=dict(color='red', dash="dash"), row=2, col=1)
fig.add_hline(y=10, line=dict(color='green', dash="dash"), row=2, col=1)

# 🎨 Layout settings
fig.update_layout(
    title="Candlestick Chart with SMA50, SMA200, Bollinger Bands & RSI",
    xaxis_rangeslider_visible=False,
    template="plotly_dark"
)

# 🚀 Show the chart
# fig.show()

# =================================================================================================
# =================================================================================================
# =================================================================================================
# =================================================================================================
# ===========# ====================================================================================
# =================================================================================================
# =================================================================================================
# =================================================================================================
# =================================================================================================
# =================================================================================================
# =================================================================================================
# =================================================================================================
# Code here
# ✅ Backtesting Strategy
capital = 100000  # Initial capital (100,000 VND)
position = 0       # Current position (1 for long, -1 for short, 0 for none)
entry_price = 0
returns = []       # Store profit/loss
for i in range(1, len(df)):
    # ✅ ENTRY CONDITIONS
    if position == 0:
        if df["RSI"].iloc[i] < 10 or df["close"].iloc[i] <= df["BB_Lower"].iloc[i]:
            position = 1  # Open Long
            entry_price = df["close"].iloc[i]

        elif df["RSI"].iloc[i] > 90 or df["close"].iloc[i] >= df["BB_Upper"].iloc[i]:
            position = -1  # Open Short
            entry_price = df["close"].iloc[i]

    # ✅ EXIT CONDITIONS
    elif position == 1:  # Close Long
        if df["RSI"].iloc[i] > 90 or df["close"].iloc[i] >= df["BB_Upper"].iloc[i]:
            profit = (df["close"].iloc[i] - entry_price) / entry_price * 100
            capital += capital * (profit / 100)
            returns.append(profit)
            position = 0

    elif position == -1:  # Close Short
        if df["RSI"].iloc[i] < 25 or df["close"].iloc[i] <= df["BB_Lower"].iloc[i]:
            profit = (entry_price - df["close"].iloc[i]) / entry_price * 100
            capital += capital * (profit / 100)
            returns.append(profit)
            position = 0

# ✅ Performance Metrics
total_return = sum(returns)
win_rate = len([x for x in returns if x > 0]) / len(returns) * 100 if returns else 0
max_drawdown = min(returns) if returns else 0
sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) if returns else 0

# ✅ Results
print(f"📈 Final Capital: {capital:.2f} VND")
print(f"✅ Total Return: {total_return:.2f}%")
print(f"🏆 Win Rate: {win_rate:.2f}%")
print(f"📉 Max Drawdown: {max_drawdown:.2f}%")
print(f"⚡ Sharpe Ratio: {sharpe_ratio:.2f}")