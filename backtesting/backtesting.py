import pandas as pd
import numpy as np
import pprint
import plotly.graph_objects as go
import plotly.subplots as sp
from data.service import *


class Backtesting:
    def __init__(
        self,
        data=None,
        period=14,
        daily_data=None,
    ):
        self.daily_data = daily_data
        self.period = period
        self.data = data

    def initiate_data(self):
        self.daily_data = DataFetcher()
        self.data = self.daily_data.fetch_data()
        self.daily_data.save_to_csv('daily_data.csv')

    def print_data(self):
        if self.data is None:
            raise TypeError("Data is not initiated")
        pprint.pp(self.data)

    def calculate_rsi(self, data=None, period=14):
        if data is None:
            data = self.data

        # Check if data is valid
        if data is None or data.empty:
            raise ValueError("Data is empty. Cannot calculate RSI.")

        # Ensure 'close' column exists
        if "close" not in data.columns:
            raise KeyError(f"Column 'close' not found in DataFrame. Available columns: {list(data.columns)}")

        delta = data["close"].diff(1)

        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = pd.Series(gain, index=data.index).rolling(window=period, min_periods=1).mean()
        avg_loss = pd.Series(loss, index=data.index).rolling(window=period, min_periods=1).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        data["RSI"] = rsi

        return data  # Return RSI as a Series

    def calculate_bollinger_bands(self, data=None, period=20, std_dev=2):
        if data is None:
            data = self.data
        if data.empty:
            return data

        data["BB_Middle"] = data["close"].rolling(window=period).mean()
        data["BB_Upper"] = data["BB_Middle"] + (std_dev * data["close"].rolling(window=period).std())
        data["BB_Lower"] = data["BB_Middle"] - (std_dev * data["close"].rolling(window=period).std())
        return data

    def calculate_sma(self, period, data=None):
        if data is None:
            data = self.data
        if data.empty:
            return data

        column_name = f"SMA{period}"
        data[column_name] = data["close"].rolling(window=period).mean()
        return data

    def apply_indicators(self):
        self.data = self.calculate_rsi()
        self.data = self.calculate_bollinger_bands()
        self.data = self.calculate_sma(50)
        self.data = self.calculate_sma(200)
        pprint.pp(self.data[:25])
        return self.data

    def backtest_strategy(self, capital=100000):
        if self.data is None or self.data.empty:
            print("No data available for backtesting.")
            return

        position = 0
        entry_price = 0
        returns = []
        trend = None

        for i in range(2, len(self.data)):
            sma_diff_prev = self.data["SMA50"].iloc[i-1] - self.data["SMA200"].iloc[i-1]
            sma_diff_now = self.data["SMA50"].iloc[i] - self.data["SMA200"].iloc[i]
            # If SMA difference changes sign, the trend possibly changes
            if sma_diff_prev < 0 and sma_diff_now > 0:
                trend = "up"
            elif sma_diff_prev > 0 and sma_diff_now < 0:
                trend = "down"
            # Entry condition
            if position == 0 and trend:
                if trend == "up" and (self.data["RSI"].iloc[i] < 10 or self.data["close"].iloc[i] <= self.data["BB_Lower"].iloc[i]):
                    position = 1
                    entry_price = self.data["close"].iloc[i]
                elif trend == "down" and (self.data["RSI"].iloc[i] > 90 or self.data["close"].iloc[i] >= self.data["BB_Upper"].iloc[i]):
                    position = -1
                    entry_price = self.data["close"].iloc[i]
            # Exit condition
            elif position == 1: #Close long
                if self.data["RSI"].iloc[i] > 90 or self.data["close"].iloc[i] >= self.data["BB_Upper"].iloc[i]:
                    profit = (self.data["close"].iloc[i] - entry_price) / entry_price * 100
                    capital += capital * (profit / 100)
                    returns.append(profit)
                    position = 0

            elif position == -1: #Close short
                if self.data["RSI"].iloc[i] < 25 or self.data["close"].iloc[i] <= self.data["BB_Lower"].iloc[i]:
                    profit = (entry_price - self.data["close"].iloc[i]) / entry_price * 100
                    capital += capital * (profit / 100)
                    returns.append(profit)
                    position = 0

        total_return = sum(returns)
        win_rate = len([x for x in returns if x > 0]) / len(returns) * 100 if returns else 0
        max_drawdown = min(returns) if returns else 0
        # Convert returns to float before calculating statistics
        # returns_float = np.array([float(r) for r in returns]) if returns else np.array([])

        # Calculate Sharpe Ratio safely
        # sharpe_ratio = np.mean(returns_float) / (np.std(returns_float) + 1e-10) if returns_float.size > 0 else 0
        sharpe_ratio = float(np.mean(returns)) / (float(np.std(returns)) + 1e-10) if returns else 0

        print(f" Final Capital: {capital:.2f} VND")
        print(f" Total Return: {total_return:.2f}%")
        print(f" Win Rate: {win_rate:.2f}%")
        print(f" Max Drawdown: {max_drawdown:.2f}%")
        print(f" Sharpe Ratio: {sharpe_ratio:.2f}")
