import pandas as pd
import numpy as np


class Backtesting:
    def __init__(
        self,
        portfolio,
        daily_data,
        period,
        no_firms,
    ):
        self.portfolio = portfolio
        self.daily_data = daily_data
        self.period = period
        self.no_firms = no_firms

    def calculate_rsi(self, data, period=14):
        delta = data["Close"].diff(1)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
        avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        data["RSI"] = rsi
        return data

    def calculate_moving_averages(self, data):
        data["MA50"] = data["Close"].rolling(window=50).mean()
        data["MA200"] = data["Close"].rolling(window=200).mean()
        return data

    def calculate_bollinger_bands(self, data, period=20, std_dev=2):
        data["BB_Middle"] = data["Close"].rolling(window=period).mean()
        data["BB_Upper"] = data["BB_Middle"] + (
            std_dev * data["Close"].rolling(window=period).std()
        )
        data["BB_Lower"] = data["BB_Middle"] - (
            std_dev * data["Close"].rolling(window=period).std()
        )
        return data

    def generate_signals(self, data):
        data["Signal"] = "HOLD"

        for i in range(1, len(data)):
            # Moving Average Crossover
            if (
                data["MA50"][i] > data["MA200"][i]
                and data["MA50"][i - 1] <= data["MA200"][i - 1]
            ):
                data.at[i, "Signal"] = "BUY"
            elif (
                data["MA50"][i] < data["MA200"][i]
                and data["MA50"][i - 1] >= data["MA200"][i - 1]
            ):
                data.at[i, "Signal"] = "SELL"

            # RSI conditions
            if data["RSI"][i] < 10:
                data.at[i, "Signal"] = "BUY"
            elif data["RSI"][i] > 90:
                data.at[i, "Signal"] = "SELL"

            # Bollinger Bands conditions
            if data["Close"][i] >= data["BB_Upper"][i]:
                data.at[i, "Signal"] = "SELL"
            elif data["Close"][i] <= data["BB_Lower"][i]:
                data.at[i, "Signal"] = "BUY"

        return data

    def backtest_strategy(self):
        self.daily_data = self.calculate_rsi(self.daily_data)
        self.daily_data = self.calculate_moving_averages(self.daily_data)
        self.daily_data = self.calculate_bollinger_bands(self.daily_data)
        self.daily_data = self.generate_signals(self.daily_data)
        return self.daily_data
