import pandas as pd
import numpy as np
import pprint
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

    def print_data(self):
        if (self.data is None):
            raise TypeError("Data is not initiated")
        pprint.pp(self.data)
        print(self.data["close"])

    def calculate_rsi(self, data=None, period=14):
        if data is None:
            data = self.data
        if data.empty:
            return data

        delta = data["close"].diff(1)
        
        gain = pd.Series(np.where(delta > 0, delta, 0), index=data.index)
        loss = pd.Series(np.where(delta < 0, -delta, 0), index=data.index)

        avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
        avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        data.loc[:, "RSI"] = rsi

        return data

    def calculate_moving_averages(self, period, data=None):
        if period is None:
            raise TypeError("No period in calculating MA")
        if data is None:
            data = self.data
        if data.empty:
            return data

        columns_name = "MA" + str(period)
        data[columns_name] = data["close"].rolling(window=period).mean()
        
        return data

    def calculate_bollinger_bands(self, data=None, period=20, std_dev=2):
        if data is None:
            data = self.data
        if data.empty:
            return data

        data["BB_Middle"] = data["close"].rolling(window=period).mean()
        data["BB_Upper"] = data["BB_Middle"] + (std_dev * data["close"].rolling(window=period).std())
        data["BB_Lower"] = data["BB_Middle"] - (std_dev * data["close"].rolling(window=period).std())
        return data

    def generate_signals(self, data=None):
        if data is None:
            data = self.data
        if data.empty:
            return data

        data["Signal"] = "HOLD"

        for i in range(1, len(data)):
            # Moving Average Crossover
            if data["MA50"][i] > data["MA200"][i] and data["MA50"][i - 1] <= data["MA200"][i - 1]:
                data.at[i, "Signal"] = "BUY"
            elif data["MA50"][i] < data["MA200"][i] and data["MA50"][i - 1] >= data["MA200"][i - 1]:
                data.at[i, "Signal"] = "SELL"

            # RSI conditions
            if data["RSI"][i] < 10:
                data.at[i, "Signal"] = "BUY"
            elif data["RSI"][i] > 90:
                data.at[i, "Signal"] = "SELL"

            # Bollinger Bands conditions
            if data["close"][i] >= data["BB_Upper"][i]:
                data.at[i, "Signal"] = "SELL"
            elif data["close"][i] <= data["BB_Lower"][i]:
                data.at[i, "Signal"] = "BUY"

        return data

    def backtest_strategy(self):
        if self.data.empty:
            print("No data available for backtesting.")
            return self.data

        self.data = self.calculate_rsi(self.data)
        self.data = self.calculate_moving_averages(self.data)
        self.data = self.calculate_bollinger_bands(self.data)
        self.data = self.generate_signals(self.data)
        return self.data
