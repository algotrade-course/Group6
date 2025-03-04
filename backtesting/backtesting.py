import pandas as pd
import numpy as np
import pprint
import plotly.graph_objects as go
import plotly.express as px
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
        self.data.reset_index(inplace=True)
        self.data["index"] = self.data.index 
        
        return self.data

    def split_data(self, train_size=0.7):
        if self.data is None or self.data.empty:
            raise ValueError("No data available to split.")
        
        split_index = int(len(self.data) * train_size)
        self.train_data = self.data.iloc[:split_index].copy()
        self.test_data = self.data.iloc[split_index:].copy()
        
        print(f"Data split: {len(self.train_data)} (train), {len(self.test_data)} (test)")

    
    def plot_chart(self):
        if self.data is None or self.data.empty:
            print("No data available for plotting.")
            return
        
        fig = sp.make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.2,
            subplot_titles=('Candlestick Chart with SMA & Bollinger Bands', 'RSI Indicator'),
            row_heights=[0.7, 0.3]
        )
        
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=self.data['date'],
            open=self.data['open'],
            high=self.data['high'],
            low=self.data['low'],
            close=self.data['close'],
            increasing_line_color='green',
            decreasing_line_color='red'
        ), row=1, col=1)
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(
            x=self.data['date'],
            y=self.data["BB_Upper"],
            mode='lines',
            name="Bollinger Upper",
            line=dict(color='purple', width=1, dash='dot')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=self.data['date'],
            y=self.data["BB_Lower"],
            mode='lines',
            name="Bollinger Lower",
            line=dict(color='purple', width=1, dash='dot')
        ), row=1, col=1)
        
        # SMA lines
        fig.add_trace(go.Scatter(
            x=self.data['date'],
            y=self.data["SMA50"],
            mode='lines',
            name="SMA 50",
            line=dict(color='blue', width=1.5)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=self.data['date'],
            y=self.data["SMA200"],
            mode='lines',
            name="SMA 200",
            line=dict(color='orange', width=1.5)
        ), row=1, col=1)
        
        # RSI Chart
        fig.add_trace(go.Scatter(
            x=self.data['date'],
            y=self.data['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='blue', width=2)
        ), row=2, col=1)
        
        # RSI Levels
        fig.add_hline(y=90, line=dict(color='red', dash="dash"), row=2, col=1)
        fig.add_hline(y=10, line=dict(color='green', dash="dash"), row=2, col=1)
        
        fig.update_layout(
            title="Candlestick Chart with SMA50, SMA200, Bollinger Bands & RSI",
            xaxis_rangeslider_visible=False,
            template="plotly_dark"
        )
        
        fig.show()


    def plot_returns(self, returns, initial_capital=100000, bar_width=0.3):
        if not returns:
            print("No transactions recorded. Cannot plot returns.")
            return
        
        # Compute cumulative capital over time
        capital_history = [initial_capital]
        for ret in returns:
            capital_history.append(capital_history[-1] * (1 + ret / 100.0))
        
        # Create a bar chart with uniform bar width
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=list(range(len(capital_history))),
            y=capital_history,
            width=[bar_width] * len(capital_history),
            text=[f"{cap:.2f}" for cap in capital_history],
            textposition='outside',
            marker={"color" : "green"}
        ))

        # Update layout for better visualization
        fig.update_layout(
            title="Asset Value After Each Transaction",
            xaxis_title="Transaction Dates",
            yaxis_title="Capital (VND)",
            showlegend=False,
        )

        fig.show()

    # Modify backtest_strategy to store returns and call plot_returns
    def backtest_strategy(self, data_test, capital=100000):
        if data_test is None or data_test.empty:
            print("No data available for backtesting.")
            return

        position = 0
        entry_price = 0
        returns = []
        trend = None

        for i in range(2, len(data_test)):
            sma_diff_prev = data_test["SMA50"].iloc[i-1] - data_test["SMA200"].iloc[i-1]
            sma_diff_now = data_test["SMA50"].iloc[i] - data_test["SMA200"].iloc[i]

            if sma_diff_prev < 0 and sma_diff_now > 0:
                trend = "up"
            elif sma_diff_prev > 0 and sma_diff_now < 0:
                trend = "down"

            if position == 0 and trend:
                if trend == "up" and (data_test["RSI"].iloc[i] < 10 or data_test["close"].iloc[i] <= data_test["BB_Lower"].iloc[i]):
                    position = 1
                    entry_price = data_test["close"].iloc[i]
                elif trend == "down" and (data_test["RSI"].iloc[i] > 90 or data_test["close"].iloc[i] >= data_test["BB_Upper"].iloc[i]):
                    position = -1
                    entry_price = data_test["close"].iloc[i]

            elif position == 1:
                if data_test["RSI"].iloc[i] > 90 or data_test["close"].iloc[i] >= data_test["BB_Upper"].iloc[i]:
                    profit = float((data_test["close"].iloc[i] - entry_price) / entry_price * 100)
                    capital += capital * (profit / 100.0)
                    returns.append(profit)
                    position = 0

            elif position == -1:
                if data_test["RSI"].iloc[i] < 25 or data_test["close"].iloc[i] <= data_test["BB_Lower"].iloc[i]:
                    profit = float((entry_price - data_test["close"].iloc[i]) / entry_price * 100)
                    capital += capital * (profit / 100.0)
                    returns.append(profit)
                    position = 0

        total_return = sum(returns)
        win_rate = len([x for x in returns if x > 0]) / len(returns) * 100 if returns else 0
        max_drawdown = min(returns) if returns else 0
        sharpe_ratio = float(np.mean(returns)) / (float(np.std(returns)) + 1e-10) if returns else 0

        print(f" Final Capital: {capital:.2f} VND")
        print(f" Total Return: {total_return:.2f}%")
        print(f" Win Rate: {win_rate:.2f}%")
        print(f" Max Drawdown: {max_drawdown:.2f}%")
        print(f" Sharpe Ratio: {sharpe_ratio:.2f}")
        
        # Call the plotting function
        self.plot_returns(returns, initial_capital=100000)
    
    def backtest_strategy_2(self, data_test, capital=100000):
        if data_test is None or data_test.empty:
            print("No data available for backtesting.")
            return
 
        # 🔥 Fix NaN & Type Issues
        data_test.dropna(inplace=True)
        data_test["RSI"] = pd.to_numeric(data_test["RSI"], errors='coerce')
        data_test["BB_Lower"] = pd.to_numeric(data_test["BB_Lower"], errors='coerce')
        data_test["close"] = pd.to_numeric(data_test["close"], errors='coerce')
 
        position = 0
        entry_price = 0
        returns = []
 
        for i in range(1, len(data_test)):
            if position == 0:
                if data_test["RSI"].iloc[i] < 25 or data_test["close"].iloc[i] <= data_test["BB_Lower"].iloc[i]:
                    position = 1  # Open Long
                    entry_price = data_test["close"].iloc[i]
 
                elif data_test["RSI"].iloc[i] > 90 or data_test["close"].iloc[i] >= data_test["BB_Upper"].iloc[i]:
                    position = -1  # Open Short
                    entry_price = data_test["close"].iloc[i]
 
            elif position == 1:  # Close Long
                if data_test["RSI"].iloc[i] > 90 or data_test["close"].iloc[i] >= data_test["BB_Upper"].iloc[i]:
                    profit = (data_test["close"].iloc[i] - entry_price) / entry_price * 100
                    capital += capital * (profit / 100)
                    returns.append(profit)
                    position = 0
 
            elif position == -1:  # Close Short
                if data_test["RSI"].iloc[i] < 25 or data_test["close"].iloc[i] <= data_test["BB_Lower"].iloc[i]:
                    profit = (entry_price - data_test["close"].iloc[i]) / entry_price * 100
                    capital += capital * (profit / 100)
                    returns.append(profit)
                    position = 0    
 
        total_return = sum(returns)
        win_rate = len([x for x in returns if x > 0]) / len(returns) * 100 if returns else 0
        max_drawdown = min(returns) if returns else 0
        sharpe_ratio = float(np.mean(returns)) / (float(np.std(returns)) + 1e-10) if returns else 0
 
        print(f" Final Capital: {capital:.2f} VND")
        print(f" Total Return: {total_return:.2f}%")
        print(f" Win Rate: {win_rate:.2f}%")
        print(f" Max Drawdown: {max_drawdown:.2f}%")
        print(f" Sharpe Ratio: {sharpe_ratio:.2f}")
 
        # Call the plotting function
        self.plot_returns(returns, initial_capital=100000)

    #Split the test case into in-sample (80%) and out-sample (20%)
    def run_backtest(self):
        print("\n--- Running Backtest (100%) ---")
        self.backtest_strategy(self.data)

        print("\n--- Split data ---")
        self.split_data()

        print("\n--- Running In-Sample Backtest (70%) ---")
        self.backtest_strategy(self.train_data)
        
        print("\n--- Running Out-of-Sample Backtest (30%) ---")
        self.backtest_strategy(self.test_data)

    def run_backtest_2(self):
        print("\n--- Running Backtest (100%) ---")
        self.backtest_strategy(self.data)

        print("\n--- Split data ---")
        self.split_data()

        print("\n--- Running In-Sample Backtest (70%) ---")
        self.backtest_strategy_2(self.train_data)
        
        print("\n--- Running Out-of-Sample Backtest (30%) ---")
        self.backtest_strategy_2(self.test_data)