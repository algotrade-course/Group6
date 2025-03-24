import pandas as pd
import numpy as np
import pprint
import mplfinance as mpf
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
        self.daily_data.save_to_csv("daily_data.csv")

    def print_data(self):
        if self.data is None:
            raise TypeError("Data is not initiated")
        pprint.pp(self.data[:100])

    def calculate_rsi(self, data=None, period=7):
        if data is None:
            data = self.data

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

        return data  # Return RSI as a Series

    def calculate_bollinger_bands(self, data=None, period=50, std_dev=2):
        if data is None:
            data = self.data
        if data.empty:
            return data

        data["BB_Middle"] = data["close"].rolling(window=period).mean()
        data["BB_Upper"] = data["BB_Middle"] + (
            std_dev * data["close"].rolling(window=period).std()
        )
        data["BB_Lower"] = data["BB_Middle"] - (
            std_dev * data["close"].rolling(window=period).std()
        )
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
        # pprint.pp(self.data[:20])
        return self.data

    def split_data(self, train_size=0.7):
        if self.data is None or self.data.empty:
            raise ValueError("No data available to split.")

        split_index = int(len(self.data) * train_size)
        self.train_data = self.data.iloc[:split_index].copy()
        self.test_data = self.data.iloc[split_index:].copy()

        print(
            f"Data split: {len(self.train_data)} (train), {len(self.test_data)} (test)"
        )

    def plot_candlestick_chart(self):
        if self.data is None or self.data.empty:
            print("No data available for plotting.")
            return

        # Ensure 'date' is datetime and set as index
        self.data["date"] = pd.to_datetime(self.data["date"])
        self.data.set_index("date", inplace=True)

        mc = mpf.make_marketcolors(
            up="green",
            down="red",  # Up (bullish) = Green, Down (bearish) = Red
            edge="inherit",  # Make candlestick edges match body color
            wick="inherit",  # Make wicks match candlestick body color
            volume="inherit",  # Optional: Match volume bars
        )

        s = mpf.make_mpf_style(marketcolors=mc, gridcolor="gray")

        # Convert price columns to float
        price_columns = ["open", "high", "low", "close"]
        self.data[price_columns] = self.data[price_columns].astype(float)

        # Plot candlestick chart (minute-level data)
        mpf.plot(
            self.data.iloc[260000:270000],
            type="candle",
            title="VN30F1M Candlestick Chart (Minute Data)",
            style=s,
            figsize=(15, 10),
            warn_too_much_data=100000,  # Increase limit if needed
            ylim=(self.data["low"].min() - 10, self.data["high"].max() + 10),
            datetime_format="%Y-%m-%d %H:%M",
        )

    def plot_chart(self):
        if self.data is None or self.data.empty:
            print("No data available for plotting.")
            return

        fig = sp.make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.5,
            subplot_titles=(
                "Candlestick Chart with SMA & Bollinger Bands",
                "RSI Indicator",
            ),
            row_heights=[0.7, 0.3],
        )

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=self.data["date"],
                open=self.data["open"],
                high=self.data["high"],
                low=self.data["low"],
                close=self.data["close"],
                increasing_line_color="green",
                decreasing_line_color="red",
            ),
            row=1,
            col=1,
        )

        # Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=self.data["date"],
                y=self.data["BB_Upper"],
                mode="lines",
                name="Bollinger Upper",
                line=dict(color="purple", width=1, dash="dot"),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=self.data["date"],
                y=self.data["BB_Lower"],
                mode="lines",
                name="Bollinger Lower",
                line=dict(color="purple", width=1, dash="dot"),
            ),
            row=1,
            col=1,
        )

        # SMA lines
        fig.add_trace(
            go.Scatter(
                x=self.data["date"],
                y=self.data["SMA50"],
                mode="lines",
                name="SMA 50",
                line=dict(color="blue", width=1.5),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=self.data["date"],
                y=self.data["SMA200"],
                mode="lines",
                name="SMA 200",
                line=dict(color="orange", width=1.5),
            ),
            row=1,
            col=1,
        )

        # RSI Chart
        fig.add_trace(
            go.Scatter(
                x=self.data["date"],
                y=self.data["RSI"],
                mode="lines",
                name="RSI",
                line=dict(color="blue", width=2),
            ),
            row=2,
            col=1,
        )

        # RSI Levels
        fig.add_hline(y=90, line=dict(color="red", dash="dash"), row=2, col=1)
        fig.add_hline(y=10, line=dict(color="green", dash="dash"), row=2, col=1)

        fig.update_layout(
            title="Candlestick Chart with SMA50, SMA200, Bollinger Bands & RSI",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
        )

        fig.show()

    def plot_returns(self, capital_map):
        if capital_map is None or not capital_map:
            print("No capital data available. Cannot plot returns.")
            return

        # Convert dictionary to DataFrame
        capital_df = pd.DataFrame(list(capital_map.items()), columns=["date", "capital"])

        # Convert date column to datetime format
        capital_df["date"] = pd.to_datetime(capital_df["date"])

        # Create a line chart using Plotly
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=capital_df["date"],
                y=capital_df["capital"],
                mode="lines",
                line=dict(color="blue", width=2),
                text=[f"{cap:.2f}" for cap in capital_df["capital"]],
                textposition="top center",
            )
        )

        # Update layout for better visualization
        fig.update_layout(
            title="Portfolio Capital Over Time",
            xaxis_title="Date",
            yaxis_title="Capital (VND)",
            xaxis=dict(type="date"),
            showlegend=False,
        )

        fig.show()


    # Modify backtest_strategy to store returns and call plot_returns
    def backtest_strategy(self, data_test, capital=100000, risk_per_trade=0.02):
        if data_test is None or data_test.empty:
            print("No data available for backtesting.")
            return

        position = 0
        entry_price = 0
        returns = []
        closing_dates = []
        trend = None
        initial_capital = capital
        capital_map = {data_test["date"].iloc[0]: capital}

        for i in range(2, len(data_test)):
            current_date = data_test["date"].iloc[i]
            
            # Existing logic to determine SMA crossover
            sma_diff_prev = data_test["SMA50"].iloc[i - 1] - data_test["SMA200"].iloc[i - 1]
            sma_diff_now = data_test["SMA50"].iloc[i] - data_test["SMA200"].iloc[i]
            
            if sma_diff_prev < 0 and sma_diff_now > 0:
                trend = "up"
            elif sma_diff_prev > 0 and sma_diff_now < 0:
                trend = "down"

            trade_size = capital * risk_per_trade

            if position == 0 and trend:
                entry_price = float(data_test["close"].iloc[i])  # Ensure float conversion
                if trend == "up" and (data_test["RSI"].iloc[i] < 15 or data_test["close"].iloc[i] <= data_test["BB_Lower"].iloc[i]):
                    position = 1
                elif trend == "down" and data_test["RSI"].iloc[i] > 90:
                    position = -1

            elif position == 1:
                if (trend == "up" and (data_test["RSI"].iloc[i] > 85 or data_test["close"].iloc[i] >= data_test["BB_Upper"].iloc[i])) or \
                (trend == "down" and data_test["RSI"].iloc[i] > 90):
                    profit = float(float(data_test["close"].iloc[i]) - float(entry_price)) * position
                    capital += (profit / float(entry_price)) * trade_size
                    returns.append(profit / float(entry_price))
                    closing_dates.append(current_date)
                    position = 0  # Exit trade

            elif position == -1:
                if (trend == "up" and data_test["RSI"].iloc[i] < 15) or (trend == "down" and (data_test["RSI"].iloc[i] < 15 or data_test["close"].iloc[i] <= data_test["BB_Lower"].iloc[i])):
                    profit = float(float(data_test["close"].iloc[i]) - float(entry_price)) * position
                    capital += (float(profit) / float(entry_price)) * float(trade_size)
                    returns.append(profit / float(entry_price))
                    closing_dates.append(current_date)
                    position = 0  # Exit trade

            capital_map[current_date] = capital

        # Max Drawdown (Peak-to-Trough Drop)
        max_capital = max(capital_map.values())
        min_capital = min(capital_map.values())
        max_drawdown = (max_capital - min_capital) / max_capital * 100

        # Win Rate
        win_rate = (
            (len([x for x in returns if x > 0]) / len(returns) * 100) if returns else 0
        )

        # Sharpe Ratio for 1-Minute Trading (Annualized)
        sharpe_ratio = (
            (
                float(np.mean(returns))
                / float(np.std(returns, ddof=1))
            )
            if returns
            else 0
        )
        

        print(f"Final Capital: {capital:.2f} VND")
        print(f"Total Return: {(capital / initial_capital) * 100 - 100:.2f}%")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Number of Transactions: {len(closing_dates)}")

        # self.plot_returns(capital_map)

        return capital_map, len(closing_dates)

    # Split the test case into in-sample (70%) and out-sample (30%)
    def run_backtest(self):
        # print("\n--- Running Backtest (100%) ---")
        # self.backtest_strategy(self.data)

        print("\n--- Split data ---")
        self.split_data()

        print("\n--- Running In-Sample Backtest (70%) ---")
        self.backtest_strategy(self.train_data)

        print("\n--- Running Out-of-Sample Backtest (30%) ---")
        self.backtest_strategy(self.test_data)
