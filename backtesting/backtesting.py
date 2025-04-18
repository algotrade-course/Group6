import pandas as pd
import numpy as np
import pprint
import mplfinance as mpf
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
from data.service import *
from evaluator import *

class Backtesting:
    def __init__(
        self,
        period_rsi,
        period_bb,
        in_sample_size,
        risk_per_trade,
        rsi_oversold,
        rsi_overbought,
        rsi_extreme_overbought,
        data=None,
        daily_data=None,
        data_in_sample=None,
        data_out_sample=None
    ):
        self.daily_data = daily_data
        self.period_rsi = period_rsi
        self.period_bb = period_bb
        self.in_sample_size = in_sample_size
        self.risk_per_trade = risk_per_trade
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.rsi_extreme_overbought = rsi_extreme_overbought
        self.data = data
        self.sharpe_ratio = None

    def initiate_data(self, use_csv=False, file_path="daily_data.csv"):
        self.daily_data = DataFetcher()
        if use_csv:
            self.daily_data.load_data_from_csv(file_path)
        else:
            self.data = self.daily_data.fetch_data()
            self.daily_data.save_to_csv(file_path)
        self.data = self.daily_data.df
        self.daily_data.print_dataset()

    def print_data(self):
        if self.data is None:
            raise TypeError("Data is not initiated")
        pprint.pp(self.data[:1000])

    def apply_indicators(self):
        self.data = calculate_rsi(self.data, self.period_rsi)
        self.data = calculate_bollinger_bands(self.data, self.period_rsi, 2)
        self.data = calculate_sma(self.data, 50)
        self.data = calculate_sma(self.data, 200)
        self.data.reset_index(inplace=True)
        self.data["index"] = self.data.index
        pprint.pp(self.data[:1000])
        return self.data    


    def split_data(self, train_size=None):
        if self.data is None or self.data.empty:
            raise ValueError("No data available to split.")
        if train_size is None:
            train_size = self.in_sample_size

        split_index = int(len(self.data) * train_size)
        self.data_in_sample = self.data.iloc[:split_index].copy()
        self.data_out_sample = self.data.iloc[split_index:].copy()

        print(
            f"Data split: {len(self.data_in_sample)} (train), {len(self.data_out_sample)} (test)"
        )
        self.data_in_sample.to_csv("data_in_sample.csv", index=False)
        self.data_out_sample.to_csv("data_out_sample.csv", index=False)

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
            self.data.iloc[211029:228328],
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

    def extract_trades(self, data_test, capital=1000000000, risk_per_trade=None):
        if data_test is None or data_test.empty:
            print("No data available to extract trades.")
            return []
        
        if risk_per_trade is None:
            risk_per_trade = self.risk_per_trade

        position = 0
        entry_price = 0
        trades = []
        trend = None
        capital_map = {data_test["date"].iloc[0]: capital}

        for i in range(2, len(data_test)):
            current_date = data_test["date"].iloc[i]

            sma_diff_prev = data_test["SMA50"].iloc[i - 1] - data_test["SMA200"].iloc[i - 1]
            sma_diff_now = data_test["SMA50"].iloc[i] - data_test["SMA200"].iloc[i]

            if sma_diff_prev < 0 and sma_diff_now > 0:
                trend = "up"
            elif sma_diff_prev > 0 and sma_diff_now < 0:
                trend = "down"

            trade_size = capital * risk_per_trade

            if position == 0 and trend:
                entry_price = float(data_test["close"].iloc[i])
                if trend == "up" and (data_test["RSI"].iloc[i] < self.rsi_oversold or data_test["close"].iloc[i] <= data_test["BB_Lower"].iloc[i]):
                    position = 1
                    date_open = current_date
                    capital_open = capital
                elif trend == "down" and data_test["RSI"].iloc[i] > self.rsi_extreme_overbought:
                    position = -1
                    date_open = current_date
                    capital_open = capital

            elif position == 1:
                if (trend == "up" and (data_test["RSI"].iloc[i] > self.rsi_overbought or data_test["close"].iloc[i] >= data_test["BB_Upper"].iloc[i])) or \
                (trend == "down" and data_test["RSI"].iloc[i] > self.rsi_extreme_overbought):
                    exit_price = float(data_test["close"].iloc[i])
                    profit = (exit_price - entry_price) - 0.47  # Apply fee after trade
                    capital += (profit / entry_price) * trade_size

                    trades.append({
                        'type': 'LONG',
                        'capital_open': capital_open,
                        'capital_close': capital,
                        'date_open': date_open,
                        'date_close': current_date
                    })

                    position = 0

            elif position == -1:
                if (trend == "up" and data_test["RSI"].iloc[i] < self.rsi_oversold) or \
                (trend == "down" and (data_test["RSI"].iloc[i] < self.rsi_oversold or data_test["close"].iloc[i] <= data_test["BB_Lower"].iloc[i])):
                    exit_price = float(data_test["close"].iloc[i])
                    profit = ((exit_price - entry_price) * position) - 0.47  # Apply fee after trade
                    capital += (profit / entry_price) * trade_size

                    trades.append({
                        'type': 'SHORT',
                        'capital_open': capital_open,
                        'capital_close': capital,
                        'date_open': date_open,
                        'date_close': current_date
                    })

                    position = 0

            capital_map[current_date] = capital

        return trades


    # Modify backtest_strategy to store returns and call plot_returns
    def backtest_strategy(self, data_test, capital=1000000000, risk_per_trade=None, print_result=False):
        if data_test is None or data_test.empty:
            print("No data available for backtesting.")
            return
        if risk_per_trade is None:
            risk_per_trade = self.risk_per_trade

        fee = 0.47
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
                if trend == "up" and (data_test["RSI"].iloc[i] < self.rsi_oversold or data_test["close"].iloc[i] <= data_test["BB_Lower"].iloc[i]):
                    position = 1
                elif trend == "down" and data_test["RSI"].iloc[i] > self.rsi_extreme_overbought:
                    position = -1

            elif position == 1:
                if (trend == "up" and (data_test["RSI"].iloc[i] > self.rsi_overbought or data_test["close"].iloc[i] >= data_test["BB_Upper"].iloc[i])) or \
                (trend == "down" and data_test["RSI"].iloc[i] > self.rsi_extreme_overbought):
                    profit = (float(data_test["close"].iloc[i]) - float(entry_price)) - fee
                    capital += (profit / float(entry_price)) * trade_size
                    returns.append(profit / float(entry_price))
                    closing_dates.append(current_date)
                    position = 0  # Exit trade


            elif position == -1:
                if (trend == "up" and data_test["RSI"].iloc[i] < self.rsi_oversold) or \
                (trend == "down" and (data_test["RSI"].iloc[i] < self.rsi_oversold or data_test["close"].iloc[i] <= data_test["BB_Lower"].iloc[i])):
                    profit = ((float(data_test["close"].iloc[i]) - float(entry_price)) * position) - fee
                    capital += (profit / float(entry_price)) * trade_size
                    returns.append(profit / float(entry_price))
                    closing_dates.append(current_date)
                    position = 0  # Exit trade

            capital_map[current_date] = capital

        # Max Drawdown (Peak-to-Trough Drop)
        max_drawdown = calculate_max_drawdown(capital_map)
        # Win Rate
        win_rate = (
            (len([x for x in returns if x > 0]) / len(returns) * 100) if returns else 0
        )
        sharpe_ratio = calculate_sharpe_ratio(returns)

        if self.sharpe_ratio is None:
            self.sharpe_ratio = sharpe_ratio
        
        
        if print_result:
            print(f"Final Capital: {capital:.2f} points")
            print(f"Total Return: {(capital / initial_capital) * 100 - 100:.2f}%")
            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Max Drawdown: {max_drawdown:.6f}%")
            print(f"Sharpe Ratio: {sharpe_ratio:.6f}")
            print(f"Number of Transactions: {len(closing_dates)}")

            self.plot_returns(capital_map)

        return capital_map, len(closing_dates)

    def run_backtest(self, extract_data = False, returns_sharp = False, print_result=False):
        # print("\n--- Running Backtest (100%) ---")
        # self.backtest_strategy(self.data)

        print("\n--- Split data ---")
        self.split_data(self.in_sample_size)
        self.backtest_strategy(self.data_in_sample, print_result=print_result)

        if extract_data:
            trades = self.extract_trades(self.data)
            trades_df = pd.DataFrame(trades)
            trades_df.to_csv("trades_output.csv", index=False)
        # print("Trades saved to trades_output.csv")
        # print(trades_df[:200])
        # self.split_data(0.8)


