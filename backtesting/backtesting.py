import pandas as pd
import numpy as np
import pprint
import mplfinance as mpf
import matplotlib.pyplot as plt
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
        stop_loss,
        take_profit,
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
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.data = data
        self.sharpe_ratio = None
        self.total_return = None
        self.fee = 0.47

    def initiate_data(self, use_csv=False, file_path="daily_data.csv"):
        self.daily_data = DataFetcher()
        if use_csv:
            self.daily_data.load_data_from_csv(file_path)
        else:
            self.data = self.daily_data.fetch_data()
            self.daily_data.save_to_csv(file_path)
        self.data = self.daily_data.df

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
        
        return self.data    


    def split_data(self, train_size=None, print_result=False):
        if self.data is None or self.data.empty:
            raise ValueError("No data available to split.")
        if train_size is None:
            train_size = self.in_sample_size

        split_index = int(len(self.data) * train_size)
        self.data_in_sample = self.data.iloc[:split_index].copy()
        self.data_out_sample = self.data.iloc[split_index:].copy()

        if print_result:
            print("\n--- Split data ---")
            print(
            f"Data split: {len(self.data_in_sample)} (train), {len(self.data_out_sample)} (test)"
        )
        
        self.data_in_sample.to_csv("data_in_sample.csv", index=False)
        self.data_out_sample.to_csv("data_out_sample.csv", index=False)

    def plot_candlestick_chart(self, output_file="all_sample_data.png"):
        if self.data is None or self.data.empty:
            print("No data available for plotting.")
            return

        # Ensure 'date' is datetime and set as index
        self.data["date"] = pd.to_datetime(self.data["date"])
        self.data.set_index("date", inplace=True)

        # Convert price columns to float
        price_columns = ["open", "high", "low", "close"]
        self.data[price_columns] = self.data[price_columns].astype(float)

        # Extract 80% of the data
        total_rows = len(self.data)
        rows_to_plot = int(total_rows * 1)
        data_to_plot = self.data.iloc[:rows_to_plot]

        # Create market colors and style
        mc = mpf.make_marketcolors(
            up="green",
            down="red",
            edge="inherit",
            wick="inherit",
            volume="inherit",
        )
        s = mpf.make_mpf_style(marketcolors=mc, gridcolor="gray")

        # Plot and save to PNG
        mpf.plot(
            data_to_plot,
            type="candle",
            title="VN30F1M Candlestick Chart (Sample Data)",
            style=s,
            figsize=(50, 10),
            warn_too_much_data=200000,
            ylim=(data_to_plot["low"].min() - 10, data_to_plot["high"].max() + 10),
            datetime_format="%Y-%m-%d %H:%M",
            savefig=dict(fname=output_file, dpi=150, bbox_inches="tight"),
        )

        print(f"Candlestick chart saved to {output_file}")


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

    


    def plot_returns(self, capital_map, output_file="returns.png"):
        if capital_map is None or not capital_map:
            print("No capital data available. Cannot plot returns.")
            return

        # Convert dictionary to DataFrame
        capital_df = pd.DataFrame(list(capital_map.items()), columns=["date", "capital"])
        capital_df["date"] = pd.to_datetime(capital_df["date"])

        # Plot using matplotlib
        plt.figure(figsize=(12, 6))
        plt.plot(capital_df["date"], capital_df["capital"], color="blue", linewidth=2)
        plt.title("Portfolio Capital Over Time")
        plt.xlabel("Date")
        plt.ylabel("Capital (VND)")
        plt.grid(True)

        # Show in a pop-up window
        plt.tight_layout()
        plt.show()

        # Save as PNG
        try:
            plt.savefig(output_file)
            print(f"Return plot saved to {output_file}")
        except Exception as e:
            print("Error saving image:", e)



    def extract_trades(self, data_test, capital=1000000000, risk_per_trade=None, fee_add=0.47):
        if data_test is None or data_test.empty:
            print("No data available to extract trades.")
            return []

        if risk_per_trade is None:
            risk_per_trade = self.risk_per_trade

        position = 0
        entry_price = 0
        trades = []
        capital_map = {data_test["date"].iloc[0]: capital}
        fee = fee_add

        for i in range(2, len(data_test)):
            current_date = data_test["date"].iloc[i]
            current_price = float(data_test["close"].iloc[i])
            trade_size = capital * risk_per_trade

            if position == 0:
                entry_price = current_price

                if data_test["RSI"].iloc[i] < self.rsi_oversold and current_price <= data_test["BB_Lower"].iloc[i]:
                    position = 1
                    date_open = current_date
                    capital_open = capital

                elif data_test["RSI"].iloc[i] > self.rsi_overbought:
                    position = -1
                    date_open = current_date
                    capital_open = capital

            elif position == 1:
                price_change = (current_price - entry_price) / entry_price

                if price_change <= -self.stop_loss or price_change >= self.take_profit or \
                data_test["RSI"].iloc[i] > self.rsi_overbought or current_price >= data_test["BB_Upper"].iloc[i]:

                    profit = (current_price - entry_price) - fee
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
                price_change = (entry_price - current_price) / entry_price

                if price_change <= -self.stop_loss or price_change >= self.take_profit or \
                data_test["RSI"].iloc[i] < self.rsi_oversold or current_price <= data_test["BB_Lower"].iloc[i]:

                    profit = ((current_price - entry_price) * position) - fee
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
    def backtest_strategy(self, data_test, capital=1000000000, fee_add = 0.47, risk_per_trade=None, print_result=False):
        if data_test is None or data_test.empty:
            print("No data available for backtesting.")
            return
        if risk_per_trade is None:
            risk_per_trade = self.risk_per_trade

        fee = fee_add
        position = 0
        entry_price = 0
        returns = []
        closing_dates = []
        initial_capital = capital
        capital_map = {data_test["date"].iloc[0]: capital}

        for i in range(2, len(data_test)):
            current_date = data_test["date"].iloc[i]
            trade_size = capital * risk_per_trade

            if position == 0:
                entry_price = float(data_test["close"].iloc[i])  # Ensure float conversion
                # Buy condition
                if data_test["RSI"].iloc[i] < self.rsi_oversold and data_test["close"].iloc[i] <= data_test["BB_Lower"].iloc[i]:
                    position = 1
                # Sell/short condition
                elif data_test["RSI"].iloc[i] > self.rsi_overbought:
                    position = -1

            elif position == 1:
                current_price = float(data_test["close"].iloc[i])
                price_change = (current_price - entry_price) / entry_price

                if price_change <= -self.stop_loss or price_change >= self.take_profit or \
                data_test["RSI"].iloc[i] > self.rsi_overbought or current_price >= data_test["BB_Upper"].iloc[i]:
                    profit = (current_price - entry_price) - fee
                    capital += (profit / entry_price) * trade_size
                    returns.append(profit / entry_price)
                    closing_dates.append(current_date)
                    position = 0

            elif position == -1:
                current_price = float(data_test["close"].iloc[i])
                price_change = (entry_price - current_price) / entry_price

                if price_change <= -self.stop_loss or price_change >= self.take_profit or \
                data_test["RSI"].iloc[i] < self.rsi_oversold or current_price <= data_test["BB_Lower"].iloc[i]:
                    profit = ((current_price - entry_price) * position) - fee
                    capital += (profit / entry_price) * trade_size
                    returns.append(profit / entry_price)
                    closing_dates.append(current_date)
                    position = 0

            capital_map[current_date] = capital

        max_drawdown = calculate_max_drawdown(capital_map)
        win_rate = (len([x for x in returns if x > 0]) / len(returns) * 100) if returns else 0
        sharpe_ratio = calculate_sharpe_ratio(returns)

        total_return = (capital / initial_capital) * 100 - 100
        number_of_trades = len(closing_dates)

        if self.sharpe_ratio is None:
            if total_return < -30:
                self.sharpe_ratio = -float('inf')
            else:
                self.sharpe_ratio = sharpe_ratio
        
        if self.total_return is None:
            if number_of_trades < 300:
                self.total_return = -float('inf')
            else:
                self.total_return = total_return

        if print_result:
            print(f"Final Capital: {capital:.2f} points")
            print(f"Total Return: {(capital / initial_capital) * 100 - 100:.2f}%")
            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Max Drawdown: {max_drawdown:.6f}%")
            print(f"Sharpe Ratio: {sharpe_ratio:.6f}")
            print(f"Number of Transactions: {len(closing_dates)}")
            self.plot_returns(capital_map)

        return capital_map, len(closing_dates)



    def run_backtest(self, extract_data = False, returns_sharp = False, print_result=False, returns_total_return = False, all_sample = False, out_sample = False):
        self.split_data(self.in_sample_size, print_result=print_result)
        if all_sample:
            #print("Executing on all sample data...")
            self.backtest_strategy(self.data, print_result=print_result)
        elif out_sample:
            #print("Executing on out sample data...")
            self.backtest_strategy(self.data_out_sample, print_result=print_result)
        else:
            #print("Executing on in sample data...")
            self.backtest_strategy(self.data_in_sample, print_result=print_result)

        if extract_data:
            trades = self.extract_trades(self.data)
            trades_df = pd.DataFrame(trades)
            trades_df.to_csv("trades_output.csv", index=False)

        if returns_sharp:
            return self.sharpe_ratio
        
        if returns_total_return:
            return self.total_return

    
    def run_backtest_no_fee(self, extract_data = False, returns_sharp = False, print_result=False, all_sample = False, out_sample = False):
        self.split_data(self.in_sample_size, print_result=print_result)
        if all_sample:
            #print("Executing on all sample data...")
            self.backtest_strategy(self.data, print_result=print_result, fee_add=0)
        elif out_sample:
            #print("Executing on out sample data...")
            self.backtest_strategy(self.data_out_sample, print_result=print_result, fee_add=0)
        else:
            #print("Executing on in sample data...")
            self.backtest_strategy(self.data_in_sample, print_result=print_result,fee_add=0)

        if extract_data:
            trades = self.extract_trades(self.data)
            trades_df = pd.DataFrame(trades)
            trades_df.to_csv("trades_output.csv", index=False)

        if returns_sharp:
            return self.sharpe_ratio

if __name__ == "__main__":
    in_sample_size = 0.8 # Percentage of data that used for the in sample test 
    period_bb = 36
    period_rsi = 6
    risk_per_trade = 0.2 # Percentage of total capital that used for each trade 
    rsi_oversold = 5
    rsi_overbought = 71
    stop_loss = 0.1
    take_profit = 0.25
    print_result = True
    backtest = Backtesting(period_rsi, period_bb, in_sample_size, risk_per_trade, rsi_oversold, rsi_overbought, stop_loss, take_profit)

    # Fetch and load data
    backtest.initiate_data(True)
    # Apply indicators (RSI, Bollinger Bands, SMA)
    backtest.apply_indicators()

    #backtest.run_backtest_no_fee(print_result=print_result)

    # Run plot chart (Still have some problem related to connection)
    backtest.plot_candlestick_chart()