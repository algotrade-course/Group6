from backtesting.backtesting import *
from data.service import *
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp

period_rsi = 7 
period_bb = 50
in_sample_size = 0.7 # Percentage of data that used for the in sample test 
risk_per_trade = 0.2 # Percentage of total capital that used for each trade 
backtest = Backtesting(period_rsi, period_bb, in_sample_size, risk_per_trade)

# Fetch and load data
backtest.initiate_data()


# Apply indicators (RSI, Bollinger Bands, SMA)
backtest.apply_indicators()
# backtest.print_data()

# # Run backtest strategy for 100% data
# # Run backtest strategy for 80% in-sample data and 20% out-sample data
backtest.run_backtest()

# Run plot chart (Still have some problem related to connection)
# backtest.plot_candlestick_chart()
