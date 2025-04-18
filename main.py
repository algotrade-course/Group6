from backtesting.backtesting import *
from data.service import *
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp

period_rsi = 14
period_bb = 70
in_sample_size = 0.7 # Percentage of data that used for the in sample test 
risk_per_trade = 0.25 # Percentage of total capital that used for each trade 
rsi_oversold = 5
rsi_overbought = 85
rsi_extreme_overbought = 90
backtest = Backtesting(period_rsi, period_bb, in_sample_size, risk_per_trade, rsi_oversold, rsi_overbought, rsi_extreme_overbought)

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
