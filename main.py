from backtesting.backtesting import *
from data.service import *
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp

backtest = Backtesting()

# Fetch and load data
backtest.initiate_data()

# Apply indicators (RSI, Bollinger Bands, SMA)
backtest.apply_indicators()

# Run backtest strategy
backtest.backtest_strategy()

# Run plot chart (Still have some problem related to connection)
# backtest.plot_chart()
