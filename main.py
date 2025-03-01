from backtesting.backtesting import *
from data.service import *


backtesting = Backtesting()
backtesting.initiate_data()
backtesting.print_data()

# print(backtesting.calculate_rsi())
# print(backtesting.calculate_moving_averages(5))

