from backtesting.backtesting import *
from data.service import *

import pprint

fetcher = DataFetcher()
df = fetcher.fetch_data()
pprint.pp(df)
