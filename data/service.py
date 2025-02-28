import pandas as pd
import psycopg2
import json
import pprint
from query import daily_data_query

class DataFetcher:
    def __init__(self, db_config_path='database.json'):
        self.db_config_path = db_config_path
        self.df = None

    def load_db_config(self):
        with open(self.db_config_path, 'rb') as fb:
            return json.load(fb)

    def fetch_data(self):
        db_info = self.load_db_config()
        with psycopg2.connect(
            host=db_info['host'],
            port=db_info['port'],
            dbname=db_info['database'],
            user=db_info['user'],
            password=db_info['password']
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(daily_data_query)
                self.df = pd.DataFrame(cur.fetchall(), columns=['date', 'symbol', 'high', 'low', 'close', 'open', 'quantity'])
        
        self.process_data()
        return self.df

    def process_data(self):
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df.set_index('date', inplace=True)
        self.df.rename(columns={'quantity': 'volume'}, inplace=True)
        price_columns = ['open', 'high', 'low', 'close']
        self.df[price_columns] = self.df[price_columns].astype(float)


fetcher = DataFetcher()
df = fetcher.fetch_data()
pprint.pp(df)