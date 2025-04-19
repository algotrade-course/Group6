import pandas as pd
import psycopg2
import json
from data.query import *

class DataFetcher:
    def __init__(self, db_config_path='data/database.json'):
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
                cur.execute(matched_data_query)
                self.df = pd.DataFrame(cur.fetchall(), columns=['date', 'symbol', 'high', 'low', 'close', 'open'])
        
        self.process_data()
        return self.df

    def process_data(self):
        self.df.columns = [col.lower() for col in self.df.columns]  # normalize column names
        if 'date' in self.df.columns:
            self.df.set_index('date', inplace=True)

    def save_to_csv(self, file_path='output.csv'):
        if self.df is not None:
            self.df.to_csv(file_path, index=True)  
            print(f"Data is already stored in {file_path}")
        else:
            print("There is no data to store")

    def print_dataset(self, num_rows=100):
        if self.df is not None:
            print(self.df.head(num_rows))
        else:
            print("No data available to display.")

    def load_data_from_csv(self, file_path='daily_data.csv'):
        try:
            self.df = pd.read_csv(file_path, parse_dates=['date'])
            self.process_data()  # Reuse the same processing (e.g. set index)
            # print(f"Data loaded from {file_path}")
        except FileNotFoundError:
            print(f"{file_path} not found.")


    



