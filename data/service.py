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

    # Modify the data
    def process_data(self):
        # Sort the database according to date
        self.df.set_index('date', inplace=True) 
        # Rename quantity to volume
        self.df.rename(columns={'quantity': 'volume'}, inplace=True)

    def save_to_csv(self, file_path='output.csv'):
        if self.df is not None:
            self.df.to_csv(file_path, index=True)  # Giữ cột 'date' làm index
            print(f"Dữ liệu đã được lưu vào {file_path}")
        else:
            print("Không có dữ liệu để lưu!")




