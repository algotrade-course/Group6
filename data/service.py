import json
import psycopg2

# Read the JSON file as byte stream and load it into Python object using load()
with open('database.json', 'rb') as fb:
    db_info = json.load(fb)

# dictionary of database info
print(db_info)
# host of the database in str
print(db_info['host'])

# create a connection
conn = psycopg2.connect(
    host=db_info['host'],
    port=db_info['port'],
    dbname=db_info['database'],
    user=db_info['user'],
    password=db_info['password']
)

# a Connection object to the database
print(conn)