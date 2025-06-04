# test_sqlite.py
import sqlite3
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect("data/raw/score.db")

# List all tables in the database
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in the database:", tables)

# Preview data from the first table (assuming there's at least one)
if tables:
    table_name = tables[0][0]
    query = f"SELECT * FROM {table_name} LIMIT 5;"
    df = pd.read_sql_query(query, conn)
    print(f"\nPreview of {table_name}:")
    print(df)

# Close the connection
conn.close()
