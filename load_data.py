import pandas as pd
import sqlite3

df = pd.read_csv("data/raw_sales.csv")
conn = sqlite3.connect("data/sales.db")
df.to_sql("sales", conn, if_exists="replace", index=False)
conn.close()

print("Loaded", len(df), "rows")
print("Columns:", df.columns.tolist())