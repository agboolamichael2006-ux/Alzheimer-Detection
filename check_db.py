import sqlite3

conn = sqlite3.connect("predictions.db")
conn.row_factory = sqlite3.Row

rows = conn.execute("SELECT id, confidence FROM predictions").fetchall()

for row in rows:
    print(row["id"], row["confidence"], type(row["confidence"]))

conn.close()