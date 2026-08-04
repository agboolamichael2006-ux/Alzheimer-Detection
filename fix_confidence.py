import sqlite3
import struct

conn = sqlite3.connect("predictions.db")
cursor = conn.cursor()

rows = cursor.execute(
    "SELECT id, confidence FROM predictions"
).fetchall()

for row in rows:

    record_id = row[0]

    confidence = row[1]

    if isinstance(confidence, bytes):

        value = struct.unpack("f", confidence)[0]

        cursor.execute(
            "UPDATE predictions SET confidence=? WHERE id=?",
            (float(value), record_id)
        )

conn.commit()
conn.close()

print("Confidence values repaired successfully.")