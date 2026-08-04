import sqlite3


# ==========================================================
# CREATE DATABASE
# ==========================================================

def create_db():

    conn = sqlite3.connect("predictions.db")

    cursor = conn.cursor()

    cursor.execute("""

    CREATE TABLE IF NOT EXISTS history(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        patient_id TEXT,

        patient_name TEXT,

        age INTEGER,

        gender TEXT,

        phone TEXT,

        address TEXT,

        image TEXT,

        diagnosis TEXT,

        prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP

    )

    """)

    conn.commit()

    conn.close()


# ==========================================================
# SAVE PREDICTION
# ==========================================================

def save_prediction(

    patient_id,
    patient_name,
    age,
    gender,
    phone,
    address,
    image,
    diagnosis

):

    conn = sqlite3.connect("predictions.db")

    cursor = conn.cursor()

    cursor.execute("""

    INSERT INTO history(

        patient_id,
        patient_name,
        age,
        gender,
        phone,
        address,
        image,
        diagnosis

    )

    VALUES(

        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?,
        ?

    )

    """,

    (

        patient_id,
        patient_name,
        age,
        gender,
        phone,
        address,
        image,
        diagnosis

    )

    )

    conn.commit()

    conn.close()


# ==========================================================
# GET HISTORY
# ==========================================================

def get_history():

    conn = sqlite3.connect("predictions.db")

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""

    SELECT *

    FROM history

    ORDER BY prediction_date DESC

    """)

    data = cursor.fetchall()

    conn.close()

    return data


# ==========================================================
# TOTAL PREDICTIONS
# ==========================================================

def total_predictions():

    conn = sqlite3.connect("predictions.db")

    cursor = conn.cursor()

    cursor.execute("""

    SELECT COUNT(*)

    FROM history

    """)

    total = cursor.fetchone()[0]

    conn.close()

    return total


# ==========================================================
# TOTAL ALZHEIMER CASES
# ==========================================================

def total_alzheimer():

    conn = sqlite3.connect("predictions.db")

    cursor = conn.cursor()

    cursor.execute("""

    SELECT COUNT(*)

    FROM history

    WHERE diagnosis='Alzheimer Detected'

    """)

    total = cursor.fetchone()[0]

    conn.close()

    return total


# ==========================================================
# TOTAL NON DEMENTED CASES
# ==========================================================

def total_non_demented():

    conn = sqlite3.connect("predictions.db")

    cursor = conn.cursor()

    cursor.execute("""

    SELECT COUNT(*)

    FROM history

    WHERE diagnosis='Non Demented'

    """)

    total = cursor.fetchone()[0]

    conn.close()

    return total