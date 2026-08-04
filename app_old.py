from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash
)

import os
import sqlite3
import numpy as np

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image



# ===================================================
# FLASK CONFIGURATION
# ===================================================

app = Flask(__name__)

app.secret_key = "alzheimer_secret_key_2026"

UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)



# ===================================================
# DATABASE
# ===================================================

DATABASE = "predictions.db"


def get_db():

    conn = sqlite3.connect(DATABASE)

    conn.row_factory = sqlite3.Row

    return conn



# ===================================================
# CREATE TABLES
# ===================================================

def create_tables():

    conn = get_db()

    cursor = conn.cursor()

    cursor.execute("""

    CREATE TABLE IF NOT EXISTS users(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        fullname TEXT NOT NULL,

        username TEXT UNIQUE NOT NULL,

        email TEXT UNIQUE NOT NULL,

        password TEXT NOT NULL

    )

    """)

    cursor.execute("""

    CREATE TABLE IF NOT EXISTS predictions(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        user_id INTEGER,

        filename TEXT,

        prediction TEXT,

        confidence REAL,

        prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY(user_id) REFERENCES users(id)

    )

    """)

    conn.commit()

    conn.close()



create_tables()



# ===================================================
# LOAD MRI MODEL
# ===================================================

MODEL_PATH = "mri_model.keras"

model = None

if os.path.exists(MODEL_PATH):

    model = load_model(MODEL_PATH)

    print("MRI Model Loaded Successfully")

else:

    print("WARNING: MRI model not found!")



# ===================================================
# MRI CLASS LABELS
# ===================================================

CLASS_NAMES = [

    "Mild Demented",

    "Moderate Demented",

    "Non Demented",

    "Very Mild Demented"

]



# ===================================================
# PREDICTION FUNCTION
# ===================================================

def predict_mri(filepath):

    img = image.load_img(

        filepath,

        target_size=(128,128)

    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    index = np.argmax(prediction)

    confidence = float(np.max(prediction))

    return CLASS_NAMES[index], confidence
# ===================================================
# HOME PAGE
# ===================================================

@app.route("/")
def home():

    return render_template("login.html")


# ===================================================
# REGISTER
# ===================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        fullname = request.form["fullname"].strip()
        username = request.form["username"].strip()
        email = request.form["email"].strip()
        password = request.form["password"]

        conn = get_db()
        cursor = conn.cursor()

        existing = cursor.execute(
            """
            SELECT * FROM users
            WHERE username = ?
            OR email = ?
            """,
            (username, email)
        ).fetchone()

        if existing:

            conn.close()

            flash("Username or Email already exists.")

            return render_template("register.html")

        hashed_password = generate_password_hash(password)

        cursor.execute(
            """
            INSERT INTO users
            (fullname, username, email, password)

            VALUES (?, ?, ?, ?)
            """,
            (
                fullname,
                username,
                email,
                hashed_password
            )
        )

        conn.commit()

        conn.close()

        flash("Account created successfully. Please login.")

        return redirect(url_for("home"))

    return render_template("register.html")


# ===================================================
# LOGIN
# ===================================================

@app.route("/login", methods=["POST"])
def login():

    username = request.form["username"].strip()
    password = request.form["password"]

    conn = get_db()

    user = conn.execute(
        """
        SELECT *
        FROM users
        WHERE username = ?
        """,
        (username,)
    ).fetchone()

    conn.close()

    if user is None:

        flash("Invalid username or password.")

        return redirect(url_for("home"))

    if not check_password_hash(
        user["password"],
        password
    ):

        flash("Invalid username or password.")

        return redirect(url_for("home"))

    session["user_id"] = user["id"]
    session["username"] = user["username"]
    session["fullname"] = user["fullname"]

    return redirect(url_for("dashboard"))


# ===================================================
# DASHBOARD
# ===================================================

@app.route("/dashboard")
def dashboard():

    if "user_id" not in session:

        return redirect(url_for("home"))

    return render_template("dashboard.html")


# ===================================================
# LOGOUT
# ===================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(url_for("home"))
# ===================================================
# PROFILE
# ===================================================

@app.route("/profile", methods=["GET", "POST"])
def profile():

    if "user_id" not in session:
        return redirect(url_for("home"))

    conn = get_db()

    if request.method == "POST":

        fullname = request.form["fullname"].strip()
        email = request.form["email"].strip()

        conn.execute(
            """
            UPDATE users
            SET fullname = ?, email = ?
            WHERE id = ?
            """,
            (fullname, email, session["user_id"])
        )

        conn.commit()

        session["fullname"] = fullname

        flash("Profile updated successfully.")

    user = conn.execute(
        """
        SELECT *
        FROM users
        WHERE id = ?
        """,
        (session["user_id"],)
    ).fetchone()

    conn.close()

    return render_template("profile.html", user=user)


# ===================================================
# UPLOAD PAGE
# ===================================================

@app.route("/upload")
def upload():

    if "user_id" not in session:
        return redirect(url_for("home"))

    return render_template("upload.html")


# ===================================================
# MRI PREDICTION
# ===================================================

@app.route("/predict", methods=["POST"])
def predict():

    if "user_id" not in session:
        return redirect(url_for("home"))

    if "mri_image" not in request.files:

        flash("Please choose an MRI image.")

        return redirect(url_for("upload"))

    file = request.files["mri_image"]

    if file.filename == "":

        flash("Please choose an MRI image.")

        return redirect(url_for("upload"))

    filename = secure_filename(file.filename)

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(filepath)

    prediction, confidence = predict_mri(filepath)

    conn = get_db()

    conn.execute(
        """
        INSERT INTO predictions
        (
            user_id,
            filename,
            prediction,
            confidence
        )

        VALUES (?, ?, ?, ?)
        """,
        (
            session["user_id"],
            filename,
            prediction,
            confidence
        )
    )

    conn.commit()

    conn.close()

    return render_template(
        "result.html",
        filename=filename,
        prediction=prediction,
        confidence=round(confidence * 100, 2)
    )
# ===================================================
# PROFILE
# ===================================================

@app.route("/profile", methods=["GET", "POST"])
def profile():

    if "user_id" not in session:
        return redirect(url_for("home"))

    conn = get_db()

    if request.method == "POST":

        fullname = request.form["fullname"].strip()
        email = request.form["email"].strip()

        conn.execute(
            """
            UPDATE users
            SET fullname = ?, email = ?
            WHERE id = ?
            """,
            (fullname, email, session["user_id"])
        )

        conn.commit()

        session["fullname"] = fullname

        flash("Profile updated successfully.")

    user = conn.execute(
        """
        SELECT *
        FROM users
        WHERE id = ?
        """,
        (session["user_id"],)
    ).fetchone()

    conn.close()

    return render_template("profile.html", user=user)


# ===================================================
# UPLOAD PAGE
# ===================================================

@app.route("/upload")
def upload():

    if "user_id" not in session:
        return redirect(url_for("home"))

    return render_template("upload.html")


# ===================================================
# MRI PREDICTION
# ===================================================

@app.route("/predict", methods=["POST"])
def predict():

    if "user_id" not in session:
        return redirect(url_for("home"))

    if "mri_image" not in request.files:

        flash("Please choose an MRI image.")

        return redirect(url_for("upload"))

    file = request.files["mri_image"]

    if file.filename == "":

        flash("Please choose an MRI image.")

        return redirect(url_for("upload"))

    filename = secure_filename(file.filename)

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(filepath)

    prediction, confidence = predict_mri(filepath)

    conn = get_db()

    conn.execute(
        """
        INSERT INTO predictions
        (
            user_id,
            filename,
            prediction,
            confidence
        )

        VALUES (?, ?, ?, ?)
        """,
        (
            session["user_id"],
            filename,
            prediction,
            confidence
        )
    )

    conn.commit()

    conn.close()

    return render_template(
        "result.html",
        filename=filename,
        prediction=prediction,
        confidence=round(confidence * 100, 2)
    )
# ===================================================
# PREDICTION HISTORY
# ===================================================

@app.route("/history")
def history():

    if "user_id" not in session:
        return redirect(url_for("home"))

    conn = get_db()

    predictions = conn.execute(
        """
        SELECT *
        FROM predictions
        WHERE user_id = ?
        ORDER BY prediction_date DESC
        """,
        (session["user_id"],)
    ).fetchall()

    conn.close()

    return render_template(
        "history.html",
        predictions=predictions
    )


# ===================================================
# REPORT PAGE
# ===================================================

@app.route("/report")
def report():

    if "user_id" not in session:
        return redirect(url_for("home"))

    return render_template("reports.html")


# ===================================================
# 404 ERROR PAGE
# ===================================================

@app.errorhandler(404)
def page_not_found(error):

    return render_template(
        "404.html"
    ), 404


# ===================================================
# RUN APPLICATION
# ===================================================

if __name__ == "__main__":

    create_tables()

    app.run(
        debug=True
    )