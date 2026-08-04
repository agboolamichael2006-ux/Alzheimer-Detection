# ==========================================================
# IMPORTS
# ==========================================================

import os
import sqlite3
import numpy as np

from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# ==========================================================
# FLASK CONFIGURATION
# ==========================================================

app = Flask(__name__)

app.secret_key = "alzheimer_secret_key_2026"

DATABASE = "predictions.db"

UPLOAD_FOLDER = os.path.join("static", "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png"
}


# ==========================================================
# DATABASE CONNECTION
# ==========================================================

def get_db():

    conn = sqlite3.connect(DATABASE)

    conn.row_factory = sqlite3.Row

    return conn


# ==========================================================
# ALLOWED FILE TYPES
# ==========================================================

def allowed_file(filename):

    return (

        "." in filename and

        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

    )


# ==========================================================
# CREATE DATABASE TABLES
# ==========================================================

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

        patient_id TEXT,

        patient_name TEXT,

        age INTEGER,

        gender TEXT,

        phone TEXT,

        address TEXT,

        image TEXT,

        diagnosis TEXT,

        confidence REAL,

        prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP

    )

    """)

    conn.commit()

    conn.close()


create_tables()


# ==========================================================
# LOAD MRI MODEL
# ==========================================================

MODEL_PATH = os.path.join("models", "mri_model.h5")

try:

    model = load_model(MODEL_PATH)

    print("✓ MRI Model Loaded Successfully")

except Exception as e:

    model = None

    print("Model Loading Error:", e)


# ==========================================================
# IMAGE PREPROCESSING
# ==========================================================

def preprocess_image(filepath):

    img = image.load_img(

        filepath,

        target_size=(128, 128)

    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(

        img_array,

        axis=0

    )

    img_array = img_array / 255.0

    return img_array


# ==========================================================
# MRI PREDICTION
# ==========================================================

def predict_image(filepath):

    if model is None:
        return None, 0

    img = preprocess_image(filepath)

    prediction = model.predict(
        img,
        verbose=0
    )[0][0]

    if prediction >= 0.5:
        diagnosis = "Alzheimer Detected"
        confidence = prediction * 100
    else:
        diagnosis = "Non Demented"
        confidence = (1 - prediction) * 100

    confidence = min(confidence, 75.00)

    return diagnosis, round(confidence, 2)
# ==========================================================
# LOGIN REQUIRED
# ==========================================================

def login_required():

    return "user_id" in session
# ==========================================================
# HOME PAGE
# ==========================================================

@app.route("/")
def home():

    if "user_id" in session:

        return redirect(url_for("dashboard"))

    return render_template("login.html")


# ==========================================================
# REGISTER
# ==========================================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        fullname = request.form["fullname"]

        username = request.form["username"]

        email = request.form["email"]

        password = request.form["password"]

        hashed_password = generate_password_hash(password)

        conn = get_db()

        try:

            conn.execute(

                """

                INSERT INTO users(

                    fullname,

                    username,

                    email,

                    password

                )

                VALUES(?,?,?,?)

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

            flash("Account created successfully.")

            return redirect(url_for("home"))

        except sqlite3.IntegrityError:

            conn.close()

            flash("Username or Email already exists.")

    return render_template("register.html")


# ==========================================================
# LOGIN
# ==========================================================

@app.route("/login", methods=["POST"])
def login():

    username = request.form["username"]

    password = request.form["password"]

    conn = get_db()

    user = conn.execute(

        """

        SELECT *

        FROM users

        WHERE username=?

        """,

        (username,)

    ).fetchone()

    conn.close()

    if user and check_password_hash(

        user["password"],

        password

    ):

        session["user_id"] = user["id"]

        session["username"] = user["username"]

        session["fullname"] = user["fullname"]

        flash("Login Successful")

        return redirect(url_for("dashboard"))

    flash("Invalid Username or Password")

    return redirect(url_for("home"))


# ==========================================================
# LOGOUT
# ==========================================================

@app.route("/logout")
def logout():

    session.clear()

    flash("Logged out successfully.")

    return redirect(url_for("home"))


# ==========================================================
# DASHBOARD
# ==========================================================

@app.route("/dashboard")
def dashboard():

    if not login_required():
        return redirect(url_for("home"))

    conn = get_db()

    total_predictions = conn.execute(
        """
        SELECT COUNT(*)
        FROM predictions
        WHERE user_id=?
        """,
        (session["user_id"],)
    ).fetchone()[0]

    total_users = conn.execute(
        "SELECT COUNT(*) FROM users"
    ).fetchone()[0]

    alzheimer_cases = conn.execute(
        """
        SELECT COUNT(*)
        FROM predictions
        WHERE user_id=?
        AND diagnosis='Alzheimer Detected'
        """,
        (session["user_id"],)
    ).fetchone()[0]

    non_demented = conn.execute(
        """
        SELECT COUNT(*)
        FROM predictions
        WHERE user_id=?
        AND diagnosis='Non Demented'
        """,
        (session["user_id"],)
    ).fetchone()[0]

    recent_predictions = conn.execute(
        """
        SELECT *
        FROM predictions
        WHERE user_id=?
        ORDER BY prediction_date DESC
        LIMIT 5
        """,
        (session["user_id"],)
    ).fetchall()

    conn.close()

    return render_template(
        "dashboard.html",
        username=session["username"],
        fullname=session["fullname"],
        total_predictions=total_predictions,
        total_users=total_users,
        alzheimer_cases=alzheimer_cases,
        non_demented=non_demented,
        recent_predictions=recent_predictions
    )
# ==========================================================
# PROFILE
# ==========================================================

@app.route("/profile", methods=["GET", "POST"])
def profile():

    if not login_required():

        return redirect(url_for("home"))

    conn = get_db()

    if request.method == "POST":

        fullname = request.form["fullname"]

        email = request.form["email"]

        conn.execute(

            """

            UPDATE users

            SET fullname=?,

                email=?

            WHERE id=?

            """,

            (

                fullname,

                email,

                session["user_id"]

            )

        )

        conn.commit()

        session["fullname"] = fullname

        flash("Profile Updated Successfully")

    user = conn.execute(

        """

        SELECT *

        FROM users

        WHERE id=?

        """,

        (

            session["user_id"],

        )

    ).fetchone()

    conn.close()

    return render_template(

        "profile.html",

        user=user

    )
# ==========================================================
# UPLOAD PAGE
# ==========================================================

@app.route("/upload")
def upload():

    if not login_required():

        return redirect(url_for("home"))

    return render_template("upload.html")


# ==========================================================
# PREDICT MRI
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    if not login_required():
        return redirect(url_for("home"))

    # ==========================
    # PATIENT DETAILS
    # ==========================

    patient_name = request.form["patient_name"]
    age = int(request.form["age"])
    gender = request.form["gender"]
    phone = request.form["phone"]
    address = request.form["address"]

    print("AGE FROM FORM:", request.form["age"])
    print("AGE VARIABLE:", age)

    # ==========================
    # GENERATE PATIENT ID
    # ==========================

    conn = get_db()

    total = conn.execute(
        "SELECT COUNT(*) FROM predictions"
    ).fetchone()[0]

    patient_id = f"PT{total + 1:03d}"

    conn.close()

    # ==========================
    # MRI IMAGE
    # ==========================

    if "image" not in request.files:
        flash("Please upload an MRI image.")
        return redirect(url_for("upload"))

    file = request.files["image"]

    if file.filename == "":
        flash("Please upload an MRI image.")
        return redirect(url_for("upload"))

    if not allowed_file(file.filename):
        flash("Only JPG, JPEG and PNG files are allowed.")
        return redirect(url_for("upload"))

    filename = secure_filename(file.filename)

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(filepath)

    # ==========================
    # AI PREDICTION
    # ==========================

    diagnosis, confidence = predict_image(filepath)

    # ==========================
    # SAVE TO DATABASE
    # ==========================

    conn = get_db()

    conn.execute(
        """
        INSERT INTO predictions
        (
            user_id,
            patient_id,
            patient_name,
            age,
            gender,
            phone,
            address,
            image,
            diagnosis,
            confidence
        )

        VALUES
        (
            ?,?,?,?,?,?,?,?,?,?
        )
        """,
        (
            session["user_id"],
            patient_id,
            patient_name,
            age,
            gender,
            phone,
            address,
            filename,
            diagnosis,
            float(confidence)
        )
    )

    conn.commit()
    conn.close()

    # ==========================
    # SHOW RESULT
    # ==========================

    return render_template(
        "result.html",
        patient_id=patient_id,
        patient_name=patient_name,
        age=age,
        gender=gender,
        phone=phone,
        address=address,
        image=filename,
        diagnosis=diagnosis,
        confidence=confidence
    )
# ==========================================================
# HISTORY
# ==========================================================

@app.route("/history")
def history():

    if not login_required():

        return redirect(url_for("home"))

    conn = get_db()

    history = conn.execute(

        """

        SELECT *

        FROM predictions

        WHERE user_id=?

        ORDER BY prediction_date DESC

        """,

        (

            session["user_id"],

        )

    ).fetchall()

    conn.close()

    return render_template(

        "history.html",

        history=history

    )


# ==========================================================
# REPORTS
# ==========================================================

@app.route("/reports")
def reports():

    if not login_required():
        return redirect(url_for("home"))

    conn = get_db()

    total_predictions = conn.execute(
        """
        SELECT COUNT(*)
        FROM predictions
        WHERE user_id=?
        """,
        (session["user_id"],)
    ).fetchone()[0]

    total_users = conn.execute(
        """
        SELECT COUNT(*)
        FROM users
        """
    ).fetchone()[0]

    alzheimer = conn.execute(
        """
        SELECT COUNT(*)
        FROM predictions
        WHERE user_id=?
        AND diagnosis='Alzheimer Detected'
        """,
        (session["user_id"],)
    ).fetchone()[0]

    non_demented = conn.execute(
        """
        SELECT COUNT(*)
        FROM predictions
        WHERE user_id=?
        AND diagnosis='Non Demented'
        """,
        (session["user_id"],)
    ).fetchone()[0]

    recent_predictions = conn.execute(
        """
        SELECT *
        FROM predictions
        WHERE user_id=?
        ORDER BY prediction_date DESC
        LIMIT 10
        """,
        (session["user_id"],)
    ).fetchall()

    conn.close()

    return render_template(
        "reports.html",
        total_predictions=total_predictions,
        total_users=total_users,
        alzheimer=alzheimer,
        non_demented=non_demented,
        recent_predictions=recent_predictions
    )
# ==========================================================
# ERROR PAGES
# ==========================================================

@app.errorhandler(404)
def page_not_found(error):

    return render_template(

        "404.html"

    ), 404


@app.errorhandler(500)
def internal_server_error(error):

    return render_template(

        "500.html"

    ), 500


# ==========================================================
# RUN APPLICATION
# ==========================================================

if __name__ == "__main__":

    app.run(

        debug=True

    )