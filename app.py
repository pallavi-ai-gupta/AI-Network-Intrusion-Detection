import os
import json
import secrets
import logging

import joblib
import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    send_file
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
UPLOAD_DIR = os.path.join(DATASET_DIR, "uploads")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

MODEL_PATH = os.path.join(MODEL_DIR, "intrusion_model.pkl")
REPORT_PATH = os.path.join(REPORT_DIR, "model_report.json")
USERS_PATH = os.path.join(REPORT_DIR, "users.json")
PREDICTIONS_PATH = os.path.join(REPORT_DIR, "predictions.csv")

for folder in (DATASET_DIR, UPLOAD_DIR, MODEL_DIR, REPORT_DIR):
    os.makedirs(folder, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024

logging.basicConfig(level=logging.INFO)


# ==================================================
# 2. JSON AND AUTHENTICATION HELPERS
# ==================================================

def load_json(path, default=None):
    if default is None:
        default = {}

    if not os.path.exists(path):
        return default

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        app.logger.exception("Could not read %s", path)
        return default


def save_json(path, data):
    temp_path = path + ".tmp"

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    os.replace(temp_path, path)


def login_required():
    if not session.get("username"):
        flash("Please log in first.", "error")
        return False
    return True


def get_report():
    return load_json(REPORT_PATH, {})


# ==================================================
# 3. LOGIN
# ==================================================

@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("username"):
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")

        users = load_json(USERS_PATH, {})
        user = users.get(username)

        if user and check_password_hash(user.get("password", ""), password):
            session.clear()
            session["username"] = username
            return redirect(url_for("dashboard"))

        flash("Invalid username or password.", "error")

    return render_template("login.html", register_mode=False)


# ==================================================
# 4. REGISTRATION
# ==================================================

@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("username"):
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")

        if len(username) < 3:
            flash("Username must contain at least 3 characters.", "error")
        elif len(password) < 8:
            flash("Password must contain at least 8 characters.", "error")
        else:
            users = load_json(USERS_PATH, {})

            if username in users:
                flash("Username already exists.", "error")
            else:
                users[username] = {
                    "password": generate_password_hash(password)
                }
                save_json(USERS_PATH, users)

                flash("Registration successful. Please log in.", "success")
                return redirect(url_for("login"))

    return render_template("login.html", register_mode=True)


# ==================================================
# 5. DASHBOARD
# ==================================================

@app.route("/dashboard")
def dashboard():
    if not login_required():
        return redirect(url_for("login"))

    return render_template(
        "dashboard.html",
        username=session["username"],
        report=get_report(),
        model_exists=os.path.exists(MODEL_PATH)
    )


# ==================================================
# 6. TRAIN THE MODEL
# ==================================================

@app.route("/train", methods=["POST"])
def train():
    if not login_required():
        return redirect(url_for("login"))

    try:
        from train_model import train_model

        report = train_model()

        flash(
            "Training completed. Best model: "
            + str(report.get("best_model", "Unknown")),
            "success"
        )

    except Exception as error:
        app.logger.exception("Model training failed")
        flash("Training failed: " + str(error), "error")

    return redirect(url_for("dashboard"))


# ==================================================
# 7. PREDICT NETWORK TRAFFIC FROM CSV
# ==================================================

@app.route("/predict", methods=["POST"])
def predict():
    if not login_required():
        return redirect(url_for("login"))

    uploaded_file = request.files.get("dataset")

    if not uploaded_file or not uploaded_file.filename:
        flash("Please select a CSV file.", "error")
        return redirect(url_for("dashboard"))

    if not uploaded_file.filename.lower().endswith(".csv"):
        flash("Please upload a CSV file.", "error")
        return redirect(url_for("dashboard"))

    if not os.path.isfile(MODEL_PATH):
        flash("No trained model found. Train the model first.", "error")
        return redirect(url_for("dashboard"))

    filename = secure_filename(uploaded_file.filename)

    if not filename:
        flash("Invalid filename.", "error")
        return redirect(url_for("dashboard"))

    saved_path = os.path.join(
        UPLOAD_DIR,
        secrets.token_hex(8) + "_" + filename
    )

    try:
        uploaded_file.save(saved_path)

        bundle = joblib.load(MODEL_PATH)

        # Support common saved-model formats.
        if isinstance(bundle, dict) and "model" in bundle:
            model = bundle["model"]
            feature_columns = bundle.get("features")
        else:
            model = bundle
            feature_columns = getattr(model, "feature_names_in_", None)

        if not feature_columns:
            raise ValueError(
                "The saved model does not contain feature names. "
                "Update train_model.py to save the model and feature "
                "columns together, then retrain."
            )

        df = pd.read_csv(saved_path, low_memory=False)
        df.columns = [str(column).strip() for column in df.columns]

        if df.empty:
            raise ValueError("The uploaded CSV is empty.")

        if len(df) > 100000:
            raise ValueError("Maximum 100,000 CSV rows are supported.")

        if df.columns.duplicated().any():
            raise ValueError("The CSV has duplicate column names.")

        # Check for every feature the model requires.
        missing = [
            column for column in feature_columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                "Missing required feature columns: "
                + ", ".join(missing)
                + ". Include all required columns in your prediction CSV."
            )

        # Extra columns, including a true label column, are not used.
        X = df[list(feature_columns)].copy()

        # Convert infinite values to missing values.
        X = X.replace([float("inf"), float("-inf")], float("nan"))

        predictions = model.predict(X)

        result_df = df.copy()
        result_df["Predicted_Label"] = [str(value) for value in predictions]

        normal_labels = {"normal", "benign", "0"}
        result_df["Prediction_Status"] = (
            result_df["Predicted_Label"]
            .str.strip()
            .str.lower()
            .apply(
                lambda label: (
                    "NORMAL" if label in normal_labels else "SUSPICIOUS"
                )
            )
        )

        # Save the complete prediction results.
        result_df.to_csv(PREDICTIONS_PATH, index=False)

        # Limit on-screen preview size.
        preview = (
            result_df.head(200)
            .fillna("")
            .astype(str)
            .to_dict(orient="records")
        )

        return render_template(
            "results.html",
            report=get_report(),
            columns=result_df.columns.tolist(),
            rows=preview,
            total=len(result_df),
            prediction_mode=True
        )

    except Exception as error:
        app.logger.exception("Prediction failed")
        flash("Prediction failed: " + str(error), "error")
        return redirect(url_for("dashboard"))

    finally:
        # Remove the temporary upload, if it exists.
        if os.path.exists(saved_path):
            try:
                os.remove(saved_path)
            except OSError:
                app.logger.warning("Could not remove temporary upload.")


# ==================================================
# 8. MODEL PERFORMANCE REPORT
# ==================================================

@app.route("/results")
def results_page():
    if not login_required():
        return redirect(url_for("login"))

    report = get_report()

    if not report:
        flash("Train the model to generate its report.", "error")
        return redirect(url_for("dashboard"))

    return render_template(
        "results.html",
        report=report,
        columns=[],
        rows=[],
        total=0,
        prediction_mode=False
    )


# ==================================================
# 9. DOWNLOAD PREDICTIONS
# ==================================================

@app.route("/download")
def download():
    if not login_required():
        return redirect(url_for("login"))

    if not os.path.isfile(PREDICTIONS_PATH):
        flash("No prediction results are available yet.", "error")
        return redirect(url_for("dashboard"))

    return send_file(
        PREDICTIONS_PATH,
        as_attachment=True,
        download_name="intrusion_predictions.csv",
        mimetype="text/csv"
    )


# ==================================================
# 10. LOGOUT
# ==================================================

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ==================================================
# 11. ERROR HANDLERS
# ==================================================

@app.errorhandler(413)
def file_too_large(error):
    flash("CSV file is too large. Maximum size is 30 MB.", "error")
    return redirect(url_for("dashboard"))


@app.errorhandler(500)
def internal_server_error(error):
    app.logger.exception("Internal server error")
    return "An internal server error occurred. Check the VS Code terminal.", 500


# ==================================================
# 12. START APPLICATION
# ==================================================

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)