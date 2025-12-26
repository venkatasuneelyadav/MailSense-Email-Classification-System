from flask import Flask, render_template, request, redirect, session, make_response
from database import init_db, create_user, login_user, save_history, fetch_history, fetch_history_all
from classifier import predict_email
import csv

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Initialize Database
init_db()


# -------------------------
# HOME → REDIRECT LOGIC
# -------------------------
@app.route("/")
def home():
    if "user_id" in session:
        return redirect("/about")
    return redirect("/login")


# -------------------------
# LOGIN PAGE
# -------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        user = login_user(username, password)

        if user:
            session["user_id"] = user[0]
            return redirect("/about")  # FIXED: Redirect to About page after login

        return render_template("login.html", error="Invalid credentials", active="login")

    return render_template("login.html", active="login")


# -------------------------
# SIGNUP PAGE
# -------------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if create_user(username, password):
            return redirect("/login")

        return render_template("signup.html", error="Username already exists", active="signup")

    return render_template("signup.html", active="signup")


# -------------------------
# ABOUT PAGE (FIRST PAGE AFTER LOGIN)
# -------------------------
@app.route("/about")
def about():
    if "user_id" not in session:
        return redirect("/login")

    return render_template("about.html", title="About Project", active="about")


# -------------------------
# DASHBOARD (PREDICTION PAGE)
# -------------------------
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    if "user_id" not in session:
        return redirect("/login")

    prediction = None
    confidence = None

    if request.method == "POST":
        subject = request.form.get("subject")
        body = request.form.get("body")

        label, conf = predict_email(subject, body)
        prediction = label.upper()
        confidence = round(conf * 100, 2)

        save_history(session["user_id"], subject, body, label, conf)

    return render_template("dashboard.html",
                           prediction=prediction,
                           confidence=confidence,
                           active="dashboard")


# -------------------------
# HISTORY PAGE
# -------------------------
@app.route("/history")
def history():
    if "user_id" not in session:
        return redirect("/login")

    records = fetch_history(session["user_id"])
    return render_template("history.html", history=records, active="history")


# -------------------------
# DOWNLOAD CSV
# -------------------------
@app.route("/download_csv")
def download_csv():
    if "user_id" not in session:
        return redirect("/login")

    rows = fetch_history(session["user_id"])

    output = [["Subject", "Body", "Prediction", "Confidence", "Timestamp"]]
    for row in rows:
        output.append(list(row))

    # Convert to CSV text
    csv_data = "\n".join([",".join([str(x).replace(",", " ") for x in row]) for row in output])

    response = make_response(csv_data)
    response.headers["Content-Disposition"] = "attachment; filename=prediction_history.csv"
    response.headers["Content-Type"] = "text/csv"

    return response


# -------------------------
# CONTACT PAGE
# -------------------------
@app.route("/contact")
def contact():
    if "user_id" not in session:
        return redirect("/login")

    return render_template("contact.html", title="Contact Us", active="contact")


# -------------------------
# LOGOUT
# -------------------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

@app.route("/team")
def team():
    if "user_id" not in session:
        return redirect("/login")
    return render_template("team.html", title="Meet the Team", active="team")



# -------------------------
# START FLASK APP
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
