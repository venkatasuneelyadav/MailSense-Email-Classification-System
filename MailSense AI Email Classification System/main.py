import streamlit as st
from auth import create_user, login_user
from classifier import predict_email, save_history, fetch_history
from database import init_db

st.set_page_config(page_title="AI Email Classifier", layout="centered")
init_db()

# ------------------------
# SESSION STATE VARIABLES
# ------------------------
if "page" not in st.session_state:
    st.session_state.page = "login"
if "user" not in st.session_state:
    st.session_state.user = None


# ------------------------
# PAGE NAVIGATION FUNCTION
# ------------------------
def go_to(page_name):
    st.session_state.page = page_name
    st.rerun()


# ------------------------
# LOGIN PAGE
# ------------------------
def login_page():
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = login_user(username, password)
        if user:
            st.session_state.user = user[0]  # store user id
            go_to("app")
        else:
            st.error("❌ Invalid username or password")

    st.write("Don't have an account?")
    if st.button("Create Account"):
        go_to("signup")


# ------------------------
# SIGNUP PAGE
# ------------------------
def signup_page():
    st.title("📝 Create Account")

    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")

    if st.button("Sign Up"):
        if create_user(username, password):
            st.success("🎉 Account created successfully!")
            st.info("Please log in now.")
            go_to("login")
        else:
            st.error("❌ Username already exists")

    if st.button("Back to Login"):
        go_to("login")


# ------------------------
# APP PAGE
# ------------------------
def app_page():

    if st.session_state.user is None:
        st.warning("Please log in first.")
        go_to("login")

    st.title("📬 AI Email Classifier")

    # LOGOUT BUTTON
    if st.button("Logout"):
        st.session_state.user = None
        go_to("login")

    subject = st.text_input("Email Subject")
    body = st.text_area("Email Body", height=200)

    if st.button("Predict"):
        label, conf = predict_email(subject, body)

        st.success(f"📌 Prediction: **{label.upper()}**")
        st.info(f"🔥 Confidence: **{conf*100:.2f}%**")

        # Save prediction to history
        save_history(st.session_state.user, subject, body, label, conf)

    # ------------------------
    # HISTORY SECTION
    # ------------------------
    st.subheader("📜 Your Prediction History")

    history = fetch_history(st.session_state.user)

    if history:
        for record in history:
            subject, body, prediction, confidence, timestamp = record

            with st.expander(f"📩 {subject[:60]}..."):
                st.write(f"**Subject:** {subject}")
                st.write(f"**Body:** {body}")
                st.write(f"**Prediction:** `{prediction.upper()}`")
                st.write(f"**Confidence:** {confidence * 100:.2f}%")
                st.write(f"**Timestamp:** {timestamp}")
    else:
        st.info("No history found yet.")


# ------------------------
# PAGE ROUTER
# ------------------------
if st.session_state.page == "login":
    login_page()
elif st.session_state.page == "signup":
    signup_page()
elif st.session_state.page == "app":
    app_page()
