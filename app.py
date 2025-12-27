import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime
import time
import os

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Student Performance System", layout="wide")

# ===================== LOAD MODEL =====================
model = joblib.load("student_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_columns = joblib.load("model_features.joblib")

# ===================== FILE HANDLING =====================
def load_users():
    if not os.path.exists("users.csv"):
        df = pd.DataFrame({
            "username": ["admin", "simon"],
            "password": ["admin123", "user123"],
            "role": ["admin", "user"]
        })
        df.to_csv("users.csv", index=False)
    return pd.read_csv("users.csv")

def load_history():
    if not os.path.exists("history.csv"):
        df = pd.DataFrame(columns=[
            "username", "math", "reading", "writing",
            "average", "prediction", "timestamp"
        ])
        df.to_csv("history.csv", index=False)
    return pd.read_csv("history.csv")

# ===================== SESSION STATE =====================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = ""
if "username" not in st.session_state:
    st.session_state.username = ""

# ===================== LOGIN PAGE =====================
def login_page():
    st.title("🔐 Login")

    users = load_users()
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = users[
            (users["username"] == username) &
            (users["password"] == password)
        ]

        if not user.empty:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = user.iloc[0]["role"]
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid username or password")

# ===================== LOGOUT =====================
def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.role = ""
    st.rerun()

# ===================== PREDICTION FUNCTION =====================
def prediction_page():
    st.subheader("📊 Student Performance Prediction")

    col1, col2 = st.columns(2)

    with col1:
        math = st.number_input("Math Score", 0, 100, 0)
        writing = st.number_input("Writing Score", 0, 100, 0)

    with col2:
        reading = st.number_input("Reading Score", 0, 100, 0)
        gender = st.selectbox("Gender", ["Female", "Male"])

    if st.button("Predict"):
        avg = (math + reading + writing) / 3

        input_dict = {
            "math score": math,
            "reading score": reading,
            "writing score": writing,
            "gender_encoded": 0 if gender == "Female" else 1
        }

        df = pd.DataFrame([input_dict])
        df = df.reindex(columns=feature_columns, fill_value=0)
        df_scaled = scaler.transform(df)

        prediction = int(model.predict(df_scaled)[0])
        label = ["Low", "Medium", "High"][prediction]

        st.success(f"Predicted Performance: **{label}**")

        # Save history
        history = load_history()
        new_row = {
            "username": st.session_state.username,
            "math": math,
            "reading": reading,
            "writing": writing,
            "average": avg,
            "prediction": label,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
        history.to_csv("history.csv", index=False)

# ===================== USER DASHBOARD =====================
def user_dashboard():
    st.subheader("👤 User Dashboard")
    history = load_history()
    user_data = history[history["username"] == st.session_state.username]

    if user_data.empty:
        st.info("No prediction history available.")
        return

    st.dataframe(user_data)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=user_data["timestamp"],
        y=user_data["average"],
        name="Average Score"
    ))

    fig.update_layout(title="Performance Over Time")
    st.plotly_chart(fig, use_container_width=True)

# ===================== ADMIN DASHBOARD =====================
def admin_dashboard():
    st.subheader("🛠 Admin Dashboard")
    history = load_history()
    users = load_users()

    st.markdown("### Registered Users")
    st.dataframe(users)

    st.markdown("### Prediction History (All Users)")
    st.dataframe(history)

    if not history.empty:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=history["prediction"],
            marker_color="skyblue"
        ))
        fig.update_layout(title="Prediction Distribution")
        st.plotly_chart(fig, use_container_width=True)

# ===================== MAIN APP =====================
if not st.session_state.logged_in:
    login_page()
else:
    st.sidebar.title("📌 Menu")
    st.sidebar.write(f"User: **{st.session_state.username}**")
    st.sidebar.write(f"Role: **{st.session_state.role}**")

    page = st.sidebar.radio("Navigate", ["Prediction", "Dashboard"])

    if st.sidebar.button("Logout"):
        logout()

    if page == "Prediction":
        prediction_page()

    if page == "Dashboard":
        if st.session_state.role == "admin":
            admin_dashboard()
        else:
            user_dashboard()
