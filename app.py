import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from datetime import datetime
import os

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Student Performance System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------- LOAD MODEL ----------------------
model = joblib.load("student_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_columns = joblib.load("model_features.joblib")

# ---------------------- SESSION STATE ----------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.role = ""

# ---------------------- LOAD USERS ----------------------
def load_users():
    return pd.read_csv("users.csv")

# ---------------------- SAVE HISTORY ----------------------
def save_history(username, math, reading, writing, avg, pred):
    history = pd.read_csv("history.csv")
    new_row = {
        "username": username,
        "math": math,
        "reading": reading,
        "writing": writing,
        "average": avg,
        "prediction": pred,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
    history.to_csv("history.csv", index=False)

# ---------------------- LOGIN PAGE ----------------------
def login_page():
    st.markdown("## 🔐 Login")

    with st.container():
        st.markdown("<div style='padding:30px;border-radius:15px;background:#f2f2f2'>", unsafe_allow_html=True)

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            users = load_users()
            user = users[
                (users["username"] == username) &
                (users["password"] == password)
            ]

            if not user.empty:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.role = user.iloc[0]["role"]
                st.success("Login successful")
                st.experimental_rerun()
            else:
                st.error("Invalid credentials")

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- DASHBOARD ----------------------
def dashboard():
    col1, col2 = st.columns([6,1])
    with col1:
        st.markdown(f"## 👋 Welcome {st.session_state.username} ({st.session_state.role})")
    with col2:
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.experimental_rerun()

    tabs = st.tabs(["🎓 Prediction", "📊 Dashboard", "📜 History"])

    # ---------------------- PREDICTION TAB ----------------------
    with tabs[0]:
        math = st.number_input("Math Score", 0, 100, 0)
        reading = st.number_input("Reading Score", 0, 100, 0)
        writing = st.number_input("Writing Score", 0, 100, 0)

        if st.button("Predict"):
            avg = (math + reading + writing) / 3

            if avg < 60:
                pe = 0
                label = "Low"
            elif avg < 80:
                pe = 1
                label = "Medium"
            else:
                pe = 2
                label = "High"

            input_df = pd.DataFrame([{
                "math score": math,
                "reading score": reading,
                "writing score": writing,
                "performance": avg,
                "gender_encoded": 0,
                "performance_encoded": pe,
                "gender_female": 1,
                "gender_male": 0
            }])

            input_df = input_df.reindex(columns=feature_columns, fill_value=0)
            scaled = scaler.transform(input_df)
            pred = int(model.predict(scaled)[0])

            st.success(f"🎯 Predicted Performance: {label}")

            save_history(
                st.session_state.username,
                math, reading, writing,
                avg, label
            )

    # ---------------------- DASHBOARD TAB ----------------------
    with tabs[1]:
        history = pd.read_csv("history.csv")

        if st.session_state.role == "user":
            history = history[history["username"] == st.session_state.username]

        st.markdown("### 📈 Performance Distribution")

        fig = go.Figure([
            go.Bar(
                x=history["prediction"].value_counts().index,
                y=history["prediction"].value_counts().values,
                marker_color=["#FF6B6B", "#FFD93D", "#4CAF50"]
            )
        ])

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📊 Average Score Trend")
        history["timestamp"] = pd.to_datetime(history["timestamp"])
        fig2 = go.Figure([
            go.Scatter(
                x=history["timestamp"],
                y=history["average"],
                mode="lines+markers"
            )
        ])
        st.plotly_chart(fig2, use_container_width=True)

    # ---------------------- HISTORY TAB ----------------------
    with tabs[2]:
        history = pd.read_csv("history.csv")

        if st.session_state.role == "user":
            history = history[history["username"] == st.session_state.username]

        st.dataframe(history, use_container_width=True)

# ---------------------- MAIN ----------------------
if not st.session_state.logged_in:
    login_page()
else:
    dashboard()
