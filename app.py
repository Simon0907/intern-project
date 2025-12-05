import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import json

# ---------------------- Load Model + Assets ----------------------
model = joblib.load("student_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_columns = joblib.load("model_features.joblib")

# ---------------------- Page Settings ----------------------
st.set_page_config(
    page_title="Student Performance Prediction",
    layout="wide",
    page_icon="📊"
)

# ---------------------- Premium CSS Styling ----------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* Background gradient */
.main {
    background: linear-gradient(135deg, #e2ebf0 0%, #f9fbff 100%);
}

/* Glassmorphism card */
.glass-card {
    backdrop-filter: blur(18px);
    background: rgba(255, 255, 255, 0.25);
    padding: 30px;
    border-radius: 22px;
    border: 1px solid rgba(255, 255, 255, 0.5);
    box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    animation: fadeIn 1s ease-in-out;
    transition: 0.3s;
}

.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 15px 40px rgba(0,0,0,0.20);
}

/* Section title */
.section-title {
    font-size: 28px;
    font-weight: 700;
    color: #1c2b4d;
    margin-bottom: 10px;
    animation: fadeIn 1.2s ease-in-out;
}

/* Main Title Styling */
.big-title {
    font-size: 48px;
    font-weight: 900;
    color: #0f1c38;
    text-shadow: 0px 2px 5px rgba(0,0,0,0.15);
    animation: fadeIn 1s ease-in-out;
}

/* Premium Button */
.stButton>button {
    width: 100%;
    border-radius: 14px;
    padding: 14px 0;
    font-size: 20px;
    font-weight: 700;
    border: none;
    color: white;
    background: linear-gradient(135deg, #6a11cb, #2575fc);
    box-shadow: 0 4px 10px rgba(0,0,0,0.25);
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 6px 18px rgba(0,0,0,0.35);
}

/* Animated Result Box */
.result-box {
    backdrop-filter: blur(12px);
    background: rgba(52, 255, 138, 0.22);
    padding: 18px;
    border-radius: 16px;
    border-left: 7px solid #16c172;
    font-size: 22px;
    font-weight: 600;
    animation: glow 2s infinite;
}

/* Fade in animation */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}

/* Glow effect */
@keyframes glow {
    0% {box-shadow: 0 0 10px #4cffba;}
    50% {box-shadow: 0 0 22px #4cffba;}
    100% {box-shadow: 0 0 10px #4cffba;}
}

</style>
""", unsafe_allow_html=True)

# ---------------------- Header Section ----------------------
st.markdown("<h1 class='big-title'>📊 Student Performance Prediction</h1>", unsafe_allow_html=True)

colA, colB = st.columns([2, 1])

with colB:
    # Add Lottie Animation
    lottie_json = {
      "v": "5.6.8",
      "fr": 30,
      "ip": 0,
      "op": 150,
      "w": 500,
      "h": 500,
      "nm": "Study",
      "ddd": 0,
      "assets": [],
      "layers": []
    }
    st_lottie(lottie_json, height=200, key="study")

with colA:
    st.write("Enter the student's details and let the AI predict performance using a trained ML model.")

# ---------------------- Input Section ----------------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<h3 class='section-title'>📝 Enter Student Details</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    math = st.number_input("📘 Math Score", 0, 100, 0)
    writing = st.number_input("✍️ Writing Score", 0, 100, 0)

with col2:
    reading = st.number_input("📖 Reading Score", 0, 100, 0)
    gender = st.selectbox("🧑 Gender", ["female", "male"])

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- Feature Engineering ----------------------
performance = (math + reading + writing) / 3
gender_encoded = 0 if gender == "female" else 1
gender_female = 1 if gender == "female" else 0
gender_male = 1 if gender == "male" else 0

# Performance encoding
pe = 0 if performance < 60 else (1 if performance < 80 else 2)

input_dict = {
    'math score': math,
    'reading score': reading,
    'writing score': writing,
    'performance': performance,
    'gender_encoded': gender_encoded,
    'performance_encoded': pe,
    'gender_female': gender_female,
    'gender_male': gender_male,
}

input_df = pd.DataFrame([input_dict])
input_df = input_df.reindex(columns=feature_columns, fill_value=0)
input_df = input_df.fillna(0)

# ---------------------- Predict Button ----------------------
predict = st.button("🔍 Predict Performance")

# ---------------------- Show Predictions ----------------------
if predict:

    scaled_input = scaler.transform(input_df)
    prediction = int(model.predict(scaled_input)[0])
    probabilities = model.predict_proba(scaled_input)[0]

    levels = ["Low", "Medium", "High"]
    final_label = levels[prediction]

    # Result with animation
    st.markdown(
        f"<div class='result-box'>🎯 Predicted Category: <b>{final_label}</b></div>",
        unsafe_allow_html=True
    )

    # ---------------------- Probability Chart ----------------------
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📈 Prediction Confidence")

    prob_df = pd.DataFrame({
        "Performance Level": levels,
        "Probability": probabilities
    }).set_index("Performance Level")

    st.bar_chart(prob_df)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------- Gauge Chart ----------------------
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📉 Performance Gauge")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=performance,
        title={"text": "Average Performance Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#001f54"},
            'steps': [
                {'range': [0, 60], 'color': "#ff4d4d"},
                {'range': [60, 80], 'color': "#ffe94a"},
                {'range': [80, 100], 'color': "#4CAF50"},
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
