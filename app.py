import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    layout="wide"
)

# --------------------------
# CUSTOM CSS (Modern UI)
# --------------------------
st.markdown("""
<style>

body {
    background-color: #f7f9fc;
}

.big-title {
    font-size: 38px;
    font-weight: 800;
    color: #1f2937;
    display: flex;
    gap: 10px;
}

.card {
    padding: 25px;
    background: white;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

.predict-btn {
    background: linear-gradient(90deg, #6366f1, #3b82f6);
    color: white;
    padding: 12px 25px;
    border-radius: 10px;
    font-size: 16px;
    border: none;
}

.result-box {
    background: #e8fce8;
    padding: 18px;
    border-radius: 10px;
    font-size: 18px;
    font-weight: 600;
    display: flex;
    gap: 10px;
    align-items: center;
}

.sub-title {
    font-size: 26px;
    font-weight: 700;
    margin-top: 20px;
    color: #374151;
    display: flex;
    gap: 10px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------
# LOAD MODELS
# --------------------------
model = joblib.load("student_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_columns = joblib.load("model_features.joblib")

# --------------------------
# TITLE
# --------------------------
st.markdown(
    "<div class='big-title'>📊 Student Performance Prediction</div>",
    unsafe_allow_html=True
)

st.write("Enter student scores to predict the performance category. The app uses ML to classify students into **Low**, **Medium**, or **High** performance.")

# --------------------------
# INPUT CARD
# --------------------------
st.markdown("<div class='sub-title'>📝 Input Student Details</div>", unsafe_allow_html=True)
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        math = st.number_input("📘 Math Score", 0, 100, 0)
        writing = st.number_input("✍️ Writing Score", 0, 100, 0)

    with col2:
        reading = st.number_input("📖 Reading Score", 0, 100, 0)
        gender = st.selectbox("🧑 Gender", ["female", "male"])

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# PREPARE DATA
# --------------------------
performance = (math + reading + writing) / 3
gender_encoded = 0 if gender == "female" else 1
gender_female = 1 if gender == "female" else 0
gender_male = 1 if gender == "male" else 0

if performance < 60:
    pe = 0
elif performance < 80:
    pe = 1
else:
    pe = 2

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

# --------------------------
# PREDICT BUTTON
# --------------------------
clicked = st.button("🔍 Predict Performance", use_container_width=True)

if clicked:
    scaled_input = scaler.transform(input_df)
    prediction = int(model.predict(scaled_input)[0])
    probabilities = model.predict_proba(scaled_input)[0]

    levels = ["Low", "Medium", "High"]
    final_label = levels[prediction]

    # RESULT BOX
    st.markdown(
        f"<div class='result-box'>🎯 Predicted Category: <b>{final_label}</b></div>",
        unsafe_allow_html=True
    )

    # --------------------------
    # PROBABILITY BAR
    # --------------------------
    st.markdown("<div class='sub-title'>📈 Prediction Confidence</div>", unsafe_allow_html=True)
    prob_df = pd.DataFrame({
        "Performance Level": levels,
        "Probability": probabilities
    })
    st.bar_chart(prob_df.set_index("Performance Level"))

    # --------------------------
    # GAUGE CHART
    # --------------------------
    st.markdown("<div class='sub-title'>📉 Performance Gauge</div>", unsafe_allow_html=True)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=performance,
        title={"text": "Average Performance Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#3b82f6"},
            'steps': [
                {'range': [0, 60], 'color': "red"},
                {'range': [60, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"},
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)
