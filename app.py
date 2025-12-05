import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

# ---------------------- Load Model + Assets ----------------------
model = joblib.load("student_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_columns = joblib.load("model_features.joblib")

# ---------------------- Page Styling ----------------------
st.set_page_config(page_title="Student Performance Prediction", layout="wide")

# Custom CSS for modern UI
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
            padding: 20px;
        }
        .stMetric {
            background: white;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0px 2px 5px rgba(0,0,0,0.1);
        }
        .input-card {
            background: white;
            padding: 25px;
            border-radius: 16px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        }
        .result-card {
            background: #e8f9f1;
            padding: 18px;
            border-radius: 12px;
            font-size: 18px;
        }
        .btn {
            width: 100%;
            padding: 12px;
            border-radius: 10px;
        }
        h1 {
            color: #202a44;
            font-weight: 900;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- HEADER ----------------------
st.markdown("<h1>📊 Student Performance Prediction</h1>", unsafe_allow_html=True)
st.write("Enter the student's scores below to predict the performance level.")

# ---------------------- Input Section ----------------------
with st.container():
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        math = st.number_input("📘 Math Score", 0, 100, 0)
        writing = st.number_input("✍️ Writing Score", 0, 100, 0)

    with col2:
        reading = st.number_input("📖 Reading Score", 0, 100, 0)
        gender = st.selectbox("🧑 Gender", ["female", "male"])

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- Feature Processing ----------------------
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
input_df = input_df.fillna(0)

# ---------------------- Predict Button ----------------------
predict = st.button("🔍 Predict Performance", use_container_width=True)

if predict:
    scaled_input = scaler.transform(input_df)
    prediction = int(model.predict(scaled_input)[0])
    probabilities = model.predict_proba(scaled_input)[0]

    levels = ["Low", "Medium", "High"]
    final_label = levels[prediction]

    # Result Box
    st.markdown(
        f"<div class='result-card'>🎯 <b>Predicted Category:</b> {final_label}</div>",
        unsafe_allow_html=True
    )

    # ---------------------- Probability Chart ----------------------
    st.subheader("📈 Prediction Confidence")

    prob_df = pd.DataFrame({
        "Performance Level": levels,
        "Probability": probabilities
    }).set_index("Performance Level")

    st.bar_chart(prob_df)

    # ---------------------- Gauge Chart ----------------------
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
