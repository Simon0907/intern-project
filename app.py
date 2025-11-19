import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

# -----------------------------------------
# Load Model + Scaler + Feature Columns
# -----------------------------------------
model = joblib.load("student_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_columns = joblib.load("model_features.joblib")

# -----------------------------------------
# Page Settings
# -----------------------------------------
st.set_page_config(page_title="Student Performance Prediction", layout="centered")
st.title("📊 Student Performance Prediction App")
st.write("Enter student scores to predict the performance level.")

# -----------------------------------------
# Input Section
# -----------------------------------------
col1, col2 = st.columns(2)

with col1:
    math = st.number_input("📘 Math Score", 0, 100, 0)
    writing = st.number_input("✍️ Writing Score", 0, 100, 0)

with col2:
    reading = st.number_input("📖 Reading Score", 0, 100, 0)
    gender = st.selectbox("🧑 Gender", ["female", "male"])

# Calculate performance
performance = (math + reading + writing) / 3

# Manual encodings (MUST match your training!)
gender_encoded = 0 if gender == "female" else 1
gender_female = 1 if gender == "female" else 0
gender_male = 1 if gender == "male" else 0

# Performance category encoding
if performance < 60:
    pe = 0
elif performance < 80:
    pe = 1
else:
    pe = 2

# -----------------------------------------
# Prepare DataFrame
# -----------------------------------------
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

# Ensure correct column order
input_df = input_df.reindex(columns=feature_columns, fill_value=0)
input_df = input_df.fillna(0)

# -----------------------------------------
# Predict Button
# -----------------------------------------
if st.button("🔍 Predict Performance"):

    # Scale input safely
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = int(model.predict(scaled_input)[0])
    probabilities = model.predict_proba(scaled_input)[0]

    levels = ["Low", "Medium", "High"]
    final_label = levels[prediction]

    st.success(f"🎯 **Predicted Category: {final_label}**")

    # -----------------------------------------
    # Probability Chart
    # -----------------------------------------
    st.subheader("📈 Prediction Confidence")
    prob_df = pd.DataFrame({
        "Performance Level": levels,
        "Probability": probabilities
    }).set_index("Performance Level")

    st.bar_chart(prob_df)

    # -----------------------------------------
    # Gauge Chart
    # -----------------------------------------
    st.subheader("📉 Performance Gauge")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=performance,
        title={"text": "Average Performance Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 60], 'color': "red"},
                {'range': [60, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"},
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)
