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

# Glassmorphism CSS
st.markdown("""
    <style>
        /* Background gradient */
        .main {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px;
        }
        
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Glassmorphism cards */
        .glass-card {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 30px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            margin-bottom: 20px;
        }
        
        /* Input fields styling */
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > select {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 12px;
            color: white;
            padding: 12px;
        }
        
        /* Labels */
        .stNumberInput > label,
        .stSelectbox > label {
            color: white !important;
            font-weight: 600;
            font-size: 16px;
        }
        
        /* Button styling */
        .stButton > button {
            background: rgba(255, 255, 255, 0.25);
            backdrop-filter: blur(10px);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 15px;
            padding: 15px 30px;
            font-weight: 600;
            font-size: 18px;
            width: 100%;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background: rgba(255, 255, 255, 0.35);
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        }
        
        /* Result card */
        .result-card {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(15px);
            padding: 25px;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: white;
            font-size: 22px;
            text-align: center;
            font-weight: 700;
            margin: 20px 0;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }
        
        /* Headers */
        h1 {
            color: white !important;
            font-weight: 900;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
            margin-bottom: 10px;
        }
        
        h2, h3 {
            color: white !important;
            font-weight: 700;
        }
        
        /* Subheader text */
        .stMarkdown p {
            color: rgba(255, 255, 255, 0.9);
            font-size: 16px;
        }
        
        /* Chart containers */
        .js-plotly-plot {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 15px;
        }
        
        /* Remove default padding */
        .block-container {
            padding-top: 2rem;
        }
        
        /* Metric cards */
        [data-testid="stMetricValue"] {
            color: white;
        }
        
        [data-testid="stMetricLabel"] {
            color: rgba(255, 255, 255, 0.8);
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- HEADER ----------------------
st.markdown("<h1>📊 Student Performance Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.9); font-size: 18px; margin-bottom: 30px;'>Enter student scores to predict performance level</p>", unsafe_allow_html=True)

# ---------------------- Input Section ----------------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

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

    # ---------------------- Charts in glass cards ----------------------
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h3>📈 Prediction Confidence</h3>", unsafe_allow_html=True)

    prob_df = pd.DataFrame({
        "Performance Level": levels,
        "Probability": probabilities
    }).set_index("Performance Level")

    st.bar_chart(prob_df)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------- Gauge Chart ----------------------
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h3>📉 Performance Gauge</h3>", unsafe_allow_html=True)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=performance,
        title={"text": "Average Performance Score", "font": {"color": "white", "size": 20}},
        number={"font": {"color": "white", "size": 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "white"},
            'bar': {'color': "rgba(255, 255, 255, 0.8)"},
            'bgcolor': "rgba(255, 255, 255, 0.1)",
            'borderwidth': 2,
            'bordercolor': "rgba(255, 255, 255, 0.3)",
            'steps': [
                {'range': [0, 60], 'color': "rgba(255, 77, 77, 0.4)"},
                {'range': [60, 80], 'color': "rgba(255, 233, 74, 0.4)"},
                {'range': [80, 100], 'color': "rgba(76, 175, 80, 0.4)"},
            ],
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )

    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
