import streamlit as st
import numpy as np

# ------------------- GLASSMORPHISM CSS -------------------

st.markdown("""
<style>

body {
    background: url('https://images.unsplash.com/photo-1507842217343-583bb7270b66?auto=format&fit=crop&w=1950&q=80')
                no-repeat fixed center;
    background-size: cover;
    font-family: 'Inter', sans-serif;
}

.glass-card {
    backdrop-filter: blur(18px);
    background: rgba(255, 255, 255, 0.18);
    border-radius: 18px;
    padding: 25px;
    border: 1px solid rgba(255, 255, 255, 0.4);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    margin-bottom: 25px;
}

.big-title {
    font-size: 42px;
    font-weight: 900;
    color: white;
    text-shadow: 0px 2px 6px rgba(0,0,0,0.3);
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    color: #fff;
    margin-bottom: 15px;
}

.predict-btn button {
    width: 100%;
    border-radius: 12px !important;
    padding: 14px 0;
    font-size: 18px;
    font-weight: 600;
    background: linear-gradient(135deg, #6a11cb, #2575fc) !important;
    color: white !important;
    border: none !important;
}

.result-box {
    backdrop-filter: blur(14px);
    background: rgba(52, 255, 138, 0.2);
    padding: 18px;
    border-radius: 14px;
    font-size: 20px;
    border-left: 5px solid #31d47a;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ------------------- HEADER -------------------

st.markdown("""
<div class="glass-card">
    <h1 class="big-title">✨ Student Performance Prediction</h1>
    <p style="font-size:18px; color:#f5f5f5;">
        A modern AI-powered tool that predicts student performance levels using Machine Learning.
    </p>
</div>
""", unsafe_allow_html=True)


# ------------------- INPUT SECTION -------------------

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<h3 class="section-title">📝 Student Details</h3>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    math = st.number_input("📘 Math Score", 0, 100, 70)
    writing = st.number_input("✍️ Writing Score", 0, 100, 67)

with col2:
    reading = st.number_input("📚 Reading Score", 0, 100, 90)
    gender = st.selectbox("🧑 Gender", ["male", "female"])

st.markdown("</div>", unsafe_allow_html=True)


# ------------------- PREDICTION BUTTON -------------------

st.markdown('<div class="glass-card">', unsafe_allow_html=True)
predict = st.button("🔍 Predict Performance", key="predict")
st.markdown("</div>", unsafe_allow_html=True)


# ------------------- OUTPUT -------------------

if predict:
    predicted = "Medium"  # ⬅ replace with model output
    st.markdown(
        f'<div class="result-box">🎯 Predicted Category: <b>{predicted}</b></div>',
        unsafe_allow_html=True
    )

    st.subheader("📈 Confidence Chart")
    st.line_chart([0.4, 0.5, 0.6, 0.8])
