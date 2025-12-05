# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Student Performance Prediction", layout="wide", page_icon="📊")

# ------------------ CSS: Premium Pink/Violet Glassmorphism ------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(120deg, #f6d3ff 0%, #f2a3ff 35%, #9a6bff 75%, #6b3bff 100%);
        background-attachment: fixed;
    }

    /* floating blurred circles */
    .float-circle {
        position: fixed;
        border-radius: 50%;
        filter: blur(70px);
        opacity: 0.55;
        z-index: 0;
    }
    #circle1 { width: 420px; height: 420px; background: #ff7bd1; top: 5%; left: 3%; }
    #circle2 { width: 380px; height: 380px; background: #a268ff; bottom: 5%; right: 5%; }

    /* glass card */
    .glass-card {
        position: relative;
        backdrop-filter: blur(14px) saturate(140%);
        -webkit-backdrop-filter: blur(14px) saturate(140%);
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 8px 30px rgba(12, 12, 14, 0.25);
        border-radius: 18px;
        padding: 24px;
        transition: transform .25s ease, box-shadow .25s ease;
    }
    .glass-card:hover { transform: translateY(-6px); box-shadow: 0 18px 45px rgba(12,12,14,0.35); }

    .title {
        font-size: 44px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 6px;
        text-shadow: 0 6px 20px rgba(0,0,0,0.25);
    }
    .subtitle {
        color: rgba(255,255,255,0.95);
        margin-bottom: 18px;
        opacity: 0.95;
    }

    /* inputs: can't modify Streamlit internals fully, but style wrappers */
    .input-wrap { padding-top: 6px; padding-bottom: 6px; }

    /* style streamlit button to be gradient */
    .stButton>button {
        background: linear-gradient(90deg,#ff6cc9,#8a6bff) !important;
        color: white !important;
        border-radius: 12px;
        padding: 12px 18px;
        font-size: 18px;
        font-weight: 700;
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
        border: none;
    }
    .stButton>button:hover { transform: translateY(-3px); box-shadow: 0 18px 40px rgba(0,0,0,0.28); }

    /* result box */
    .result-box {
        background: linear-gradient(90deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        border-left: 6px solid #9bffb8;
        padding: 18px;
        border-radius: 12px;
        font-size: 20px;
        font-weight: 700;
        color: #ffffff;
        margin-top: 12px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.18);
    }

    /* section titles */
    .section-title {
        color: #fff;
        font-size: 22px;
        font-weight: 700;
        margin-top: 18px;
        margin-bottom: 12px;
    }

    /* small helper */
    .muted { color: rgba(255,255,255,0.9); font-size: 14px; margin-bottom: 10px; }

    </style>

    <div id="circle1" class="float-circle"></div>
    <div id="circle2" class="float-circle"></div>
    """,
    unsafe_allow_html=True,
)

# ------------------ Load model files (with helpful error UI) ------------------
model_path = Path("student_model.joblib")
scaler_path = Path("scaler.joblib")
features_path = Path("model_features.joblib")

load_ok = True
try:
    model = joblib.load(model_path)
except Exception as e:
    load_ok = False
    st.error(
        "❌ Could not load `student_model.joblib`. Make sure the file exists in the app folder and is pushed to GitHub."
    )

try:
    scaler = joblib.load(scaler_path)
except Exception as e:
    load_ok = False
    st.error(
        "❌ Could not load `scaler.joblib`. Make sure the file exists in the app folder and is pushed to GitHub."
    )

try:
    feature_columns = joblib.load(features_path)
except Exception as e:
    load_ok = False
    st.error(
        "❌ Could not load `model_features.joblib`. Make sure the file exists in the app folder and is pushed to GitHub."
    )

# If loading failed, stop and show instructions
if not load_ok:
    st.stop()

# ------------------ Header ------------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<div style='display:flex;align-items:center;gap:18px;'>", unsafe_allow_html=True)
st.markdown("<div><h1 class='title'>✨ Student Performance Prediction</h1>"
            "<div class='subtitle'>Predict performance category (Low / Medium / High) using three scores.</div></div>",
            unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ------------------ Input Card ------------------
st.markdown("<div class='glass-card' style='margin-top:20px;'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>📝 Input Student Details</div>", unsafe_allow_html=True)
st.markdown("<div class='muted'>Provide Math, Reading and Writing scores (0 - 100). Gender is optional for the model.</div>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    with st.container():
        st.markdown("<div class='input-wrap'>", unsafe_allow_html=True)
        math = st.number_input("📘 Math Score", min_value=0, max_value=100, value=70, step=1)
        writing = st.number_input("✍️ Writing Score", min_value=0, max_value=100, value=67, step=1)
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown("<div class='input-wrap'>", unsafe_allow_html=True)
        reading = st.number_input("📖 Reading Score", min_value=0, max_value=100, value=90, step=1)
        gender = st.selectbox("🧑 Gender", ["female", "male"])
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)  # close input card

# ------------------ Prepare features ------------------
performance = (math + reading + writing) / 3.0
gender_encoded = 0 if gender == "female" else 1
gender_female = 1 if gender == "female" else 0
gender_male = 1 if gender == "male" else 0
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
# Align columns to training order (safe)
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# ------------------ Predict button ------------------
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
predict_clicked = st.button("🔮 Predict Performance")

# ------------------ If clicked: predict and show premium UI ------------------
if predict_clicked:
    # safety: ensure no NaNs
    input_df = input_df.fillna(0)

    # scale & predict
    try:
        scaled = scaler.transform(input_df)
    except Exception as e:
        st.error("Scaler transform failed. Confirm the scaler was fitted on the same feature order used here.")
        raise

    try:
        ypred = model.predict(scaled)
        yprob = model.predict_proba(scaled)[0]
    except Exception as e:
        st.error("Model prediction failed. Confirm model compatibility with the features provided.")
        raise

    label_map = ["Low", "Medium", "High"]
    pred_label = label_map[int(ypred[0])]

    # Result box
    st.markdown(
        f"<div class='glass-card' style='margin-top:12px;'>"
        f"<div class='result-box'>🎯 Predicted Category: <span style='color:#ffd86b'> {pred_label} </span></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Probability bar
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📈 Prediction Confidence")
    prob_df = pd.DataFrame({"Performance Level": label_map, "Probability": yprob}).set_index("Performance Level")
    st.bar_chart(prob_df)
    st.markdown("</div>", unsafe_allow_html=True)

    # Gauge (Plotly)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📉 Performance Gauge")
    gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=performance,
        number={'suffix': ""},
        title={'text': "Average Performance Score"},
        delta={'reference': 75, 'increasing': {'color': "#00cc88"}, 'decreasing': {'color': "#ff4d4d"}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#ff6cc9"},
            'bgcolor': "rgba(0,0,0,0)",
            'steps': [
                {'range': [0, 60], 'color': "#ff6b6b"},
                {'range': [60, 80], 'color': "#ffd86b"},
                {'range': [80, 100], 'color': "#7effb2"},
            ],
            'threshold': {
                'line': {'color': "white", 'width': 2},
                'thickness': 0.75,
                'value': performance
            }
        }
    ))
    gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(gauge, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Detailed probabilities table (small)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.write("Confidence (detailed):")
    probs_display = pd.DataFrame({"Level": label_map, "Probability": (yprob * 100).round(2)})
    probs_display = probs_display.set_index("Level")
    st.table(probs_display)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ Footer / Tips ------------------
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
st.markdown(
    "<div style='color: rgba(255,255,255,0.92); font-size:13px; opacity:0.95'>"
    "Tip: Push changes to GitHub then restart Streamlit Cloud (Manage app → Restart) if your UI doesn't update immediately."
    "</div>", unsafe_allow_html=True
)
