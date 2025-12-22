import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from datetime import datetime
import time

# ---------------------- Load Model & Assets ----------------------
model = joblib.load("student_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_columns = joblib.load("model_features.joblib")

# ---------------------- Page Styling ----------------------
st.set_page_config(
    page_title="Student Performance Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern Dark Theme with Cyan Accents
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700;800&family=Geist+Mono:wght@400;500&display=swap');
        
        * {
            font-family: 'Geist', sans-serif;
        }
        
        /* Dark gradient background */
        .main {
            background: linear-gradient(135deg, #0f172a 0%, #1a2847 50%, #0d1f38 100%);
            min-height: 100vh;
        }
        
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1a2847 50%, #0d1f38 100%);
        }
        
        /* Animations */
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes slideInLeft {
            from {
                opacity: 0;
                transform: translateX(-30px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        @keyframes slideInRight {
            from {
                opacity: 0;
                transform: translateX(30px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        @keyframes scaleIn {
            from {
                opacity: 0;
                transform: scale(0.95);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.8; }
        }
        
        @keyframes shimmer {
            0% {
                background-position: -1000px 0;
            }
            100% {
                background-position: 1000px 0;
            }
        }
        
        /* Main title */
        h1 {
            color: #ffffff !important;
            font-weight: 800;
            text-align: center;
            font-size: 3em;
            margin-bottom: 10px;
            animation: fadeInDown 0.8s ease-out;
            letter-spacing: -1px;
        }
        
        /* Subtitle */
        .subtitle {
            text-align: center;
            color: #94a3b8;
            font-size: 1.1em;
            margin-bottom: 40px;
            animation: fadeInUp 0.8s ease-out 0.2s both;
            font-weight: 500;
            letter-spacing: 0.3px;
        }
        
        /* Premium card styling */
        .card {
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 20px;
            animation: fadeInUp 0.8s ease-out;
            transition: all 0.3s ease;
        }
        
        .card:hover {
            border-color: rgba(34, 211, 238, 0.3);
            background: rgba(30, 41, 59, 0.95);
        }
        
        /* Input section */
        .input-card {
            background: rgba(30, 41, 59, 0.6);
            border: 2px solid rgba(34, 211, 238, 0.2);
            border-radius: 16px;
            padding: 40px;
            margin-bottom: 30px;
            animation: fadeInUp 0.8s ease-out 0.2s both;
            transition: all 0.3s ease;
        }
        
        .input-card:hover {
            border-color: rgba(34, 211, 238, 0.4);
            background: rgba(30, 41, 59, 0.8);
        }
        
        /* Input labels */
        .stNumberInput > label,
        .stSelectbox > label {
            color: #e2e8f0 !important;
            font-weight: 600;
            font-size: 0.95em;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 0.8em;
        }
        
        /* Input fields */
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > select {
            background: rgba(15, 23, 42, 0.6) !important;
            border: 1.5px solid rgba(148, 163, 184, 0.2) !important;
            border-radius: 10px;
            color: #e2e8f0 !important;
            padding: 12px 14px !important;
            font-size: 0.95em;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .stNumberInput > div > div > input:focus,
        .stSelectbox > div > div > select:focus {
            background: rgba(15, 23, 42, 0.8) !important;
            border: 1.5px solid #22d3ee !important;
            box-shadow: 0 0 20px rgba(34, 211, 238, 0.1);
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%);
            color: white !important;
            border: none !important;
            border-radius: 12px;
            padding: 14px 40px !important;
            font-weight: 700;
            font-size: 1em;
            width: 100%;
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            text-transform: uppercase;
            letter-spacing: 0.8px;
            box-shadow: 0 4px 15px rgba(34, 211, 238, 0.25);
            animation: fadeInUp 0.8s ease-out 0.3s both;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #06b6d4 0%, #0ea5e9 100%);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(34, 211, 238, 0.35);
        }
        
        .stButton > button:active {
            transform: translateY(0);
        }
        
        /* Result card */
        .result-card {
            background: linear-gradient(135deg, rgba(34, 211, 238, 0.1), rgba(6, 182, 212, 0.05));
            border: 2px solid rgba(34, 211, 238, 0.4);
            border-radius: 16px;
            padding: 40px;
            text-align: center;
            margin: 30px 0;
            animation: scaleIn 0.6s ease-out;
        }
        
        .result-label {
            color: #94a3b8;
            font-size: 0.9em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 15px;
        }
        
        .result-badge {
            display: inline-block;
            color: white;
            padding: 16px 40px;
            border-radius: 12px;
            font-weight: 800;
            font-size: 2em;
            margin: 20px 0;
            animation: slideInRight 0.6s ease-out;
        }
        
        .badge-high {
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(34, 197, 94, 0.1));
            border: 2px solid rgba(34, 197, 94, 0.4);
            color: #86efac;
        }
        
        .badge-medium {
            background: linear-gradient(135deg, rgba(234, 179, 8, 0.2), rgba(234, 179, 8, 0.1));
            border: 2px solid rgba(234, 179, 8, 0.4);
            color: #fcd34d;
        }
        
        .badge-low {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(239, 68, 68, 0.1));
            border: 2px solid rgba(239, 68, 68, 0.4);
            color: #fca5a5;
        }
        
        /* Score cards */
        .score-card {
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 14px;
            padding: 25px;
            text-align: center;
            animation: fadeInUp 0.6s ease-out;
            transition: all 0.3s ease;
        }
        
        .score-card:hover {
            border-color: rgba(34, 211, 238, 0.3);
            background: rgba(30, 41, 59, 0.95);
            transform: translateY(-4px);
        }
        
        .score-icon {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .score-label {
            color: #94a3b8;
            font-size: 0.85em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }
        
        .score-value {
            color: #22d3ee;
            font-size: 2.8em;
            font-weight: 800;
        }
        
        /* Section headers */
        h2, h3 {
            color: #f1f5f9 !important;
            font-weight: 700;
            margin-top: 40px;
            margin-bottom: 25px;
            animation: fadeInUp 0.6s ease-out;
            letter-spacing: -0.5px;
        }
        
        h2 {
            font-size: 2em;
        }
        
        h3 {
            font-size: 1.5em;
        }
        
        /* Chart wrapper */
        .chart-wrapper {
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 16px;
            padding: 30px;
            margin: 25px 0;
            animation: fadeInUp 0.8s ease-out;
        }
        
        /* Text styling */
        p, span {
            color: #cbd5e1;
        }
        
        /* Insight cards */
        .insight-card {
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 14px;
            padding: 25px;
            animation: fadeInUp 0.8s ease-out;
            transition: all 0.3s ease;
        }
        
        .insight-card:hover {
            border-color: rgba(34, 211, 238, 0.3);
            background: rgba(30, 41, 59, 0.95);
        }
        
        .insight-card h4 {
            color: #22d3ee;
            font-size: 1.2em;
            margin-top: 0;
            margin-bottom: 15px;
            font-weight: 700;
        }
        
        .insight-card p {
            color: #cbd5e1;
            line-height: 1.6;
        }
        
        /* Block container */
        .block-container {
            padding-top: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        /* Remove default divider color */
        hr {
            border: none;
            height: 1px;
            background: rgba(148, 163, 184, 0.1);
            margin: 30px 0;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(15, 23, 42, 0.5);
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(34, 211, 238, 0.3);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(34, 211, 238, 0.5);
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- HEADER ----------------------
st.markdown("<h1>📊 Student Performance Prediction</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>AI-Powered Academic Performance Analysis System</div>",
    unsafe_allow_html=True
)

# ---------------------- Input Section ----------------------
st.markdown("<div class='input-card'>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("<p style='font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.5px; color: #94a3b8; font-weight: 600; margin-bottom: 8px;'>📘 Math Score</p>", unsafe_allow_html=True)
    math = st.number_input("Math Score", 0, 100, 0, label_visibility="collapsed")
    
    st.markdown("<p style='font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.5px; color: #94a3b8; font-weight: 600; margin-bottom: 8px; margin-top: 20px;'>✍️ Writing Score</p>", unsafe_allow_html=True)
    writing = st.number_input("Writing Score", 0, 100, 0, label_visibility="collapsed")

with col2:
    st.markdown("<p style='font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.5px; color: #94a3b8; font-weight: 600; margin-bottom: 8px;'>📖 Reading Score</p>", unsafe_allow_html=True)
    reading = st.number_input("Reading Score", 0, 100, 0, label_visibility="collapsed")
    
    st.markdown("<p style='font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.5px; color: #94a3b8; font-weight: 600; margin-bottom: 8px; margin-top: 20px;'>🧑 Student Gender</p>", unsafe_allow_html=True)
    gender = st.selectbox("Gender", ["Female", "Male"], label_visibility="collapsed")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- Feature Processing ----------------------
performance = (math + reading + writing) / 3

gender_encoded = 0 if gender.lower() == "female" else 1
gender_female = 1 if gender.lower() == "female" else 0
gender_male = 1 if gender.lower() == "male" else 0

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
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict = st.button("🔍 Predict Performance", use_container_width=True)

if predict:
    with st.spinner(""):
        progress_bar = st.progress(0)
        
        for i in range(100):
            progress_bar.progress(i + 1)
            time.sleep(0.015)
        
        progress_bar.empty()
    
    # Get predictions
    scaled_input = scaler.transform(input_df)
    prediction = int(model.predict(scaled_input)[0])
    probabilities = model.predict_proba(scaled_input)[0]

    levels = ["Low", "Medium", "High"]
    final_label = levels[prediction]
    
    # Badge styles
    badge_classes = {
        "Low": "badge-low",
        "Medium": "badge-medium",
        "High": "badge-high"
    }

    # ==================== RESULTS SECTION ====================
    st.markdown("<h2 style='text-align: center;'>📈 Prediction Results</h2>", unsafe_allow_html=True)
    
    # Result card
    st.markdown(
        f"""
        <div class='result-card'>
            <div class='result-label'>Predicted Category</div>
            <div class='result-badge {badge_classes[final_label]}'>{final_label}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # ==================== SCORE OVERVIEW ====================
    st.markdown("<h3>📊 Score Overview</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div class='score-card'>
                <div class='score-icon'>📘</div>
                <div class='score-label'>Math</div>
                <div class='score-value'>{math}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div class='score-card'>
                <div class='score-icon'>📖</div>
                <div class='score-label'>Reading</div>
                <div class='score-value'>{reading}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div class='score-card'>
                <div class='score-icon'>✍️</div>
                <div class='score-label'>Writing</div>
                <div class='score-value'>{writing}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            f"""
            <div class='score-card'>
                <div class='score-icon'>⭐</div>
                <div class='score-label'>Average</div>
                <div class='score-value'>{performance:.1f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # ==================== CONFIDENCE CHART ====================
    st.markdown("<div class='chart-wrapper'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top: 0;'>📊 Prediction Confidence</h3>", unsafe_allow_html=True)

    prob_df = pd.DataFrame({
        "Performance Level": levels,
        "Probability %": [round(p * 100, 1) for p in probabilities]
    })

    fig_bar = go.Figure(data=[
        go.Bar(
            x=prob_df["Performance Level"],
            y=prob_df["Probability %"],
            marker=dict(
                color=['#fc8181', '#fcd34d', '#86efac'],
                line=dict(color='rgba(255,255,255,0.2)', width=2)
            ),
            text=[f"{p:.1f}%" for p in prob_df["Probability %"]],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>',
        )
    ])
    
    fig_bar.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1', size=12),
        xaxis=dict(
            showgrid=False,
            showline=False,
            color='#cbd5e1'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(148, 163, 184, 0.1)',
            showline=False,
            color='#cbd5e1'
        ),
        margin=dict(t=20, b=20, l=20, r=20),
        height=350,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ==================== GAUGE CHART ====================
    st.markdown("<div class='chart-wrapper'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top: 0;'>📉 Performance Gauge</h3>", unsafe_allow_html=True)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=performance,
        delta={'reference': 70, 'suffix': ' vs Target'},
        title={"text": "Average Performance", "font": {"color": "#cbd5e1", "size": 18}},
        number={"font": {"color": "#22d3ee", "size": 48}, "suffix": "/100"},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickcolor': "#94a3b8",
                'tickfont': {'color': '#94a3b8'}
            },
            'bar': {'color': "#22d3ee"},
            'bgcolor': "rgba(148, 163, 184, 0.1)",
            'borderwidth': 2,
            'bordercolor': "rgba(148, 163, 184, 0.3)",
            'steps': [
                {'range': [0, 60], 'color': "rgba(239, 68, 68, 0.1)"},
                {'range': [60, 80], 'color': "rgba(234, 179, 8, 0.1)"},
                {'range': [80, 100], 'color': "rgba(34, 197, 94, 0.1)"},
            ],
        }
    ))
    
    fig_gauge.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#cbd5e1'),
        margin=dict(t=40, b=40, l=40, r=40),
        height=400
    )

    st.plotly_chart(fig_gauge, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ==================== INSIGHTS SECTION ====================
    st.markdown("<h3>💡 Performance Insights</h3>", unsafe_allow_html=True)
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
        st.markdown(f"<h4>📌 Category Analysis</h4>", unsafe_allow_html=True)
        
        if final_label == "High":
            st.markdown(
                "Your student demonstrates **excellent academic performance** with strong, consistent scores across all subjects. This indicates strong command of core concepts and excellent study habits.",
                unsafe_allow_html=True
            )
        elif final_label == "Medium":
            st.markdown(
                "Your student shows **solid academic abilities** with balanced performance across subjects. Focused improvement in weaker areas could lead to higher overall performance.",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "Your student would benefit from **additional support and targeted practice**. Focusing on foundational concepts in weaker subjects will help improve overall performance.",
                unsafe_allow_html=True
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with insights_col2:
        st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
        st.markdown(f"<h4>📊 Subject Strengths</h4>", unsafe_allow_html=True)
        
        scores = {"Math": math, "Reading": reading, "Writing": writing}
        strongest = max(scores, key=scores.get)
        weakest = min(scores, key=scores.get)
        
        st.markdown(
            f"<p><strong>🏆 Strongest:</strong> {strongest} ({scores[strongest]}/100)</p>"
            f"<p><strong>📈 Focus Area:</strong> {weakest} ({scores[weakest]}/100)</p>"
            f"<p><strong>🎯 Recommendation:</strong> Maintain {strongest} excellence while improving {weakest}</p>",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
