import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from datetime import datetime
import time

# ---------------------- Load Model + Assets ----------------------
model = joblib.load("student_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_columns = joblib.load("model_features.joblib")

# ---------------------- Page Styling ----------------------
st.set_page_config(
    page_title="Student Performance Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium Glassmorphism + Animations CSS
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        
        * {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        
        /* Animated gradient background */
        .main {
            background: linear-gradient(-45deg, #667eea, #764ba2, #5f72bd, #667eea);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
            min-height: 100vh;
            padding: 40px 20px;
        }
        
        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .stApp {
            background: linear-gradient(-45deg, #667eea, #764ba2, #5f72bd, #667eea);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
        }
        
        /* Fade in animation */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* Slide in animation */
        @keyframes slideInRight {
            from {
                opacity: 0;
                transform: translateX(50px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        /* Scale animation */
        @keyframes scaleIn {
            from {
                opacity: 0;
                transform: scale(0.8);
            }
            to {
                opacity: 1;
                transform: scale(1);
            }
        }
        
        /* Pulse animation */
        @keyframes pulse {
            0%, 100% {
                opacity: 1;
            }
            50% {
                opacity: 0.7;
            }
        }
        
        /* Floating animation */
        @keyframes float {
            0%, 100% {
                transform: translateY(0px);
            }
            50% {
                transform: translateY(-10px);
            }
        }
        
        /* Glow animation */
        @keyframes glow {
            0%, 100% {
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37), 0 0 20px rgba(255, 255, 255, 0.1);
            }
            50% {
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.5), 0 0 40px rgba(255, 255, 255, 0.2);
            }
        }
        
        /* Title styling with animation */
        h1 {
            color: white !important;
            font-weight: 800;
            text-align: center;
            text-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
            font-size: 3.5em;
            margin-bottom: 10px;
            animation: fadeInUp 0.8s ease-out;
        }
        
        /* Subtitle */
        .subtitle {
            text-align: center;
            color: rgba(255, 255, 255, 0.95);
            font-size: 1.1em;
            margin-bottom: 40px;
            animation: fadeInUp 0.8s ease-out 0.2s both;
            font-weight: 500;
            letter-spacing: 0.5px;
        }
        
        /* Premium glass card */
        .glass-card {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.25);
            padding: 35px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            margin-bottom: 25px;
            animation: fadeInUp 0.8s ease-out 0.3s both;
            transition: all 0.3s ease;
        }
        
        .glass-card:hover {
            background: rgba(255, 255, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.35);
            transform: translateY(-5px);
            box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.45);
        }
        
        /* Input section card */
        .input-section {
            background: rgba(255, 255, 255, 0.12);
            backdrop-filter: blur(25px);
            -webkit-backdrop-filter: blur(25px);
            border-radius: 28px;
            border: 1.5px solid rgba(255, 255, 255, 0.2);
            padding: 40px;
            box-shadow: inset 0 0 30px rgba(255, 255, 255, 0.1), 0 8px 32px rgba(31, 38, 135, 0.37);
            margin-bottom: 30px;
            animation: fadeInUp 0.8s ease-out 0.25s both;
        }
        
        /* Input labels with styling */
        .stNumberInput > label,
        .stSelectbox > label {
            color: white !important;
            font-weight: 700;
            font-size: 1.1em;
            margin-bottom: 10px;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        }
        
        /* Input fields styling */
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > select {
            background: rgba(255, 255, 255, 0.25) !important;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1.5px solid rgba(255, 255, 255, 0.3) !important;
            border-radius: 14px;
            color: white !important;
            padding: 14px 16px !important;
            font-size: 1em;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .stNumberInput > div > div > input:focus,
        .stSelectbox > div > div > select:focus {
            background: rgba(255, 255, 255, 0.35) !important;
            border: 1.5px solid rgba(255, 255, 255, 0.5) !important;
            box-shadow: 0 0 20px rgba(255, 255, 255, 0.2);
        }
        
        /* Input placeholder */
        .stNumberInput > div > div > input::placeholder {
            color: rgba(255, 255, 255, 0.6);
        }
        
        /* Premium button styling */
        .stButton > button {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.3), rgba(255, 255, 255, 0.15));
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
            color: white !important;
            border: 1.5px solid rgba(255, 255, 255, 0.4) !important;
            border-radius: 18px;
            padding: 16px 40px !important;
            font-weight: 700;
            font-size: 1.1em;
            width: 100%;
            transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            text-transform: uppercase;
            letter-spacing: 1.2px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.3);
            animation: fadeInUp 0.8s ease-out 0.4s both;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.4), rgba(255, 255, 255, 0.25));
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 12px 40px rgba(31, 38, 135, 0.45);
            border: 1.5px solid rgba(255, 255, 255, 0.6) !important;
        }
        
        .stButton > button:active {
            transform: translateY(-1px) scale(0.98);
        }
        
        /* Result card - Premium */
        .result-card {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.1));
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1.5px solid rgba(255, 255, 255, 0.35);
            padding: 35px;
            border-radius: 24px;
            color: white;
            font-size: 1.8em;
            text-align: center;
            font-weight: 800;
            margin: 30px 0;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37), inset 0 0 20px rgba(255, 255, 255, 0.1);
            animation: scaleIn 0.6s ease-out;
            letter-spacing: 0.5px;
        }
        
        /* Category badge */
        .category-badge {
            display: inline-block;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.25), rgba(255, 255, 255, 0.1));
            border: 1.5px solid rgba(255, 255, 255, 0.4);
            color: white;
            padding: 12px 30px;
            border-radius: 50px;
            font-weight: 700;
            font-size: 1.3em;
            margin: 10px 0;
            animation: slideInRight 0.6s ease-out 0.2s both;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        
        /* Performance score card */
        .score-card {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.25);
            padding: 25px;
            text-align: center;
            color: white;
            margin: 15px;
            animation: scaleIn 0.6s ease-out;
            transition: all 0.3s ease;
        }
        
        .score-card:hover {
            transform: translateY(-8px);
            background: rgba(255, 255, 255, 0.2);
            box-shadow: 0 12px 32px rgba(31, 38, 135, 0.4);
        }
        
        .score-value {
            font-size: 2.5em;
            font-weight: 800;
            margin: 10px 0;
            animation: float 3s ease-in-out infinite;
        }
        
        .score-label {
            font-size: 1em;
            font-weight: 600;
            opacity: 0.9;
        }
        
        /* Section headers */
        h2, h3 {
            color: white !important;
            font-weight: 700;
            text-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
            margin-top: 30px;
            margin-bottom: 20px;
            animation: fadeInUp 0.6s ease-out;
        }
        
        /* Progress bar animation */
        .progress-bar {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            height: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, rgba(255, 255, 255, 0.6), rgba(255, 255, 255, 0.3));
            border-radius: 20px;
            animation: slideInRight 1s ease-out;
        }
        
        /* Chart wrapper */
        .chart-wrapper {
            background: rgba(255, 255, 255, 0.12);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 30px;
            margin: 20px 0;
            animation: fadeInUp 0.8s ease-out;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        }
        
        /* Remove default Streamlit padding */
        .block-container {
            padding-top: 2rem;
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        /* Text styling */
        p, span {
            color: rgba(255, 255, 255, 0.95);
        }
        
        /* Tooltip styling */
        .tooltip {
            background: rgba(0, 0, 0, 0.3);
            color: white;
            padding: 10px 15px;
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            font-size: 0.95em;
            margin-top: 10px;
        }
        
        /* Loading animation */
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        /* Decorative elements */
        .decoration {
            position: fixed;
            border-radius: 50%;
            opacity: 0.1;
            pointer-events: none;
            animation: float 8s ease-in-out infinite;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- HEADER with Animation ----------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1>📊 Student Performance Prediction</h1>", unsafe_allow_html=True)

st.markdown(
    "<div class='subtitle'>🎓 Predict Academic Excellence with Advanced AI Analytics</div>",
    unsafe_allow_html=True
)

# ---------------------- Input Section ----------------------
st.markdown("<div class='input-section'>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("<p style='font-size: 0.95em; opacity: 0.9; margin-bottom: 5px;'>📘 Math Score</p>", unsafe_allow_html=True)
    math = st.number_input("Math Score", 0, 100, 0, label_visibility="collapsed")
    
    st.markdown("<p style='font-size: 0.95em; opacity: 0.9; margin-bottom: 5px; margin-top: 15px;'>✍️ Writing Score</p>", unsafe_allow_html=True)
    writing = st.number_input("Writing Score", 0, 100, 0, label_visibility="collapsed")

with col2:
    st.markdown("<p style='font-size: 0.95em; opacity: 0.9; margin-bottom: 5px;'>📖 Reading Score</p>", unsafe_allow_html=True)
    reading = st.number_input("Reading Score", 0, 100, 0, label_visibility="collapsed")
    
    st.markdown("<p style='font-size: 0.95em; opacity: 0.9; margin-bottom: 5px; margin-top: 15px;'>🧑 Student Gender</p>", unsafe_allow_html=True)
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
    # Simulate processing with animation
    with st.spinner(""):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(100):
            progress_bar.progress(i + 1)
            time.sleep(0.02)
        
        status_text.empty()
        progress_bar.empty()
    
    # Get predictions
    scaled_input = scaler.transform(input_df)
    prediction = int(model.predict(scaled_input)[0])
    probabilities = model.predict_proba(scaled_input)[0]

    levels = ["Low", "Medium", "High"]
    final_label = levels[prediction]
    
    # Color mapping
    colors = {
        "Low": "#FF6B6B",
        "Medium": "#FFD93D",
        "High": "#4CAF50"
    }

    # ==================== RESULTS SECTION ====================
    st.markdown("<h2 style='text-align: center; margin-top: 40px;'>📈 Prediction Results</h2>", unsafe_allow_html=True)
    
    # Animated result card
    st.markdown(
        f"""
        <div class='result-card'>
            <div style='font-size: 1em; opacity: 0.85; margin-bottom: 15px;'>🎯 PREDICTED CATEGORY</div>
            <div class='category-badge' style='background: linear-gradient(135deg, {colors[final_label]}40, {colors[final_label]}20);'>{final_label.upper()}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # ==================== SCORE OVERVIEW ====================
    st.markdown("<h3 style='text-align: center;'>📊 Score Overview</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div class='score-card'>
                <div class='score-value'>📘</div>
                <div class='score-label'>Math</div>
                <div class='score-value' style='font-size: 2em; animation: none;'>{math}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div class='score-card'>
                <div class='score-value'>📖</div>
                <div class='score-label'>Reading</div>
                <div class='score-value' style='font-size: 2em; animation: none;'>{reading}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div class='score-card'>
                <div class='score-value'>✍️</div>
                <div class='score-label'>Writing</div>
                <div class='score-value' style='font-size: 2em; animation: none;'>{writing}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            f"""
            <div class='score-card'>
                <div class='score-value'>⭐</div>
                <div class='score-label'>Average</div>
                <div class='score-value' style='font-size: 2em; animation: none;'>{performance:.1f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # ==================== CONFIDENCE CHART ====================
    st.markdown("<div class='chart-wrapper'>", unsafe_allow_html=True)
    st.markdown("<h3>📈 Prediction Confidence Breakdown</h3>", unsafe_allow_html=True)

    prob_df = pd.DataFrame({
        "Performance Level": levels,
        "Probability %": [round(p * 100, 1) for p in probabilities]
    })

    fig_bar = go.Figure(data=[
        go.Bar(
            x=prob_df["Performance Level"],
            y=prob_df["Probability %"],
            marker=dict(
                color=['#FF6B6B', '#FFD93D', '#4CAF50'],
                line=dict(color='rgba(255,255,255,0.3)', width=2)
            ),
            text=[f"{p:.1f}%" for p in prob_df["Probability %"]],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>',
        )
    ])
    
    fig_bar.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12, family='Plus Jakarta Sans'),
        xaxis=dict(
            showgrid=False,
            showline=False,
            linewidth=0,
            color='white'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            showline=False,
            color='white'
        ),
        margin=dict(t=20, b=20, l=20, r=20),
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ==================== GAUGE CHART ====================
    st.markdown("<div class='chart-wrapper'>", unsafe_allow_html=True)
    st.markdown("<h3>📉 Performance Gauge</h3>", unsafe_allow_html=True)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=performance,
        delta={'reference': 70, 'suffix': ' vs Average'},
        title={"text": "Average Performance Score", "font": {"color": "white", "size": 20}},
        number={"font": {"color": "white", "size": 50}, "suffix": "/100"},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickcolor': "rgba(255,255,255,0.5)",
                'tickfont': {'color': 'white'}
            },
            'bar': {'color': "rgba(255, 255, 255, 0.9)"},
            'bgcolor': "rgba(255, 255, 255, 0.1)",
            'borderwidth': 2,
            'bordercolor': "rgba(255, 255, 255, 0.3)",
            'steps': [
                {'range': [0, 60], 'color': "rgba(255, 107, 107, 0.3)"},
                {'range': [60, 80], 'color': "rgba(255, 217, 61, 0.3)"},
                {'range': [80, 100], 'color': "rgba(76, 175, 80, 0.3)"},
            ],
        }
    ))
    
    fig_gauge.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', family='Plus Jakarta Sans'),
        margin=dict(t=40, b=40, l=40, r=40),
        height=450
    )

    st.plotly_chart(fig_gauge, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ==================== INSIGHTS SECTION ====================
    st.markdown("<h3>💡 Performance Insights</h3>", unsafe_allow_html=True)
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color: white; margin-top: 0;'>📌 Category Analysis</h4>", unsafe_allow_html=True)
        
        if final_label == "High":
            st.markdown(
                "✨ **Excellent Performance!** Your student demonstrates exceptional academic performance with strong scores across all subjects. Keep up the excellent work!",
                unsafe_allow_html=True
            )
        elif final_label == "Medium":
            st.markdown(
                "💪 **Good Performance!** Your student shows solid academic abilities. There's room for improvement in specific areas to achieve higher performance levels.",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "📚 **Development Opportunity!** Your student would benefit from additional support and practice. Focus on improving foundational concepts in weaker subjects.",
                unsafe_allow_html=True
            )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with insights_col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color: white; margin-top: 0;'>📊 Subject Strengths</h4>", unsafe_allow_html=True)
        
        scores = {"Math": math, "Reading": reading, "Writing": writing}
        strongest = max(scores, key=scores.get)
        weakest = min(scores, key=scores.get)
        
        st.markdown(
            f"🏆 **Strongest:** {strongest} ({scores[strongest]}/100)<br>"
            f"📈 **Focus Area:** {weakest} ({scores[weakest]}/100)<br>"
            f"🎯 **Recommendation:** Emphasize improvement in {weakest} while maintaining {strongest} performance.",
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
