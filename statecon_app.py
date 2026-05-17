"""
StatEcon Analyzer Pro - LIVE EDITION
Created by Morris Mutinda
AI-Powered Statistical & Economic Analysis Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import requests
import time
import warnings
warnings.filterwarnings('ignore')

try:
    from streamlit_lottie import st_lottie
    LOTTIE_AVAILABLE = True
except ImportError:
    LOTTIE_AVAILABLE = False

# ============================================================
# CREATOR INFO
# ============================================================
CREATOR_NAME = "Morris Mutinda"
CREATOR_PHONE = "+254 705 481 616"
CREATOR_TITLE = "Applied Statistics & Economics"
CREATOR_TAGLINE = "Turning Data into Decisions"

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="StatEcon Pro | by Morris Mutinda",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': f"""
        # StatEcon Analyzer Pro
        
        **Created by:** {CREATOR_NAME}  
        **Field:** {CREATOR_TITLE}  
        **Contact:** {CREATOR_PHONE}
        
        AI-Powered Statistical & Economic Analysis Platform
        """
    }
)

# ============================================================
# LIVELY CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    html, body, [class*="css"] {
        font-family: Poppins, sans-serif !important;
    }
    
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 20s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main .block-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem 3rem !important;
        margin-top: 1rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.2);
    }
    
    .brand-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 200% 200%;
        animation: shimmer 5s ease infinite;
        padding: 2.5rem;
        border-radius: 25px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    @keyframes shimmer {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .brand-title {
        color: white;
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        text-align: center;
        text-shadow: 3px 3px 10px rgba(0,0,0,0.3);
        animation: fadeInDown 1s ease-out;
    }
    
    .brand-subtitle {
        color: rgba(255,255,255,0.95);
        font-size: 1.3rem;
        text-align: center;
        margin-top: 0.8rem;
        font-weight: 300;
        animation: fadeInUp 1s ease-out 0.3s both;
    }
    
    .creator-badge {
        display: inline-block;
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        color: #2c3e50;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        margin-top: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.5); }
        50% { box-shadow: 0 0 40px rgba(102, 126, 234, 0.8); }
    }
    
    .ai-insight {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #ff6b6b;
        color: #2c3e50;
        font-weight: 500;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        animation: fadeIn 0.6s ease-out;
        transition: all 0.3s ease;
    }
    
    .ai-insight:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    .ai-positive {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #27ae60;
        color: #2c3e50;
        font-weight: 500;
        animation: fadeIn 0.6s ease-out;
        transition: all 0.3s ease;
    }
    
    .ai-positive:hover {
        transform: translateY(-5px) scale(1.02);
    }
    
    .ai-warning {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #e67e22;
        color: #2c3e50;
        font-weight: 500;
        animation: fadeIn 0.6s ease-out;
        transition: all 0.3s ease;
    }
    
    .ai-warning:hover {
        transform: translateY(-5px) scale(1.02);
    }
    
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 5px solid #667eea;
        transition: all 0.3s ease;
        animation: fadeIn 0.5s ease-out;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 10px;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        transform: translateY(-3px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        animation: glow 2s ease-in-out infinite;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 2.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.5);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.1);
        border: 2px dashed rgba(255, 255, 255, 0.4);
        border-radius: 15px;
        padding: 1rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: scale(1.02);
    }
    
    .feature-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        text-align: center;
        transition: all 0.4s ease;
        height: 100%;
        animation: fadeInUp 0.8s ease-out;
    }
    
    .feature-card:hover {
        transform: translateY(-10px) scale(1.03);
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
    }
    
    .feature-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        animation: float 3s ease-in-out infinite;
        display: inline-block;
    }
    
    .feature-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    .stat-counter {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        animation: pulse 3s ease-in-out infinite;
    }
    
    .stat-number {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        color: white;
    }
    
    .stat-label {
        font-size: 0.95rem;
        opacity: 0.9;
        margin: 0;
        color: white;
    }
    
    .achievement-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        color: #2c3e50;
        border-radius: 25px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 0.3rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .contact-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #7f8c8d;
        font-size: 0.9rem;
        border-top: 2px solid #ecf0f1;
        margin-top: 3rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
    }
    
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: rgba(255,255,255,0.1); }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# AI INSIGHTS ENGINE
# ============================================================

def generate_ai_insights(data, numeric_cols, categorical_cols):
    insights = []
    n_rows = data.shape[0]
    
    if n_rows < 30:
        insights.append({'type': 'warning', 'icon': '⚠️', 'title': 'Small Sample Size',
            'message': f"Only {n_rows} observations. Consider collecting more data (30+ recommended)."})
    elif n_rows > 1000:
        insights.append({'type': 'positive', 'icon': '✅', 'title': 'Robust Sample Size',
            'message': f"Excellent! {n_rows} observations is ideal for reliable analysis."})
    
    missing_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
    if missing_pct > 10:
        insights.append({'type': 'warning', 'icon': '🔴', 'title': 'High Missing Data',
            'message': f"{missing_pct:.1f}% missing. Consider imputation techniques."})
    elif missing_pct == 0:
        insights.append({'type': 'positive', 'icon': '✨', 'title': 'Complete Dataset',
            'message': "No missing values — perfect for analysis!"})
    
    for col in numeric_cols:
        series = data[col].dropna()
        if len(series) > 0 and series.mean() != 0:
            cv = abs(series.std() / series.mean() * 100)
            if cv > 50:
                insights.append({'type': 'insight', 'icon': '📊',
                    'title': f'High Variability in {col}',
                    'message': f"{col} shows extreme variability (CV = {cv:.1f}%)."})
    
    for col in numeric_cols:
        series = data[col].dropna()
        if len(series) >= 8:
            skew = series.skew()
            if abs(skew) > 1:
                direction = "right" if skew > 0 else "left"
                insights.append({'type': 'insight', 'icon': '📈',
                    'title': f'{col} is Skewed',
                    'message': f"Heavily {direction}-skewed (skew = {skew:.2f})."})
    
    if len(numeric_cols) >= 2:
        corr = data[numeric_cols].corr()
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                r = corr.iloc[i, j]
                if abs(r) > 0.9:
                    insights.append({'type': 'warning', 'icon': '⚡',
                        'title': 'Multicollinearity Alert',
                        'message': f"{numeric_cols[i]} and {numeric_cols[j]} (r = {r:.3f})."})
                elif abs(r) > 0.7:
                    rel = "positive" if r > 0 else "negative"
                    insights.append({'type': 'insight', 'icon': '🔗',
                        'title': 'Strong Relationship',
                        'message': f"Strong {rel} correlation: {numeric_cols[i]} & {numeric_cols[j]} (r = {r:.3f})."})
    
    if categorical_cols and numeric_cols:
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        if data[cat_col].nunique() < 50:
            top = data.loc[data[num_col].idxmax(), cat_col]
            bot = data.loc[data[num_col].idxmin(), cat_col]
            top_val = data[num_col].max()
            bot_val = data[num_col].min()
            if bot_val != 0:
                ratio = top_val / bot_val
                insights.append({'type': 'insight', 'icon': '🏆',
                    'title': 'Performance Gap',
                    'message': f"'{top}' leads ({top_val:.2f}) vs '{bot}' ({bot_val:.2f}) — {ratio:.1f}x gap."})
    
    return insights

def display_insights(insights):
    if not insights:
        st.info("No insights generated.")
        return
    for insight in insights:
        css_class = {'positive': 'ai-positive', 'warning': 'ai-warning', 'insight': 'ai-insight'}.get(insight['type'], 'ai-insight')
        st.markdown(f"""
            <div class="{css_class}">
                <strong style="font-size: 1.1rem;">{insight['icon']} {insight['title']}</strong><br>
                <span style="font-size: 0.95rem;">{insight['message']}</span>
            </div>
        """, unsafe_allow_html=True)

# ============================================================
# BRANDED HEADER
# ============================================================
st.markdown(f"""
<div class="brand-header">
    <h1 class="brand-title">📊 StatEcon Analyzer Pro</h1>
    <p class="brand-subtitle">✨ AI-Powered Statistical & Economic Analysis ✨</p>
    <div style="text-align: center;">
        <span class="creator-badge">
            👨‍💻 Created by {CREATOR_NAME} | {CREATOR_TAGLINE}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(f"""
        <div style="text-align: center; padding: 1rem;">
            <h1 style="font-size: 2rem; margin: 0;">📊</h1>
            <h2 style="margin-top: 0.5rem;">StatEcon Pro</h2>
            <p style="font-size: 0.85rem; opacity: 0.9;">by {CREATOR_NAME}</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📁 Upload Your Data")
    
    upload_method = st.radio("Upload method:", ["📤 Upload File", "📋 Paste Path"], key="upload_method")
    
    data = None
    
    if upload_method == "📤 Upload File":
        uploaded_file = st.file_uploader("Drop CSV/Excel here", type=['csv', 'xlsx', 'xls'], key="file_up")
        if uploaded_file:
            try:
                with st.spinner("🎨 Loading..."):
                    time.sleep(0.5)
                    if uploaded_file.name.endswith('.csv'):
                        data = pd.read_csv(uploaded_file)
                    else:
                        data = pd.read_excel(uploaded_file)
                st.success(f"✅ {uploaded_file.name}")
                st.balloons()
            except Exception as e:
                st.error(f"❌ {e}")
    else:
        file_path = st.text_input("File path:", placeholder="C:\\path\\to\\file.csv", key="file_path")
        if file_path:
            try:
                file_path = file_path.strip('"').strip("'")
                with st.spinner("🎨 Loading..."):
                    if file_path.endswith('.csv'):
                        data = pd.read_csv(file_path)
                    else:
                        data = pd.read_excel(file_path)
                st.success("✅ Loaded!")
                st.balloons()
            except Exception as e:
                st.error(f"❌ {e}")
    
    st.markdown("---")
    
    if data is not None:
        st.markdown("### 📊 Dataset Info")
        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{data.shape[0]:,}")
        c2.metric("Cols", data.shape[1])
        
        st.markdown("---")
        st.markdown("### 🎨 Customize")
        chart_theme = st.selectbox("Theme",
            ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"], key="theme")
        color_scheme = st.selectbox("Colors",
            ["Viridis", "Plasma", "Inferno", "Magma", "Rainbow", "Turbo"], key="colors")
    else:
        chart_theme = "plotly"
        color_scheme = "Viridis"
    
    st.markdown("---")
    st.markdown(f"""
        <div class="contact-card">
            <h4 style="color: white; margin: 0;">📞 Contact Me</h4>
            <p style="color: white; margin: 0.5rem 0; font-size: 0.9rem;">
                <strong>{CREATOR_NAME}</strong><br>
                {CREATOR_TITLE}<br>
                📱 {CREATOR_PHONE}
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="text-align: center; padding: 1rem; font-size: 0.75rem; opacity: 0.7;">
            <p>🚀 v3.0 LIVE EDITION</p>
            <p>© 2025 All Rights Reserved</p>
        </div>
    """, unsafe_allow_html=True)

# ============================================================
# MAIN CONTENT
# ============================================================
if data is None:
    st.markdown(f"""
        <div style="text-align: center; padding: 2rem 0;">
            <h2 style="font-size: 2rem; color: #667eea;">
                👋 Karibu! Welcome to StatEcon Pro
            </h2>
            <p style="font-size: 1.1rem; color: #6c757d;">
                Created with ❤️ by Morris Mutinda from Kenya 🇰🇪
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Trusted by Data Lovers Worldwide")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class="stat-counter">
            <p class="stat-number">10+</p>
            <p class="stat-label">Analysis Modules</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="stat-counter">
            <p class="stat-number">∞</p>
            <p class="stat-label">AI Insights</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="stat-counter">
            <p class="stat-number">100%</p>
            <p class="stat-label">Free Forever</p>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="stat-counter">
            <p class="stat-number">24/7</p>
            <p class="stat-label">Available</p>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ✨ Features That Will Wow You")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="feature-card">
            <div class="feature-icon">🤖</div>
            <h3 class="feature-title">AI Auto-Insights</h3>
            <p>Smart algorithms discover patterns automatically.</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="feature-card">
            <div class="feature-icon">📊</div>
            <h3 class="feature-title">Pro Analytics</h3>
            <p>Descriptive stats to advanced econometrics.</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="feature-card">
            <div class="feature-icon">🎨</div>
            <h3 class="feature-title">Beautiful Charts</h3>
            <p>Interactive Plotly visualizations.</p>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="feature-card">
            <div class="feature-icon">📈</div>
            <h3 class="feature-title">Regression Magic</h3>
            <p>OLS with AI-powered interpretation.</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="feature-card">
            <div class="feature-icon">💰</div>
            <h3 class="feature-title">Smart Finance</h3>
            <p>NPV, IRR with AI recommendations.</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="feature-card">
            <div class="feature-icon">🔬</div>
            <h3 class="feature-title">Hypothesis Tests</h3>
            <p>T-tests, ANOVA with auto conclusions.</p>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 20px; color: white; animation: pulse 2s ease-in-out infinite;">
            <h2 style="color: white; margin: 0;">👈 Upload Your Data Now!</h2>
            <p style="font-size: 1.1rem; margin-top: 0.5rem;">
                Get instant AI-powered insights in seconds
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎯 Perfect For")
    st.markdown("""
        <div style="text-align: center;">
            <span class="achievement-badge">🎓 Students</span>
            <span class="achievement-badge">📊 Analysts</span>
            <span class="achievement-badge">💼 Researchers</span>
            <span class="achievement-badge">💰 Investors</span>
            <span class="achievement-badge">🏢 Consultants</span>
            <span class="achievement-badge">📈 Economists</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class="contact-card" style="max-width: 600px; margin: 2rem auto;">
            <h2 style="color: white; margin: 0;">👨‍💻 Meet the Creator</h2>
            <h3 style="color: white; margin: 0.5rem 0;">{CREATOR_NAME}</h3>
            <p style="color: white; margin: 0.5rem 0;">
                BSc Applied Statistics & Economics<br>
                📱 {CREATOR_PHONE}
            </p>
            <p style="color: white; font-style: italic; margin-top: 1rem;">
                "{CREATOR_TAGLINE}"
            </p>
        </div>
    """, unsafe_allow_html=True)

else:
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    
    if 'first_load' not in st.session_state:
        st.session_state.first_load = True
        st.balloons()
    
    st.markdown("### 📊 Your Data at a Glance")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="stat-counter">
            <p class="stat-number">{data.shape[0]:,}</p>
            <p class="stat-label">📋 Rows</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="stat-counter">
            <p class="stat-number">{data.shape[1]}</p>
            <p class="stat-label">📊 Columns</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="stat-counter">
            <p class="stat-number">{len(numeric_cols)}</p>
            <p class="stat-label">🔢 Numeric</p>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="stat-counter">
            <p class="stat-number">{len(categorical_cols)}</p>
            <p class="stat-label">📝 Categorical</p>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.expander("🤖 **AI AUTO-INSIGHTS** — Click to see what I discovered!", expanded=True):
        with st.spinner("🧠 AI is analyzing..."):
            time.sleep(0.5)
            insights = generate_ai_insights(data, numeric_cols, categorical_cols)
        if insights:
            st.markdown(f"### 🎯 Found {len(insights)} insights!")
            display_insights(insights)
    
    tabs = st.tabs(["📋 Overview", "📊 Stats", "🔗 Correlations", "🎯 Outliers",
        "🏆 Rankings", "📈 Regression", "🔬 Tests", "📉 Charts", "💰 NPV/IRR", "📥 Export"])
    
       with tabs[0]:
        st.markdown("## 📋 Data Overview")
        info_df = pd.DataFrame({
            'Column': data.columns, 'Type': [str(t) for t in data.dtypes],
            'Missing': data.isnull().sum().values,
            'Missing %': (data.isnull().sum() / len(data) * 100).round(2).values,
            'Unique': data.nunique().values
        })
        st.dataframe(info_df, use_container_width=True)
        st.markdown("### 👀 Preview")
        n_rows = st.slider("Rows:", 5, min(100, len(data)), 10, key="prev_slider")
        st.dataframe(data.head(n_rows), use_container_width=True)
    
    with tabs[1]:
        st.markdown("## 📊 Descriptive Statistics")
        if numeric_cols:
            summary = data[numeric_cols].describe().T
            summary['skewness'] = data[numeric_cols].skew()
            summary['kurtosis'] = data[numeric_cols].kurtosis()
            st.dataframe(summary.round(3), use_container_width=True)
            
            selected_col = st.selectbox("Variable:", numeric_cols, key="desc_var")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean", f"{data[selected_col].mean():.2f}")
            c2.metric("Median", f"{data[selected_col].median():.2f}")
            c3.metric("Std Dev", f"{data[selected_col].std():.2f}")
            c4.metric("Range", f"{data[selected_col].max() - data[selected_col].min():.2f}")
            
            fig = px.histogram(data, x=selected_col, nbins=30, marginal="box",
                              template=chart_theme, color_discrete_sequence=['#667eea'])
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:
        st.markdown("## 🔗 Correlation Analysis")
        if len(numeric_cols) >= 2:
            method = st.selectbox("Method:", ["pearson", "spearman", "kendall"], key="corr_m")
            corr = data[numeric_cols].corr(method=method)
            fig = px.imshow(corr, text_auto='.2f', color_continuous_scale=color_scheme,
                          template=chart_theme, zmin=-1, zmax=1)
            st.plotly_chart(fig, use_container_width=True)
            
            c1, c2 = st.columns(2)
            x_var = c1.selectbox("X:", numeric_cols, key='sc_x')
            y_var = c2.selectbox("Y:", numeric_cols, index=min(1, len(numeric_cols)-1), key='sc_y')
            fig = px.scatter(data, x=x_var, y=y_var, trendline="ols",
                           template=chart_theme, color_discrete_sequence=['#667eea'])
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[3]:
        st.markdown("## 🎯 Outlier Detection")
        if numeric_cols:
            outlier_summary = []
            for col in numeric_cols:
                series = data[col].dropna()
                Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                mask = (series < lower) | (series > upper)
                outlier_summary.append({'Variable': col, 'Outliers': int(mask.sum()),
                    'Outlier %': round(mask.sum() / len(series) * 100, 2),
                    'Lower': round(lower, 2), 'Upper': round(upper, 2)})
            st.dataframe(pd.DataFrame(outlier_summary), use_container_width=True)
            
            selected = st.multiselect("Variables:", numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))], key="out_sel")
            if selected:
                fig = go.Figure()
                for col in selected:
                    fig.add_trace(go.Box(y=data[col], name=col, boxpoints='outliers'))
                fig.update_layout(template=chart_theme)
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[4]:
        st.markdown("## 🏆 Performance Rankings")
        if categorical_cols and numeric_cols:
            c1, c2, c3 = st.columns(3)
            cat = c1.selectbox("Category:", categorical_cols, key="rk_cat")
            metric = c2.selectbox("Metric:", numeric_cols, key="rk_met")
            n = c3.number_input("Top N:", 3, 20, 5, key="rk_n")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"### 🥇 Top {n}")
                st.dataframe(data.nlargest(n, metric)[[cat, metric]].reset_index(drop=True), use_container_width=True)
            with c2:
                st.markdown(f"### 📉 Bottom {n}")
                st.dataframe(data.nsmallest(n, metric)[[cat, metric]].reset_index(drop=True), use_container_width=True)
            
            sorted_data = data.sort_values(metric, ascending=True)
            fig = px.bar(sorted_data, x=metric, y=cat, orientation='h',
                        template=chart_theme, color=metric, color_continuous_scale=color_scheme)
            fig.update_layout(height=max(400, len(data) * 30))
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[5]:
        st.markdown("## 📈 Regression Analysis")
        if len(numeric_cols) >= 2:
            c1, c2 = st.columns(2)
            y_col = c1.selectbox("Y:", numeric_cols, key="reg_y")
            x_opts = [c for c in numeric_cols if c != y_col]
            x_cols = c2.multiselect("X:", x_opts, default=x_opts[:min(2, len(x_opts))], key="reg_x")
            
            if x_cols and st.button("🚀 Run Regression", type="primary", key="run_reg"):
                with st.spinner("🧮 Running..."):
                    time.sleep(0.5)
                    y = data[y_col].dropna()
                    X = data[x_cols].loc[y.index]
                    mask = X.notna().all(axis=1)
                    y, X = y[mask], X[mask]
                    X = sm.add_constant(X)
                    model = OLS(y, X).fit()
                    
                    st.success("✅ Done!")
                    st.balloons()
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("R²", f"{model.rsquared:.4f}")
                    c2.metric("Adj R²", f"{model.rsquared_adj:.4f}")
                    c3.metric("F p-val", f"{model.f_pvalue:.6f}")
                    c4.metric("N", int(model.nobs))
                    
                    coef_df = pd.DataFrame({
                        'Coef': model.params, 'Std Err': model.bse,
                        't-stat': model.tvalues, 'p-value': model.pvalues,
                        'Significant': model.pvalues < 0.05
                    })
                    st.dataframe(coef_df.round(4), use_container_width=True)
                    
                    if model.rsquared > 0.7:
                        st.markdown(f"""<div class="ai-positive">
                            <strong>🎯 Excellent!</strong> R² = {model.rsquared:.2%}
                        </div>""", unsafe_allow_html=True)
                    elif model.rsquared > 0.5:
                        st.markdown(f"""<div class="ai-insight">
                            <strong>📊 Moderate Fit.</strong> R² = {model.rsquared:.2%}
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div class="ai-warning">
                            <strong>⚠️ Weak Fit.</strong> R² = {model.rsquared:.2%}
                        </div>""", unsafe_allow_html=True)
    
    with tabs[6]:
        st.markdown("## 🔬 Hypothesis Testing")
        test = st.selectbox("Test:", ["One-Sample t-Test", "Independent t-Test", "ANOVA"], key="hyp")
        
        if test == "One-Sample t-Test" and numeric_cols:
            col = st.selectbox("Variable:", numeric_cols, key="os_v")
            mu = st.number_input("Mean:", value=0.0, key="os_m")
            if st.button("Run", key="r_os"):
                stat, p = stats.ttest_1samp(data[col].dropna(), mu)
                c1, c2 = st.columns(2)
                c1.metric("t-stat", f"{stat:.4f}")
                c2.metric("p-value", f"{p:.6f}")
                if p < 0.05:
                    st.success("✅ Reject H₀")
                    st.balloons()
                else:
                    st.info("ℹ️ Fail to reject H₀")
        
        elif test == "ANOVA" and categorical_cols and numeric_cols:
            cat = st.selectbox("Group:", categorical_cols, key="an_c")
            val = st.selectbox("Value:", numeric_cols, key="an_v")
            if st.button("Run", key="r_an"):
                groups = [g[val].values for n, g in data.groupby(cat)]
                stat, p = stats.f_oneway(*groups)
                c1, c2 = st.columns(2)
                c1.metric("F-stat", f"{stat:.4f}")
                c2.metric("p-value", f"{p:.6f}")
                if p < 0.05:
                    st.success("✅ Reject H₀")
                else:
                    st.info("ℹ️ Fail to reject H₀")
    
    with tabs[7]:
        st.markdown("## 📉 Visualizations")
        viz = st.selectbox("Chart:", ["Histogram", "Bar", "Scatter", "Pie", "3D Scatter"], key="viz")
        
        if viz == "Histogram" and numeric_cols:
            col = st.selectbox("Var:", numeric_cols, key="vh_v")
            fig = px.histogram(data, x=col, nbins=30, template=chart_theme,
                              color_discrete_sequence=['#667eea'])
            st.plotly_chart(fig, use_container_width=True)
        elif viz == "Bar" and categorical_cols and numeric_cols:
            c1, c2 = st.columns(2)
            x = c1.selectbox("Cat:", categorical_cols, key="vb_c")
            y = c2.selectbox("Val:", numeric_cols, key="vb_v")
            grouped = data.groupby(x)[y].mean().reset_index().sort_values(y, ascending=False)
            fig = px.bar(grouped, x=x, y=y, template=chart_theme,
                        color=y, color_continuous_scale=color_scheme)
            st.plotly_chart(fig, use_container_width=True)
        elif viz == "Scatter" and len(numeric_cols) >= 2:
            c1, c2 = st.columns(2)
            x = c1.selectbox("X:", numeric_cols, key='vs_x')
            y = c2.selectbox("Y:", numeric_cols, index=1, key='vs_y')
            fig = px.scatter(data, x=x, y=y, template=chart_theme, trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
        elif viz == "Pie" and categorical_cols:
            cat = st.selectbox("Cat:", categorical_cols, key="vp_c")
            counts = data[cat].value_counts().head(10)
            fig = px.pie(values=counts.values, names=counts.index, hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        elif viz == "3D Scatter" and len(numeric_cols) >= 3:
            c1, c2, c3 = st.columns(3)
            x = c1.selectbox("X:", numeric_cols, key='v3_x')
            y = c2.selectbox("Y:", numeric_cols, index=1, key='v3_y')
            z = c3.selectbox("Z:", numeric_cols, index=2, key='v3_z')
            fig = px.scatter_3d(data, x=x, y=y, z=z, color=z,
                              color_continuous_scale=color_scheme)
            st.plotly_chart(fig, use_container_width=True)
    
    with tabs[8]:
        st.markdown("## 💰 Cost-Benefit Analysis")
        c1, c2 = st.columns(2)
        cf_str = c1.text_area("Cash flows:", value="-100000, 25000, 30000, 35000, 40000, 45000", key="cba_cf")
        rate = c2.slider("Rate (%):", 0.0, 30.0, 10.0, key="cba_r") / 100
        
        if st.button("💡 Calculate", type="primary", key="cba_b"):
            with st.spinner("💰 Calculating..."):
                time.sleep(0.5)
                cfs = [float(x.strip()) for x in cf_str.split(',')]
                npv = sum(cf / (1 + rate)**t for t, cf in enumerate(cfs))
                
                r = 0.1
                for _ in range(1000):
                    n = sum(cf / (1 + r)**t for t, cf in enumerate(cfs))
                    d = sum(-t * cf / (1 + r)**(t+1) for t, cf in enumerate(cfs))
                    if abs(d) < 1e-12: break
                    r -= n / d
                    if abs(n) < 1e-6: break
                
                c1, c2, c3 = st.columns(3)
                c1.metric("💵 NPV", f"${npv:,.2f}")
                c2.metric("📈 IRR", f"{r*100:.2f}%")
                c3.metric("Decision", "✅ ACCEPT" if npv > 0 else "❌ REJECT")
                
                if npv > 0:
                    st.markdown(f"""<div class="ai-positive">
                        <strong>🤖 Accept!</strong> NPV positive (${npv:,.2f})
                    </div>""", unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown(f"""<div class="ai-warning">
                        <strong>🤖 Reject.</strong> NPV negative (${npv:,.2f})
                    </div>""", unsafe_allow_html=True)
                
                fig = go.Figure(go.Bar(x=list(range(len(cfs))), y=cfs,
                    marker_color=['#e74c3c' if c<0 else '#27ae60' for c in cfs],
                    text=[f'${c:,.0f}' for c in cfs], textposition='outside'))
                fig.update_layout(title="💰 Cash Flow Timeline", template=chart_theme)
                st.plotly_chart(fig, use_container_width=True)
    
    with tabs[9]:
        st.markdown("## 📥 Export Results")
        c1, c2, c3 = st.columns(3)
        with c1:
            if numeric_cols:
                st.download_button("📊 Stats", data[numeric_cols].describe().T.to_csv(),
                    "stats.csv", "text/csv", key="dl_s")
        with c2:
            if len(numeric_cols) >= 2:
                st.download_button("🔗 Correlations", data[numeric_cols].corr().to_csv(),
                    "corr.csv", "text/csv", key="dl_c")
        with c3:
            st.download_button("📄 Data", data.to_csv(index=False),
                "data.csv", "text/csv", key="dl_d")
        
        if st.button("📝 Generate AI Report", key="ai_rep"):
            with st.spinner("🤖 Crafting..."):
                time.sleep(1)
                insights = generate_ai_insights(data, numeric_cols, categorical_cols)
                report = f"""
STATECON ANALYZER PRO - AI REPORT
{'=' * 70}
Created by: {CREATOR_NAME}
Contact: {CREATOR_PHONE}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET: {data.shape[0]} rows x {data.shape[1]} cols

AI INSIGHTS:
{'-' * 70}
"""
                for i, ins in enumerate(insights, 1):
                    report += f"\n{i}. {ins['icon']} {ins['title']}\n   {ins['message']}\n"
                report += f"\n\nSTATISTICS:\n{'-' * 70}\n"
                report += data[numeric_cols].describe().to_string() if numeric_cols else 'N/A'
                report += f"\n\n{'=' * 70}\nPowered by StatEcon Pro v3.0\nCreated by {CREATOR_NAME}\n"
            
            st.download_button("💾 Download Report", report,
                f"report_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
                "text/plain", key="dl_r")
            st.success("✅ Report ready!")
            st.balloons()

# ============================================================
# FOOTER WITH YOUR BRANDING
# ============================================================
st.markdown(f"""
<div class="footer">
    <p><strong>📊 StatEcon Analyzer Pro v3.0 — LIVE EDITION</strong></p>
    <p>👨‍💻 Designed & Developed by <strong>{CREATOR_NAME}</strong></p>
    <p>📱 {CREATOR_PHONE} | 🎓 {CREATOR_TITLE}</p>
    <p>✨ Made with ❤️ in Kenya 🇰🇪 | © 2025 All Rights Reserved ✨</p>
</div>
""", unsafe_allow_html=True)
