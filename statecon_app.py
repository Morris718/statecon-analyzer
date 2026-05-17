import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import time
import warnings
warnings.filterwarnings('ignore')

CREATOR_NAME = "Morris Mutinda"
CREATOR_PHONE = "+254 705 481 616"
CREATOR_TITLE = "BSc Applied Statistics & Economics"
CREATOR_TAGLINE = "Turning Data into Decisions"

st.set_page_config(
    page_title="StatEcon Pro | by Morris Mutinda",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
html, body, [class*="css"] { font-family: Poppins, sans-serif !important; }
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
}
.brand-subtitle {
    color: rgba(255,255,255,0.95);
    font-size: 1.3rem;
    text-align: center;
    margin-top: 0.8rem;
    font-weight: 300;
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
}
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}
.ai-insight, .ai-positive, .ai-warning {
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    color: #2c3e50;
    font-weight: 500;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    animation: fadeIn 0.6s ease-out;
    transition: all 0.3s ease;
}
.ai-insight {
    background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
    border-left: 5px solid #ff6b6b;
}
.ai-positive {
    background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
    border-left: 5px solid #27ae60;
}
.ai-warning {
    background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
    border-left: 5px solid #e67e22;
}
.ai-insight:hover, .ai-positive:hover, .ai-warning:hover {
    transform: translateY(-5px) scale(1.02);
}
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    padding: 1.2rem;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    border-left: 5px solid #667eea;
    transition: all 0.3s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-5px);
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
    background: rgba(102, 126, 234, 0.1);
    padding: 10px;
    border-radius: 15px;
}
.stTabs [data-baseweb="tab"] {
    background-color: white;
    border-radius: 12px;
    padding: 12px 24px;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
}
.stButton button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 2.5rem;
    font-weight: 600;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
}
.stButton button:hover {
    transform: translateY(-3px) scale(1.02);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
}
[data-testid="stSidebar"] * { color: white !important; }
.feature-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    text-align: center;
    transition: all 0.4s ease;
    height: 100%;
}
.feature-card:hover {
    transform: translateY(-10px) scale(1.03);
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
</style>
""", unsafe_allow_html=True)


def generate_ai_insights(data, numeric_cols, categorical_cols):
    insights = []
    n_rows = data.shape[0]
    if n_rows < 30:
        insights.append({'type': 'warning', 'icon': '⚠️', 'title': 'Small Sample',
            'message': f"Only {n_rows} observations. Consider 30+ for robust analysis."})
    elif n_rows > 1000:
        insights.append({'type': 'positive', 'icon': '✅', 'title': 'Robust Sample',
            'message': f"Excellent! {n_rows} observations is ideal."})
    missing_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
    if missing_pct > 10:
        insights.append({'type': 'warning', 'icon': '🔴', 'title': 'High Missing Data',
            'message': f"{missing_pct:.1f}% missing. Consider imputation."})
    elif missing_pct == 0:
        insights.append({'type': 'positive', 'icon': '✨', 'title': 'Complete Dataset',
            'message': "No missing values - perfect!"})
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
                        'message': f"{numeric_cols[i]} & {numeric_cols[j]} (r = {r:.3f})."})
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
                    'message': f"'{top}' ({top_val:.2f}) vs '{bot}' ({bot_val:.2f}) - {ratio:.1f}x gap."})
    return insights


def display_insights(insights):
    if not insights:
        st.info("No insights generated.")
        return
    for insight in insights:
        css_class = {'positive': 'ai-positive', 'warning': 'ai-warning', 'insight': 'ai-insight'}.get(insight['type'], 'ai-insight')
        st.markdown(f"""<div class="{css_class}">
            <strong style="font-size: 1.1rem;">{insight['icon']} {insight['title']}</strong><br>
            <span style="font-size: 0.95rem;">{insight['message']}</span>
            </div>""", unsafe_allow_html=True)


st.markdown(f"""<div class="brand-header">
    <h1 class="brand-title">📊 StatEcon Analyzer Pro</h1>
    <p class="brand-subtitle">✨ AI-Powered Statistical & Economic Analysis ✨</p>
    <div style="text-align: center;">
        <span class="creator-badge">👨‍💻 Created by {CREATOR_NAME} | {CREATOR_TAGLINE}</span>
    </div>
</div>""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown(f"""<div style="text-align: center; padding: 1rem;">
        <h1 style="font-size: 2rem; margin: 0;">📊</h1>
        <h2 style="margin-top: 0.5rem;">StatEcon Pro</h2>
        <p style="font-size: 0.85rem; opacity: 0.9;">by {CREATOR_NAME}</p>
        </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📁 Upload Your Data")
    upload_method = st.radio("Upload method:", ["📤 Upload File", "📋 Paste Path"], key="up_method")
    data = None
    if upload_method == "📤 Upload File":
        uploaded_file = st.file_uploader("Drop CSV/Excel here", type=['csv', 'xlsx', 'xls'], key="file_up")
        if uploaded_file:
            try:
                with st.spinner("🎨 Loading..."):
                    time.sleep(0.3)
                    if uploaded_file.name.endswith('.csv'):
                        data = pd.read_csv(uploaded_file)
                    else:
                        data = pd.read_excel(uploaded_file)
                st.success(f"✅ {uploaded_file.name}")
                st.balloons()
            except Exception as e:
                st.error(f"❌ {e}")
    else:
        file_path = st.text_input("File path:", key="file_path")
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
        chart_theme = st.selectbox("Theme", ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"], key="theme")
        color_scheme = st.selectbox("Colors", ["Viridis", "Plasma", "Inferno", "Magma", "Rainbow", "Turbo"], key="colors")
    else:
        chart_theme = "plotly"
        color_scheme = "Viridis"
    st.markdown("---")
    st.markdown(f"""<div class="contact-card">
        <h4 style="color: white; margin: 0;">📞 Contact Me</h4>
        <p style="color: white; margin: 0.5rem 0; font-size: 0.9rem;">
        <strong>{CREATOR_NAME}</strong><br>
        {CREATOR_TITLE}<br>
        📱 {CREATOR_PHONE}
        </p></div>""", unsafe_allow_html=True)
    st.markdown("""<div style="text-align: center; padding: 1rem; font-size: 0.75rem; opacity: 0.7;">
        <p>🚀 v3.0 LIVE EDITION</p>
        <p>© 2025 All Rights Reserved</p>
        </div>""", unsafe_allow_html=True)


if data is None:
    st.markdown(f"""<div style="text-align: center; padding: 2rem 0;">
        <h2 style="font-size: 2rem; color: #667eea;">👋 Karibu! Welcome to StatEcon Pro</h2>
        <p style="font-size: 1.1rem; color: #6c757d;">Created with ❤️ by {CREATOR_NAME} from Kenya 🇰🇪</p>
        </div>""", unsafe_allow_html=True)
    st.markdown("### 🎯 Trusted by Data Lovers Worldwide")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class="stat-counter"><p class="stat-number">10+</p><p class="stat-label">Modules</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="stat-counter"><p class="stat-number">∞</p><p class="stat-label">AI Insights</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="stat-counter"><p class="stat-number">100%</p><p class="stat-label">Free</p></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="stat-counter"><p class="stat-number">24/7</p><p class="stat-label">Available</p></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ✨ Features")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="feature-card"><div class="feature-icon">🤖</div>
            <h3 class="feature-title">AI Insights</h3><p>Smart algorithms discover patterns automatically.</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="feature-card"><div class="feature-icon">📊</div>
            <h3 class="feature-title">Pro Analytics</h3><p>From basics to advanced econometrics.</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="feature-card"><div class="feature-icon">🎨</div>
            <h3 class="feature-title">Beautiful Charts</h3><p>Interactive Plotly visualizations.</p></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="feature-card"><div class="feature-icon">📈</div>
            <h3 class="feature-title">Regression</h3><p>OLS with AI interpretation.</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="feature-card"><div class="feature-icon">💰</div>
            <h3 class="feature-title">Smart Finance</h3><p>NPV, IRR with recommendations.</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="feature-card"><div class="feature-icon">🔬</div>
            <h3 class="feature-title">Hypothesis Tests</h3><p>T-tests, ANOVA with conclusions.</p></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px; color: white;">
        <h2 style="color: white; margin: 0;">👈 Upload Your Data Now!</h2>
        <p style="font-size: 1.1rem; margin-top: 0.5rem;">Get instant AI insights</p>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎯 Perfect For")
    st.markdown("""<div style="text-align: center;">
        <span class="achievement-badge">🎓 Students</span>
        <span class="achievement-badge">📊 Analysts</span>
        <span class="achievement-badge">💼 Researchers</span>
        <span class="achievement-badge">💰 Investors</span>
        <span class="achievement-badge">🏢 Consultants</span>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""<div class="contact-card" style="max-width: 600px; margin: 2rem auto;">
        <h2 style="color: white; margin: 0;">👨‍💻 Meet the Creator</h2>
        <h3 style="color: white; margin: 0.5rem 0;">{CREATOR_NAME}</h3>
        <p style="color: white;">{CREATOR_TITLE}<br>📱 {CREATOR_PHONE}</p>
        <p style="color: white; font-style: italic; margin-top: 1rem;">"{CREATOR_TAGLINE}"</p>
        </div>""", unsafe_allow_html=True)

else:
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    if 'first_load' not in st.session_state:
        st.session_state.first_load = True
        st.balloons()
    st.markdown("### 📊 Your Data at a Glance")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="stat-counter"><p class="stat-number">{data.shape[0]:,}</p><p class="stat-label">📋 Rows</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="stat-counter"><p class="stat-number">{data.shape[1]}</p><p class="stat-label">📊 Columns</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="stat-counter"><p class="stat-number">{len(numeric_cols)}</p><p class="stat-label">🔢 Numeric</p></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="stat-counter"><p class="stat-number">{len(categorical_cols)}</p><p class="stat-label">📝 Categorical</p></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🤖 **AI AUTO-INSIGHTS**", expanded=True):
        with st.spinner("🧠 AI is analyzing..."):
            time.sleep(0.3)
            insights = generate_ai_insights(data, numeric_cols, categorical_cols)
        if insights:
            st.markdown(f"### 🎯 Found {len(insights)} insights!")
            display_insights(insights)
    tabs = st.tabs(["📋 Overview", "📊 Stats", "🔗 Correlations", "🎯 Outliers",
        "🏆 Rankings", "📈 Regression", "🔬 Tests", "📉 Charts", "💰 NPV/IRR", "📥 Export"])
    with tabs[0]:
        st.markdown("## 📋 Data Overview")
        info_df = pd.DataFrame({
            'Column': data.columns,
            'Type': [str(t) for t in data.dtypes],
            'Missing': data.isnull().sum().values,
            'Missing %': (data.isnull().sum() / len(data) * 100).round(2).values,
            'Unique': data.nunique().values
        })
        st.dataframe(info_df, use_container_width=True)
        n_rows = st.slider("Rows:", 5, min(100, len(data)), 10, key="prev")
        st.dataframe(data.head(n_rows), use_container_width=True)
    with tabs[1]:
        st.markdown("## 📊 Descriptive Statistics")
        if numeric_cols:
            summary = data[numeric_cols].describe().T
            summary['skewness'] = data[numeric_cols].skew()
            summary['kurtosis'] = data[numeric_cols].kurtosis()
            st.dataframe(summary.round(3), use_container_width=True)
            selected = st.selectbox("Variable:", numeric_cols, key="dv")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean", f"{data[selected].mean():.2f}")
            c2.metric("Median", f"{data[selected].median():.2f}")
            c3.metric("Std", f"{data[selected].std():.2f}")
            c4.metric("Range", f"{data[selected].max() - data[selected].min():.2f}")
            fig = px.histogram(data, x=selected, nbins=30, marginal="box",
                template=chart_theme, color_discrete_sequence=['#667eea'])
            st.plotly_chart(fig, use_container_width=True)
    with tabs[2]:
        st.markdown("## 🔗 Correlation Analysis")
        if len(numeric_cols) >= 2:
            method = st.selectbox("Method:", ["pearson", "spearman", "kendall"], key="cm")
            corr = data[numeric_cols].corr(method=method)
            fig = px.imshow(corr, text_auto='.2f', color_continuous_scale=color_scheme,
                template=chart_theme, zmin=-1, zmax=1)
            st.plotly_chart(fig, use_container_width=True)
            c1, c2 = st.columns(2)
            x = c1.selectbox("X:", numeric_cols, key='cx')
            y = c2.selectbox("Y:", numeric_cols, index=min(1, len(numeric_cols)-1), key='cy')
            fig = px.scatter(data, x=x, y=y, trendline="ols", template=chart_theme)
            st.plotly_chart(fig, use_container_width=True)
    with tabs[3]:
        st.markdown("## 🎯 Outlier Detection")
        if numeric_cols:
            out_list = []
            for col in numeric_cols:
                s = data[col].dropna()
                Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
                IQR = Q3 - Q1
                low, up = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                mask = (s < low) | (s > up)
                out_list.append({'Variable': col, 'Outliers': int(mask.sum()),
                    '%': round(mask.sum()/len(s)*100, 2),
                    'Lower': round(low, 2), 'Upper': round(up, 2)})
            st.dataframe(pd.DataFrame(out_list), use_container_width=True)
            sel = st.multiselect("Variables:", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))], key="os")
            if sel:
                fig = go.Figure()
                for col in sel:
                    fig.add_trace(go.Box(y=data[col], name=col, boxpoints='outliers'))
                fig.update_layout(template=chart_theme)
                st.plotly_chart(fig, use_container_width=True)
    with tabs[4]:
        st.markdown("## 🏆 Rankings")
        if categorical_cols and numeric_cols:
            c1, c2, c3 = st.columns(3)
            cat = c1.selectbox("Category:", categorical_cols, key="rc")
            met = c2.selectbox("Metric:", numeric_cols, key="rm")
            n = c3.number_input("Top N:", 3, 20, 5, key="rn")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"### 🥇 Top {n}")
                st.dataframe(data.nlargest(n, met)[[cat, met]].reset_index(drop=True), use_container_width=True)
            with c2:
                st.markdown(f"### 📉 Bottom {n}")
                st.dataframe(data.nsmallest(n, met)[[cat, met]].reset_index(drop=True), use_container_width=True)
            sd = data.sort_values(met)
            fig = px.bar(sd, x=met, y=cat, orientation='h', template=chart_theme,
                color=met, color_continuous_scale=color_scheme)
            fig.update_layout(height=max(400, len(data) * 30))
            st.plotly_chart(fig, use_container_width=True)
    with tabs[5]:
        st.markdown("## 📈 Regression Analysis")
        if len(numeric_cols) >= 2:
            c1, c2 = st.columns(2)
            y_col = c1.selectbox("Y:", numeric_cols, key="ry")
            x_opts = [c for c in numeric_cols if c != y_col]
            x_cols = c2.multiselect("X:", x_opts, default=x_opts[:min(2, len(x_opts))], key="rx")
            if x_cols and st.button("🚀 Run Regression", type="primary", key="rr"):
                with st.spinner("🧮 Running..."):
                    time.sleep(0.3)
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
                        't': model.tvalues, 'p': model.pvalues,
                        'Sig': model.pvalues < 0.05})
                    st.dataframe(coef_df.round(4), use_container_width=True)
                    if model.rsquared > 0.7:
                        st.markdown(f"""<div class="ai-positive"><strong>🎯 Excellent!</strong> R² = {model.rsquared:.2%}</div>""", unsafe_allow_html=True)
                    elif model.rsquared > 0.5:
                        st.markdown(f"""<div class="ai-insight"><strong>📊 Moderate.</strong> R² = {model.rsquared:.2%}</div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div class="ai-warning"><strong>⚠️ Weak.</strong> R² = {model.rsquared:.2%}</div>""", unsafe_allow_html=True)
    with tabs[6]:
        st.markdown("## 🔬 Hypothesis Testing")
        test = st.selectbox("Test:", ["One-Sample t-Test", "Independent t-Test", "ANOVA"], key="ht")
        if test == "One-Sample t-Test" and numeric_cols:
            col = st.selectbox("Variable:", numeric_cols, key="osv")
            mu = st.number_input("Mean:", value=0.0, key="osm")
            if st.button("Run", key="ros"):
                stat, p = stats.ttest_1samp(data[col].dropna(), mu)
                c1, c2 = st.columns(2)
                c1.metric("t-stat", f"{stat:.4f}")
                c2.metric("p-value", f"{p:.6f}")
                if p < 0.05:
                    st.success("✅ Reject H₀")
                    st.balloons()
                else:
                    st.info("ℹ️ Fail to reject H₀")
        elif test == "Independent t-Test" and len(numeric_cols) >= 2:
            c1, c2 = st.columns(2)
            v1 = c1.selectbox("Var 1:", numeric_cols, key='iv1')
            v2 = c2.selectbox("Var 2:", numeric_cols, index=1, key='iv2')
            if st.button("Run", key="rit"):
                stat, p = stats.ttest_ind(data[v1].dropna(), data[v2].dropna())
                c1, c2 = st.columns(2)
                c1.metric("t-stat", f"{stat:.4f}")
                c2.metric("p-value", f"{p:.6f}")
                if p < 0.05:
                    st.success("✅ Reject H₀")
                else:
                    st.info("ℹ️ Fail to reject H₀")
        elif test == "ANOVA" and categorical_cols and numeric_cols:
            cat = st.selectbox("Group:", categorical_cols, key="ac")
            val = st.selectbox("Value:", numeric_cols, key="av")
            if st.button("Run", key="ran"):
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
        viz = st.selectbox("Chart:", ["Histogram", "Bar", "Scatter", "Pie", "3D Scatter"], key="vt")
        if viz == "Histogram" and numeric_cols:
            col = st.selectbox("Var:", numeric_cols, key="vhv")
            fig = px.histogram(data, x=col, nbins=30, template=chart_theme, color_discrete_sequence=['#667eea'])
            st.plotly_chart(fig, use_container_width=True)
        elif viz == "Bar" and categorical_cols and numeric_cols:
            c1, c2 = st.columns(2)
            x = c1.selectbox("Cat:", categorical_cols, key="vbc")
            y = c2.selectbox("Val:", numeric_cols, key="vbv")
            g = data.groupby(x)[y].mean().reset_index().sort_values(y, ascending=False)
            fig = px.bar(g, x=x, y=y, template=chart_theme, color=y, color_continuous_scale=color_scheme)
            st.plotly_chart(fig, use_container_width=True)
        elif viz == "Scatter" and len(numeric_cols) >= 2:
            c1, c2 = st.columns(2)
            x = c1.selectbox("X:", numeric_cols, key='vsx')
            y = c2.selectbox("Y:", numeric_cols, index=1, key='vsy')
            fig = px.scatter(data, x=x, y=y, template=chart_theme, trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
        elif viz == "Pie" and categorical_cols:
            cat = st.selectbox("Cat:", categorical_cols, key="vpc")
            counts = data[cat].value_counts().head(10)
            fig = px.pie(values=counts.values, names=counts.index, hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        elif viz == "3D Scatter" and len(numeric_cols) >= 3:
            c1, c2, c3 = st.columns(3)
            x = c1.selectbox("X:", numeric_cols, key='v3x')
            y = c2.selectbox("Y:", numeric_cols, index=1, key='v3y')
            z = c3.selectbox("Z:", numeric_cols, index=2, key='v3z')
            fig = px.scatter_3d(data, x=x, y=y, z=z, color=z, color_continuous_scale=color_scheme)
            st.plotly_chart(fig, use_container_width=True)
    with tabs[8]:
        st.markdown("## 💰 Cost-Benefit Analysis")
        c1, c2 = st.columns(2)
        cf = c1.text_area("Cash flows:", value="-100000, 25000, 30000, 35000, 40000, 45000", key="cf")
        rate = c2.slider("Rate (%):", 0.0, 30.0, 10.0, key="cr") / 100
        if st.button("💡 Calculate", type="primary", key="cb"):
            with st.spinner("💰 Calculating..."):
                time.sleep(0.3)
                cfs = [float(x.strip()) for x in cf.split(',')]
                npv = sum(c / (1 + rate)**t for t, c in enumerate(cfs))
                r = 0.1
                for _ in range(1000):
                    n = sum(c / (1 + r)**t for t, c in enumerate(cfs))
                    d = sum(-t * c / (1 + r)**(t+1) for t, c in enumerate(cfs))
                    if abs(d) < 1e-12:
                        break
                    r -= n / d
                    if abs(n) < 1e-6:
                        break
                c1, c2, c3 = st.columns(3)
                c1.metric("💵 NPV", f"${npv:,.2f}")
                c2.metric("📈 IRR", f"{r*100:.2f}%")
                c3.metric("Decision", "✅ ACCEPT" if npv > 0 else "❌ REJECT")
                if npv > 0:
                    st.markdown(f"""<div class="ai-positive"><strong>🤖 Accept!</strong> NPV: ${npv:,.2f}</div>""", unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown(f"""<div class="ai-warning"><strong>🤖 Reject.</strong> NPV: ${npv:,.2f}</div>""", unsafe_allow_html=True)
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
                st.download_button("📊 Stats", data[numeric_cols].describe().T.to_csv(), "stats.csv", "text/csv", key="ds")
        with c2:
            if len(numeric_cols) >= 2:
                st.download_button("🔗 Correlations", data[numeric_cols].corr().to_csv(), "corr.csv", "text/csv", key="dc")
        with c3:
            st.download_button("📄 Data", data.to_csv(index=False), "data.csv", "text/csv", key="dd")
        if st.button("📝 Generate AI Report", key="ar"):
            with st.spinner("🤖 Crafting..."):
                time.sleep(0.5)
                insights = generate_ai_insights(data, numeric_cols, categorical_cols)
                report = f"""STATECON ANALYZER PRO - AI REPORT
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
                f"report_{pd.Timestamp.now().strftime('%Y%m%d')}.txt", "text/plain", key="dr")
            st.success("✅ Report ready!")
            st.balloons()

st.markdown(f"""<div class="footer">
    <p><strong>📊 StatEcon Analyzer Pro v3.0 — LIVE EDITION</strong></p>
    <p>👨‍💻 Designed & Developed by <strong>{CREATOR_NAME}</strong></p>
    <p>📱 {CREATOR_PHONE} | 🎓 {CREATOR_TITLE}</p>
    <p>✨ Made with ❤️ in Kenya 🇰🇪 | © 2025 All Rights Reserved ✨</p>
    </div>""", unsafe_allow_html=True)