"""
StatEcon Analyzer Pro
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
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIGURATION & BRANDING
# ============================================================
st.set_page_config(
    page_title="StatEcon Analyzer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/statecon',
        'Report a bug': 'https://github.com/yourusername/statecon/issues',
        'About': "# StatEcon Analyzer Pro\nBuilt by [Your Name] | BSc Applied Statistics & Economics"
    }
)

# ============================================================
# CUSTOM CSS — Professional Branding
# ============================================================
st.markdown("""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom header */
    .brand-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .brand-title {
        color: white;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .brand-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        text-align: center;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* AI Insight Cards */
    .ai-insight {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #ff6b6b;
        color: #2c3e50;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .ai-positive {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #27ae60;
        color: #2c3e50;
        font-weight: 500;
    }
    
    .ai-warning {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #e67e22;
        color: #2c3e50;
        font-weight: 500;
    }
    
    /* Metric cards */
    [data-testid="stMetric"] {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f8f9fa;
        padding: 8px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px;
        padding: 10px 18px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Success/Info boxes */
    .stSuccess {
        border-radius: 10px;
        border-left: 5px solid #27ae60;
    }
    
    .stInfo {
        border-radius: 10px;
        border-left: 5px solid #3498db;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #7f8c8d;
        font-size: 0.9rem;
        border-top: 1px solid #ecf0f1;
        margin-top: 3rem;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# BRANDED HEADER
# ============================================================
st.markdown("""
    <div class="brand-header">
        <h1 class="brand-title">📊 StatEcon Analyzer Pro</h1>
        <p class="brand-subtitle">AI-Powered Statistical & Economic Analysis Platform</p>
    </div>
""", unsafe_allow_html=True)

# ============================================================
# AI INSIGHTS ENGINE
# ============================================================

def generate_ai_insights(data, numeric_cols, categorical_cols):
    """Generate intelligent insights about the dataset."""
    insights = []
    
    # 1. Dataset size insight
    n_rows = data.shape[0]
    if n_rows < 30:
        insights.append({
            'type': 'warning',
            'icon': '⚠️',
            'title': 'Small Sample Size',
            'message': f"Your dataset has only {n_rows} observations. Statistical tests may have limited power. Consider collecting more data for robust conclusions (minimum 30 recommended)."
        })
    elif n_rows > 1000:
        insights.append({
            'type': 'positive',
            'icon': '✅',
            'title': 'Robust Sample Size',
            'message': f"Excellent! With {n_rows} observations, your dataset is large enough for reliable statistical analysis and machine learning applications."
        })
    
    # 2. Missing data insight
    missing_pct = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
    if missing_pct > 10:
        insights.append({
            'type': 'warning',
            'icon': '🔴',
            'title': 'High Missing Data',
            'message': f"{missing_pct:.1f}% of your data is missing. Consider imputation techniques (mean, median, or KNN) or investigate the cause of missingness."
        })
    elif missing_pct == 0:
        insights.append({
            'type': 'positive',
            'icon': '✨',
            'title': 'Complete Dataset',
            'message': "Your dataset has no missing values — perfect for analysis!"
        })
    
    # 3. Variability insights
    if numeric_cols:
        for col in numeric_cols:
            series = data[col].dropna()
            if len(series) > 0 and series.mean() != 0:
                cv = abs(series.std() / series.mean() * 100)
                if cv > 50:
                    insights.append({
                        'type': 'insight',
                        'icon': '📊',
                        'title': f'High Variability in {col}',
                        'message': f"{col} shows extreme variability (CV = {cv:.1f}%). This suggests significant disparities across observations — investigate the drivers of this variation."
                    })
    
    # 4. Distribution insights (skewness)
    for col in numeric_cols:
        series = data[col].dropna()
        if len(series) >= 8:
            skew = series.skew()
            if abs(skew) > 1:
                direction = "right" if skew > 0 else "left"
                insights.append({
                    'type': 'insight',
                    'icon': '📈',
                    'title': f'{col} is Highly Skewed',
                    'message': f"Distribution is heavily {direction}-skewed (skew = {skew:.2f}). Consider log/sqrt transformation before regression analysis."
                })
    
    # 5. Correlation insights
    if len(numeric_cols) >= 2:
        corr = data[numeric_cols].corr()
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                r = corr.iloc[i, j]
                if abs(r) > 0.9:
                    insights.append({
                        'type': 'warning',
                        'icon': '⚡',
                        'title': 'Multicollinearity Alert',
                        'message': f"{numeric_cols[i]} and {numeric_cols[j]} are almost perfectly correlated (r = {r:.3f}). They may be measuring the same thing. Avoid using both in regression."
                    })
                elif abs(r) > 0.7:
                    relationship = "positive" if r > 0 else "negative"
                    insights.append({
                        'type': 'insight',
                        'icon': '🔗',
                        'title': 'Strong Relationship Detected',
                        'message': f"Strong {relationship} correlation between {numeric_cols[i]} and {numeric_cols[j]} (r = {r:.3f}). This is a key relationship worth investigating."
                    })
    
    # 6. Outlier insights
    for col in numeric_cols:
        series = data[col].dropna()
        if len(series) > 0:
            Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((series < Q1 - 1.5*IQR) | (series > Q3 + 1.5*IQR)).sum()
            if outliers > 0:
                pct = outliers / len(series) * 100
                if pct > 10:
                    insights.append({
                        'type': 'warning',
                        'icon': '🎯',
                        'title': f'Many Outliers in {col}',
                        'message': f"{outliers} outliers detected ({pct:.1f}%). Investigate whether these are data errors or genuinely extreme cases worth studying separately."
                    })
    
    # 7. Top performer insight
    if categorical_cols and numeric_cols:
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]
        if data[cat_col].nunique() < 50:
            top_performer = data.loc[data[num_col].idxmax(), cat_col]
            bottom_performer = data.loc[data[num_col].idxmin(), cat_col]
            top_val = data[num_col].max()
            bottom_val = data[num_col].min()
            
            if bottom_val != 0:
                ratio = top_val / bottom_val
                insights.append({
                    'type': 'insight',
                    'icon': '🏆',
                    'title': 'Performance Gap Detected',
                    'message': f"'{top_performer}' leads in {num_col} ({top_val:.2f}), while '{bottom_performer}' lags ({bottom_val:.2f}) — a {ratio:.1f}x gap. Investigate what drives this disparity."
                })
    
    return insights


def interpret_regression(model, x_cols):
    """Generate AI interpretation of regression results."""
    interpretations = []
    
    # Model fit
    r2 = model.rsquared
    if r2 > 0.8:
        interpretations.append({
            'type': 'positive',
            'icon': '🎯',
            'message': f"**Excellent model fit!** R² = {r2:.2%} means your predictors explain {r2:.1%} of the variation. This is a very strong model."
        })
    elif r2 > 0.5:
        interpretations.append({
            'type': 'insight',
            'icon': '📊',
            'message': f"**Moderate model fit.** R² = {r2:.2%}. Your model captures meaningful patterns but there's room for improvement. Consider adding more predictors."
        })
    elif r2 > 0.3:
        interpretations.append({
            'type': 'warning',
            'icon': '⚠️',
            'message': f"**Weak model fit.** R² = {r2:.2%}. Your predictors explain only a small portion of variance. The relationship may be non-linear or you're missing key variables."
        })
    else:
        interpretations.append({
            'type': 'warning',
            'icon': '🔴',
            'message': f"**Very weak model.** R² = {r2:.2%}. This model has little predictive power. Reconsider your variable selection or try non-linear models."
        })
    
    # F-test
    if model.f_pvalue < 0.05:
        interpretations.append({
            'type': 'positive',
            'icon': '✅',
            'message': f"**Model is statistically significant** (F p-value = {model.f_pvalue:.4f}). The predictors collectively have a real effect on the outcome."
        })
    else:
        interpretations.append({
            'type': 'warning',
            'icon': '❌',
            'message': f"**Model is NOT statistically significant** (F p-value = {model.f_pvalue:.4f}). The relationship may be due to chance."
        })
    
    # Individual coefficients
    sig_vars = []
    insig_vars = []
    for var in x_cols:
        if var in model.pvalues.index:
            if model.pvalues[var] < 0.05:
                coef = model.params[var]
                direction = "increases" if coef > 0 else "decreases"
                sig_vars.append(f"**{var}** ({direction} outcome by {abs(coef):.3f} per unit, p={model.pvalues[var]:.4f})")
            else:
                insig_vars.append(var)
    
    if sig_vars:
        interpretations.append({
            'type': 'positive',
            'icon': '⭐',
            'message': f"**Significant predictors:**\n" + "\n".join([f"- {v}" for v in sig_vars])
        })
    
    if insig_vars:
        interpretations.append({
            'type': 'insight',
            'icon': '💡',
            'message': f"**Not statistically significant:** {', '.join(insig_vars)}. Consider removing them or collecting more data."
        })
    
    return interpretations


def display_insights(insights):
    """Display AI insights with appropriate styling."""
    if not insights:
        st.info("No specific insights generated. Your data looks standard.")
        return
    
    for insight in insights:
        css_class = {
            'positive': 'ai-positive',
            'warning': 'ai-warning',
            'insight': 'ai-insight'
        }.get(insight['type'], 'ai-insight')
        
        st.markdown(f"""
            <div class="{css_class}">
                <strong>{insight['icon']} {insight['title']}</strong><br>
                {insight['message']}
            </div>
        """, unsafe_allow_html=True)


# ============================================================
# SIDEBAR - FILE UPLOAD
# ============================================================
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2 style="color: #667eea;">📊 StatEcon Pro</h2>
            <p style="color: #7f8c8d; font-size: 0.85rem;">Your AI Analytics Companion</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📁 Upload Your Data")
    
    upload_method = st.radio(
        "Upload method:",
        ["📤 Upload File", "📋 Paste Path"],
        key="upload_method_radio"
    )
    
    data = None
    
    if upload_method == "📤 Upload File":
        uploaded_file = st.file_uploader(
            "Drop CSV/Excel here",
            type=['csv', 'xlsx', 'xls'],
            key="file_uploader_main"
        )
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                st.success(f"✅ {uploaded_file.name}")
            except Exception as e:
                st.error(f"❌ {e}")
    else:
        file_path = st.text_input(
            "File path:",
            placeholder="C:\\path\\to\\file.csv",
            key="file_path_input"
        )
        if file_path:
            try:
                file_path = file_path.strip('"').strip("'")
                if file_path.endswith('.csv'):
                    data = pd.read_csv(file_path)
                else:
                    data = pd.read_excel(file_path)
                st.success("✅ Loaded!")
            except Exception as e:
                st.error(f"❌ {e}")
    
    st.markdown("---")
    
    if data is not None:
        st.markdown("### 📊 Dataset Info")
        col1, col2 = st.columns(2)
        col1.metric("Rows", data.shape[0])
        col2.metric("Cols", data.shape[1])
        
        st.markdown("---")
        st.markdown("### 🎨 Customize")
        chart_theme = st.selectbox(
            "Theme",
            ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"],
            key="chart_theme_select"
        )
    else:
        chart_theme = "plotly"
    
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; font-size: 0.8rem; color: #7f8c8d;">
            <p>🚀 <strong>StatEcon Pro v2.0</strong></p>
            <p>Built with ❤️ for Data Analysts</p>
            <p>© 2025 All Rights Reserved</p>
        </div>
    """, unsafe_allow_html=True)


# ============================================================
# MAIN CONTENT
# ============================================================
if data is None:
    # Landing page
    st.markdown("### 👋 Welcome to StatEcon Analyzer Pro")
    st.info("📂 **Upload your CSV or Excel file from the sidebar to begin your analysis journey!**")
    
    # Feature showcase
    st.markdown("## ✨ What Makes Us Different")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        ### 🤖 AI-Powered Insights
        Auto-generated explanations of patterns, anomalies, and relationships in your data.
        """)
    with c2:
        st.markdown("""
        ### 📊 Professional Analytics
        From basic descriptives to advanced econometrics — all in one platform.
        """)
    with c3:
        st.markdown("""
        ### 🎨 Beautiful Visualizations
        Interactive charts powered by Plotly — publication ready.
        """)
    
    st.markdown("---")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("#### 📋\n**10 Analysis Tabs**")
    c2.markdown("#### 📈\n**OLS Regression**")
    c3.markdown("#### 🔬\n**Hypothesis Tests**")
    c4.markdown("#### 💰\n**NPV / IRR**")
    
    st.markdown("---")
    st.markdown("### 🎯 Perfect For")
    st.markdown("""
    - 🎓 **Students** — Statistics, Economics, Finance projects
    - 📊 **Analysts** — Quick data exploration and reporting
    - 💼 **Researchers** — Hypothesis testing and modeling
    - 💰 **Investors** — Cost-benefit and risk analysis
    """)

else:
    # ============================================================
    # AI INSIGHTS BANNER (always shown when data loaded)
    # ============================================================
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    
    with st.expander("🤖 **AI AUTO-INSIGHTS** — Click to see what I discovered about your data", expanded=True):
        insights = generate_ai_insights(data, numeric_cols, categorical_cols)
        if insights:
            st.markdown(f"### Found {len(insights)} key insights in your data:")
            display_insights(insights)
        else:
            st.info("Upload data to receive AI insights!")
    
    # ============================================================
    # ANALYSIS TABS
    # ============================================================
    tabs = st.tabs([
        "📋 Overview",
        "📊 Descriptive",
        "🔗 Correlations",
        "🎯 Outliers",
        "🏆 Rankings",
        "📈 Regression",
        "🔬 Tests",
        "📉 Charts",
        "💰 NPV/IRR",
        "📥 Export"
    ])
    
    # ========== TAB 1: OVERVIEW ==========
    with tabs[0]:
        st.header("📋 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", data.shape[0])
        col2.metric("Total Columns", data.shape[1])
        col3.metric("Numeric Cols", len(numeric_cols))
        col4.metric("Categorical Cols", len(categorical_cols))
        
        st.markdown("### 📌 Column Information")
        info_df = pd.DataFrame({
            'Column': data.columns,
            'Type': [str(t) for t in data.dtypes],
            'Missing': data.isnull().sum().values,
            'Missing %': (data.isnull().sum() / len(data) * 100).round(2).values,
            'Unique': data.nunique().values
        })
        st.dataframe(info_df, use_container_width=True)
        
        st.markdown("### 👀 Data Preview")
        n_rows = st.slider("Rows to show:", 5, min(100, len(data)), 10, key="preview_slider")
        st.dataframe(data.head(n_rows), use_container_width=True)
    
    # ========== TAB 2: DESCRIPTIVE STATS ==========
    with tabs[1]:
        st.header("📊 Descriptive Statistics")
        
        if not numeric_cols:
            st.warning("No numeric columns found.")
        else:
            summary = data[numeric_cols].describe().T
            summary['variance'] = data[numeric_cols].var()
            summary['skewness'] = data[numeric_cols].skew()
            summary['kurtosis'] = data[numeric_cols].kurtosis()
            summary['cv_%'] = (data[numeric_cols].std() / data[numeric_cols].mean() * 100).abs()
            st.dataframe(summary.round(3), use_container_width=True)
            
            st.markdown("### 🎯 Explore a Variable")
            selected_col = st.selectbox("Variable:", numeric_cols, key="desc_variable_select")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean", f"{data[selected_col].mean():.2f}")
            c2.metric("Median", f"{data[selected_col].median():.2f}")
            c3.metric("Std Dev", f"{data[selected_col].std():.2f}")
            c4.metric("Range", f"{data[selected_col].max() - data[selected_col].min():.2f}")
            
            # AI insight for this variable
            series = data[selected_col].dropna()
            skew = series.skew()
            if abs(skew) > 1:
                direction = "right" if skew > 0 else "left"
                st.markdown(f"""
                    <div class="ai-insight">
                        <strong>🤖 AI Insight:</strong> {selected_col} is {direction}-skewed (skew={skew:.2f}). 
                        The mean ({series.mean():.2f}) and median ({series.median():.2f}) differ significantly — 
                        the median may be a better measure of central tendency here.
                    </div>
                """, unsafe_allow_html=True)
            
            fig = px.histogram(data, x=selected_col, nbins=20, 
                               title=f'Distribution of {selected_col}',
                               template=chart_theme, marginal="box",
                               color_discrete_sequence=['#667eea'])
            st.plotly_chart(fig, use_container_width=True)
    
    # ========== TAB 3: CORRELATIONS ==========
    with tabs[2]:
        st.header("🔗 Correlation Analysis")
        
        if len(numeric_cols) < 2:
            st.warning("Need ≥2 numeric columns.")
        else:
            method = st.selectbox("Method:", ["pearson", "spearman", "kendall"], key="corr_method_select")
            corr = data[numeric_cols].corr(method=method)
            
            fig = px.imshow(corr, text_auto='.2f', aspect="auto",
                           color_continuous_scale='RdBu_r',
                           title=f'{method.capitalize()} Correlation Matrix',
                           template=chart_theme, zmin=-1, zmax=1)
            st.plotly_chart(fig, use_container_width=True)
            
            # AI Insight for correlations
            strong = []
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    r = corr.iloc[i, j]
                    if abs(r) > 0.5:
                        strong.append({
                            'Var1': numeric_cols[i],
                            'Var2': numeric_cols[j],
                            'r': round(r, 4),
                            'Strength': 'Very Strong' if abs(r) > 0.8 else 'Strong' if abs(r) > 0.7 else 'Moderate'
                        })
            
            if strong:
                st.markdown("### 🔥 Strong Correlations Found")
                st.dataframe(pd.DataFrame(strong), use_container_width=True)
                
                # AI interpretation
                top_corr = max(strong, key=lambda x: abs(x['r']))
                st.markdown(f"""
                    <div class="ai-insight">
                        <strong>🤖 AI Interpretation:</strong> The strongest relationship is between 
                        <b>{top_corr['Var1']}</b> and <b>{top_corr['Var2']}</b> (r = {top_corr['r']}). 
                        {'This very high correlation suggests potential multicollinearity — avoid using both in regression.' if abs(top_corr['r']) > 0.9 else 'This suggests these variables move together meaningfully.'}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No strong correlations (|r| > 0.5) found.")
            
            st.markdown("### 📍 Scatter Plot Explorer")
            c1, c2 = st.columns(2)
            x_var = c1.selectbox("X:", numeric_cols, key='corr_scatter_x')
            y_var = c2.selectbox("Y:", numeric_cols, index=min(1, len(numeric_cols)-1), key='corr_scatter_y')
            
            color_var = None
            if categorical_cols:
                color_choice = st.selectbox("Color by:", ["None"] + categorical_cols, key='corr_color_select')
                if color_choice != "None":
                    color_var = color_choice
            
            fig = px.scatter(data, x=x_var, y=y_var, color=color_var,
                           trendline="ols", template=chart_theme,
                           title=f'{y_var} vs {x_var}')
            st.plotly_chart(fig, use_container_width=True)
    
    # ========== TAB 4: OUTLIERS ==========
    with tabs[3]:
        st.header("🎯 Outlier Detection")
        
        if not numeric_cols:
            st.warning("No numeric columns.")
        else:
            method = st.radio("Method:", ["IQR (1.5x)", "Z-Score (>3)"], key="outlier_method_radio")
            
            outlier_summary = []
            for col in numeric_cols:
                series = data[col].dropna()
                if method == "IQR (1.5x)":
                    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                    mask = (series < lower) | (series > upper)
                else:
                    z_scores = np.abs(stats.zscore(series))
                    mask = z_scores > 3
                    lower = series.mean() - 3*series.std()
                    upper = series.mean() + 3*series.std()
                
                outlier_summary.append({
                    'Variable': col,
                    'Outliers': int(mask.sum()),
                    'Outlier %': round(mask.sum() / len(series) * 100, 2),
                    'Lower': round(lower, 2),
                    'Upper': round(upper, 2)
                })
            
            df_outliers = pd.DataFrame(outlier_summary)
            st.dataframe(df_outliers, use_container_width=True)
            
            # AI Insight
            total_outliers = df_outliers['Outliers'].sum()
            if total_outliers > 0:
                worst_col = df_outliers.loc[df_outliers['Outliers'].idxmax(), 'Variable']
                st.markdown(f"""
                    <div class="ai-warning">
                        <strong>🤖 AI Alert:</strong> Found <b>{total_outliers}</b> outliers across your data. 
                        <b>{worst_col}</b> has the most extreme values. Consider: (1) Are these data entry errors? 
                        (2) Are they genuine extreme cases worth studying? (3) Should you use robust statistical methods?
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### 📦 Box Plot Visualization")
            selected = st.multiselect("Variables:", numeric_cols, 
                                     default=numeric_cols[:min(5, len(numeric_cols))],
                                     key="outlier_boxplot_select")
            if selected:
                fig = go.Figure()
                for col in selected:
                    fig.add_trace(go.Box(y=data[col], name=col, boxpoints='outliers',
                                        marker_color='#667eea'))
                fig.update_layout(template=chart_theme, title="Box Plots with Outliers")
                st.plotly_chart(fig, use_container_width=True)
    
    # ========== TAB 5: RANKINGS ==========
    with tabs[4]:
        st.header("🏆 Top & Bottom Performers")
        
        if not categorical_cols or not numeric_cols:
            st.warning("Need both categorical and numeric columns.")
        else:
            c1, c2, c3 = st.columns(3)
            cat = c1.selectbox("Category:", categorical_cols, key="rank_category_select")
            metric = c2.selectbox("Metric:", numeric_cols, key="rank_metric_select")
            n = c3.number_input("Top/Bottom N:", 3, 20, 5, key="rank_n_input")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"### 🥇 Top {n}")
                top_n = data.nlargest(n, metric)[[cat, metric]].reset_index(drop=True)
                top_n.index = top_n.index + 1
                st.dataframe(top_n, use_container_width=True)
            with c2:
                st.markdown(f"### 📉 Bottom {n}")
                bottom_n = data.nsmallest(n, metric)[[cat, metric]].reset_index(drop=True)
                bottom_n.index = bottom_n.index + 1
                st.dataframe(bottom_n, use_container_width=True)
            
            # AI insight
            top_val = data[metric].max()
            bottom_val = data[metric].min()
            top_name = data.loc[data[metric].idxmax(), cat]
            bottom_name = data.loc[data[metric].idxmin(), cat]
            
            if bottom_val != 0:
                ratio = top_val / bottom_val
                st.markdown(f"""
                    <div class="ai-insight">
                        <strong>🤖 AI Insight:</strong> <b>{top_name}</b> outperforms <b>{bottom_name}</b> by 
                        <b>{ratio:.1f}x</b> in {metric}. This significant gap warrants investigation — 
                        what factors make {top_name} successful that could be replicated?
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"### 📊 Full Ranking")
            sorted_data = data.sort_values(metric, ascending=True)
            fig = px.bar(sorted_data, x=metric, y=cat, orientation='h',
                        template=chart_theme, color=metric,
                        color_continuous_scale='Viridis')
            fig.update_layout(height=max(400, len(data) * 30))
            st.plotly_chart(fig, use_container_width=True)
    
    # ========== TAB 6: REGRESSION ==========
    with tabs[5]:
        st.header("📈 Regression Analysis")
        
        if len(numeric_cols) < 2:
            st.warning("Need ≥2 numeric columns.")
        else:
            c1, c2 = st.columns(2)
            y_col = c1.selectbox("Y (Dependent):", numeric_cols, key="reg_y_select")
            x_options = [c for c in numeric_cols if c != y_col]
            x_cols = c2.multiselect("X (Independent):", x_options, 
                                   default=x_options[:min(2, len(x_options))],
                                   key="reg_x_multiselect")
            
            if x_cols and st.button("🚀 Run Regression", type="primary", key="run_regression_btn"):
                try:
                    y = data[y_col].dropna()
                    X = data[x_cols].loc[y.index]
                    mask = X.notna().all(axis=1)
                    y, X = y[mask], X[mask]
                    X = sm.add_constant(X)
                    model = OLS(y, X).fit()
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("R²", f"{model.rsquared:.4f}")
                    c2.metric("Adj R²", f"{model.rsquared_adj:.4f}")
                    c3.metric("F p-value", f"{model.f_pvalue:.6f}")
                    c4.metric("N", int(model.nobs))
                    
                    st.markdown("### 📊 Coefficients")
                    coef_df = pd.DataFrame({
                        'Coefficient': model.params,
                        'Std Error': model.bse,
                        't-stat': model.tvalues,
                        'p-value': model.pvalues,
                        'Significant (5%)': model.pvalues < 0.05
                    })
                    st.dataframe(coef_df.round(4), use_container_width=True)
                    
                    # AI Interpretation
                    st.markdown("### 🤖 AI Interpretation")
                    interpretations = interpret_regression(model, x_cols)
                    for interp in interpretations:
                        css_class = {'positive': 'ai-positive', 'warning': 'ai-warning', 'insight': 'ai-insight'}.get(interp['type'], 'ai-insight')
                        st.markdown(f"""
                            <div class="{css_class}">
                                <strong>{interp['icon']}</strong> {interp['message']}
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("### 📉 Residuals Plot")
                    fig = px.scatter(x=model.fittedvalues, y=model.resid,
                                    labels={'x': 'Fitted', 'y': 'Residuals'},
                                    template=chart_theme)
                    fig.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # ========== TAB 7: HYPOTHESIS TESTS ==========
    with tabs[6]:
        st.header("🔬 Hypothesis Testing")
        
        test = st.selectbox("Test type:", 
                           ["One-Sample t-Test", "Independent t-Test", "ANOVA"],
                           key="hyp_test_select")
        
        if test == "One-Sample t-Test" and numeric_cols:
            col = st.selectbox("Variable:", numeric_cols, key="onesample_variable_select")
            mu = st.number_input("Hypothesized mean:", value=0.0, key="onesample_mu_input")
            if st.button("Run Test", key="run_onesample_btn"):
                stat, p = stats.ttest_1samp(data[col].dropna(), mu)
                c1, c2, c3 = st.columns(3)
                c1.metric("t-stat", f"{stat:.4f}")
                c2.metric("p-value", f"{p:.6f}")
                c3.metric("Sample Mean", f"{data[col].mean():.4f}")
                if p < 0.05:
                    st.markdown(f"""
                        <div class="ai-positive">
                            <strong>🤖 AI Conclusion:</strong> ✅ Reject H₀. The mean of {col} ({data[col].mean():.2f}) 
                            is <b>significantly different</b> from {mu} (p={p:.4f}).
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="ai-insight">
                            <strong>🤖 AI Conclusion:</strong> Cannot reject H₀. No statistically significant 
                            difference between mean ({data[col].mean():.2f}) and {mu}.
                        </div>
                    """, unsafe_allow_html=True)
        
        elif test == "Independent t-Test" and len(numeric_cols) >= 2:
            c1, c2 = st.columns(2)
            var1 = c1.selectbox("Var 1:", numeric_cols, key='indep_v1_select')
            var2 = c2.selectbox("Var 2:", numeric_cols, index=1, key='indep_v2_select')
            if st.button("Run Test", key="run_indep_btn"):
                stat, p = stats.ttest_ind(data[var1].dropna(), data[var2].dropna())
                c1, c2 = st.columns(2)
                c1.metric("t-stat", f"{stat:.4f}")
                c2.metric("p-value", f"{p:.6f}")
                if p < 0.05:
                    st.success(f"✅ Means are significantly different (p={p:.4f})")
                else:
                    st.info(f"ℹ️ No significant difference (p={p:.4f})")
        
        elif test == "ANOVA" and categorical_cols and numeric_cols:
            cat = st.selectbox("Group:", categorical_cols, key="anova_cat_select")
            val = st.selectbox("Value:", numeric_cols, key="anova_val_select")
            if st.button("Run ANOVA", key="run_anova_btn"):
                groups = [g[val].values for n, g in data.groupby(cat)]
                stat, p = stats.f_oneway(*groups)
                c1, c2 = st.columns(2)
                c1.metric("F-stat", f"{stat:.4f}")
                c2.metric("p-value", f"{p:.6f}")
                if p < 0.05:
                    st.success(f"✅ Group means differ significantly (p={p:.4f})")
                else:
                    st.info(f"ℹ️ No significant group differences (p={p:.4f})")
    
    # ========== TAB 8: VISUALIZATIONS ==========
    with tabs[7]:
        st.header("📉 Visualizations")
        
        viz_type = st.selectbox("Chart:", 
                               ["Histogram", "Box Plot", "Scatter", "Bar", "Line", "Pie"],
                               key="viz_type_select")
        
        if viz_type == "Histogram" and numeric_cols:
            col = st.selectbox("Var:", numeric_cols, key="viz_hist_select")
            bins = st.slider("Bins:", 5, 100, 20, key="viz_hist_bins")
            fig = px.histogram(data, x=col, nbins=bins, template=chart_theme,
                              color_discrete_sequence=['#667eea'])
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Box Plot" and numeric_cols:
            cols = st.multiselect("Vars:", numeric_cols, default=numeric_cols[:3],
                                 key="viz_box_multiselect")
            if cols:
                fig = go.Figure()
                for c in cols:
                    fig.add_trace(go.Box(y=data[c], name=c))
                fig.update_layout(template=chart_theme)
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Scatter" and len(numeric_cols) >= 2:
            c1, c2 = st.columns(2)
            x = c1.selectbox("X:", numeric_cols, key='viz_scatter_x')
            y = c2.selectbox("Y:", numeric_cols, index=1, key='viz_scatter_y')
            fig = px.scatter(data, x=x, y=y, template=chart_theme, trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Bar" and categorical_cols and numeric_cols:
            c1, c2 = st.columns(2)
            x = c1.selectbox("Cat:", categorical_cols, key="viz_bar_cat")
            y = c2.selectbox("Val:", numeric_cols, key="viz_bar_val")
            agg = st.selectbox("Agg:", ["mean", "sum", "median", "max", "min"], key="viz_bar_agg")
            grouped = data.groupby(x)[y].agg(agg).reset_index()
            fig = px.bar(grouped, x=x, y=y, template=chart_theme)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Line" and numeric_cols:
            cols = st.multiselect("Vars:", numeric_cols, default=numeric_cols[:2],
                                 key="viz_line_multiselect")
            if cols:
                fig = px.line(data, y=cols, template=chart_theme)
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Pie" and categorical_cols:
            cat = st.selectbox("Cat:", categorical_cols, key="viz_pie_select")
            counts = data[cat].value_counts().head(10)
            fig = px.pie(values=counts.values, names=counts.index, template=chart_theme)
            st.plotly_chart(fig, use_container_width=True)
    
    # ========== TAB 9: COST-BENEFIT ==========
    with tabs[8]:
        st.header("💰 Cost-Benefit Analysis")
        
        c1, c2 = st.columns(2)
        cash_flows_str = c1.text_area("Cash flows (comma-separated):",
                                       value="-100000, 25000, 30000, 35000, 40000, 45000",
                                       key="cba_cashflows_input")
        discount_rate = c2.slider("Discount Rate (%):", 0.0, 30.0, 10.0, key="cba_rate_slider") / 100
        
        if st.button("💡 Calculate", type="primary", key="cba_calc_btn"):
            try:
                cash_flows = [float(x.strip()) for x in cash_flows_str.split(',')]
                npv = sum(cf / (1 + discount_rate)**t for t, cf in enumerate(cash_flows))
                
                rate = 0.1
                for _ in range(1000):
                    npv_calc = sum(cf / (1 + rate)**t for t, cf in enumerate(cash_flows))
                    dnpv = sum(-t * cf / (1 + rate)**(t+1) for t, cf in enumerate(cash_flows))
                    if abs(dnpv) < 1e-12:
                        break
                    rate -= npv_calc / dnpv
                    if abs(npv_calc) < 1e-6:
                        break
                
                c1, c2, c3 = st.columns(3)
                c1.metric("NPV", f"${npv:,.2f}")
                c2.metric("IRR", f"{rate*100:.2f}%")
                c3.metric("Decision", "✅ ACCEPT" if npv > 0 else "❌ REJECT")
                
                # AI Insight
                if npv > 0:
                    st.markdown(f"""
                        <div class="ai-positive">
                            <strong>🤖 AI Recommendation:</strong> ✅ <b>Accept this project.</b> 
                            NPV is positive (${npv:,.2f}), meaning the project creates value at the {discount_rate*100:.0f}% discount rate. 
                            The IRR ({rate*100:.2f}%) exceeds your cost of capital — a strong signal.
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="ai-warning">
                            <strong>🤖 AI Recommendation:</strong> ❌ <b>Reject this project.</b> 
                            NPV is negative (${npv:,.2f}). The project destroys value at this discount rate. 
                            Consider renegotiating costs or seeking alternative opportunities.
                        </div>
                    """, unsafe_allow_html=True)
                
                periods = list(range(len(cash_flows)))
                fig = go.Figure()
                fig.add_trace(go.Bar(x=periods, y=cash_flows,
                                    marker_color=['#e74c3c' if cf < 0 else '#27ae60' for cf in cash_flows],
                                    text=[f'${cf:,.0f}' for cf in cash_flows],
                                    textposition='outside'))
                fig.update_layout(title="Cash Flow Timeline", template=chart_theme)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
    
    # ========== TAB 10: EXPORT ==========
    with tabs[9]:
        st.header("📥 Export Results")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            if numeric_cols:
                csv = data[numeric_cols].describe().T.to_csv()
                st.download_button("📊 Statistics", csv, "stats.csv", "text/csv", key="dl_stats")
        with c2:
            if len(numeric_cols) >= 2:
                csv = data[numeric_cols].corr().to_csv()
                st.download_button("🔗 Correlations", csv, "correlations.csv", "text/csv", key="dl_corr")
        with c3:
            csv = data.to_csv(index=False)
            st.download_button("📄 Data", csv, "data.csv", "text/csv", key="dl_data")
        
        st.markdown("### 🤖 AI-Generated Report")
        if st.button("📝 Generate Smart Report", key="ai_report_btn"):
            insights = generate_ai_insights(data, numeric_cols, categorical_cols)
            
            report = f"""
STATECON ANALYZER PRO — AI-GENERATED REPORT
{'=' * 70}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
{'-' * 70}
Dataset: {data.shape[0]} observations × {data.shape[1]} variables
Numeric: {len(numeric_cols)} | Categorical: {len(categorical_cols)}

🤖 AI-DISCOVERED INSIGHTS
{'-' * 70}
"""
            for i, ins in enumerate(insights, 1):
                report += f"\n{i}. {ins['icon']} {ins['title']}\n   {ins['message']}\n"
            
            report += f"""

DESCRIPTIVE STATISTICS
{'-' * 70}
{data[numeric_cols].describe().to_string() if numeric_cols else 'N/A'}

CORRELATIONS
{'-' * 70}
{data[numeric_cols].corr().to_string() if len(numeric_cols) >= 2 else 'N/A'}

END OF REPORT
{'=' * 70}
Powered by StatEcon Analyzer Pro
            """
            st.download_button("💾 Download AI Report", report, 
                             f"ai_report_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
                             "text/plain", key="dl_ai_report")
            st.success("✅ AI Report ready!")

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
    <div class="footer">
        <p><strong>StatEcon Analyzer Pro</strong> | AI-Powered Analytics Platform</p>
        <p>Built with ❤️ using Python, Streamlit & Plotly | © 2025</p>
    </div>
""", unsafe_allow_html=True)
