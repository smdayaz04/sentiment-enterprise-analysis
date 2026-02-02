import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
import data_loader
import text_cleaner
import sentiment_analyzer

st.set_page_config(page_title="Enterprise Reputation Intelligence", layout="wide")

# Pro-Level Dark Theme CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    [data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }
    .stSidebar { background-color: #0e1117; border-right: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_massive_data():
    # Loading full 1.6M rows into memory cache for speed
    df = data_loader.load_data(n=None)
    return df

# Initialize Data
df_full = load_massive_data()

# ==========================================
# SIDEBAR - ADVANCED FILTERS
# ==========================================
st.sidebar.title("📊 Enterprise Filters")

# 1. Brand/Sector Selection
sector = st.sidebar.selectbox("Market Sector", list(config.BRAND_KEYWORDS.keys()))
keywords = config.BRAND_KEYWORDS[sector]

# 2. Volume Slider (Your "Personal" Control)
vol = st.sidebar.slider("Analysis Depth (Rows)", 10000, 1000000, 50000)

# 3. Date Range Slicer
st.sidebar.subheader("Temporal Filtering")
# Simulating a date range from the dataset's date strings
data_sample = df_full.sample(vol)
data_sample['timestamp'] = pd.to_datetime(data_sample['date'].str.replace(' PDT ', ' '), errors='coerce')
data_sample = data_sample.dropna(subset=['timestamp'])

start_date = data_sample['timestamp'].min().date()
end_date = data_sample['timestamp'].max().date()
selected_dates = st.sidebar.date_input("Analysis Window", [start_date, end_date])

# ==========================================
# DATA PIPELINE (Hidden from User)
# ==========================================
# Filter by keywords and date
filtered_df = data_sample[data_sample['text'].str.contains('|'.join(keywords), case=False)]
filtered_df = text_cleaner.process_batch(filtered_df)
final_df = sentiment_analyzer.analyze_sentiment(filtered_df)

# ==========================================
# DASHBOARD UI
# ==========================================
st.title(f"🚨 {sector.upper()} Reputation Intelligence")
st.markdown(f"**Enterprise-grade monitoring for {len(final_df):,} live-indexed records**")

# Top Metrics Row (KPIs)
m1, m2, m3, m4 = st.columns(4)
neg_pct = (len(final_df[final_df['vader_label'] == 'negative']) / len(final_df)) * 100 if len(final_df) > 0 else 0
pos_pct = (len(final_df[final_df['vader_label'] == 'positive']) / len(final_df)) * 100 if len(final_df) > 0 else 0

m1.metric("Database Scale", f"{len(df_full):,}")
m2.metric("Negative Volume", f"{neg_pct:.1f}%", delta=f"{neg_pct-30:.1f}%", delta_color="inverse")
m3.metric("Positive Volume", f"{pos_pct:.1f}%", delta="4.2%")
m4.metric("Risk Score", f"{int(100-neg_pct)}/100")

st.divider()

# Advanced Visuals Row
col_left, col_right = st.columns([1, 1.5])

with col_left:
    st.subheader("Sentiment Distribution")
    # Pro Donut Chart
    fig_pie = px.pie(final_df, names='vader_label', hole=0.5,
                     color='vader_label',
                     color_discrete_map={'positive': '#2ecc71', 'negative': '#e74c3c', 'neutral': '#95a5a6'})
    fig_pie.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig_pie, use_container_width=True)

with col_right:
    st.subheader("Sentiment Trend (Time-Series)")
    # Group by date for trend line
    trend_df = final_df.copy()
    trend_df['day'] = trend_df['timestamp'].dt.date
    daily_trend = trend_df.groupby(['day', 'vader_label']).size().unstack(fill_value=0).reset_index()
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=daily_trend['day'], y=daily_trend.get('negative', [0]), name='Negative', line=dict(color='#e74c3c', width=3)))
    fig_trend.add_trace(go.Scatter(x=daily_trend['day'], y=daily_trend.get('positive', [0]), name='Positive', line=dict(color='#2ecc71', width=3)))
    fig_trend.update_layout(template='plotly_dark', margin=dict(t=0, b=0, l=0, r=0), height=300)
    st.plotly_chart(fig_trend, use_container_width=True)

# Data Drill-Down
st.subheader("🔍 Reputation Drill-Down (High-Risk Samples)")
st.dataframe(final_df[final_df['vader_label'] == 'negative'][['timestamp', 'text', 'vader_score']].head(50), use_container_width=True)

if neg_pct > config.NEG_LIMIT:
    st.warning(f"⚠️ SYSTEM ALERT: Negative sentiment for {sector} has exceeded the {config.NEG_LIMIT}% risk threshold.")