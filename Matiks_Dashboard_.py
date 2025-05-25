import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Custom CSS for professional styling, theme-aware
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: var(--background-color, #f5f6f5);
    }
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: var(--secondary-background-color, #e6f0fa);
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: bold;
        color: var(--text-color, #003087);
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: var(--primary-color, #b3d4fc);
    }
    /* Headings */
    h1 {
        color: var(--text-color, #003087);
        font-family: 'Arial', sans-serif;
    }
    h2, h3 {
        color: var(--text-color, #005b96);
        font-family: 'Arial', sans-serif;
    }
    /* Metric cards */
    .stMetric {
        background-color: var(--secondary-background-color, #e6f0fa);
        border-radius: 10px;
        padding: 10px;
        color: var(--text-color, #333333);
    }
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: var(--secondary-background-color, #e6f0fa);
    }
    /* Explanation boxes */
    .explanation {
        background-color: var(--secondary-background-color, #f8f9fa);
        color: var(--text-color, #333333);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border: 1px solid var(--border-color, #dcdcdc);
    }
    /* Ensure readability in dark theme */
    [data-theme="dark"] .explanation {
        background-color: #2c2f33;
        color: #e0e0e0;
        border: 1px solid #444444;
    }
</style>
""", unsafe_allow_html=True)

# Title and Introduction
st.title("📊 Matiks Data Analyst Dashboard")
st.markdown("Welcome to the Matiks Dashboard! This app analyzes user behavior, monetization, and engagement for the Matiks gaming platform, providing insights into Daily Active Users (DAU), Weekly Active Users (WAU), Monthly Active Users (MAU), revenue trends, retention, and user segmentation.")

# Sidebar Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Section", ["Approach", "KPI Visualizations", "Insights"])

# Load and Clean Data
@st.cache_data
def load_data():
    df = pd.read_csv('/content/cleaned_matiks_data.csv')
    df['Signup_Date'] = pd.to_datetime(df['Signup_Date'], errors='coerce')
    df['Last_Login'] = pd.to_datetime(df['Last_Login'], errors='coerce')
    df['Total_Hours_Played'] = df['Total_Hours_Played'].apply(lambda x: max(x, 0))
    df['Avg_Session_Duration_Min'] = df['Avg_Session_Duration_Min'].apply(lambda x: max(x, 0))
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Error: 'cleaned_matiks_data.csv' not found in /content/. Please ensure the file is uploaded to Google Colab.")
    st.stop()

# Calculate KPIs
def calculate_kpis(df):
    # DAU, WAU, MAU
    dau = df.groupby(df['Last_Login'].dt.date)['User_ID'].nunique().reset_index()
    dau.columns = ['Date', 'DAU']
    df['Week'] = df['Last_Login'].dt.to_period('W').apply(lambda r: r.start_time)
    wau = df.groupby('Week')['User_ID'].nunique().reset_index()
    wau.columns = ['Week', 'WAU']
    df['Month'] = df['Last_Login'].dt.to_period('M').apply(lambda r: r.start_time)
    mau = df.groupby('Month')['User_ID'].nunique().reset_index()
    mau.columns = ['Month', 'MAU']
    
    # Revenue Trend
    df['Signup_Month'] = df['Signup_Date'].dt.to_period('M').apply(lambda r: r.start_time)
    revenue_trend = df.groupby('Signup_Month')['Total_Revenue_USD'].sum().reset_index()
    
    # Breakdowns
    device_breakdown = df.groupby('Device_Type').agg({'User_ID': 'nunique', 'Total_Revenue_USD': 'sum'}).reset_index()
    device_breakdown.columns = ['Device_Type', 'Unique_Users', 'Total_Revenue']
    segment_breakdown = df.groupby('Subscription_Tier').agg({'User_ID': 'nunique', 'Total_Revenue_USD': 'sum'}).reset_index()
    segment_breakdown.columns = ['Subscription_Tier', 'Unique_Users', 'Total_Revenue']
    game_mode_breakdown = df.groupby('Preferred_Game_Mode').agg({'User_ID': 'nunique', 'Total_Revenue_USD': 'sum'}).reset_index()
    game_mode_breakdown.columns = ['Game_Mode', 'Unique_Users', 'Total_Revenue']
    
    # Retention Rate (Day 7)
    df['Day7_Login'] = df['Last_Login'] <= df['Signup_Date'] + timedelta(days=7)
    retention_day7 = df.groupby(df['Signup_Date'].dt.to_period('M'))['Day7_Login'].mean().reset_index()
    retention_day7['Day7_Retention_Rate'] = retention_day7['Day7_Login'] * 100
    retention_day7['Signup_Date'] = retention_day7['Signup_Date'].astype(str)  # Fix Period serialization
    
    # ARPU and ARPPU
    arpu = df['Total_Revenue_USD'].sum() / df['User_ID'].nunique()
    paying_users = df[df['In_Game_Purchases_Count'] > 0]
    arppu = paying_users['Total_Revenue_USD'].sum() / paying_users['User_ID'].nunique() if paying_users['User_ID'].nunique() > 0 else 0
    
    # Purchase Conversion Rate
    purchase_conversion = (paying_users['User_ID'].nunique() / df['User_ID'].nunique()) * 100
    
    # Stickiness Ratio
    avg_dau_per_month = dau.groupby(dau['Date'].apply(lambda x: pd.to_datetime(x).to_period('M').start_time))['DAU'].mean().reset_index()
    avg_dau_per_month.columns = ['Month', 'Avg_DAU']
    stickiness = avg_dau_per_month.merge(mau, on='Month')
    stickiness['Stickiness_Ratio'] = stickiness['Avg_DAU'] / stickiness['MAU']
    stickiness['Month'] = stickiness['Month'].astype(str)  # Fix Period serialization
    
    # Revenue per Session
    df['Revenue_Per_Session'] = df['Total_Revenue_USD'] / df['Total_Play_Sessions']
    revenue_per_session = df.groupby('Game_Title')['Revenue_Per_Session'].mean().reset_index()
    
    # Rank Tier Advancement Rate
    rank_order = {'Bronze': 1, 'Silver': 2, 'Gold': 3, 'Platinum': 4, 'Diamond': 5}
    df['Rank_Value'] = df['Rank_Tier'].map(rank_order)
    high_rank_users = df[df['Rank_Value'] >= 3]['User_ID'].nunique()
    rank_advancement_rate = (high_rank_users / df['User_ID'].nunique()) * 100
    
    # User Segmentation (Clustering)
    features = df[['Total_Hours_Played', 'Total_Revenue_USD']].dropna()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_features)
    cluster_summary = df.groupby('Cluster').agg({
        'Total_Hours_Played': 'mean',
        'Total_Revenue_USD': 'mean',
        'User_ID': 'count',
        'Subscription_Tier': lambda x: x.mode()[0]
    }).reset_index()
    cluster_summary.columns = ['Cluster', 'Avg_Hours_Played', 'Avg_Revenue', 'User_Count', 'Common_Subscription_Tier']
    
    return {
        'dau': dau, 'wau': wau, 'mau': mau, 'revenue_trend': revenue_trend,
        'device_breakdown': device_breakdown, 'segment_breakdown': segment_breakdown,
        'game_mode_breakdown': game_mode_breakdown, 'retention_day7': retention_day7,
        'arpu': arpu, 'arppu': arppu, 'purchase_conversion': purchase_conversion,
        'stickiness': stickiness, 'revenue_per_session': revenue_per_session,
        'rank_advancement_rate': rank_advancement_rate, 'cluster_summary': cluster_summary,
        'df': df
    }

kpis = calculate_kpis(df)

# Page Content
if page == "Approach":
    st.header("Approach")
    st.markdown("""
    <div class="explanation">
    <h3>Data Analysis Approach</h3>
    <p><b>Data Loading and Cleaning:</b> Dates are converted to datetime, and negative values in playtime and session duration are set to zero for consistency.</p>
    <p><b>KPI Calculations:</b> Key metrics include:</p>
    <ul>
        <li><b>Activity Metrics:</b> DAU, WAU, MAU, and Stickiness Ratio to measure user engagement frequency.</li>
        <li><b>Revenue Metrics:</b> Revenue trends, ARPU, ARPPU, Revenue per Session, and Purchase Conversion Rate to assess monetization.</li>
        <li><b>Engagement Metrics:</b> Retention Rate (Day 7) and Rank Tier Advancement Rate to evaluate user retention and progression.</li>
        <li><b>Segmentation:</b> Breakdowns by Device Type, Subscription Tier, Game Mode, and clustering to identify high-value users.</li>
    </ul>
    <p><b>Visualizations:</b> Interactive Plotly charts display trends and breakdowns, with a professional layout for easy exploration.</p>
    <p><b>Relevance:</b> The KPIs address user behavior, churn risks, high-value user characteristics, and retention/revenue improvement opportunities, aligning with the Matiks task objectives.</p>
    </div>
    """, unsafe_allow_html=True)

elif page == "KPI Visualizations":
    st.header("KPI Visualizations")
    tab1, tab2, tab3, tab4 = st.tabs(["Activity Metrics", "Revenue Metrics", "Engagement Metrics", "Segmentation"])
    
    with tab1:
        st.subheader("Activity Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average DAU (Last Month)", f"{kpis['dau']['DAU'].tail(30).mean():.1f}")
        with col2:
            st.metric("Average WAU (Last Month)", f"{kpis['wau']['WAU'].tail(4).mean():.1f}")
        with col3:
            st.metric("Latest MAU", f"{kpis['mau']['MAU'].iloc[-1]}")
        
        fig_dau = px.line(kpis['dau'], x='Date', y='DAU', title='Daily Active Users (DAU)')
        st.plotly_chart(fig_dau, use_container_width=True)
        fig_wau = px.line(kpis['wau'], x='Week', y='WAU', title='Weekly Active Users (WAU)')
        st.plotly_chart(fig_wau, use_container_width=True)
        fig_mau = px.line(kpis['mau'], x='Month', y='MAU', title='Monthly Active Users (MAU)')
        st.plotly_chart(fig_mau, use_container_width=True)
        fig_stickiness = px.line(kpis['stickiness'], x='Month', y='Stickiness_Ratio', title='Stickiness Ratio (DAU/MAU)')
        st.plotly_chart(fig_stickiness, use_container_width=True)
    
    with tab2:
        st.subheader("Revenue Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ARPU", f"${kpis['arpu']:.2f}")
        with col2:
            st.metric("ARPPU", f"${kpis['arppu']:.2f}")
        with col3:
            st.metric("Purchase Conversion Rate", f"{kpis['purchase_conversion']:.2f}%")
        
        fig_revenue = px.line(kpis['revenue_trend'], x='Signup_Month', y='Total_Revenue_USD', title='Revenue Trend by Signup Month')
        st.plotly_chart(fig_revenue, use_container_width=True)
        fig_device = px.bar(kpis['device_breakdown'], x='Device_Type', y='Total_Revenue', title='Revenue by Device Type')
        st.plotly_chart(fig_device, use_container_width=True)
        fig_segment = px.bar(kpis['segment_breakdown'], x='Subscription_Tier', y='Total_Revenue', title='Revenue by Subscription Tier')
        st.plotly_chart(fig_segment, use_container_width=True)
        fig_game_mode = px.bar(kpis['game_mode_breakdown'], x='Game_Mode', y='Total_Revenue', title='Revenue by Game Mode')
        st.plotly_chart(fig_game_mode, use_container_width=True)
        fig_rev_session = px.bar(kpis['revenue_per_session'], x='Game_Title', y='Revenue_Per_Session', title='Revenue per Session by Game Title')
        st.plotly_chart(fig_rev_session, use_container_width=True)
    
    with tab3:
        st.subheader("Engagement Metrics")
        st.metric("Day 7 Retention Rate (Latest Cohort)", f"{kpis['retention_day7']['Day7_Retention_Rate'].iloc[-1]:.2f}%")
        st.metric("Rank Tier Advancement Rate (Gold+)", f"{kpis['rank_advancement_rate']:.2f}%")
        
        fig_retention = px.line(kpis['retention_day7'], x='Signup_Date', y='Day7_Retention_Rate', title='Day 7 Retention Rate by Signup Month')
        st.plotly_chart(fig_retention, use_container_width=True)
    
    with tab4:
        st.subheader("User Segmentation")
        fig_cluster = px.scatter(kpis['df'], x='Total_Hours_Played', y='Total_Revenue_USD', color='Cluster', title='User Segmentation by Engagement and Revenue')
        st.plotly_chart(fig_cluster, use_container_width=True)
        st.write("Cluster Summary:")
        st.dataframe(kpis['cluster_summary'])

elif page == "Insights":
    st.header("Insights")
    st.markdown("""
    <div class="explanation">
    <h3>Key Insights and Recommendations</h3>
    <p><b>Activity Insights:</b> High DAU/MAU stickiness ratios in certain months suggest strong user engagement. Focus on replicating successful engagement strategies (e.g., events, updates) from high-stickiness periods.</p>
    <p><b>Revenue Insights:</b> Premium subscription tiers and specific game modes drive the most revenue. Promote premium upgrades and enhance high-revenue game modes with new features.</p>
    <p><b>Engagement Insights:</b> Low Day 7 retention rates in early cohorts indicate onboarding issues. Improve tutorials or early rewards to boost retention. High rank advancement rates suggest progression mechanics are engaging.</p>
    <p><b>Segmentation Insights:</b> Clustering reveals high-revenue, high-engagement users often have premium subscriptions. Target these users with exclusive content to maximize revenue.</p>
    <p><b>Recommendations:</b></p>
    <ul>
        <li>Enhance onboarding to improve Day 7 retention.</li>
        <li>Increase in-game purchase prompts for users in high-engagement clusters.</li>
        <li>Prioritize marketing for high-revenue game titles and device types.</li>
        <li>Analyze referral sources further to optimize acquisition channels.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)