# aqi_dashboard_dark.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

st.set_page_config(page_title="AQI Dashboard — Dark", layout="wide")

# ---------------------------
# Styling for dark theme (REPLACED)
# ---------------------------
dark_css = """
<style>
/* ⭐ MAKE INPUT BOX TEXT BLACK ⭐ */
[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] * {
    color: #000 !important;
}

[data-testid="stSidebar"] .stMultiSelect div[data-baseweb="select"] * {
    color: #000 !important;
}

[data-testid="stSidebar"] input {
    color: #000 !important;
}

[data-testid="stSidebar"] .css-16idsys {
    color: #000 !important;
}

[data-testid="stSidebar"] .st-af {
    color: #000 !important;
}

[data-testid="stSidebar"] .css-1n76uvr {
    color: #000 !important;
}


/* ⭐ MAKE SIDEBAR TEXT WHITE ⭐ */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] div {
    color: #ffffff !important;
}

/* page background */
[data-testid="stAppViewContainer"] {
  background: #0b0f14;
  color: #e6eef6;
}
/* sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg,#071021,#0d1720);
  color: #e6eef6;
  padding-top: 1.25rem;
  border-right: 1px solid rgba(255,255,255,0.03);
}
/* header */
.stTitle {
  color: #f5f9ff;
}
/* cards & metric spacing */
.css-1d391kg {  /* streamlit metric container class — may vary across versions */
  background: rgba(255,255,255,0.02);
  border-radius: 12px;
  padding: 14px;
}

/* ⭐ MAKE METRIC VALUES PURE WHITE ⭐ */
[data-testid="stMetricValue"] {
  color: #ffffff !important;
  font-weight: 600;
}

/* metric labels slightly lighter */
[data-testid="stMetricLabel"] {
  color: #d9e2ec !important;
}

/* make other primary headings stand out */
h1, h2, h3 {
  color: #f5f9ff;
}

/* hide Streamlit footer (optional) */
footer {visibility: hidden;}
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)

# Plotly dark theme
px.defaults.template = "plotly_dark"
px.defaults.color_continuous_scale = px.colors.sequential.Blues_r

# ---------------------------
# Data (your sample + generated time series)
# ---------------------------
city_data = {
    "Pune": {"AQI": 188, "CO": 12.9, "NO2": 26.4, "O3": 27.6, "PM25": 146},
    "Mumbai": {"AQI": 142, "CO": 10.2, "NO2": 31.5, "O3": 22.4, "PM25": 95},
    "Delhi": {"AQI": 312, "CO": 15.4, "NO2": 52.8, "O3": 33.5, "PM25": 260},
    "Chennai": {"AQI": 98, "CO": 7.4, "NO2": 18.3, "O3": 20.6, "PM25": 70},
    "Kolkata": {"AQI": 168, "CO": 11.7, "NO2": 27.9, "O3": 25.3, "PM25": 130},
    "Hyderabad": {"AQI": 121, "CO": 9.2, "NO2": 22.1, "O3": 24.5, "PM25": 100}
}

cities = list(city_data.keys())

# Generate sample daily time series for last 90 days per city (simulated)
def generate_timeseries(base_aqi):
    rng = pd.date_range(end=datetime.today(), periods=90)
    # simulate seasonality + noise
    seasonal = 20*np.sin(np.linspace(0, 3*np.pi, len(rng)))
    trend = np.linspace(-5, 5, len(rng))
    noise = np.random.normal(0, 10, len(rng))
    vals = np.clip(base_aqi + seasonal + trend + noise, 10, 500)
    df = pd.DataFrame({"date": rng, "AQI": vals})
    # pollutant breakdown (simulated proportions)
    df["PM25"] = np.clip(vals * (0.5 + np.random.normal(0, 0.05, len(rng))), 1, 400)
    df["NO2"] = np.clip(vals * (0.15 + np.random.normal(0, 0.03, len(rng))), 1, 200)
    df["O3"] = np.clip(vals * (0.12 + np.random.normal(0, 0.03, len(rng))), 1, 200)
    df["CO"] = np.clip(vals * (0.08 + np.random.normal(0, 0.02, len(rng))), 0.1, 50)
    return df

timeseries = {c: generate_timeseries(city_data[c]["AQI"]) for c in cities}

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.markdown("## Controls")
city = st.sidebar.selectbox("Select primary city", cities, index=0)
compare_cities = st.sidebar.multiselect(
    "Compare with (choose up to 3)",
    [c for c in cities if c != city],
    default=[c for c in cities if c != city][:2],
    max_selections=3
)
metric = st.sidebar.selectbox("Metric for trend / gauge", ["AQI", "PM25", "NO2", "O3", "CO"])
smooth = st.sidebar.checkbox("Smooth trend (7-day rolling)", value=True)
range_days = st.sidebar.slider("Days to show (history)", min_value=14, max_value=90, value=60, step=1)
download = st.sidebar.button("Download city data CSV")

# ---------------------------
# Top bar: Title + badges
# ---------------------------
col_title, col_spacer = st.columns([9,1])
with col_title:
    st.markdown("### 🌫️ AQI Dashboard — Modern Dark")
    st.markdown(f"**Selected city:** `{city}`  •  Compare: `{', '.join(compare_cities) or 'None'}`")
with col_spacer:
    pass

# ---------------------------
# Cards / KPIs
# ---------------------------
k1, k2, k3, k4, k5 = st.columns([1.3,1.1,1.1,1.1,1.1])
data = city_data[city]

k1.metric("AQI (now)", int(data["AQI"]))
k2.metric("PM2.5", f"{data['PM25']} µg/m³")
k3.metric("NO₂", f"{data['NO2']} ppb")
k4.metric("O₃", f"{data['O3']} ppb")
k5.metric("CO", f"{data['CO']} ppm")

# Gauge indicator (Plotly Indicator) for current AQI
g1, g2 = st.columns([2,1])
with g1:
    # Trend chart and area stacked
    df = timeseries[city].tail(range_days).copy()
    if smooth:
        df["AQI_smooth"] = df["AQI"].rolling(7, min_periods=1).mean()
        trend_col = "AQI_smooth"
    else:
        trend_col = "AQI"

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=df["date"], y=df[trend_col], mode="lines", name=f"{city} {metric}",
        line=dict(width=3)
    ))
    # add comparisons
    for cc in compare_cities[:3]:
        dfc = timeseries[cc].tail(range_days).copy()
        if smooth:
            dfc["AQI_smooth"] = dfc["AQI"].rolling(7, min_periods=1).mean()
            yc = dfc["AQI_smooth"]
        else:
            yc = dfc["AQI"]
        fig_trend.add_trace(go.Scatter(x=dfc["date"], y=yc, mode="lines", name=cc, line=dict(width=2, dash='dot')))

    fig_trend.update_layout(
        title=f"{city} — {metric} trend (last {range_days} days)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=50, l=10, r=10, b=10),
        hovermode="x unified"
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with g2:
    # Gauge for AQI
    curr = city_data[city]["AQI"]
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=curr,
        title={"text": f"{city} AQI now"},
        delta={"reference": np.mean([city_data[c]["AQI"] for c in cities]), "position": "bottom"},
        gauge={
            "axis": {"range": [0, 500]},
            "bar": {"color": "#66ccff"},
            "steps": [
                {"range": [0, 50], "color": "#2ECC71"},
                {"range": [50, 100], "color": "#F1C40F"},
                {"range": [100, 200], "color": "#E67E22"},
                {"range": [200, 300], "color": "#E74C3C"},
                {"range": [300, 500], "color": "#8E44AD"}
            ],
            "threshold": {"line": {"color": "white", "width": 4}, "thickness": 0.75, "value": curr}
        }
    ))
    fig_gauge.update_layout(margin=dict(t=20,b=0,l=0,r=0), height=380)
    st.plotly_chart(fig_gauge, use_container_width=True)

# ---------------------------
# Pollutant radar + histogram + heatmap
# ---------------------------
r1, r2 = st.columns([1.4,1])
with r1:
    st.subheader(f"{city} Pollutant Profile (recent average)")
    recent = timeseries[city].tail(14).mean()
    radar_df = pd.DataFrame({
        "pollutant": ["PM2.5", "NO2", "O3", "CO"],
        "value": [recent["PM25"], recent["NO2"], recent["O3"], recent["CO"]]
    })
    # radar / polar chart
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=radar_df["value"].tolist(),
        theta=radar_df["pollutant"].tolist(),
        fill='toself',
        name=city
    ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False, margin=dict(t=20,b=10))
    st.plotly_chart(fig_radar, use_container_width=True)

with r2:
    st.subheader("Pollutant distribution (last 60 days)")
    hist_df = timeseries[city].tail(60)
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=hist_df["PM25"], name="PM2.5", nbinsx=25, opacity=0.75))
    fig_hist.add_trace(go.Histogram(x=hist_df["NO2"], name="NO2", nbinsx=25, opacity=0.75))
    fig_hist.update_layout(barmode='overlay', legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"), margin=dict(t=20))
    st.plotly_chart(fig_hist, use_container_width=True)

# Heatmap comparing cities by latest AQI and PM2.5
st.subheader("City comparison heatmap (latest values)")
heat_df = pd.DataFrame({
    "City": cities,
    "AQI": [city_data[c]["AQI"] for c in cities],
    "PM25": [city_data[c]["PM25"] for c in cities]
}).set_index("City")
fig_heat = px.imshow(heat_df.T, text_auto=True, aspect="auto", origin='lower')
fig_heat.update_layout(margin=dict(t=20,b=10))
st.plotly_chart(fig_heat, use_container_width=True)

# ---------------------------
# Top 10 Countries by AQI (improved)
# ---------------------------
st.subheader("Top 10 Countries by AQI (sample)")
top10 = pd.DataFrame({
    "Country": ["India", "Iran", "Vietnam", "Saudi Arabia", "Bangladesh", "Nepal", "Indonesia", "Iraq", "China", "Chile"],
    "AQI": [168, 152, 141, 139, 132, 120, 110, 108, 105, 100]
})
fig_top10 = px.bar(top10.sort_values("AQI", ascending=True), x="AQI", y="Country", orientation="h")
fig_top10.update_layout(margin=dict(t=10,b=10))
st.plotly_chart(fig_top10, use_container_width=True)

# ---------------------------
# Detailed table / raw data + download
# ---------------------------
with st.expander("Show raw timeseries for selected city"):
    st.dataframe(timeseries[city].sort_values("date", ascending=False).reset_index(drop=True))

# CSV download
csv_buf = io.StringIO()
df_full = timeseries[city].copy()
df_full.to_csv(csv_buf, index=False)
csv_bytes = csv_buf.getvalue().encode()
st.download_button("Download selected city timeseries (CSV)", data=csv_bytes, file_name=f"{city}_timeseries.csv", mime="text/csv")

# ---------------------------
# Footer / notes
# ---------------------------
st.markdown("---")
st.markdown(
    "Dashboard theme: **Modern Dark** • Smooth accents • Data simulated for demo purposes. "
    "Replace `city_data` and `timeseries` with your real API / dataset to reflect real measurements."
)

# Optional: small help section
with st.expander("How to replace with real data"):
    st.markdown("""
    1. Replace `city_data` dict with your real current/latest values (AQI, PM2.5, NO2, O3, CO).  
    2. Replace `generate_timeseries` with your historical records (date-indexed).  
    3. Ensure pollutant units are correct (µg/m³ / ppb / ppm).  
    4. If you have a larger dataset, consider caching with `@st.cache_data`.
    """)
