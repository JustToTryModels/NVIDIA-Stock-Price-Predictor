import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ==============================
# 🔹 Page Configuration
# ==============================
st.set_page_config(
    page_title="NVIDIA Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# 🔹 Global CSS Styling
# ==============================
st.markdown("""
<style>
    /* ── Import Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Root Variables ── */
    :root {
        --nvidia-green: #76b900;
        --nvidia-dark:  #1a1a2e;
        --accent-blue:  #00d4ff;
        --accent-pink:  #ff006e;
        --bg-dark:      #0e0e1a;
        --card-bg:      #16213e;
        --card-border:  #0f3460;
        --text-primary: #ffffff;
        --text-secondary: #a0aec0;
        --success:      #00e676;
        --warning:      #ffd600;
        --danger:       #ff1744;
    }

    /* ── Global Reset ── */
    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0e0e1a 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
    }

    /* ── Hide Streamlit Branding ── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1a 0%, #1a1a2e 100%);
        border-right: 1px solid #0f3460;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #76b900 !important;
    }

    /* ── Hero Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 40%, #0f3460 100%);
        border: 1px solid #0f3460;
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5);
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(118,185,0,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        bottom: -30%;
        left: -5%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(0,212,255,0.1) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #76b900, #00d4ff, #76b900);
        background-size: 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        animation: shimmer 3s infinite linear;
    }
    @keyframes shimmer {
        0%   { background-position: 0% }
        100% { background-position: 200% }
    }
    .hero-subtitle {
        color: #a0aec0;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(118,185,0,0.15);
        border: 1px solid #76b900;
        color: #76b900;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-right: 8px;
        margin-top: 1rem;
    }

    /* ── Metric Cards ── */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #16213e, #1a1a2e);
        border: 1px solid #0f3460;
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(118,185,0,0.2);
        border-color: #76b900;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #76b900, #00d4ff);
        border-radius: 16px 16px 0 0;
    }
    .metric-icon { font-size: 1.8rem; margin-bottom: 0.4rem; }
    .metric-label {
        color: #a0aec0;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        color: #ffffff;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .metric-delta-pos { color: #00e676; font-size: 0.82rem; font-weight: 600; }
    .metric-delta-neg { color: #ff1744; font-size: 0.82rem; font-weight: 600; }

    /* ── Section Headers ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #0f3460;
    }
    .section-header-icon {
        background: linear-gradient(135deg, #76b900, #00d4ff);
        border-radius: 10px;
        width: 38px;
        height: 38px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
    }
    .section-header-text {
        color: #ffffff;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 0;
    }
    .section-header-sub {
        color: #a0aec0;
        font-size: 0.85rem;
        margin: 0;
    }

    /* ── Info / Warning / Success Boxes ── */
    .info-box {
        background: rgba(0,212,255,0.08);
        border-left: 4px solid #00d4ff;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #e2e8f0;
        font-size: 0.9rem;
    }
    .success-box {
        background: rgba(0,230,118,0.08);
        border-left: 4px solid #00e676;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #e2e8f0;
        font-size: 0.9rem;
    }
    .warning-box {
        background: rgba(255,214,0,0.08);
        border-left: 4px solid #ffd600;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        color: #e2e8f0;
        font-size: 0.9rem;
    }

    /* ── Prediction Table ── */
    .pred-table-wrap {
        background: linear-gradient(135deg, #16213e, #1a1a2e);
        border: 1px solid #0f3460;
        border-radius: 16px;
        overflow: hidden;
        margin-top: 1rem;
    }
    .pred-table-header {
        background: linear-gradient(90deg, #76b900, #00d4ff);
        padding: 0.75rem 1.5rem;
        color: #0e0e1a;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #76b900, #5a8f00);
        color: #0e0e1a !important;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 700;
        cursor: pointer;
        transition: all 0.25s ease;
        letter-spacing: 0.03em;
        width: 100%;
        box-shadow: 0 4px 20px rgba(118,185,0,0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #8fd400, #76b900);
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(118,185,0,0.5);
    }
    .stButton > button:active {
        transform: translateY(0px);
    }

    /* ── Slider ── */
    .stSlider [data-baseweb="slider"] {
        padding: 0.5rem 0;
    }
    .stSlider [data-testid="stThumbValue"] {
        background: #76b900;
        color: #0e0e1a;
        font-weight: 700;
        border-radius: 6px;
        padding: 2px 8px;
    }

    /* ── Selectbox / Inputs ── */
    .stSelectbox [data-baseweb="select"] > div,
    .stMultiSelect [data-baseweb="select"] > div {
        background: #16213e;
        border: 1px solid #0f3460;
        border-radius: 10px;
        color: white;
    }
    .stTextInput > div > div {
        background: #16213e;
        border: 1px solid #0f3460;
        border-radius: 10px;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #16213e;
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        border: 1px solid #0f3460;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #a0aec0;
        border-radius: 10px;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #76b900, #5a8f00) !important;
        color: #0e0e1a !important;
    }

    /* ── DataFrames ── */
    .stDataFrame {
        border: 1px solid #0f3460;
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: #16213e !important;
        border: 1px solid #0f3460 !important;
        border-radius: 12px !important;
        color: #a0aec0 !important;
        font-weight: 600;
    }
    .streamlit-expanderContent {
        background: #1a1a2e !important;
        border: 1px solid #0f3460 !important;
        border-radius: 0 0 12px 12px !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: #76b900 !important;
    }

    /* ── Status Pill ── */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-live {
        background: rgba(0,230,118,0.15);
        border: 1px solid #00e676;
        color: #00e676;
    }
    .status-dot {
        width: 8px; height: 8px;
        background: #00e676;
        border-radius: 50%;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%,100% { opacity: 1; }
        50%      { opacity: 0.3; }
    }

    /* ── Footer ── */
    .custom-footer {
        text-align: center;
        color: #4a5568;
        font-size: 0.78rem;
        padding: 2rem 0 1rem;
        border-top: 1px solid #0f3460;
        margin-top: 3rem;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0e0e1a; }
    ::-webkit-scrollbar-thumb { background: #0f3460; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #76b900; }
</style>
""", unsafe_allow_html=True)

# ==============================
# 🔹 Load Model (Cached)
# ==============================
@st.cache_resource
def load_nvidia_model():
    model_file = 'LSTM_Model/NVIDIA_LSTM_LB(5)_U(150)_RMSE(1.32).keras'
    try:
        model = load_model(model_file)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# ==============================
# 🔹 Fetch Stock Data (Cached)
# ==============================
@st.cache_data(ttl=300)
def get_stock_data(ticker='NVDA'):
    try:
        data = yf.download(ticker, period='max', auto_adjust=True)
        return data
    except Exception as e:
        st.error(f"❌ Error fetching stock data: {e}")
        return None

@st.cache_data(ttl=60)
def get_live_quote(ticker='NVDA'):
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        return {
            'price':       round(float(info.last_price), 2),
            'prev_close':  round(float(info.previous_close), 2),
            'day_high':    round(float(info.day_high), 2),
            'day_low':     round(float(info.day_low), 2),
            'volume':      int(info.three_month_average_volume),
            'mkt_cap':     float(info.market_cap),
        }
    except Exception:
        return None

# ==============================
# 🔹 Business Day Generator
# ==============================
def generate_business_days(start_date, num_days):
    return pd.bdate_range(start=start_date, periods=num_days).tolist()

# ==============================
# 🔹 Prediction Function
# ==============================
def predict_next_business_days(model, data, look_back=5, days=5):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    last_sequence = data_scaled[-look_back:]
    predictions = []

    for _ in range(days):
        X_input = np.reshape(last_sequence, (1, look_back, 1))
        pred    = model.predict(X_input, verbose=0)
        predictions.append(pred[0, 0])
        last_sequence = np.append(last_sequence[1:], pred, axis=0)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

# ==============================
# 🔹 Formatting Helpers
# ==============================
def fmt_currency(val):
    return f"${val:,.2f}"

def fmt_volume(val):
    if val >= 1_000_000_000:
        return f"{val/1_000_000_000:.2f}B"
    if val >= 1_000_000:
        return f"{val/1_000_000:.2f}M"
    return f"{val:,}"

def fmt_mktcap(val):
    if val >= 1_000_000_000_000:
        return f"${val/1_000_000_000_000:.2f}T"
    if val >= 1_000_000_000:
        return f"${val/1_000_000_000:.2f}B"
    return f"${val:,.0f}"

# ==============================
# 🔹 Plotly Theme Helper
# ==============================
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor ='rgba(0,0,0,0)',
    font         =dict(family='Inter', color='#a0aec0', size=12),
    xaxis        =dict(
        gridcolor='rgba(255,255,255,0.05)',
        linecolor='#0f3460',
        zerolinecolor='rgba(0,0,0,0)',
        showspikes=True, spikecolor='#76b900', spikethickness=1,
    ),
    yaxis        =dict(
        gridcolor='rgba(255,255,255,0.05)',
        linecolor='#0f3460',
        zerolinecolor='rgba(0,0,0,0)',
        showspikes=True, spikecolor='#76b900', spikethickness=1,
    ),
    legend=dict(
        bgcolor='rgba(22,33,62,0.8)',
        bordercolor='#0f3460',
        borderwidth=1,
        font=dict(color='#e2e8f0'),
    ),
    hovermode='x unified',
    hoverlabel=dict(
        bgcolor='#16213e',
        bordercolor='#0f3460',
        font=dict(color='white', size=12),
    ),
    margin=dict(l=10, r=10, t=50, b=10),
)

# ──────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:2.5rem;'>🤖</div>
        <h2 style='color:#76b900; font-weight:800; margin:0.3rem 0;'>AI Predictor</h2>
        <p style='color:#a0aec0; font-size:0.8rem; margin:0;'>LSTM Neural Network</p>
    </div>
    <hr style='border-color:#0f3460; margin:1rem 0;'>
    """, unsafe_allow_html=True)

    st.markdown("#### ⚙️ Forecast Settings")
    num_days = st.slider("📅 Business Days to Forecast", 1, 30, 5,
                         help="Number of future trading days to predict")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📊 Chart Settings")
    chart_type = st.selectbox("Historical Chart Style",
                              ["Candlestick", "Line", "Area"],
                              index=0)
    history_window = st.selectbox("History Window",
                                  ["3 Months", "6 Months", "1 Year", "2 Years", "5 Years", "All"],
                                  index=2)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📐 Technical Indicators")
    show_ma20  = st.checkbox("20-Day MA",  value=True)
    show_ma50  = st.checkbox("50-Day MA",  value=True)
    show_ma200 = st.checkbox("200-Day MA", value=False)
    show_vol   = st.checkbox("Volume Bars", value=True)

    st.markdown("<hr style='border-color:#0f3460; margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:rgba(118,185,0,0.08); border:1px solid #76b900;
                border-radius:12px; padding:1rem; margin-top:0.5rem;'>
        <p style='color:#76b900; font-weight:700; margin:0 0 0.5rem 0; font-size:0.9rem;'>
            🧠 Model Info
        </p>
        <p style='color:#a0aec0; font-size:0.78rem; margin:0.2rem 0;'>Architecture: LSTM</p>
        <p style='color:#a0aec0; font-size:0.78rem; margin:0.2rem 0;'>Units: 150</p>
        <p style='color:#a0aec0; font-size:0.78rem; margin:0.2rem 0;'>Look-back: 5 days</p>
        <p style='color:#a0aec0; font-size:0.78rem; margin:0.2rem 0;'>RMSE: $1.32</p>
        <p style='color:#a0aec0; font-size:0.78rem; margin:0.2rem 0;'>Ticker: NVDA</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    refresh_btn = st.button("🔄 Refresh Market Data")

# ──────────────────────────────────────────────
#  LOAD DATA & MODEL
# ──────────────────────────────────────────────
model      = load_nvidia_model()
stock_data = get_stock_data('NVDA')
quote      = get_live_quote('NVDA')

if refresh_btn:
    st.cache_data.clear()
    st.rerun()

# ──────────────────────────────────────────────
#  HERO BANNER
# ──────────────────────────────────────────────
col_logo, col_hero = st.columns([1, 4])
with col_logo:
    st.markdown("<br>", unsafe_allow_html=True)
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/NVIDIA_logo.svg/1920px-NVIDIA_logo.svg.png",
        width=180
    )
with col_hero:
    live_price_str = fmt_currency(quote['price']) if quote else "N/A"
    change         = round(quote['price'] - quote['prev_close'], 2) if quote else 0
    change_pct     = round((change / quote['prev_close']) * 100, 2) if quote else 0
    delta_color    = "#00e676" if change >= 0 else "#ff1744"
    delta_arrow    = "▲" if change >= 0 else "▼"

    st.markdown(f"""
    <div class="hero-banner">
        <div style="display:flex; align-items:center; gap:1rem; flex-wrap:wrap;">
            <div>
                <h1 class="hero-title">Stock Price Predictor</h1>
                <p class="hero-subtitle">
                    LSTM Deep Learning · Real-Time Data · Interactive Analytics
                </p>
                <div style="margin-top:0.8rem;">
                    <span class="hero-badge">🤖 AI Powered</span>
                    <span class="hero-badge">📡 Live Data</span>
                    <span class="hero-badge">⚡ NVDA</span>
                </div>
            </div>
            <div style="margin-left:auto; text-align:right;">
                <div class="status-pill status-live" style="margin-bottom:0.5rem;">
                    <span class="status-dot"></span> MARKET DATA
                </div>
                <div style="font-size:2.2rem; font-weight:800; color:#fff;">
                    {live_price_str}
                </div>
                <div style="color:{delta_color}; font-size:1rem; font-weight:600;">
                    {delta_arrow} ${abs(change):,.2f} ({abs(change_pct):.2f}%)
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  LIVE METRIC CARDS
# ──────────────────────────────────────────────
if quote:
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "💰", "Current Price",    fmt_currency(quote['price']),
         f"{delta_arrow} {abs(change_pct):.2f}%", change >= 0),
        (c2, "📈", "Day High",         fmt_currency(quote['day_high']),
         "Today's high", None),
        (c3, "📉", "Day Low",          fmt_currency(quote['day_low']),
         "Today's low", None),
        (c4, "🏢", "Market Cap",       fmt_mktcap(quote['mkt_cap']),
         "Total market cap", None),
    ]
    for col, icon, label, value, delta, pos in cards:
        with col:
            if pos is True:
                delta_cls = "metric-delta-pos"
            elif pos is False:
                delta_cls = "metric-delta-neg"
            else:
                delta_cls = "metric-label"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="{delta_cls}">{delta}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  HELPER: filter by history window
# ──────────────────────────────────────────────
def filter_by_window(df, window):
    now = df.index[-1]
    map_ = {
        "3 Months": now - pd.DateOffset(months=3),
        "6 Months": now - pd.DateOffset(months=6),
        "1 Year":   now - pd.DateOffset(years=1),
        "2 Years":  now - pd.DateOffset(years=2),
        "5 Years":  now - pd.DateOffset(years=5),
        "All":      df.index[0],
    }
    return df[df.index >= map_[window]]

# ──────────────────────────────────────────────
#  MAIN TABS
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Market Overview",
    "🔮  AI Forecast",
    "📋  Data Explorer",
    "ℹ️   Model Info",
])

# ════════════════════════════════════════════════
#  TAB 1 – MARKET OVERVIEW
# ════════════════════════════════════════════════
with tab1:
    if stock_data is None or stock_data.empty:
        st.error("❌ Could not load stock data.")
    else:
        df_view = filter_by_window(stock_data, history_window)

        # ── Section Header ──
        st.markdown("""
        <div class="section-header">
            <div class="section-header-icon">📊</div>
            <div>
                <p class="section-header-text">Price Chart — NVDA</p>
                <p class="section-header-sub">Historical NVIDIA Corporation stock price</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Compute MAs ──
        close_col = df_view['Close'].squeeze()
        ma20  = close_col.rolling(20).mean()
        ma50  = close_col.rolling(50).mean()
        ma200 = close_col.rolling(200).mean()

        # ── Build Chart ──
        if show_vol:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.75, 0.25], vertical_spacing=0.03)
        else:
            fig = go.Figure()

        def add_trace(trace, row=None, col=None):
            if show_vol:
                fig.add_trace(trace, row=row, col=col)
            else:
                fig.add_trace(trace)

        main_row = 1 if show_vol else None
        main_col = 1 if show_vol else None

        # ── Candlestick / Line / Area ──
        if chart_type == "Candlestick":
            add_trace(go.Candlestick(
                x     = df_view.index,
                open  = df_view['Open'].squeeze(),
                high  = df_view['High'].squeeze(),
                low   = df_view['Low'].squeeze(),
                close = close_col,
                name  = 'NVDA',
                increasing_line_color='#00e676',
                decreasing_line_color='#ff1744',
                increasing_fillcolor ='rgba(0,230,118,0.3)',
                decreasing_fillcolor ='rgba(255,23,68,0.3)',
            ), row=main_row, col=main_col)

        elif chart_type == "Line":
            add_trace(go.Scatter(
                x=df_view.index, y=close_col, mode='lines',
                name='Close Price',
                line=dict(color='#76b900', width=2),
                fill=None,
            ), row=main_row, col=main_col)

        else:  # Area
            add_trace(go.Scatter(
                x=df_view.index, y=close_col, mode='lines',
                name='Close Price',
                line=dict(color='#76b900', width=2),
                fill='tozeroy',
                fillcolor='rgba(118,185,0,0.1)',
            ), row=main_row, col=main_col)

        # ── Moving Averages ──
        if show_ma20:
            add_trace(go.Scatter(
                x=df_view.index, y=ma20, mode='lines',
                name='MA 20', line=dict(color='#ffd600', width=1.5, dash='dot'),
            ), row=main_row, col=main_col)
        if show_ma50:
            add_trace(go.Scatter(
                x=df_view.index, y=ma50, mode='lines',
                name='MA 50', line=dict(color='#00d4ff', width=1.5, dash='dot'),
            ), row=main_row, col=main_col)
        if show_ma200:
            add_trace(go.Scatter(
                x=df_view.index, y=ma200, mode='lines',
                name='MA 200', line=dict(color='#ff006e', width=1.5, dash='dot'),
            ), row=main_row, col=main_col)

        # ── Volume ──
        if show_vol:
            vol    = df_view['Volume'].squeeze()
            colors = ['rgba(0,230,118,0.6)' if c >= o else 'rgba(255,23,68,0.6)'
                      for c, o in zip(close_col, df_view['Open'].squeeze())]
            fig.add_trace(go.Bar(
                x=df_view.index, y=vol,
                name='Volume', marker_color=colors, showlegend=False,
            ), row=2, col=1)

        # ── Layout ──
        layout_upd = dict(**PLOTLY_LAYOUT,
                          title=dict(text=f'NVIDIA (NVDA) — {history_window}',
                                     font=dict(size=16, color='#ffffff')),
                          height=520 if show_vol else 420)
        if show_vol:
            layout_upd['xaxis2'] = dict(gridcolor='rgba(255,255,255,0.05)',
                                        linecolor='#0f3460')
            layout_upd['yaxis2'] = dict(gridcolor='rgba(255,255,255,0.05)',
                                        linecolor='#0f3460', title='Volume')
        fig.update_layout(**layout_upd)
        if chart_type == "Candlestick":
            fig.update_layout(xaxis_rangeslider_visible=False)

        st.plotly_chart(fig, use_container_width=True)

        # ── Statistics Row ──
        st.markdown("""
        <div class="section-header">
            <div class="section-header-icon">📐</div>
            <div>
                <p class="section-header-text">Statistical Summary</p>
                <p class="section-header-sub">Key metrics for the selected window</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        close_s = df_view['Close'].squeeze()
        s1, s2, s3, s4, s5 = st.columns(5)
        stats = [
            (s1, "📊", "Period High",   fmt_currency(float(close_s.max()))),
            (s2, "📉", "Period Low",    fmt_currency(float(close_s.min()))),
            (s3, "📈", "Period Return", f"{((float(close_s.iloc[-1])/float(close_s.iloc[0]))-1)*100:+.2f}%"),
            (s4, "〰️", "Volatility",   f"{float(close_s.pct_change().std()*np.sqrt(252)*100):.2f}%"),
            (s5, "📅", "Trading Days",  str(len(close_s))),
        ]
        for col, icon, label, val in stats:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="font-size:1.3rem;">{val}</div>
                </div>
                """, unsafe_allow_html=True)

# ════════════════════════════════════════════════
#  TAB 2 – AI FORECAST
# ════════════════════════════════════════════════
with tab2:

    st.markdown("""
    <div class="section-header">
        <div class="section-header-icon">🔮</div>
        <div>
            <p class="section-header-text">AI Price Forecast</p>
            <p class="section-header-sub">LSTM neural network multi-step prediction</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        📅 <strong>Today:</strong> {datetime.now().strftime('%A, %B %d, %Y')} &nbsp;|&nbsp;
        🎯 <strong>Forecasting:</strong> Next <strong>{num_days}</strong> business day(s) for NVDA
    </div>
    """, unsafe_allow_html=True)

    # ── Session State ──
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None

    # ── Predict Button ──
    col_btn, col_gap = st.columns([1, 2])
    with col_btn:
        run_pred = st.button(f"🚀 Run Forecast — {num_days} Day{'s' if num_days>1 else ''}",
                             key='forecast-btn')

    if run_pred:
        if model is None:
            st.error("❌ Model not loaded.")
        elif stock_data is None or stock_data.empty:
            st.error("❌ Stock data unavailable.")
        else:
            with st.spinner("🤖 LSTM model is generating predictions…"):
                close_prices    = stock_data['Close'].values.reshape(-1, 1)
                dates           = stock_data.index
                predictions     = predict_next_business_days(model, close_prices,
                                                             look_back=5, days=num_days)
                last_date       = dates[-1]
                prediction_dates = generate_business_days(last_date + timedelta(days=1), num_days)

                st.session_state.prediction_results = {
                    'close_prices':      close_prices,
                    'dates':             dates,
                    'predictions':       predictions,
                    'prediction_dates':  prediction_dates,
                    'num_days':          num_days,
                    'last_actual_price': float(close_prices[-1]),
                }

            st.markdown("""
            <div class="success-box">
                ✅ <strong>Forecast complete!</strong> Scroll down to explore the results.
            </div>
            """, unsafe_allow_html=True)

    # ── Display Results ──
    if st.session_state.prediction_results:
        res              = st.session_state.prediction_results
        predictions      = res['predictions']
        prediction_dates = res['prediction_dates']
        stored_num_days  = res['num_days']
        close_prices     = res['close_prices']
        dates            = res['dates']
        last_actual      = res['last_actual_price']

        # ── Summary Metrics ──
        pred_min  = float(predictions.min())
        pred_max  = float(predictions.max())
        pred_last = float(predictions[-1])
        overall_chg     = pred_last - last_actual
        overall_chg_pct = (overall_chg / last_actual) * 100

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🎯</div>
                <div class="metric-label">Day 1 Forecast</div>
                <div class="metric-value">{fmt_currency(float(predictions[0]))}</div>
                <div class="{'metric-delta-pos' if predictions[0]>=last_actual else 'metric-delta-neg'}">
                    {'▲' if predictions[0]>=last_actual else '▼'}
                    {abs(float(predictions[0])-last_actual):.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">📈</div>
                <div class="metric-label">Period High</div>
                <div class="metric-value">{fmt_currency(pred_max)}</div>
                <div class="metric-label" style="margin-top:4px;">Forecast peak</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">📉</div>
                <div class="metric-label">Period Low</div>
                <div class="metric-value">{fmt_currency(pred_min)}</div>
                <div class="metric-label" style="margin-top:4px;">Forecast trough</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🏁</div>
                <div class="metric-label">Final Day Target</div>
                <div class="metric-value">{fmt_currency(pred_last)}</div>
                <div class="{'metric-delta-pos' if overall_chg>=0 else 'metric-delta-neg'}">
                    {'▲' if overall_chg>=0 else '▼'}
                    {abs(overall_chg_pct):.2f}% vs today
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Chart: Context + Forecast ──
        st.markdown("""
        <div class="section-header">
            <div class="section-header-icon">📈</div>
            <div>
                <p class="section-header-text">Historical Context + Forecast</p>
                <p class="section-header-sub">Last 60 days of actual prices with AI predictions</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        context_n  = 60
        hist_dates = dates[-context_n:]
        hist_close = close_prices[-context_n:].flatten()

        bridge_x = [hist_dates[-1], prediction_dates[0]]
        bridge_y = [hist_close[-1], float(predictions[0])]

        fig_fc = go.Figure()

        # Historical area
        fig_fc.add_trace(go.Scatter(
            x=hist_dates, y=hist_close, mode='lines',
            name='Historical', line=dict(color='#76b900', width=2.5),
            fill='tozeroy', fillcolor='rgba(118,185,0,0.07)',
        ))
        # Bridge
        fig_fc.add_trace(go.Scatter(
            x=bridge_x, y=bridge_y, mode='lines',
            name='_bridge', showlegend=False,
            line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dot'),
        ))
        # Forecast line
        fig_fc.add_trace(go.Scatter(
            x=prediction_dates, y=predictions.flatten(), mode='lines+markers',
            name='AI Forecast',
            line=dict(color='#ff006e', width=3),
            marker=dict(size=9, color='#ff006e',
                        line=dict(color='white', width=2),
                        symbol='circle'),
            fill='tozeroy', fillcolor='rgba(255,0,110,0.07)',
        ))
        # Confidence band (±2%)
        upper = predictions.flatten() * 1.02
        lower = predictions.flatten() * 0.98
        fig_fc.add_trace(go.Scatter(
            x=list(prediction_dates) + list(prediction_dates[::-1]),
            y=list(upper) + list(lower[::-1]),
            fill='toself', fillcolor='rgba(255,0,110,0.06)',
            line=dict(color='rgba(0,0,0,0)'),
            name='±2% Band', showlegend=True,
        ))

        # Vertical divider
        fig_fc.add_vline(
            x=hist_dates[-1], line_width=1,
            line_dash='dash', line_color='rgba(255,255,255,0.3)',
            annotation_text='Today',
            annotation_font_color='#a0aec0',
        )

        fig_fc.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text=f'NVDA — {stored_num_days}-Day AI Forecast',
                       font=dict(size=16, color='#fff')),
            height=440,
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # ── Chart: Forecast Only Bar ──
        st.markdown("""
        <div class="section-header">
            <div class="section-header-icon">🎯</div>
            <div>
                <p class="section-header-text">Forecast Breakdown</p>
                <p class="section-header-sub">Day-by-day predicted prices</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        bar_colors = ['#00e676' if p >= last_actual else '#ff1744'
                      for p in predictions.flatten()]

        fig_bar = go.Figure(go.Bar(
            x=[d.strftime('%b %d') for d in prediction_dates],
            y=predictions.flatten(),
            marker_color=bar_colors,
            marker_line_color='rgba(255,255,255,0.3)',
            marker_line_width=1,
            text=[fmt_currency(p) for p in predictions.flatten()],
            textposition='outside',
            textfont=dict(color='white', size=12, family='Inter'),
            name='Predicted Price',
        ))

        # Baseline
        fig_bar.add_hline(
            y=last_actual, line_dash='dot',
            line_color='rgba(255,255,255,0.5)', line_width=1.5,
            annotation_text=f'Today: {fmt_currency(last_actual)}',
            annotation_font_color='#a0aec0',
        )

        fig_bar.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text='Daily Forecast Prices',
                       font=dict(size=16, color='#fff')),
            height=380,
            yaxis=dict(
                **PLOTLY_LAYOUT['yaxis'],
                range=[min(pred_min, last_actual) * 0.97,
                       max(pred_max, last_actual) * 1.03],
                tickprefix='$',
            ),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Prediction Table ──
        st.markdown("""
        <div class="section-header">
            <div class="section-header-icon">📋</div>
            <div>
                <p class="section-header-text">Forecast Data Table</p>
                <p class="section-header-sub">Full numeric breakdown</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        pred_df = pd.DataFrame({
            'Day':             range(1, stored_num_days + 1),
            'Date':            [d.strftime('%Y-%m-%d (%A)') for d in prediction_dates],
            'Predicted Price': [fmt_currency(p) for p in predictions.flatten()],
            'vs Today ($)':    [f"{'▲' if p>=last_actual else '▼'} {abs(p-last_actual):.2f}"
                                for p in predictions.flatten()],
            'vs Today (%)':    [f"{'▲' if p>=last_actual else '▼'} {abs((p-last_actual)/last_actual*100):.2f}%"
                                for p in predictions.flatten()],
        })

        st.dataframe(
            pred_df.set_index('Day'),
            use_container_width=True,
            height=min(60 + 35 * stored_num_days, 420),
        )

        # ── Disclaimer ──
        st.markdown("""
        <div class="warning-box">
            ⚠️ <strong>Disclaimer:</strong> These predictions are generated by an AI model
            for educational purposes only and do <strong>not</strong> constitute financial advice.
            Past performance does not guarantee future results. Always consult a qualified
            financial advisor before making investment decisions.
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════
#  TAB 3 – DATA EXPLORER
# ════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="section-header">
        <div class="section-header-icon">📋</div>
        <div>
            <p class="section-header-text">Data Explorer</p>
            <p class="section-header-sub">Browse, filter, and analyze raw market data</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if stock_data is None or stock_data.empty:
        st.error("❌ No data available.")
    else:
        df_exp = filter_by_window(stock_data, history_window).copy()
        df_exp.index = pd.to_datetime(df_exp.index)

        # ── Quick Stats ──
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">📅</div>
                <div class="metric-label">Total Records</div>
                <div class="metric-value">{len(df_exp):,}</div>
            </div>
            """, unsafe_allow_html=True)
        with e2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">📆</div>
                <div class="metric-label">From</div>
                <div class="metric-value" style="font-size:1.1rem;">
                    {df_exp.index[0].strftime('%b %d, %Y')}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with e3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">📆</div>
                <div class="metric-label">To</div>
                <div class="metric-value" style="font-size:1.1rem;">
                    {df_exp.index[-1].strftime('%b %d, %Y')}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with e4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">💵</div>
                <div class="metric-label">Latest Close</div>
                <div class="metric-value">
                    {fmt_currency(float(df_exp['Close'].iloc[-1]))}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Raw Data Table ──
        st.markdown("##### 📄 Raw OHLCV Data")
        df_display = df_exp.copy()
        df_display.index = df_display.index.strftime('%Y-%m-%d')
        st.dataframe(
            df_display.sort_index(ascending=False),
            use_container_width=True,
            height=400,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Returns Distribution ──
        st.markdown("""
        <div class="section-header">
            <div class="section-header-icon">📊</div>
            <div>
                <p class="section-header-text">Daily Returns Distribution</p>
                <p class="section-header-sub">Histogram of daily percentage changes</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        returns = df_exp['Close'].squeeze().pct_change().dropna() * 100

        fig_hist = go.Figure(go.Histogram(
            x=returns,
            nbinsx=80,
            marker_color='#76b900',
            marker_line_color='rgba(0,0,0,0)',
            opacity=0.85,
            name='Daily Returns',
        ))
        fig_hist.add_vline(x=0, line_dash='dash',
                           line_color='rgba(255,255,255,0.4)', line_width=1.5)
        fig_hist.add_vline(x=float(returns.mean()),
                           line_dash='dot', line_color='#00d4ff', line_width=1.5,
                           annotation_text=f'Mean: {float(returns.mean()):.3f}%',
                           annotation_font_color='#00d4ff')

        fig_hist.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text='Distribution of Daily Returns (%)',
                       font=dict(size=16, color='#fff')),
            height=350,
            bargap=0.05,
            xaxis=dict(**PLOTLY_LAYOUT['xaxis'], ticksuffix='%'),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # ── Correlation Heatmap (OHLCV) ──
        with st.expander("📐 OHLCV Correlation Matrix", expanded=False):
            corr_df = df_exp[['Open','High','Low','Close','Volume']].copy()
            corr_df.columns = ['Open','High','Low','Close','Volume']
            corr = corr_df.corr()

            fig_corr = px.imshow(
                corr,
                color_continuous_scale=[[0, '#ff006e'], [0.5, '#16213e'], [1, '#76b900']],
                zmin=-1, zmax=1,
                text_auto='.2f',
                aspect='auto',
            )
            fig_corr.update_layout(
                **PLOTLY_LAYOUT,
                title=dict(text='Correlation Matrix', font=dict(size=16, color='#fff')),
                height=350,
                coloraxis_colorbar=dict(
                    tickfont=dict(color='#a0aec0'),
                    title=dict(text='r', font=dict(color='#a0aec0')),
                ),
            )
            st.plotly_chart(fig_corr, use_container_width=True)

        # ── Download ──
        st.markdown("<br>", unsafe_allow_html=True)
        csv = df_exp.to_csv().encode('utf-8')
        st.download_button(
            label="⬇️  Download CSV",
            data=csv,
            file_name=f"NVDA_{history_window.replace(' ','_')}.csv",
            mime='text/csv',
        )

# ════════════════════════════════════════════════
#  TAB 4 – MODEL INFO
# ════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class="section-header">
        <div class="section-header-icon">🧠</div>
        <div>
            <p class="section-header-text">Model Architecture & Info</p>
            <p class="section-header-sub">Details about the LSTM forecasting model</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_info, col_arch = st.columns([1, 1])

    with col_info:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#16213e,#1a1a2e);
                    border:1px solid #0f3460; border-radius:16px; padding:1.5rem;">
            <h4 style="color:#76b900; margin-top:0;">⚡ Model Specifications</h4>
            <table style="width:100%; border-collapse:collapse; color:#e2e8f0; font-size:0.9rem;">
                <tr style="border-bottom:1px solid #0f3460;">
                    <td style="padding:10px 0; color:#a0aec0; font-weight:600;">Architecture</td>
                    <td style="padding:10px 0; text-align:right;">LSTM (Long Short-Term Memory)</td>
                </tr>
                <tr style="border-bottom:1px solid #0f3460;">
                    <td style="padding:10px 0; color:#a0aec0; font-weight:600;">LSTM Units</td>
                    <td style="padding:10px 0; text-align:right; color:#76b900;">150</td>
                </tr>
                <tr style="border-bottom:1px solid #0f3460;">
                    <td style="padding:10px 0; color:#a0aec0; font-weight:600;">Look-back Window</td>
                    <td style="padding:10px 0; text-align:right; color:#76b900;">5 days</td>
                </tr>
                <tr style="border-bottom:1px solid #0f3460;">
                    <td style="padding:10px 0; color:#a0aec0; font-weight:600;">RMSE</td>
                    <td style="padding:10px 0; text-align:right; color:#00e676;">$1.32</td>
                </tr>
                <tr style="border-bottom:1px solid #0f3460;">
                    <td style="padding:10px 0; color:#a0aec0; font-weight:600;">Ticker</td>
                    <td style="padding:10px 0; text-align:right;">NVDA (NASDAQ)</td>
                </tr>
                <tr style="border-bottom:1px solid #0f3460;">
                    <td style="padding:10px 0; color:#a0aec0; font-weight:600;">Feature</td>
                    <td style="padding:10px 0; text-align:right;">Adjusted Close Price</td>
                </tr>
                <tr style="border-bottom:1px solid #0f3460;">
                    <td style="padding:10px 0; color:#a0aec0; font-weight:600;">Normalisation</td>
                    <td style="padding:10px 0; text-align:right;">MinMax [0, 1]</td>
                </tr>
                <tr>
                    <td style="padding:10px 0; color:#a0aec0; font-weight:600;">Model Status</td>
                    <td style="padding:10px 0; text-align:right;">
                        <span style="color:#00e676; font-weight:700;">
                            {'✅ Loaded' if model else '❌ Not Loaded'}
                        </span>
                    </td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col_arch:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#16213e,#1a1a2e);
                    border:1px solid #0f3460; border-radius:16px; padding:1.5rem; height:100%;">
            <h4 style="color:#76b900; margin-top:0;">🏗️ Architecture Diagram</h4>
        """, unsafe_allow_html=True)

        layers = [
            ("Input Layer",   "Sequence: (batch, 5, 1)",    "#00d4ff"),
            ("LSTM Layer",    "150 units, return_seq=False", "#76b900"),
            ("Dense Layer",   "1 unit (price output)",       "#ff006e"),
            ("Output",        "Next-day close price ($)",    "#ffd600"),
        ]
        for name, desc, color in layers:
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.03); border:1px solid {color}33;
                        border-left:4px solid {color}; border-radius:10px;
                        padding:0.8rem 1rem; margin-bottom:0.6rem;">
                <div style="color:{color}; font-weight:700; font-size:0.9rem;">{name}</div>
                <div style="color:#a0aec0; font-size:0.8rem; margin-top:2px;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── How It Works ──
    st.markdown("""
    <div class="section-header">
        <div class="section-header-icon">❓</div>
        <div>
            <p class="section-header-text">How It Works</p>
            <p class="section-header-sub">Step-by-step prediction pipeline</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("1", "Data Collection",   "Fetches complete NVDA historical data from Yahoo Finance in real-time.", "#76b900"),
        ("2", "Normalisation",     "Close prices are scaled to [0,1] using MinMaxScaler before model inference.", "#00d4ff"),
        ("3", "Sequence Input",    "The last 5 trading day prices form the input sequence (look-back window).", "#ffd600"),
        ("4", "LSTM Inference",    "The 150-unit LSTM processes the sequence and outputs the next price.", "#ff006e"),
        ("5", "Iterative Forecast","Each prediction becomes part of the next input sequence (autoregression).", "#76b900"),
        ("6", "Denormalisation",   "Predictions are inverse-transformed back to actual dollar prices.", "#00d4ff"),
    ]

    cols = st.columns(3)
    for i, (num, title, desc, color) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="metric-card" style="text-align:left; padding:1.2rem;">
                <div style="width:32px; height:32px; border-radius:50%;
                            background:{color}22; border:2px solid {color};
                            display:flex; align-items:center; justify-content:center;
                            color:{color}; font-weight:800; font-size:0.9rem;
                            margin-bottom:0.7rem;">{num}</div>
                <div style="color:#fff; font-weight:700; font-size:0.9rem;
                            margin-bottom:0.4rem;">{title}</div>
                <div style="color:#a0aec0; font-size:0.8rem; line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  FOOTER
# ──────────────────────────────────────────────
st.markdown(f"""
<div class="custom-footer">
    <strong style="color:#76b900;">NVIDIA Stock Price Predictor</strong> &nbsp;·&nbsp;
    Powered by LSTM Deep Learning &nbsp;·&nbsp;
    Data via Yahoo Finance &nbsp;·&nbsp;
    Built with Streamlit
    <br><br>
    <span>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC</span>
    <br>
    <span>⚠️ For educational purposes only. Not financial advice.</span>
</div>
""", unsafe_allow_html=True)
