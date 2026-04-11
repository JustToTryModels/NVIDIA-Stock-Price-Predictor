# app.py
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

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

    /* ── Root & Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        padding: 2rem 3rem 3rem 3rem;
        max-width: 1400px;
    }

    /* ── Hide Streamlit Branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Sidebar Styling ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        border-right: 1px solid rgba(118, 185, 0, 0.2);
    }

    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #76b900 !important;
    }

    /* ── Main Background ── */
    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 50%, #0a0f1a 100%);
        color: #e0e0e0;
    }

    /* ── Hero Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a3e 40%, #0d1f3c 100%);
        border: 1px solid rgba(118, 185, 0, 0.3);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255,255,255,0.05);
    }

    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(118, 185, 0, 0.08) 0%, transparent 70%);
        pointer-events: none;
    }

    .hero-banner::after {
        content: '';
        position: absolute;
        bottom: -30%;
        left: -5%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(0, 100, 255, 0.06) 0%, transparent 70%);
        pointer-events: none;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #76b900, #a8e063, #76b900);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.1;
        animation: shimmer 3s linear infinite;
    }

    @keyframes shimmer {
        to { background-position: 200% center; }
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #8899aa;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(118, 185, 0, 0.15);
        border: 1px solid rgba(118, 185, 0, 0.4);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #76b900;
        margin-top: 1rem;
    }

    /* ── Metric Cards ── */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(118, 185, 0, 0.2);
        border-radius: 16px;
        padding: 1.4rem 1.2rem;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.3);
        border-color: rgba(118, 185, 0, 0.5);
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #76b900, #a8e063);
        border-radius: 16px 16px 0 0;
    }

    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #667788;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
    }

    .metric-value {
        font-size: 1.6rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1;
    }

    .metric-delta {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }

    .metric-delta.positive { color: #76b900; }
    .metric-delta.negative { color: #ff4b4b; }

    .metric-icon {
        font-size: 1.5rem;
        position: absolute;
        top: 1.2rem;
        right: 1.2rem;
        opacity: 0.6;
    }

    /* ── Section Headers ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1.2rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(118, 185, 0, 0.15);
    }

    .section-header h2 {
        font-size: 1.3rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }

    .section-icon {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #76b900, #a8e063);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        flex-shrink: 0;
    }

    /* ── Predict Button ── */
    .stButton > button {
        background: linear-gradient(90deg, #76b900 0%, #a8e063 50%, #76b900 100%);
        background-size: 200% auto;
        color: #0a0a0f !important;
        border: none;
        border-radius: 14px;
        padding: 14px 32px;
        font-size: 1rem !important;
        font-weight: 700 !important;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        letter-spacing: 0.02em;
        box-shadow: 0 4px 15px rgba(118, 185, 0, 0.3);
        text-transform: uppercase;
    }

    .stButton > button:hover {
        background-position: right center;
        box-shadow: 0 6px 25px rgba(118, 185, 0, 0.5);
        transform: translateY(-2px);
    }

    .stButton > button:active {
        transform: translateY(0px);
    }

    /* ── Slider ── */
    .stSlider [data-testid="stThumbValue"] {
        background: #76b900 !important;
        color: white !important;
    }

    /* ── Cards / Containers ── */
    .content-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(118, 185, 0, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }

    /* ── Prediction Table ── */
    .pred-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin-top: 1rem;
    }

    .pred-table thead tr {
        background: linear-gradient(90deg, rgba(118,185,0,0.2), rgba(118,185,0,0.1));
    }

    .pred-table th {
        padding: 12px 16px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #76b900;
        text-align: left;
        border-bottom: 1px solid rgba(118,185,0,0.2);
    }

    .pred-table td {
        padding: 12px 16px;
        font-size: 0.9rem;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        color: #d0d0d0;
    }

    .pred-table tr:hover td {
        background: rgba(118,185,0,0.05);
    }

    .pred-table tr:last-child td {
        border-bottom: none;
    }

    .price-positive { color: #76b900; font-weight: 600; }
    .price-negative { color: #ff4b4b; font-weight: 600; }

    /* ── Info Box ── */
    .info-box {
        background: rgba(118, 185, 0, 0.08);
        border: 1px solid rgba(118, 185, 0, 0.25);
        border-left: 4px solid #76b900;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        font-size: 0.88rem;
        color: #b0c4b8;
        line-height: 1.6;
    }

    .warning-box {
        background: rgba(255, 165, 0, 0.08);
        border: 1px solid rgba(255, 165, 0, 0.25);
        border-left: 4px solid #ffa500;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        font-size: 0.88rem;
        color: #c4b8a0;
        line-height: 1.6;
    }

    /* ── Status Indicators ── */
    .status-live {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #76b900;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        background: #76b900;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.8); }
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        border: 1px solid rgba(118,185,0,0.1);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 20px;
        font-weight: 600;
        font-size: 0.88rem;
        color: #667788 !important;
        background: transparent;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(118,185,0,0.2), rgba(168,224,99,0.1)) !important;
        color: #76b900 !important;
        border: 1px solid rgba(118,185,0,0.3) !important;
    }

    /* ── Dataframe ── */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(118,185,0,0.15);
    }

    /* ── Sidebar Elements ── */
    .sidebar-section {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(118,185,0,0.15);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .sidebar-logo {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 1px solid rgba(118,185,0,0.15);
        margin-bottom: 1rem;
    }

    /* ── Divider ── */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(118,185,0,0.3), transparent);
        margin: 2rem 0;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0f0f1a; }
    ::-webkit-scrollbar-thumb { background: #76b900; border-radius: 3px; }

    /* ── Responsive metric grid ── */
    @media (max-width: 768px) {
        .metric-grid { grid-template-columns: repeat(2, 1fr); }
        .hero-title { font-size: 2rem; }
        .main .block-container { padding: 1rem; }
    }
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
        return None


# ==============================
# 🔹 Fetch Stock Data (Cached)
# ==============================
@st.cache_data(ttl=300)
def get_stock_data(ticker='NVDA'):
    try:
        data = yf.download(ticker, period='max', auto_adjust=True)
        info = yf.Ticker(ticker).info
        return data, info
    except Exception as e:
        return None, None


# ==============================
# 🔹 Generate Business Days
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
        prediction = model.predict(X_input, verbose=0)
        predictions.append(prediction[0, 0])
        last_sequence = np.append(last_sequence[1:], prediction, axis=0)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions


# ==============================
# 🔹 Helper: Format Large Numbers
# ==============================
def format_large_number(num):
    if num is None:
        return "N/A"
    try:
        num = float(num)
        if num >= 1e12:
            return f"${num/1e12:.2f}T"
        elif num >= 1e9:
            return f"${num/1e9:.2f}B"
        elif num >= 1e6:
            return f"${num/1e6:.2f}M"
        else:
            return f"${num:,.2f}"
    except:
        return "N/A"


# ==============================
# 🔹 Plotly Theme Config
# ==============================
PLOTLY_THEME = {
    "paper_bgcolor": "rgba(15,15,26,0)",
    "plot_bgcolor": "rgba(15,15,26,0)",
    "font_color": "#a0aabb",
    "gridcolor": "rgba(255,255,255,0.05)",
    "zerolinecolor": "rgba(255,255,255,0.08)",
}


# ──────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/NVIDIA_logo.svg/1920px-NVIDIA_logo.svg.png"
             width="180" style="filter: brightness(1.1);">
        <p style="color:#667788; font-size:0.75rem; margin-top:8px; margin-bottom:0;">
            AI-Powered Stock Prediction
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <p style="font-size:0.7rem; font-weight:700; text-transform:uppercase;
                  letter-spacing:0.1em; color:#76b900; margin:0 0 8px 0;">
            ⚙️ Forecast Settings
        </p>
    """, unsafe_allow_html=True)

    num_days = st.slider(
        "Business Days to Forecast",
        min_value=1, max_value=30, value=5, step=1,
        help="Select how many trading days ahead to predict."
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # Chart options
    st.markdown("""
    <div class="sidebar-section">
        <p style="font-size:0.7rem; font-weight:700; text-transform:uppercase;
                  letter-spacing:0.1em; color:#76b900; margin:0 0 8px 0;">
            📊 Chart Options
        </p>
    """, unsafe_allow_html=True)

    chart_type = st.selectbox(
        "Historical Chart Type",
        ["Candlestick", "Line", "Area"],
        help="Choose how to display the historical price data."
    )

    history_period = st.selectbox(
        "History Window",
        ["3 Months", "6 Months", "1 Year", "2 Years", "5 Years", "All Time"],
        index=2
    )

    show_volume = st.toggle("Show Volume", value=True)
    show_ma = st.toggle("Show Moving Averages", value=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Model info
    st.markdown("""
    <div class="sidebar-section">
        <p style="font-size:0.7rem; font-weight:700; text-transform:uppercase;
                  letter-spacing:0.1em; color:#76b900; margin:0 0 10px 0;">
            🧠 Model Info
        </p>
        <table style="width:100%; font-size:0.78rem; border-collapse:collapse;">
            <tr>
                <td style="color:#667788; padding:3px 0;">Architecture</td>
                <td style="color:#d0d0d0; text-align:right;">LSTM</td>
            </tr>
            <tr>
                <td style="color:#667788; padding:3px 0;">Look-back</td>
                <td style="color:#d0d0d0; text-align:right;">5 Days</td>
            </tr>
            <tr>
                <td style="color:#667788; padding:3px 0;">Units</td>
                <td style="color:#d0d0d0; text-align:right;">150</td>
            </tr>
            <tr>
                <td style="color:#667788; padding:3px 0;">RMSE</td>
                <td style="color:#76b900; text-align:right; font-weight:700;">1.32</td>
            </tr>
            <tr>
                <td style="color:#667788; padding:3px 0;">Ticker</td>
                <td style="color:#d0d0d0; text-align:right;">NVDA</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="warning-box" style="font-size:0.72rem; margin-top:0.5rem;">
        ⚠️ <strong>Disclaimer:</strong> Predictions are for educational purposes only.
        Do not use as financial advice. Past performance does not guarantee future results.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <p style="text-align:center; color:#334455; font-size:0.7rem; margin-top:1rem;">
        Last updated: {datetime.now().strftime('%b %d, %Y %H:%M')}
    </p>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────────────
stock = 'NVDA'

# ── Hero Banner ──
st.markdown(f"""
<div class="hero-banner">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:1rem;">
        <div>
            <h1 class="hero-title">NVIDIA Stock Predictor</h1>
            <p class="hero-subtitle">
                Deep Learning–powered forecasting using Long Short-Term Memory neural networks
            </p>
            <div style="display:flex; gap:10px; flex-wrap:wrap; margin-top:1rem;">
                <div class="hero-badge">🟢 NASDAQ: NVDA</div>
                <div class="hero-badge">🧠 LSTM Model</div>
                <div class="hero-badge">📅 Forecasting {num_days} Days</div>
            </div>
        </div>
        <div style="text-align:right;">
            <div class="status-live">
                <div class="status-dot"></div>
                LIVE DATA
            </div>
            <p style="color:#445566; font-size:0.78rem; margin-top:6px;">
                {datetime.now().strftime('%A, %B %d %Y')}
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Load model & data ──
model = load_nvidia_model()
stock_data, stock_info = get_stock_data(stock)

# ── Model status banner ──
if model is None:
    st.markdown("""
    <div style="background:rgba(255,75,75,0.1); border:1px solid rgba(255,75,75,0.3);
                border-left:4px solid #ff4b4b; border-radius:8px; padding:1rem 1.2rem; margin-bottom:1rem;">
        ❌ <strong style="color:#ff4b4b;">Model not loaded.</strong>
        <span style="color:#c0a0a0; font-size:0.88rem;"> Ensure the LSTM model file exists at
        <code>LSTM_Model/NVIDIA_LSTM_LB(5)_U(150)_RMSE(1.32).keras</code></span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="info-box">
        ✅ <strong>Model loaded successfully.</strong>
        LSTM network with 150 units and 5-day look-back window is ready for inference.
    </div>
    """, unsafe_allow_html=True)

# ── Key Metrics ──
if stock_data is not None and not stock_data.empty:
    try:
        latest_close = float(stock_data['Close'].iloc[-1])
        prev_close   = float(stock_data['Close'].iloc[-2])
        daily_chg    = latest_close - prev_close
        daily_pct    = (daily_chg / prev_close) * 100
        high_52w     = float(stock_data['Close'].tail(252).max())
        low_52w      = float(stock_data['Close'].tail(252).min())
        mkt_cap      = stock_info.get('marketCap') if stock_info else None
        volume       = stock_info.get('volume') if stock_info else None

        delta_class  = "positive" if daily_chg >= 0 else "negative"
        delta_arrow  = "▲" if daily_chg >= 0 else "▼"

        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-icon">💵</div>
                <div class="metric-label">Current Price</div>
                <div class="metric-value">${latest_close:,.2f}</div>
                <div class="metric-delta {delta_class}">
                    {delta_arrow} {abs(daily_chg):.2f} ({abs(daily_pct):.2f}%) today
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-icon">📈</div>
                <div class="metric-label">52-Week High</div>
                <div class="metric-value">${high_52w:,.2f}</div>
                <div class="metric-delta {'positive' if latest_close < high_52w else 'negative'}">
                    {'↓ ' + f"{((high_52w - latest_close)/high_52w)*100:.1f}% below peak"
                     if latest_close < high_52w else '✓ At 52-week high'}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-icon">📉</div>
                <div class="metric-label">52-Week Low</div>
                <div class="metric-value">${low_52w:,.2f}</div>
                <div class="metric-delta positive">
                    ↑ {((latest_close - low_52w)/low_52w)*100:.1f}% above low
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-icon">🏦</div>
                <div class="metric-label">Market Cap</div>
                <div class="metric-value">{format_large_number(mkt_cap)}</div>
                <div class="metric-delta positive">NASDAQ Listed</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        pass


# ──────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None


# ──────────────────────────────────────────────────────
# PREDICT BUTTON
# ──────────────────────────────────────────────────────
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_clicked = st.button(
        f"🚀 Predict Next {num_days} Business Day{'s' if num_days > 1 else ''}",
        key='forecast-button',
        use_container_width=True
    )

if predict_clicked:
    if model is None:
        st.error("❌ Model not loaded. Cannot make predictions.")
    elif stock_data is None or stock_data.empty:
        st.error("❌ Failed to load stock data.")
    else:
        with st.spinner("🧠 Running LSTM inference..."):
            close_prices = stock_data['Close'].values.reshape(-1, 1)
            dates = stock_data.index

            predictions = predict_next_business_days(
                model, close_prices, look_back=5, days=num_days
            )

            last_date = dates[-1]
            prediction_dates = generate_business_days(
                last_date + timedelta(days=1), num_days
            )

            st.session_state.prediction_results = {
                'stock_data': stock_data,
                'close_prices': close_prices,
                'dates': dates,
                'predictions': predictions,
                'prediction_dates': prediction_dates,
                'num_days': num_days,
                'stock': stock,
                'chart_type': chart_type,
                'history_period': history_period,
                'show_volume': show_volume,
                'show_ma': show_ma,
            }

        st.success(f"✅ Forecast complete! Predicted {num_days} trading day(s) ahead.")


# ──────────────────────────────────────────────────────
# RESULTS
# ──────────────────────────────────────────────────────
if st.session_state.prediction_results is not None:
    r = st.session_state.prediction_results

    stock_data_r    = r['stock_data']
    close_prices    = r['close_prices']
    dates           = r['dates']
    predictions     = r['predictions']
    prediction_dates = r['prediction_dates']
    stored_num_days = r['num_days']
    stored_stock    = r['stock']
    c_type          = r.get('chart_type', 'Line')
    h_period        = r.get('history_period', '1 Year')
    s_volume        = r.get('show_volume', True)
    s_ma            = r.get('show_ma', True)

    # ── Filter history window ──
    period_map = {
        "3 Months": 63, "6 Months": 126, "1 Year": 252,
        "2 Years": 504, "5 Years": 1260, "All Time": len(stock_data_r)
    }
    window = period_map.get(h_period, 252)
    plot_data = stock_data_r.iloc[-window:]

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════
    # TABS
    # ═══════════════════════════════════════
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Price Charts", "🔮 Forecast", "📋 Data Table", "📈 Analytics"
    ])

    # ─────────────────────────────────────
    # TAB 1 – HISTORICAL CHARTS
    # ─────────────────────────────────────
    with tab1:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">📊</div>
            <h2>Historical Price Chart</h2>
        </div>
        """, unsafe_allow_html=True)

        # Build figure
        rows = 2 if s_volume else 1
        row_heights = [0.7, 0.3] if s_volume else [1.0]

        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            row_heights=row_heights,
            vertical_spacing=0.04
        )

        # ── Price trace ──
        if c_type == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=plot_data.index,
                open=plot_data['Open'].squeeze(),
                high=plot_data['High'].squeeze(),
                low=plot_data['Low'].squeeze(),
                close=plot_data['Close'].squeeze(),
                name="OHLC",
                increasing_line_color='#76b900',
                decreasing_line_color='#ff4b4b',
                increasing_fillcolor='rgba(118,185,0,0.6)',
                decreasing_fillcolor='rgba(255,75,75,0.6)',
            ), row=1, col=1)

        elif c_type == "Area":
            fig.add_trace(go.Scatter(
                x=plot_data.index,
                y=plot_data['Close'].squeeze(),
                name="Close Price",
                line=dict(color='#76b900', width=2),
                fill='tozeroy',
                fillcolor='rgba(118,185,0,0.08)',
                hovertemplate='<b>%{x|%b %d, %Y}</b><br>Price: $%{y:,.2f}<extra></extra>'
            ), row=1, col=1)

        else:  # Line
            fig.add_trace(go.Scatter(
                x=plot_data.index,
                y=plot_data['Close'].squeeze(),
                name="Close Price",
                line=dict(color='#76b900', width=2),
                hovertemplate='<b>%{x|%b %d, %Y}</b><br>Price: $%{y:,.2f}<extra></extra>'
            ), row=1, col=1)

        # ── Moving Averages ──
        if s_ma and c_type != "Candlestick":
            for period, color, dash in [(20, '#f0a500', 'solid'), (50, '#00bfff', 'dot')]:
                ma = plot_data['Close'].squeeze().rolling(period).mean()
                fig.add_trace(go.Scatter(
                    x=plot_data.index, y=ma,
                    name=f"MA{period}",
                    line=dict(color=color, width=1.2, dash=dash),
                    opacity=0.8,
                    hovertemplate=f'MA{period}: $%{{y:,.2f}}<extra></extra>'
                ), row=1, col=1)
        elif s_ma and c_type == "Candlestick":
            for period, color, dash in [(20, '#f0a500', 'solid'), (50, '#00bfff', 'dot')]:
                ma = plot_data['Close'].squeeze().rolling(period).mean()
                fig.add_trace(go.Scatter(
                    x=plot_data.index, y=ma,
                    name=f"MA{period}",
                    line=dict(color=color, width=1.5, dash=dash),
                    opacity=0.9,
                ), row=1, col=1)

        # ── Volume ──
        if s_volume and rows == 2:
            colors = [
                '#76b900' if float(plot_data['Close'].squeeze().iloc[i]) >=
                             float(plot_data['Open'].squeeze().iloc[i])
                else '#ff4b4b'
                for i in range(len(plot_data))
            ]
            fig.add_trace(go.Bar(
                x=plot_data.index,
                y=plot_data['Volume'].squeeze(),
                name="Volume",
                marker_color=colors,
                opacity=0.6,
                hovertemplate='<b>%{x|%b %d, %Y}</b><br>Volume: %{y:,.0f}<extra></extra>'
            ), row=2, col=1)

        fig.update_layout(
            paper_bgcolor=PLOTLY_THEME["paper_bgcolor"],
            plot_bgcolor=PLOTLY_THEME["plot_bgcolor"],
            font=dict(color=PLOTLY_THEME["font_color"], family='Inter'),
            height=520,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(118,185,0,0.2)", borderwidth=1
            ),
            hovermode='x unified',
            xaxis_rangeslider_visible=False,
        )

        for row_i in range(1, rows + 1):
            fig.update_xaxes(
                showgrid=True, gridcolor=PLOTLY_THEME["gridcolor"],
                zeroline=False, linecolor="rgba(255,255,255,0.05)",
                row=row_i, col=1
            )
            fig.update_yaxes(
                showgrid=True, gridcolor=PLOTLY_THEME["gridcolor"],
                zeroline=False, linecolor="rgba(255,255,255,0.05)",
                row=row_i, col=1
            )

        if s_volume:
            fig.update_yaxes(title_text="Price (USD)", row=1, col=1,
                             title_font=dict(size=11))
            fig.update_yaxes(title_text="Volume", row=2, col=1,
                             title_font=dict(size=11))

        st.plotly_chart(fig, use_container_width=True)


    # ─────────────────────────────────────
    # TAB 2 – FORECAST
    # ─────────────────────────────────────
    with tab2:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">🔮</div>
            <h2>LSTM Price Forecast</h2>
        </div>
        """, unsafe_allow_html=True)

        pred_vals = predictions.flatten()
        current_price = float(close_prices[-1])

        # ── Forecast summary metrics ──
        max_pred = float(np.max(pred_vals))
        min_pred = float(np.min(pred_vals))
        avg_pred = float(np.mean(pred_vals))
        final_pred = float(pred_vals[-1])
        overall_chg = final_pred - current_price
        overall_pct = (overall_chg / current_price) * 100

        c1, c2, c3, c4 = st.columns(4)
        cards = [
            (c1, "Final Forecast", f"${final_pred:,.2f}",
             f"{'▲' if overall_chg>=0 else '▼'} {abs(overall_pct):.2f}% vs today",
             "positive" if overall_chg >= 0 else "negative"),
            (c2, "Forecast High",  f"${max_pred:,.2f}",
             f"Peak predicted price", "positive"),
            (c3, "Forecast Low",   f"${min_pred:,.2f}",
             f"Trough predicted price", "negative"),
            (c4, "Forecast Avg",   f"${avg_pred:,.2f}",
             f"Mean over {stored_num_days} day(s)", "positive"),
        ]
        for col, label, val, delta, cls in cards:
            with col:
                st.markdown(f"""
                <div class="metric-card" style="margin-bottom:1rem;">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="font-size:1.3rem;">{val}</div>
                    <div class="metric-delta {cls}">{delta}</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Combined Chart ──
        context_window = min(60, len(dates))
        hist_dates_ctx = dates[-context_window:]
        hist_price_ctx = close_prices[-context_window:].flatten()

        fig2 = go.Figure()

        # Historical context
        fig2.add_trace(go.Scatter(
            x=hist_dates_ctx,
            y=hist_price_ctx,
            name="Historical",
            line=dict(color='#76b900', width=2),
            hovertemplate='<b>%{x|%b %d, %Y}</b><br>Price: $%{y:,.2f}<extra></extra>'
        ))

        # Bridge line
        bridge_x = [hist_dates_ctx[-1], prediction_dates[0]]
        bridge_y = [hist_price_ctx[-1], pred_vals[0]]
        fig2.add_trace(go.Scatter(
            x=bridge_x, y=bridge_y,
            mode='lines',
            line=dict(color='rgba(255,165,0,0.5)', width=1.5, dash='dot'),
            showlegend=False, hoverinfo='skip'
        ))

        # Confidence band (±3% simple heuristic)
        upper = pred_vals * 1.03
        lower = pred_vals * 0.97
        fig2.add_trace(go.Scatter(
            x=list(prediction_dates) + list(prediction_dates[::-1]),
            y=list(upper) + list(lower[::-1]),
            fill='toself',
            fillcolor='rgba(255,165,0,0.07)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Band (±3%)',
            hoverinfo='skip'
        ))

        # Prediction line
        fig2.add_trace(go.Scatter(
            x=prediction_dates,
            y=pred_vals,
            name="Predicted",
            mode='lines+markers',
            line=dict(color='#ffa500', width=2.5, dash='dash'),
            marker=dict(size=8, color='#ffa500',
                        line=dict(color='#ffffff', width=1.5)),
            hovertemplate='<b>%{x|%b %d, %Y}</b><br>Predicted: $%{y:,.2f}<extra></extra>'
        ))

        # Today marker
        fig2.add_vline(
            x=hist_dates_ctx[-1], line_dash="dot",
            line_color="rgba(255,255,255,0.2)", line_width=1.5
        )
        fig2.add_annotation(
            x=hist_dates_ctx[-1], y=hist_price_ctx[-1],
            text="Today", showarrow=True, arrowhead=2,
            arrowcolor="#76b900", font=dict(color="#76b900", size=11),
            bgcolor="rgba(0,0,0,0.5)", bordercolor="#76b900", borderwidth=1,
            ax=30, ay=-30
        )

        fig2.update_layout(
            paper_bgcolor=PLOTLY_THEME["paper_bgcolor"],
            plot_bgcolor=PLOTLY_THEME["plot_bgcolor"],
            font=dict(color=PLOTLY_THEME["font_color"], family='Inter'),
            height=420,
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,165,0,0.2)", borderwidth=1
            ),
            hovermode='x unified',
            xaxis=dict(showgrid=True, gridcolor=PLOTLY_THEME["gridcolor"]),
            yaxis=dict(showgrid=True, gridcolor=PLOTLY_THEME["gridcolor"],
                       title="Price (USD)")
        )

        st.plotly_chart(fig2, use_container_width=True)

        # ── Day-by-day prediction chart ──
        st.markdown("""
        <div class="section-header" style="margin-top:1rem;">
            <div class="section-icon">📅</div>
            <h2>Day-by-Day Forecast</h2>
        </div>
        """, unsafe_allow_html=True)

        day_labels = [pd.Timestamp(d).strftime('%a %b %d') for d in prediction_dates]
        prev_prices = [current_price] + list(pred_vals[:-1])
        bar_colors = ['#76b900' if pred_vals[i] >= prev_prices[i] else '#ff4b4b'
                      for i in range(len(pred_vals))]

        fig3 = go.Figure()

        fig3.add_trace(go.Bar(
            x=day_labels,
            y=pred_vals,
            marker_color=bar_colors,
            marker_line_color='rgba(255,255,255,0.1)',
            marker_line_width=1,
            name='Predicted Price',
            hovertemplate='<b>%{x}</b><br>Price: $%{y:,.2f}<extra></extra>',
            text=[f'${v:,.2f}' for v in pred_vals],
            textposition='outside',
            textfont=dict(size=11, color='#d0d0d0')
        ))

        fig3.add_hline(
            y=current_price,
            line_dash="dot", line_color="rgba(255,255,255,0.3)", line_width=1.5,
            annotation_text=f"Current: ${current_price:,.2f}",
            annotation_position="top right",
            annotation_font=dict(color="rgba(255,255,255,0.5)", size=10)
        )

        fig3.update_layout(
            paper_bgcolor=PLOTLY_THEME["paper_bgcolor"],
            plot_bgcolor=PLOTLY_THEME["plot_bgcolor"],
            font=dict(color=PLOTLY_THEME["font_color"], family='Inter'),
            height=340,
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=False,
            xaxis=dict(showgrid=False),
            yaxis=dict(
                showgrid=True, gridcolor=PLOTLY_THEME["gridcolor"],
                title="Price (USD)",
                range=[min(pred_vals)*0.97, max(pred_vals)*1.05]
            )
        )

        st.plotly_chart(fig3, use_container_width=True)

        # ── Prediction table ──
        st.markdown("""
        <div class="section-header" style="margin-top:0.5rem;">
            <div class="section-icon">📋</div>
            <h2>Forecast Detail</h2>
        </div>
        """, unsafe_allow_html=True)

        rows_html = ""
        prev = current_price
        for i, (d, p) in enumerate(zip(prediction_dates, pred_vals)):
            chg = p - prev
            pct = (chg / prev) * 100
            arrow = "▲" if chg >= 0 else "▼"
            cls   = "price-positive" if chg >= 0 else "price-negative"
            rows_html += f"""
            <tr>
                <td style="color:#667788;">{i+1}</td>
                <td><strong>{pd.Timestamp(d).strftime('%A, %B %d %Y')}</strong></td>
                <td><strong>${p:,.2f}</strong></td>
                <td class="{cls}">{arrow} ${abs(chg):.2f}</td>
                <td class="{cls}">{arrow} {abs(pct):.2f}%</td>
                <td>{'🟢 Bullish' if chg >= 0 else '🔴 Bearish'}</td>
            </tr>
            """
            prev = p

        st.markdown(f"""
        <div class="content-card">
            <table class="pred-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Date</th>
                        <th>Predicted Price</th>
                        <th>Change ($)</th>
                        <th>Change (%)</th>
                        <th>Signal</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # ── Download button ──
        pred_df = pd.DataFrame({
            'Date': [pd.Timestamp(d).strftime('%Y-%m-%d') for d in prediction_dates],
            'Predicted_Price': pred_vals.round(2)
        })
        csv = pred_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Forecast CSV",
            data=csv,
            file_name=f"NVDA_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )


    # ─────────────────────────────────────
    # TAB 3 – DATA TABLE
    # ─────────────────────────────────────
    with tab3:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">📋</div>
            <h2>Historical OHLCV Data</h2>
        </div>
        """, unsafe_allow_html=True)

        col_f1, col_f2 = st.columns([3, 1])
        with col_f1:
            search_rows = st.slider(
                "Rows to display", 10, 500, 100, step=10,
                key='row_slider'
            )
        with col_f2:
            sort_order = st.selectbox("Sort", ["Newest First", "Oldest First"],
                                      key='sort_order')

        display_df = stock_data_r.copy()
        display_df = display_df.sort_index(
            ascending=(sort_order == "Oldest First")
        ).head(search_rows)

        # Round for display
        for col in ['Open','High','Low','Close']:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(2)

        st.dataframe(
            display_df,
            use_container_width=True,
            height=420
        )

        csv2 = stock_data_r.to_csv().encode('utf-8')
        st.download_button(
            "⬇️ Download Full Historical Data",
            data=csv2,
            file_name=f"NVDA_historical_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv',
            key='hist_download'
        )


    # ─────────────────────────────────────
    # TAB 4 – ANALYTICS
    # ─────────────────────────────────────
    with tab4:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">📈</div>
            <h2>Price Analytics</h2>
        </div>
        """, unsafe_allow_html=True)

        analysis_data = stock_data_r['Close'].squeeze()

        col_a1, col_a2 = st.columns(2)

        # ── Returns Distribution ──
        with col_a1:
            st.markdown("**📊 Daily Returns Distribution (1Y)**")
            returns_1y = analysis_data.tail(252).pct_change().dropna() * 100

            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=returns_1y,
                nbinsx=60,
                marker_color='#76b900',
                marker_line_color='rgba(0,0,0,0.3)',
                marker_line_width=0.5,
                opacity=0.85,
                name="Daily Returns",
                hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
            ))
            fig_hist.add_vline(
                x=float(returns_1y.mean()),
                line_dash="dash", line_color="#ffa500",
                annotation_text=f"Mean: {returns_1y.mean():.2f}%",
                annotation_font=dict(color="#ffa500", size=10)
            )
            fig_hist.update_layout(
                paper_bgcolor=PLOTLY_THEME["paper_bgcolor"],
                plot_bgcolor=PLOTLY_THEME["plot_bgcolor"],
                font=dict(color=PLOTLY_THEME["font_color"], family='Inter'),
                height=300,
                margin=dict(l=5, r=5, t=10, b=5),
                showlegend=False,
                xaxis=dict(title="Return (%)", showgrid=True,
                           gridcolor=PLOTLY_THEME["gridcolor"]),
                yaxis=dict(title="Count", showgrid=True,
                           gridcolor=PLOTLY_THEME["gridcolor"])
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # ── Rolling Volatility ──
        with col_a2:
            st.markdown("**📉 30-Day Rolling Volatility**")
            vol_30 = analysis_data.pct_change().rolling(30).std() * np.sqrt(252) * 100
            vol_plot = vol_30.tail(252)

            fig_vol = go.Figure()
            fig_vol.add_trace(go.Scatter(
                x=vol_plot.index,
                y=vol_plot.values,
                fill='tozeroy',
                fillcolor='rgba(255,75,75,0.1)',
                line=dict(color='#ff4b4b', width=2),
                name='Annualised Vol',
                hovertemplate='<b>%{x|%b %d, %Y}</b><br>Vol: %{y:.1f}%<extra></extra>'
            ))
            fig_vol.update_layout(
                paper_bgcolor=PLOTLY_THEME["paper_bgcolor"],
                plot_bgcolor=PLOTLY_THEME["plot_bgcolor"],
                font=dict(color=PLOTLY_THEME["font_color"], family='Inter'),
                height=300,
                margin=dict(l=5, r=5, t=10, b=5),
                showlegend=False,
                xaxis=dict(showgrid=True, gridcolor=PLOTLY_THEME["gridcolor"]),
                yaxis=dict(title="Volatility (%)", showgrid=True,
                           gridcolor=PLOTLY_THEME["gridcolor"])
            )
            st.plotly_chart(fig_vol, use_container_width=True)

        # ── Cumulative Return ──
        st.markdown("**📈 Cumulative Return Over Time**")
        cum_ret_data = analysis_data.pct_change().dropna()
        cum_ret = (1 + cum_ret_data).cumprod() * 100 - 100

        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=cum_ret.index,
            y=cum_ret.values,
            fill='tozeroy',
            fillcolor='rgba(118,185,0,0.08)',
            line=dict(color='#76b900', width=2),
            name='Cumulative Return',
            hovertemplate='<b>%{x|%b %d, %Y}</b><br>Return: %{y:.1f}%<extra></extra>'
        ))
        fig_cum.update_layout(
            paper_bgcolor=PLOTLY_THEME["paper_bgcolor"],
            plot_bgcolor=PLOTLY_THEME["plot_bgcolor"],
            font=dict(color=PLOTLY_THEME["font_color"], family='Inter'),
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            hovermode='x unified',
            xaxis=dict(showgrid=True, gridcolor=PLOTLY_THEME["gridcolor"]),
            yaxis=dict(title="Cumulative Return (%)",
                       showgrid=True, gridcolor=PLOTLY_THEME["gridcolor"])
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        # ── Statistics Summary ──
        st.markdown("""
        <div class="section-header" style="margin-top:0.5rem;">
            <div class="section-icon">📊</div>
            <h2>Statistical Summary</h2>
        </div>
        """, unsafe_allow_html=True)

        returns_all = analysis_data.pct_change().dropna() * 100
        stats = {
            "Total Return (All Time)": f"{float((analysis_data.iloc[-1]/analysis_data.iloc[0]-1)*100):,.1f}%",
            "Annualised Volatility":   f"{float(returns_all.std()*np.sqrt(252)):.2f}%",
            "Max Drawdown":            f"{float(((analysis_data/analysis_data.cummax())-1).min()*100):.2f}%",
            "Best Daily Return":       f"{float(returns_all.max()):.2f}%",
            "Worst Daily Return":      f"{float(returns_all.min()):.2f}%",
            "Avg Daily Return":        f"{float(returns_all.mean()):.3f}%",
            "Sharpe Ratio (approx)":   f"{float(returns_all.mean()/returns_all.std()*np.sqrt(252)):.2f}",
            "Data Points":             f"{len(analysis_data):,} trading days",
        }

        cols_s = st.columns(4)
        for i, (k, v) in enumerate(stats.items()):
            with cols_s[i % 4]:
                st.markdown(f"""
                <div class="metric-card" style="margin-bottom:1rem;">
                    <div class="metric-label">{k}</div>
                    <div class="metric-value" style="font-size:1.1rem;">{v}</div>
                </div>
                """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

st.markdown(f"""
<div style="text-align:center; padding:1.5rem 0 2rem 0;">
    <p style="color:#334455; font-size:0.78rem; margin:0;">
        NVIDIA Stock Predictor &nbsp;|&nbsp; Built with Streamlit &amp; TensorFlow
        &nbsp;|&nbsp; Data via Yahoo Finance
        &nbsp;|&nbsp; {datetime.now().strftime('%Y')}
    </p>
    <p style="color:#2a3444; font-size:0.7rem; margin-top:6px;">
        ⚠️ For educational purposes only. Not financial advice.
    </p>
</div>
""", unsafe_allow_html=True)
