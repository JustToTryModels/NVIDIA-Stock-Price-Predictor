# app.py
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

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
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Root Variables ── */
    :root {
        --bg-primary:     #0a0e1a;
        --bg-secondary:   #111827;
        --bg-card:        #1a2235;
        --bg-card-hover:  #1e2a40;
        --accent-green:   #76b900;
        --accent-blue:    #00d4ff;
        --accent-purple:  #8b5cf6;
        --accent-red:     #ff4560;
        --accent-orange:  #ff8c00;
        --text-primary:   #f0f4ff;
        --text-secondary: #8892a4;
        --border-color:   #2a3550;
        --shadow:         0 4px 24px rgba(0,0,0,0.5);
    }

    /* ── Global Reset ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }

    /* ── Hide Streamlit Branding ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Main Container ── */
    .main .block-container {
        padding: 1.5rem 2.5rem 2rem;
        max-width: 1400px;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-color);
    }
    [data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }

    /* ── Hero Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #0d1b2e 0%, #0a1628 40%, #111827 100%);
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow);
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(118,185,0,0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        bottom: -80px; left: 40%;
        width: 250px; height: 250px;
        background: radial-gradient(circle, rgba(0,212,255,0.07) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #76b900, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: var(--text-secondary);
        margin-top: 0.6rem;
        font-weight: 400;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(118,185,0,0.15);
        border: 1px solid rgba(118,185,0,0.4);
        color: var(--accent-green);
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 1rem;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
        border-color: rgba(118,185,0,0.4);
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 4px; height: 100%;
        border-radius: 4px 0 0 4px;
    }
    .metric-card.green::before  { background: var(--accent-green);  }
    .metric-card.blue::before   { background: var(--accent-blue);   }
    .metric-card.purple::before { background: var(--accent-purple); }
    .metric-card.orange::before { background: var(--accent-orange); }
    .metric-card.red::before    { background: var(--accent-red);    }

    .metric-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--text-secondary);
        margin-bottom: 0.4rem;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1;
    }
    .metric-delta {
        font-size: 0.8rem;
        margin-top: 0.35rem;
        font-weight: 500;
    }
    .metric-delta.positive { color: var(--accent-green); }
    .metric-delta.negative { color: var(--accent-red);   }
    .metric-icon {
        font-size: 1.5rem;
        position: absolute;
        top: 1.2rem; right: 1.2rem;
        opacity: 0.6;
    }

    /* ── Section Headers ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin: 1.8rem 0 1rem;
    }
    .section-header h2 {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
    }
    .section-line {
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, var(--border-color), transparent);
    }
    .section-icon {
        width: 32px; height: 32px;
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem;
    }
    .section-icon.green  { background: rgba(118,185,0,0.15); }
    .section-icon.blue   { background: rgba(0,212,255,0.15); }
    .section-icon.purple { background: rgba(139,92,246,0.15); }

    /* ── Prediction Table ── */
    .pred-table-wrapper {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        overflow: hidden;
    }
    .pred-table-wrapper table {
        width: 100%;
        border-collapse: collapse;
    }
    .pred-table-wrapper th {
        background: rgba(118,185,0,0.1);
        color: var(--accent-green);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 1rem 1.2rem;
        text-align: left;
        border-bottom: 1px solid var(--border-color);
    }
    .pred-table-wrapper td {
        padding: 0.85rem 1.2rem;
        border-bottom: 1px solid rgba(42,53,80,0.5);
        font-size: 0.92rem;
        color: var(--text-primary);
    }
    .pred-table-wrapper tr:last-child td { border-bottom: none; }
    .pred-table-wrapper tr:hover td { background: rgba(255,255,255,0.03); }

    .price-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .price-up   { background: rgba(118,185,0,0.15); color: var(--accent-green); }
    .price-down { background: rgba(255,69,96,0.15);  color: var(--accent-red);   }
    .price-flat { background: rgba(0,212,255,0.15);  color: var(--accent-blue);  }

    .day-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px; height: 28px;
        border-radius: 8px;
        background: rgba(139,92,246,0.2);
        color: var(--accent-purple);
        font-size: 0.8rem;
        font-weight: 700;
        margin-right: 0.5rem;
    }

    /* ── Forecast Slider ── */
    .stSlider > div > div > div > div {
        background: var(--accent-green) !important;
    }

    /* ── Forecast Button ── */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #76b900, #00a86b) !important;
        color: #000 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.3px !important;
        width: 100% !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 15px rgba(118,185,0,0.3) !important;
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(118,185,0,0.45) !important;
    }
    div[data-testid="stButton"] > button:active {
        transform: translateY(0) !important;
    }

    /* ── Info Box ── */
    .info-box {
        background: rgba(0,212,255,0.07);
        border: 1px solid rgba(0,212,255,0.25);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        font-size: 0.88rem;
        color: var(--text-secondary);
        line-height: 1.6;
    }
    .info-box strong { color: var(--accent-blue); }

    /* ── Warning Box ── */
    .warn-box {
        background: rgba(255,140,0,0.08);
        border: 1px solid rgba(255,140,0,0.3);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        font-size: 0.85rem;
        color: #c8a060;
        line-height: 1.6;
    }

    /* ── Sidebar Components ── */
    .sidebar-logo {
        text-align: center;
        padding: 1.5rem 0 1rem;
    }
    .sidebar-section {
        background: rgba(255,255,255,0.04);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.8rem 0;
    }
    .sidebar-section-title {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: var(--text-secondary);
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    .sidebar-stat {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.4rem 0;
        border-bottom: 1px solid rgba(42,53,80,0.5);
        font-size: 0.85rem;
    }
    .sidebar-stat:last-child { border-bottom: none; }
    .sidebar-stat-label { color: var(--text-secondary); }
    .sidebar-stat-value { color: var(--text-primary); font-weight: 600; }

    /* ── Divider ── */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-color), transparent);
        margin: 1.5rem 0;
    }

    /* ── Status Dot ── */
    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        background: var(--accent-green);
        animation: pulse-dot 2s infinite;
        margin-right: 6px;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50%       { opacity: 0.5; transform: scale(1.3); }
    }

    /* ── Dataframe Overrides ── */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        overflow: hidden;
    }
    .dataframe { background: var(--bg-card) !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-secondary); }
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover { background: #3a4560; }

    /* ── Plotly Charts ── */
    .js-plotly-plot .plotly { border-radius: 16px; }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: var(--accent-green) !important; }
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
def get_live_info(ticker='NVDA'):
    try:
        t    = yf.Ticker(ticker)
        info = t.info
        return info
    except Exception:
        return {}


# ==============================
# 🔹 Helpers
# ==============================
def generate_business_days(start_date, num_days):
    return pd.bdate_range(start=start_date, periods=num_days).tolist()


def predict_next_business_days(model, data, look_back=5, days=5):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    last_sequence = data_scaled[-look_back:]
    predictions   = []
    for _ in range(days):
        X_input    = np.reshape(last_sequence, (1, look_back, 1))
        prediction = model.predict(X_input, verbose=0)
        predictions.append(prediction[0, 0])
        last_sequence = np.append(last_sequence[1:], prediction, axis=0)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions


def fmt_currency(val):
    if val is None:
        return "N/A"
    if val >= 1e12:
        return f"${val/1e12:.2f}T"
    if val >= 1e9:
        return f"${val/1e9:.2f}B"
    if val >= 1e6:
        return f"${val/1e6:.2f}M"
    return f"${val:,.2f}"


def fmt_pct(val):
    if val is None:
        return "N/A"
    return f"{val*100:.2f}%" if abs(val) < 10 else f"{val:.2f}%"


# ==============================
# 🔹 Plotly Theme Helper
# ==============================
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(17,24,39,0.0)',
    plot_bgcolor='rgba(17,24,39,0.0)',
    font=dict(family='Inter', color='#8892a4', size=11),
    xaxis=dict(
        showgrid=True, gridcolor='rgba(42,53,80,0.6)',
        zeroline=False, showline=False,
        tickfont=dict(color='#8892a4'),
        title_font=dict(color='#8892a4')
    ),
    yaxis=dict(
        showgrid=True, gridcolor='rgba(42,53,80,0.6)',
        zeroline=False, showline=False,
        tickfont=dict(color='#8892a4'),
        title_font=dict(color='#8892a4')
    ),
    legend=dict(
        bgcolor='rgba(26,34,53,0.8)',
        bordercolor='rgba(42,53,80,0.8)',
        borderwidth=1,
        font=dict(color='#c8d0e0', size=11)
    ),
    margin=dict(l=20, r=20, t=50, b=20),
    hovermode='x unified',
    hoverlabel=dict(
        bgcolor='#1a2235',
        bordercolor='#2a3550',
        font=dict(color='#f0f4ff', size=12)
    )
)


# ==============================
# 🔹 Load model eagerly
# ==============================
model = load_nvidia_model()


# ==============================
# 🔹 Session State Init
# ==============================
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'stock_data_loaded' not in st.session_state:
    st.session_state.stock_data_loaded = False


# ╔══════════════════════════════════════════╗
# ║              SIDEBAR                     ║
# ╚══════════════════════════════════════════╝
with st.sidebar:

    # Logo
    st.markdown("""
        <div class="sidebar-logo">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/NVIDIA_logo.svg/1920px-NVIDIA_logo.svg.png"
                 width="160" style="filter: brightness(1.1);">
            <p style="color:#8892a4; font-size:0.75rem; margin-top:0.5rem; letter-spacing:0.5px;">
                AI-Powered Stock Prediction
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # Model status
    status_color = "#76b900" if model else "#ff4560"
    status_text  = "Model Loaded" if model else "Model Error"
    st.markdown(f"""
        <div class="sidebar-section">
            <div class="sidebar-section-title">System Status</div>
            <div class="sidebar-stat">
                <span class="sidebar-stat-label">
                    <span class="status-dot" style="background:{status_color};"></span>
                    LSTM Model
                </span>
                <span class="sidebar-stat-value" style="color:{status_color};">{status_text}</span>
            </div>
            <div class="sidebar-stat">
                <span class="sidebar-stat-label">
                    <span class="status-dot"></span>Data Feed
                </span>
                <span class="sidebar-stat-value" style="color:#76b900;">Live</span>
            </div>
            <div class="sidebar-stat">
                <span class="sidebar-stat-label">Look-back Window</span>
                <span class="sidebar-stat-value">5 Days</span>
            </div>
            <div class="sidebar-stat">
                <span class="sidebar-stat-label">Model RMSE</span>
                <span class="sidebar-stat-value" style="color:#00d4ff;">1.32</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Live Quote
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="sidebar-section-title" style="padding-left:4px;">
            📡 Live Quote — NVDA
        </div>
    """, unsafe_allow_html=True)

    with st.spinner("Fetching live data…"):
        info = get_live_info('NVDA')

    if info:
        price      = info.get('currentPrice') or info.get('regularMarketPrice')
        prev_close = info.get('previousClose') or info.get('regularMarketPreviousClose')
        mkt_cap    = info.get('marketCap')
        pe         = info.get('trailingPE')
        volume     = info.get('volume') or info.get('regularMarketVolume')
        week52_hi  = info.get('fiftyTwoWeekHigh')
        week52_lo  = info.get('fiftyTwoWeekLow')

        change     = ((price - prev_close) / prev_close * 100) if price and prev_close else None
        chg_color  = "#76b900" if (change and change >= 0) else "#ff4560"
        chg_arrow  = "▲" if (change and change >= 0) else "▼"
        chg_str    = f"{chg_arrow} {abs(change):.2f}%" if change else "N/A"

        st.markdown(f"""
            <div class="sidebar-section">
                <div style="font-size:1.8rem; font-weight:800; color:#f0f4ff;">
                    ${price:,.2f}
                </div>
                <div style="color:{chg_color}; font-size:0.9rem; font-weight:600; margin-bottom:0.8rem;">
                    {chg_str} today
                </div>
                <div class="sidebar-stat">
                    <span class="sidebar-stat-label">Market Cap</span>
                    <span class="sidebar-stat-value">{fmt_currency(mkt_cap)}</span>
                </div>
                <div class="sidebar-stat">
                    <span class="sidebar-stat-label">P/E Ratio</span>
                    <span class="sidebar-stat-value">{f"{pe:.1f}x" if pe else "N/A"}</span>
                </div>
                <div class="sidebar-stat">
                    <span class="sidebar-stat-label">Volume</span>
                    <span class="sidebar-stat-value">{fmt_currency(volume).replace("$","") if volume else "N/A"}</span>
                </div>
                <div class="sidebar-stat">
                    <span class="sidebar-stat-label">52W High</span>
                    <span class="sidebar-stat-value" style="color:#76b900;">${week52_hi:,.2f}</span>
                </div>
                <div class="sidebar-stat">
                    <span class="sidebar-stat-label">52W Low</span>
                    <span class="sidebar-stat-value" style="color:#ff4560;">${week52_lo:,.2f}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-box">Live data unavailable.</div>',
                    unsafe_allow_html=True)

    # Forecast controls
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="sidebar-section-title" style="padding-left:4px;">
            ⚙️ Forecast Settings
        </div>
    """, unsafe_allow_html=True)

    num_days = st.slider(
        "Business Days to Forecast",
        min_value=1, max_value=30, value=5, step=1,
        help="Number of future trading days the model will predict."
    )

    chart_lookback = st.select_slider(
        "Historical View",
        options=["3M", "6M", "1Y", "2Y", "5Y", "All"],
        value="1Y",
        help="How much historical data to display on the chart."
    )

    show_volume   = st.toggle("Show Volume Bars",  value=True)
    show_ma       = st.toggle("Show Moving Averages", value=True)
    show_bollinger = st.toggle("Show Bollinger Bands", value=False)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="warn-box">
            ⚠️ <strong>Disclaimer:</strong> Predictions are generated by an
            LSTM model for educational purposes only. This is not financial advice.
        </div>
    """, unsafe_allow_html=True)

    predict_btn = st.button(
        f"🚀  Run {num_days}-Day Forecast",
        key='forecast-button',
        use_container_width=True
    )


# ╔══════════════════════════════════════════╗
# ║              MAIN CONTENT                ║
# ╚══════════════════════════════════════════╝

# ── Hero Banner ──────────────────────────────
current_date = datetime.now().strftime('%A, %B %d %Y')
current_time = datetime.now().strftime('%H:%M UTC')

st.markdown(f"""
    <div class="hero-banner">
        <span class="hero-badge">🤖 LSTM Neural Network · NVDA</span>
        <h1 class="hero-title">NVIDIA Stock Price Predictor</h1>
        <p class="hero-subtitle">
            Deep learning–powered forecasting using historical OHLCV data · Updated {current_date} · {current_time}
        </p>
    </div>
""", unsafe_allow_html=True)


# ── Run Prediction ────────────────────────────
if predict_btn:
    if model is None:
        st.error("❌ Model not loaded. Please check the model file path.")
    else:
        with st.spinner("⚙️ Fetching market data and running inference…"):
            stock_data = get_stock_data('NVDA')

        if stock_data is None or stock_data.empty:
            st.error("❌ Failed to retrieve stock data. Check your connection.")
        else:
            with st.spinner("🧠 Generating predictions…"):
                close_prices = stock_data['Close'].values.reshape(-1, 1)
                dates        = stock_data.index

                predictions      = predict_next_business_days(
                    model, close_prices, look_back=5, days=num_days
                )
                last_date        = dates[-1]
                prediction_dates = generate_business_days(
                    last_date + timedelta(days=1), num_days
                )

            st.session_state.prediction_results = {
                'stock_data':      stock_data,
                'close_prices':    close_prices,
                'dates':           dates,
                'predictions':     predictions,
                'prediction_dates': prediction_dates,
                'num_days':        num_days,
                'chart_lookback':  chart_lookback,
                'show_volume':     show_volume,
                'show_ma':         show_ma,
                'show_bollinger':  show_bollinger,
            }
            st.success("✅ Forecast generated successfully!")


# ── Display Results ───────────────────────────
if st.session_state.prediction_results is not None:
    r = st.session_state.prediction_results

    stock_data       = r['stock_data']
    close_prices     = r['close_prices']
    dates            = r['dates']
    predictions      = r['predictions']
    prediction_dates = r['prediction_dates']
    stored_num_days  = r['num_days']
    stored_lookback  = r.get('chart_lookback', '1Y')
    stored_volume    = r.get('show_volume', True)
    stored_ma        = r.get('show_ma', True)
    stored_bollinger = r.get('show_bollinger', False)

    flat_preds = predictions.flatten()
    last_price = float(close_prices[-1])
    max_pred   = float(flat_preds.max())
    min_pred   = float(flat_preds.min())
    avg_pred   = float(flat_preds.mean())
    pct_change = (flat_preds[-1] - last_price) / last_price * 100

    # ─── KPI Metrics ─────────────────────────────
    st.markdown("""
        <div class="section-header">
            <div class="section-icon green">📊</div>
            <h2>Forecast Summary</h2>
            <div class="section-line"></div>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (c1, "Last Close Price", f"${last_price:,.2f}", None,                   "green",  "💵"),
        (c2, f"Day {stored_num_days} Forecast", f"${flat_preds[-1]:,.2f}",
         f"{pct_change:+.2f}% vs last close",
         "green" if pct_change >= 0 else "red",
         "🎯"),
        (c3, "Forecast High",  f"${max_pred:,.2f}",
         f"+{(max_pred-last_price)/last_price*100:.2f}% upside", "blue",   "📈"),
        (c4, "Forecast Low",   f"${min_pred:,.2f}",
         f"{(min_pred-last_price)/last_price*100:.2f}% downside", "orange", "📉"),
        (c5, "Forecast Avg",   f"${avg_pred:,.2f}",
         f"Over {stored_num_days} days",                           "purple", "〜"),
    ]

    for col, label, value, delta, color, icon in metrics:
        delta_class = ""
        if delta:
            delta_class = "positive" if "+" in delta else "negative"
        delta_html = (
            f'<div class="metric-delta {delta_class}">{delta}</div>'
            if delta else ""
        )
        with col:
            st.markdown(f"""
                <div class="metric-card {color}">
                    <span class="metric-icon">{icon}</span>
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    {delta_html}
                </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ─── Filter historical data by lookback ──────
    lookback_map = {
        "3M": 63, "6M": 126, "1Y": 252,
        "2Y": 504, "5Y": 1260, "All": len(dates)
    }
    n_pts = lookback_map.get(stored_lookback, 252)
    hist_dates  = dates[-n_pts:]
    hist_prices = close_prices[-n_pts:].flatten()

    # ─── Chart 1: Full Overview ───────────────────
    st.markdown("""
        <div class="section-header">
            <div class="section-icon blue">📈</div>
            <h2>Price Chart with Forecast</h2>
            <div class="section-line"></div>
        </div>
    """, unsafe_allow_html=True)

    df_hist = pd.DataFrame({'Date': hist_dates, 'Close': hist_prices})
    df_hist['MA20']  = df_hist['Close'].rolling(20).mean()
    df_hist['MA50']  = df_hist['Close'].rolling(50).mean()
    df_hist['Upper'] = df_hist['Close'].rolling(20).mean() + 2 * df_hist['Close'].rolling(20).std()
    df_hist['Lower'] = df_hist['Close'].rolling(20).mean() - 2 * df_hist['Close'].rolling(20).std()

    pred_df = pd.DataFrame({
        'Date':  prediction_dates,
        'Price': flat_preds
    })

    # Candlestick data
    ohlc = stock_data[['Open', 'High', 'Low', 'Close']].iloc[-n_pts:].copy()
    for col in ['Open', 'High', 'Low', 'Close']:
        ohlc[col] = pd.to_numeric(ohlc[col].squeeze(), errors='coerce')

    rows    = 2 if stored_volume else 1
    row_hts = [0.75, 0.25] if stored_volume else [1.0]

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_hts
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=ohlc.index,
        open=ohlc['Open'], high=ohlc['High'],
        low=ohlc['Low'],   close=ohlc['Close'],
        name='OHLC',
        increasing_line_color='#76b900',
        decreasing_line_color='#ff4560',
        increasing_fillcolor='rgba(118,185,0,0.7)',
        decreasing_fillcolor='rgba(255,69,96,0.7)',
    ), row=1, col=1)

    # Bollinger Bands
    if stored_bollinger:
        fig.add_trace(go.Scatter(
            x=df_hist['Date'], y=df_hist['Upper'],
            name='Upper BB', line=dict(color='rgba(139,92,246,0.5)', width=1, dash='dot'),
            fill=None, showlegend=True
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_hist['Date'], y=df_hist['Lower'],
            name='Lower BB', line=dict(color='rgba(139,92,246,0.5)', width=1, dash='dot'),
            fill='tonexty', fillcolor='rgba(139,92,246,0.05)', showlegend=True
        ), row=1, col=1)

    # Moving Averages
    if stored_ma:
        fig.add_trace(go.Scatter(
            x=df_hist['Date'], y=df_hist['MA20'],
            name='MA 20', line=dict(color='#00d4ff', width=1.2),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df_hist['Date'], y=df_hist['MA50'],
            name='MA 50', line=dict(color='#ff8c00', width=1.2),
        ), row=1, col=1)

    # Bridge line (last historical → first prediction)
    bridge_x = [hist_dates[-1], prediction_dates[0]]
    bridge_y = [hist_prices[-1], flat_preds[0]]
    fig.add_trace(go.Scatter(
        x=bridge_x, y=bridge_y,
        mode='lines',
        line=dict(color='rgba(255,165,0,0.4)', width=1.5, dash='dash'),
        showlegend=False,
        hoverinfo='skip'
    ), row=1, col=1)

    # Prediction zone fill
    pred_upper = flat_preds * 1.02
    pred_lower = flat_preds * 0.98
    fig.add_trace(go.Scatter(
        x=list(prediction_dates) + list(reversed(prediction_dates)),
        y=list(pred_upper) + list(reversed(pred_lower)),
        fill='toself',
        fillcolor='rgba(255,165,0,0.07)',
        line=dict(color='rgba(0,0,0,0)'),
        name='±2% Confidence',
        hoverinfo='skip',
        showlegend=True
    ), row=1, col=1)

    # Prediction line
    fig.add_trace(go.Scatter(
        x=prediction_dates, y=flat_preds,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#ff8c00', width=2.5, dash='dash'),
        marker=dict(
            color='#ff8c00', size=8,
            line=dict(color='#fff', width=1.5)
        ),
        customdata=[[f"Day {i+1}"] for i in range(len(flat_preds))],
        hovertemplate="<b>%{customdata[0]}</b><br>Date: %{x|%b %d, %Y}<br>Price: $%{y:,.2f}<extra></extra>"
    ), row=1, col=1)

    # Vertical separator line
    fig.add_vline(
        x=hist_dates[-1],
        line_width=1.5, line_dash="dash",
        line_color="rgba(255,255,255,0.2)",
        annotation_text="Forecast Start",
        annotation_font_color="rgba(255,255,255,0.5)",
        annotation_font_size=11
    )

    # Volume bars
    if stored_volume:
        vol = stock_data['Volume'].iloc[-n_pts:].squeeze()
        colors = [
            '#76b900' if float(stock_data['Close'].iloc[i]) >= float(stock_data['Open'].iloc[i])
            else '#ff4560'
            for i in range(-n_pts, 0)
        ]
        fig.add_trace(go.Bar(
            x=ohlc.index, y=vol,
            name='Volume',
            marker_color=colors,
            marker_line_width=0,
            opacity=0.6
        ), row=2, col=1)
        fig.update_yaxes(title_text='Volume', row=2, col=1,
                         title_font=dict(size=10))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=f'<b>NVDA</b> — Historical & {stored_num_days}-Day Forecast',
            font=dict(size=15, color='#f0f4ff')
        ),
        height=560 if stored_volume else 460,
        xaxis_rangeslider_visible=False,
        yaxis_title='Price (USD)',
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ─── Chart 2 + Table (side by side) ──────────
    st.markdown("""
        <div class="section-header">
            <div class="section-icon purple">🎯</div>
            <h2>Detailed Forecast</h2>
            <div class="section-line"></div>
        </div>
    """, unsafe_allow_html=True)

    col_chart, col_table = st.columns([1.1, 0.9], gap="large")

    with col_chart:
        # Zoomed forecast bar chart
        colors_bar = []
        for i, p in enumerate(flat_preds):
            ref = last_price if i == 0 else flat_preds[i-1]
            colors_bar.append('#76b900' if p >= ref else '#ff4560')

        fig2 = go.Figure()

        # Background reference line
        fig2.add_hline(
            y=last_price,
            line_dash="dot", line_color="rgba(255,255,255,0.25)",
            line_width=1.5,
            annotation_text=f"Last Close ${last_price:,.2f}",
            annotation_font_color="rgba(255,255,255,0.4)",
            annotation_font_size=11,
            annotation_position="bottom right"
        )

        fig2.add_trace(go.Bar(
            x=[f"Day {i+1}<br>{pd.Timestamp(d).strftime('%b %d')}"
               for i, d in enumerate(prediction_dates)],
            y=flat_preds,
            marker=dict(
                color=colors_bar,
                opacity=0.85,
                line=dict(width=0),
                cornerradius=6
            ),
            name='Forecast',
            customdata=[[d.strftime('%A, %B %d %Y'), p]
                        for d, p in zip(prediction_dates, flat_preds)],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Predicted: <b>$%{customdata[1]:,.2f}</b><extra></extra>"
            )
        ))

        fig2.add_trace(go.Scatter(
            x=[f"Day {i+1}<br>{pd.Timestamp(d).strftime('%b %d')}"
               for i, d in enumerate(prediction_dates)],
            y=flat_preds,
            mode='lines+markers',
            line=dict(color='rgba(255,255,255,0.3)', width=1.5),
            marker=dict(color='white', size=6,
                        line=dict(color='#1a2235', width=1.5)),
            showlegend=False,
            hoverinfo='skip'
        ))

        fig2.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(
                text=f'<b>Day-by-Day Price Forecast</b>',
                font=dict(size=14, color='#f0f4ff')
            ),
            height=400,
            yaxis=dict(
                **PLOTLY_LAYOUT['yaxis'],
                range=[min(flat_preds) * 0.985, max(flat_preds) * 1.015],
                tickprefix='$',
                tickformat=',.2f'
            ),
            showlegend=False,
            bargap=0.25
        )

        st.plotly_chart(fig2, use_container_width=True)

    with col_table:
        st.markdown("<br>", unsafe_allow_html=True)

        # Build HTML table
        rows_html = ""
        for i, (d, p) in enumerate(zip(prediction_dates, flat_preds)):
            ref        = last_price if i == 0 else flat_preds[i-1]
            chg        = p - ref
            chg_pct    = chg / ref * 100
            pill_class = "price-up" if chg >= 0 else "price-down"
            arrow      = "▲" if chg >= 0 else "▼"
            chg_str    = f"{arrow} {abs(chg_pct):.2f}%"
            dow        = pd.Timestamp(d).strftime('%a')
            date_str   = pd.Timestamp(d).strftime('%b %d, %Y')

            rows_html += f"""
            <tr>
                <td>
                    <span class="day-badge">{i+1}</span>
                    <span style="color:#8892a4; font-size:0.75rem;">{dow}</span>
                    {date_str}
                </td>
                <td>
                    <span class="price-pill {pill_class}">${p:,.2f}</span>
                </td>
                <td>
                    <span class="{'price-up' if chg >= 0 else 'price-down'}"
                          style="font-size:0.82rem; font-weight:600;">
                        {chg_str}
                    </span>
                </td>
            </tr>
            """

        st.markdown(f"""
            <div class="pred-table-wrapper">
                <table>
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Predicted Price</th>
                            <th>Change</th>
                        </tr>
                    </thead>
                    <tbody>{rows_html}</tbody>
                </table>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    # ─── Chart 3: Cumulative Return ───────────────
    with st.expander("📊 Cumulative Return Projection", expanded=False):
        cum_prices = np.concatenate([[last_price], flat_preds])
        cum_return = (cum_prices / last_price - 1) * 100
        cum_dates  = [dates[-1]] + list(prediction_dates)

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=cum_dates, y=cum_return,
            mode='lines+markers',
            fill='tozeroy',
            fillcolor='rgba(118,185,0,0.08)',
            line=dict(color='#76b900', width=2.5),
            marker=dict(color='#76b900', size=7,
                        line=dict(color='#fff', width=1.5)),
            customdata=[[f"${cum_prices[i]:,.2f}", f"{r:.2f}%"]
                        for i, r in enumerate(cum_return)],
            hovertemplate=(
                "<b>%{x|%b %d, %Y}</b><br>"
                "Price: %{customdata[0]}<br>"
                "Return: <b>%{customdata[1]}</b><extra></extra>"
            )
        ))
        fig3.add_hline(y=0, line_dash="dot",
                       line_color="rgba(255,255,255,0.2)", line_width=1)
        fig3.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(
                text='<b>Cumulative Return from Last Close</b>',
                font=dict(size=14, color='#f0f4ff')
            ),
            height=320,
            yaxis=dict(**PLOTLY_LAYOUT['yaxis'], ticksuffix='%'),
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ─── Historical Data Table ────────────────────
    with st.expander("🗃️ Full Historical Data", expanded=False):
        st.markdown("""
            <div class="section-header" style="margin-top:0;">
                <div class="section-icon green">📋</div>
                <h2>Historical OHLCV Data</h2>
                <div class="section-line"></div>
            </div>
        """, unsafe_allow_html=True)

        display_df = stock_data.copy()
        display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d')
        st.dataframe(
            display_df.iloc[::-1],
            height=380,
            use_container_width=True
        )

    # ─── Download Button ──────────────────────────
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    pred_export = pd.DataFrame({
        'Date':            [d.strftime('%Y-%m-%d') for d in prediction_dates],
        'Predicted_Price': [round(p, 4) for p in flat_preds],
        'Change_vs_Prev':  [
            round(flat_preds[i] - (last_price if i == 0 else flat_preds[i-1]), 4)
            for i in range(len(flat_preds))
        ],
        'Change_Pct':      [
            round(
                (flat_preds[i] - (last_price if i == 0 else flat_preds[i-1])) /
                (last_price if i == 0 else flat_preds[i-1]) * 100, 4
            )
            for i in range(len(flat_preds))
        ]
    })

    dc1, dc2, dc3 = st.columns([1, 1, 1])
    with dc2:
        st.download_button(
            label="⬇️  Export Predictions as CSV",
            data=pred_export.to_csv(index=False),
            file_name=f"NVDA_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv',
            use_container_width=True
        )

else:
    # ── Empty State ───────────────────────────────
    st.markdown("""
        <div style="
            text-align: center;
            padding: 5rem 2rem;
            background: rgba(26,34,53,0.4);
            border: 1px dashed rgba(42,53,80,0.8);
            border-radius: 20px;
            margin-top: 1rem;
        ">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📡</div>
            <h3 style="color: #f0f4ff; font-size: 1.4rem; margin-bottom: 0.5rem;">
                Ready to Forecast
            </h3>
            <p style="color: #8892a4; font-size: 0.95rem; max-width: 460px; margin: 0 auto 1.5rem;">
                Configure your forecast settings in the sidebar and click
                <strong style="color:#76b900;">Run Forecast</strong> to generate
                AI-powered price predictions for NVIDIA stock.
            </p>
            <div style="
                display: inline-flex; gap: 2rem;
                background: rgba(42,53,80,0.3);
                border-radius: 12px; padding: 1rem 2rem;
                font-size: 0.85rem; color: #8892a4;
            ">
                <span>🧠 LSTM Model</span>
                <span>📊 Up to 30-Day Horizon</span>
                <span>⚡ Real-time Data</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
