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
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Root Variables ── */
    :root {
        --primary:   #76b900;
        --secondary: #1a1a2e;
        --accent:    #00d4ff;
        --card-bg:   rgba(255,255,255,0.03);
        --border:    rgba(118,185,0,0.25);
        --text-main: #e8e8e8;
        --text-muted:#a0a0b0;
    }

    /* ── Global Reset ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── App Background ── */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1117 50%, #0a1628 100%);
        color: var(--text-main);
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #111827 100%);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--primary);
    }

    /* ── Hide default Streamlit chrome ── */
    #MainMenu, footer, header {visibility: hidden;}

    /* ── Metric Cards ── */
    [data-testid="metric-container"] {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }
    [data-testid="metric-container"] label {
        color: var(--text-muted) !important;
        font-size: 0.75rem !important;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: var(--primary) !important;
        font-size: 1.6rem !important;
        font-weight: 700;
    }
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        font-size: 0.85rem !important;
        font-weight: 500;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #76b900 0%, #5a8f00 100%);
        color: #0a0a1a !important;
        border: none;
        border-radius: 12px;
        padding: 14px 32px;
        font-size: 0.95rem !important;
        font-weight: 700;
        letter-spacing: 0.04em;
        cursor: pointer;
        transition: all 0.25s ease;
        width: 100%;
        text-transform: uppercase;
        box-shadow: 0 4px 20px rgba(118,185,0,0.25);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(118,185,0,0.4);
        background: linear-gradient(135deg, #8fd400 0%, #76b900 100%);
    }
    .stButton > button:active {
        transform: translateY(0px);
        box-shadow: 0 2px 10px rgba(118,185,0,0.2);
    }

    /* ── Sliders ── */
    .stSlider [data-baseweb="slider"] {
        margin-top: 8px;
    }
    .stSlider [data-testid="stThumbValue"] {
        background: var(--primary) !important;
        color: #0a0a1a !important;
        font-weight: 700;
        border-radius: 8px;
    }

    /* ── DataFrames ── */
    .stDataFrame {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }
    .stDataFrame [data-testid="stDataFrameResizable"] {
        border-radius: 12px;
    }

    /* ── Selectbox / Radio ── */
    .stSelectbox [data-baseweb="select"] > div {
        background: var(--card-bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text-main) !important;
    }

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        background: var(--card-bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        font-weight: 600;
        color: var(--text-main) !important;
    }

    /* ── Custom Card ── */
    .pro-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(118,185,0,0.20);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(10px);
        margin-bottom: 16px;
        transition: border-color 0.3s ease;
    }
    .pro-card:hover {
        border-color: rgba(118,185,0,0.50);
    }

    /* ── Section Headers ── */
    .section-header {
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--primary);
        letter-spacing: 0.02em;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-header::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(118,185,0,0.4), transparent);
    }

    /* ── Hero Banner ── */
    .hero-banner {
        background: linear-gradient(135deg,
            rgba(118,185,0,0.12) 0%,
            rgba(0,212,255,0.06) 50%,
            rgba(118,185,0,0.04) 100%);
        border: 1px solid rgba(118,185,0,0.30);
        border-radius: 20px;
        padding: 40px 32px;
        text-align: center;
        position: relative;
        overflow: hidden;
        margin-bottom: 28px;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(118,185,0,0.05) 0%, transparent 60%);
        animation: pulse-bg 4s ease-in-out infinite;
    }
    @keyframes pulse-bg {
        0%,100% { transform: scale(1); opacity: 0.5; }
        50%      { transform: scale(1.1); opacity: 1; }
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #76b900, #00d4ff, #76b900);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shine 3s linear infinite;
        margin-bottom: 8px;
        position: relative;
        z-index: 1;
    }
    @keyframes shine {
        0%   { background-position: 0% center; }
        100% { background-position: 200% center; }
    }
    .hero-subtitle {
        font-size: 1rem;
        color: var(--text-muted);
        font-weight: 400;
        position: relative;
        z-index: 1;
    }

    /* ── Status Badges ── */
    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .badge-green  { background: rgba(118,185,0,0.15);  color: #76b900;  border: 1px solid rgba(118,185,0,0.3);  }
    .badge-red    { background: rgba(255,75,75,0.15);   color: #ff6b6b;  border: 1px solid rgba(255,75,75,0.3);  }
    .badge-blue   { background: rgba(0,212,255,0.15);   color: #00d4ff;  border: 1px solid rgba(0,212,255,0.3);  }

    /* ── Info / Warning / Success Boxes ── */
    .info-box {
        background: rgba(0,212,255,0.08);
        border-left: 4px solid #00d4ff;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin: 12px 0;
        font-size: 0.9rem;
        color: var(--text-main);
    }
    .warn-box {
        background: rgba(255,170,0,0.08);
        border-left: 4px solid #ffaa00;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin: 12px 0;
        font-size: 0.9rem;
        color: var(--text-main);
    }
    .success-box {
        background: rgba(118,185,0,0.08);
        border-left: 4px solid #76b900;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin: 12px 0;
        font-size: 0.9rem;
        color: var(--text-main);
    }

    /* ── Tab Styling ── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--card-bg);
        border-radius: 12px;
        border: 1px solid var(--border);
        padding: 6px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 0.875rem;
        color: var(--text-muted) !important;
        letter-spacing: 0.03em;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #76b900, #5a8f00) !important;
        color: #0a0a1a !important;
    }

    /* ── Progress Bar ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #76b900, #00d4ff);
        border-radius: 10px;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar       { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #76b900; border-radius: 3px; }

    /* ── Divider ── */
    hr { border-color: rgba(118,185,0,0.15) !important; }

    /* ── Tooltip helper ── */
    .tooltip-text {
        font-size: 0.78rem;
        color: var(--text-muted);
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================
# 🔹 Plotly Dark Theme Helper
# ==============================
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#e8e8e8', size=12),
    xaxis=dict(
        showgrid=True, gridcolor='rgba(255,255,255,0.05)',
        showline=False, zeroline=False,
        tickfont=dict(size=11, color='#a0a0b0'),
    ),
    yaxis=dict(
        showgrid=True, gridcolor='rgba(255,255,255,0.05)',
        showline=False, zeroline=False,
        tickfont=dict(size=11, color='#a0a0b0'),
    ),
    legend=dict(
        bgcolor='rgba(0,0,0,0)',
        bordercolor='rgba(118,185,0,0.3)',
        borderwidth=1,
        font=dict(size=12),
    ),
    margin=dict(l=20, r=20, t=50, b=20),
    hovermode='x unified',
)


# ==============================
# 🔹 Load Model
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
# 🔹 Fetch Stock Data
# ==============================
@st.cache_data(ttl=900)          # refresh every 15 min
def get_stock_data(ticker='NVDA'):
    try:
        data = yf.download(ticker, period='max', auto_adjust=True)
        return data
    except Exception as e:
        st.error(f"❌ Error fetching stock data: {e}")
        return None


@st.cache_data(ttl=300)
def get_realtime_quote(ticker='NVDA'):
    """Fetch real-time ticker info."""
    try:
        t    = yf.Ticker(ticker)
        info = t.info
        return info
    except Exception:
        return {}


# ==============================
# 🔹 Business Day Generator
# ==============================
def generate_business_days(start_date, num_days):
    return pd.bdate_range(start=start_date, periods=num_days).tolist()


# ==============================
# 🔹 Prediction Function
# ==============================
def predict_next_business_days(model, data, look_back=5, days=5):
    scaler      = MinMaxScaler(feature_range=(0, 1))
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


# ==============================
# 🔹 Technical Indicator Helpers
# ==============================
def compute_rsi(series, window=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast   = series.ewm(span=fast,   adjust=False).mean()
    ema_slow   = series.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_bollinger(series, window=20):
    sma    = series.rolling(window).mean()
    std    = series.rolling(window).std()
    upper  = sma + 2 * std
    lower  = sma - 2 * std
    return upper, sma, lower


# ==============================
# 🔹 Session State Init
# ==============================
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False


# ==============================
# 🔹 Load Model (once)
# ==============================
model = load_nvidia_model()
st.session_state.model_loaded = model is not None


# ==============================
# ╔══════════════════════════╗
# ║       SIDEBAR            ║
# ╚══════════════════════════╝
# ==============================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 24px 0;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/NVIDIA_logo.svg/1920px-NVIDIA_logo.svg.png"
             style="width:80%; filter: drop-shadow(0 0 12px rgba(118,185,0,0.5));">
        <p style="color:#a0a0b0; font-size:0.75rem; margin-top:12px; letter-spacing:0.08em; text-transform:uppercase;">
            AI-Powered Stock Forecasting
        </p>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    # ── Model Status ──
    st.markdown("### 🤖 Model Status")
    if st.session_state.model_loaded:
        st.markdown('<span class="badge badge-green">✓ Model Loaded</span>', unsafe_allow_html=True)
        st.markdown("""
        <div class="pro-card" style="margin-top:12px;">
            <p style="margin:0; font-size:0.82rem; color:#a0a0b0;">
                <b style="color:#76b900;">Architecture:</b> LSTM<br>
                <b style="color:#76b900;">Look-Back:</b> 5 days<br>
                <b style="color:#76b900;">Units:</b> 150<br>
                <b style="color:#76b900;">RMSE:</b> $1.32
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-red">✗ Model Unavailable</span>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Forecast Settings ──
    st.markdown("### ⚙️ Forecast Settings")

    num_days = st.slider(
        "Forecast Horizon (Business Days)",
        min_value=1, max_value=30, value=5,
        help="Number of business days ahead to predict"
    )
    st.markdown(f'<p class="tooltip-text">Forecasting <b style="color:#76b900;">{num_days}</b> business day{"s" if num_days > 1 else ""} ahead</p>',
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chart Settings ──
    st.markdown("### 📊 Chart Settings")

    history_period = st.selectbox(
        "Historical Display Window",
        options=["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years", "All"],
        index=3,
        help="How much historical data to show on the chart"
    )

    show_volume      = st.toggle("Show Volume",      value=True)
    show_ma          = st.toggle("Show Moving Avgs", value=True)
    show_bollinger   = st.toggle("Show Bollinger Bands", value=False)
    show_rsi         = st.toggle("Show RSI",         value=False)
    show_macd        = st.toggle("Show MACD",        value=False)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Data Info ──
    st.markdown("""
    <div style="font-size:0.78rem; color:#606070; text-align:center; padding-bottom:8px;">
        Data provided by Yahoo Finance.<br>
        Predictions are for educational purposes only.<br>
        <b style="color:#76b900;">Not financial advice.</b>
    </div>
    """, unsafe_allow_html=True)


# ==============================
# ╔══════════════════════════╗
# ║       MAIN CONTENT       ║
# ╚══════════════════════════╝
# ==============================

# ── Hero Banner ──
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-title">📈 NVIDIA Stock Predictor</div>
    <div class="hero-subtitle">
        LSTM Deep Learning · Real-Time Data · Interactive Analytics
    </div>
    <div style="margin-top:16px;">
        <span class="badge badge-green">NVDA · NASDAQ</span>&nbsp;
        <span class="badge badge-blue">LSTM Model</span>&nbsp;
        <span class="badge badge-green">Live Data</span>
    </div>
    <div style="margin-top:14px; font-size:0.82rem; color:#606070;">
        📅 {datetime.now().strftime('%A, %B %d, %Y  ·  %H:%M UTC')}
    </div>
</div>
""", unsafe_allow_html=True)


# ── Live Metrics Row ──
with st.spinner("Fetching live market data..."):
    info       = get_realtime_quote('NVDA')
    stock_data = get_stock_data('NVDA')

if info and stock_data is not None and not stock_data.empty:

    current_price  = info.get('currentPrice') or info.get('regularMarketPrice', float('nan'))
    prev_close     = info.get('previousClose', float('nan'))
    day_high       = info.get('dayHigh',       float('nan'))
    day_low        = info.get('dayLow',        float('nan'))
    market_cap     = info.get('marketCap',     float('nan'))
    volume         = info.get('volume',        float('nan'))
    fifty_two_high = info.get('fiftyTwoWeekHigh', float('nan'))
    fifty_two_low  = info.get('fiftyTwoWeekLow',  float('nan'))

    price_change   = current_price - prev_close if (current_price and prev_close) else 0
    price_change_p = (price_change / prev_close * 100) if prev_close else 0

    def fmt_large(n):
        if not n or np.isnan(n):  return "N/A"
        if n >= 1e12: return f"${n/1e12:.2f}T"
        if n >= 1e9:  return f"${n/1e9:.2f}B"
        if n >= 1e6:  return f"${n/1e6:.2f}M"
        return f"${n:,.0f}"

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric(
            "💲 Current Price",
            f"${current_price:,.2f}" if current_price else "N/A",
            f"{price_change:+.2f} ({price_change_p:+.2f}%)"
        )
    with col2:
        st.metric("📈 Day High",  f"${day_high:,.2f}"  if day_high  else "N/A")
    with col3:
        st.metric("📉 Day Low",   f"${day_low:,.2f}"   if day_low   else "N/A")
    with col4:
        st.metric("🏦 Market Cap", fmt_large(market_cap))
    with col5:
        st.metric("📊 52W High",  f"${fifty_two_high:,.2f}" if fifty_two_high else "N/A")
    with col6:
        st.metric("📊 52W Low",   f"${fifty_two_low:,.2f}"  if fifty_two_low  else "N/A")

else:
    st.markdown('<div class="warn-box">⚠️ Could not fetch live quote data. Charts will still work.</div>',
                unsafe_allow_html=True)
    stock_data = get_stock_data('NVDA')

st.markdown("<hr>", unsafe_allow_html=True)


# ==============================
# 🔹 Filter historical data by chosen window
# ==============================
def filter_data_by_period(df, period_label):
    now = df.index[-1]
    mapping = {
        "1 Month":  now - pd.DateOffset(months=1),
        "3 Months": now - pd.DateOffset(months=3),
        "6 Months": now - pd.DateOffset(months=6),
        "1 Year":   now - pd.DateOffset(years=1),
        "2 Years":  now - pd.DateOffset(years=2),
        "5 Years":  now - pd.DateOffset(years=5),
        "All":      df.index[0],
    }
    cutoff = mapping.get(period_label, df.index[0])
    return df[df.index >= cutoff]


# ==============================
# 🔹 Main Tab Layout
# ==============================
tab_charts, tab_predict, tab_technicals, tab_data, tab_about = st.tabs([
    "📊 Market Overview",
    "🔮 AI Forecast",
    "🔬 Technical Analysis",
    "📋 Raw Data",
    "ℹ️ About"
])


# ╔══════════════════════════════╗
# ║   TAB 1 — MARKET OVERVIEW   ║
# ╚══════════════════════════════╝
with tab_charts:
    if stock_data is not None and not stock_data.empty:

        chart_data = filter_data_by_period(stock_data, history_period)

        close_s  = chart_data['Close'].squeeze()
        open_s   = chart_data['Open'].squeeze()
        high_s   = chart_data['High'].squeeze()
        low_s    = chart_data['Low'].squeeze()
        volume_s = chart_data['Volume'].squeeze()

        # ── Candlestick + Volume ──
        st.markdown('<div class="section-header">🕯️ Price Chart</div>', unsafe_allow_html=True)

        rows   = 2 if show_volume else 1
        r_h    = [0.75, 0.25] if show_volume else [1.0]
        specs  = [[{"secondary_y": False}]] * rows

        fig_main = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=r_h,
            specs=specs,
        )

        # Candlestick
        fig_main.add_trace(
            go.Candlestick(
                x=chart_data.index,
                open=open_s, high=high_s, low=low_s, close=close_s,
                name="OHLC",
                increasing_line_color='#76b900',
                decreasing_line_color='#ff4b4b',
                increasing_fillcolor='rgba(118,185,0,0.7)',
                decreasing_fillcolor='rgba(255,75,75,0.7)',
            ),
            row=1, col=1
        )

        # Moving Averages
        if show_ma:
            for window, color, name in [
                (20,  '#00d4ff', 'MA 20'),
                (50,  '#ff9500', 'MA 50'),
                (200, '#b040ff', 'MA 200'),
            ]:
                if len(close_s) >= window:
                    ma = close_s.rolling(window).mean()
                    fig_main.add_trace(
                        go.Scatter(
                            x=chart_data.index, y=ma,
                            mode='lines', name=name,
                            line=dict(color=color, width=1.5),
                            opacity=0.85,
                        ),
                        row=1, col=1
                    )

        # Bollinger Bands
        if show_bollinger and len(close_s) >= 20:
            bb_up, bb_mid, bb_low = compute_bollinger(close_s)
            fig_main.add_trace(
                go.Scatter(x=chart_data.index, y=bb_up, mode='lines',
                           name='BB Upper', line=dict(color='rgba(255,170,0,0.5)', width=1, dash='dot')),
                row=1, col=1
            )
            fig_main.add_trace(
                go.Scatter(x=chart_data.index, y=bb_low, mode='lines',
                           name='BB Lower', line=dict(color='rgba(255,170,0,0.5)', width=1, dash='dot'),
                           fill='tonexty', fillcolor='rgba(255,170,0,0.04)'),
                row=1, col=1
            )

        # Volume Bars
        if show_volume:
            colors_vol = ['#76b900' if c >= o else '#ff4b4b'
                          for c, o in zip(close_s, open_s)]
            fig_main.add_trace(
                go.Bar(
                    x=chart_data.index, y=volume_s,
                    name='Volume',
                    marker_color=colors_vol,
                    opacity=0.6,
                ),
                row=2, col=1
            )
            fig_main.update_yaxes(title_text="Volume", row=2, col=1,
                                  title_font=dict(size=10, color='#a0a0b0'))

        # Layout
        layout = {**PLOTLY_LAYOUT,
                  'height': 560 if show_volume else 440,
                  'title': dict(
                      text=f"<b>NVDA</b> · {history_period}",
                      font=dict(size=15, color='#76b900'),
                      x=0.01
                  ),
                  'xaxis_rangeslider_visible': False,
                  }
        fig_main.update_layout(**layout)
        st.plotly_chart(fig_main, use_container_width=True)

        # ── Return Distribution ──
        st.markdown('<div class="section-header">📊 Daily Return Distribution</div>',
                    unsafe_allow_html=True)

        returns = close_s.pct_change().dropna() * 100
        fig_ret = go.Figure()
        fig_ret.add_trace(go.Histogram(
            x=returns,
            nbinsx=80,
            marker_color='#76b900',
            opacity=0.8,
            name='Daily Returns',
        ))
        fig_ret.add_vline(x=returns.mean(), line_color='#00d4ff',
                          line_dash='dash', line_width=2,
                          annotation_text=f"Mean: {returns.mean():.3f}%",
                          annotation_font_color='#00d4ff')
        fig_ret.update_layout(**{**PLOTLY_LAYOUT,
                                 'height': 320,
                                 'title': "Distribution of Daily Returns (%)",
                                 'xaxis_title': "Return (%)",
                                 'yaxis_title': "Frequency"})
        st.plotly_chart(fig_ret, use_container_width=True)

        # ── Stats Row ──
        st.markdown('<div class="section-header">📐 Statistical Summary</div>',
                    unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Mean Daily Return", f"{returns.mean():.4f}%")
        with c2:
            st.metric("Std Deviation",     f"{returns.std():.4f}%")
        with c3:
            st.metric("Max Gain",          f"{returns.max():.2f}%")
        with c4:
            st.metric("Max Loss",          f"{returns.min():.2f}%")

    else:
        st.markdown('<div class="warn-box">⚠️ Unable to load stock data.</div>',
                    unsafe_allow_html=True)


# ╔════════════════════════════╗
# ║   TAB 2 — AI FORECAST     ║
# ╚════════════════════════════╝
with tab_predict:
    st.markdown('<div class="section-header">🔮 AI Price Forecast</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        🤖 The LSTM model will predict the next <b style="color:#76b900;">{num_days}</b> business
        day{"s" if num_days > 1 else ""} using the last 5 days of closing prices as input.
        Adjust the horizon in the sidebar.
    </div>
    """, unsafe_allow_html=True)

    col_btn, col_gap = st.columns([1, 2])
    with col_btn:
        run_prediction = st.button(
            f"🚀 Run {num_days}-Day Forecast",
            key='forecast-button'
        )

    if run_prediction:
        if not st.session_state.model_loaded:
            st.markdown('<div class="warn-box">❌ Model not loaded. Cannot make predictions.</div>',
                        unsafe_allow_html=True)
        elif stock_data is None or stock_data.empty:
            st.markdown('<div class="warn-box">❌ Failed to load stock data.</div>',
                        unsafe_allow_html=True)
        else:
            close_prices = stock_data['Close'].values.reshape(-1, 1)

            with st.spinner("🧠 Running LSTM inference..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    progress_bar.progress(i + 1)

                predictions      = predict_next_business_days(model, close_prices,
                                                              look_back=5, days=num_days)
                last_date        = stock_data.index[-1]
                prediction_dates = generate_business_days(last_date + timedelta(days=1), num_days)

            progress_bar.empty()

            st.session_state.prediction_results = {
                'stock_data':       stock_data,
                'close_prices':     close_prices,
                'dates':            stock_data.index,
                'predictions':      predictions,
                'prediction_dates': prediction_dates,
                'num_days':         num_days,
                'stock':            'NVDA',
            }
            st.markdown('<div class="success-box">✅ Forecast complete! Results displayed below.</div>',
                        unsafe_allow_html=True)

    # ── Display Results ──
    if st.session_state.prediction_results is not None:
        res  = st.session_state.prediction_results
        preds       = res['predictions'].flatten()
        pred_dates  = res['prediction_dates']
        close_arr   = res['close_prices'].flatten()
        hist_dates  = res['dates']
        n_days      = res['num_days']

        last_actual   = close_arr[-1]
        pred_final    = preds[-1]
        pred_change   = pred_final - last_actual
        pred_change_p = pred_change / last_actual * 100

        # ── Summary KPIs ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">📊 Forecast Summary</div>', unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Last Close",          f"${last_actual:,.2f}")
        with k2:
            st.metric(f"Day {n_days} Target", f"${pred_final:,.2f}",
                      f"{pred_change:+.2f} ({pred_change_p:+.2f}%)")
        with k3:
            st.metric("Forecast High",        f"${preds.max():,.2f}")
        with k4:
            st.metric("Forecast Low",         f"${preds.min():,.2f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Full Historical + Forecast Chart ──
        st.markdown('<div class="section-header">📈 Historical + Forecast</div>',
                    unsafe_allow_html=True)

        # Limit history display to 1 year for readability
        cutoff     = hist_dates[-1] - pd.DateOffset(years=1)
        mask       = hist_dates >= cutoff
        plot_dates = hist_dates[mask]
        plot_close = close_arr[mask.values] if hasattr(mask, 'values') else close_arr[mask]

        fig_combo = go.Figure()

        # Historical line
        fig_combo.add_trace(go.Scatter(
            x=plot_dates, y=plot_close,
            mode='lines', name='Historical Close',
            line=dict(color='#76b900', width=2),
            fill='tozeroy',
            fillcolor='rgba(118,185,0,0.06)',
        ))

        # Connector
        conn_x = [plot_dates[-1], pred_dates[0]]
        conn_y = [plot_close[-1],  preds[0]]
        fig_combo.add_trace(go.Scatter(
            x=conn_x, y=conn_y,
            mode='lines', name='',
            line=dict(color='rgba(0,212,255,0.4)', width=1.5, dash='dot'),
            showlegend=False,
        ))

        # Prediction band (uncertainty cone)
        uncertainty = np.linspace(0, preds.std() * 1.5, len(pred_dates))
        upper_band  = preds + uncertainty
        lower_band  = preds - uncertainty

        fig_combo.add_trace(go.Scatter(
            x=list(pred_dates) + list(pred_dates[::-1]),
            y=list(upper_band) + list(lower_band[::-1]),
            fill='toself',
            fillcolor='rgba(0,212,255,0.07)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Uncertainty Band',
            showlegend=True,
        ))

        # Forecast line
        fig_combo.add_trace(go.Scatter(
            x=pred_dates, y=preds,
            mode='lines+markers', name='AI Forecast',
            line=dict(color='#00d4ff', width=2.5, dash='dash'),
            marker=dict(size=8, color='#00d4ff',
                        line=dict(color='#0a0a1a', width=2)),
        ))

        # Vertical separator
        fig_combo.add_vline(
            x=hist_dates[-1],
            line_dash="dot",
            line_color="rgba(255,255,255,0.2)",
            annotation_text="Today",
            annotation_position="top",
            annotation_font_color='#a0a0b0',
            annotation_font_size=11,
        )

        fig_combo.update_layout(**{**PLOTLY_LAYOUT,
                                   'height': 480,
                                   'title': '<b>NVDA</b> — Historical Prices & AI Forecast',
                                   'yaxis_title': 'Price (USD)',
                                   'xaxis_title': 'Date'})
        st.plotly_chart(fig_combo, use_container_width=True)

        # ── Forecast-Only Chart ──
        st.markdown('<div class="section-header">🔭 Forecast Detail</div>',
                    unsafe_allow_html=True)

        direction_colors = ['#76b900' if p >= last_actual else '#ff4b4b' for p in preds]

        fig_fcast = go.Figure()
        fig_fcast.add_trace(go.Bar(
            x=pred_dates, y=preds,
            name='Predicted Price',
            marker_color=direction_colors,
            opacity=0.8,
            text=[f"${p:,.2f}" for p in preds],
            textposition='outside',
            textfont=dict(size=11, color='#e8e8e8'),
        ))
        fig_fcast.add_hline(
            y=last_actual,
            line_dash='dot',
            line_color='rgba(255,255,255,0.3)',
            annotation_text=f"Last Close: ${last_actual:,.2f}",
            annotation_position='bottom right',
            annotation_font_color='#a0a0b0',
        )
        fig_fcast.update_layout(**{**PLOTLY_LAYOUT,
                                   'height': 380,
                                   'title': f'<b>Predicted Prices — Next {n_days} Business Day{"s" if n_days>1 else ""}</b>',
                                   'yaxis_title': 'Price (USD)',
                                   'yaxis_range': [min(preds) * 0.97, max(preds) * 1.03]})
        st.plotly_chart(fig_fcast, use_container_width=True)

        # ── Prediction Table ──
        st.markdown('<div class="section-header">📋 Forecast Table</div>',
                    unsafe_allow_html=True)

        prediction_df = pd.DataFrame({
            'Date':              [d.strftime('%Y-%m-%d') for d in pred_dates],
            'Predicted Price':   [f"${p:,.2f}" for p in preds],
            'Change vs Last':    [f"{p - last_actual:+.2f}" for p in preds],
            'Change %':          [f"{(p - last_actual)/last_actual*100:+.2f}%" for p in preds],
            'Direction':         ['🟢 Bullish' if p >= last_actual else '🔴 Bearish' for p in preds],
        })

        st.dataframe(
            prediction_df,
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("""
        <div class="warn-box">
            ⚠️ <b>Disclaimer:</b> These predictions are generated by an LSTM model trained on
            historical price data. They are for <b>educational and demonstrative purposes only</b>
            and should <b>not</b> be used as financial advice. Always consult a qualified
            financial professional before making investment decisions.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding: 60px 20px;">
            <div style="font-size:4rem; margin-bottom:16px;">🤖</div>
            <div style="font-size:1.2rem; color:#a0a0b0; font-weight:500;">
                Click <b style="color:#76b900;">Run Forecast</b> to generate AI predictions
            </div>
            <div style="font-size:0.85rem; color:#606070; margin-top:8px;">
                Adjust forecast horizon in the sidebar
            </div>
        </div>
        """, unsafe_allow_html=True)


# ╔════════════════════════════════╗
# ║   TAB 3 — TECHNICAL ANALYSIS  ║
# ╚════════════════════════════════╝
with tab_technicals:
    if stock_data is not None and not stock_data.empty:

        chart_data_ta = filter_data_by_period(stock_data, history_period)
        close_ta      = chart_data_ta['Close'].squeeze()

        st.markdown('<div class="section-header">🔬 Technical Indicators</div>',
                    unsafe_allow_html=True)

        # ── RSI ──
        with st.expander("📉 RSI — Relative Strength Index", expanded=show_rsi):
            rsi = compute_rsi(close_ta)
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=chart_data_ta.index, y=rsi,
                mode='lines', name='RSI',
                line=dict(color='#76b900', width=2),
                fill='tozeroy',
                fillcolor='rgba(118,185,0,0.07)',
            ))
            fig_rsi.add_hline(y=70, line_color='#ff4b4b', line_dash='dot',
                              annotation_text='Overbought (70)',
                              annotation_position='bottom right',
                              annotation_font_color='#ff4b4b')
            fig_rsi.add_hline(y=30, line_color='#76b900', line_dash='dot',
                              annotation_text='Oversold (30)',
                              annotation_position='top right',
                              annotation_font_color='#76b900')
            fig_rsi.add_hrect(y0=70, y1=100, fillcolor='rgba(255,75,75,0.05)',
                              line_width=0)
            fig_rsi.add_hrect(y0=0, y1=30, fillcolor='rgba(118,185,0,0.05)',
                              line_width=0)
            fig_rsi.update_layout(**{**PLOTLY_LAYOUT,
                                     'height': 300, 'yaxis_range': [0, 100],
                                     'title': 'RSI (14)',
                                     'yaxis_title': 'RSI'})
            st.plotly_chart(fig_rsi, use_container_width=True)

            latest_rsi = rsi.dropna().iloc[-1]
            if latest_rsi > 70:
                st.markdown(f'<div class="warn-box">⚠️ RSI = <b>{latest_rsi:.1f}</b> — Stock appears <b>Overbought</b>.</div>', unsafe_allow_html=True)
            elif latest_rsi < 30:
                st.markdown(f'<div class="success-box">🟢 RSI = <b>{latest_rsi:.1f}</b> — Stock appears <b>Oversold</b> (potential buying opportunity).</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="info-box">ℹ️ RSI = <b>{latest_rsi:.1f}</b> — Stock is in <b>Neutral</b> territory.</div>', unsafe_allow_html=True)

        # ── MACD ──
        with st.expander("📊 MACD — Moving Average Convergence Divergence", expanded=show_macd):
            macd_line, signal_line, histogram = compute_macd(close_ta)
            fig_macd = go.Figure()
            colors_macd = ['#76b900' if h >= 0 else '#ff4b4b' for h in histogram]
            fig_macd.add_trace(go.Bar(
                x=chart_data_ta.index, y=histogram,
                name='Histogram', marker_color=colors_macd, opacity=0.7,
            ))
            fig_macd.add_trace(go.Scatter(
                x=chart_data_ta.index, y=macd_line,
                mode='lines', name='MACD',
                line=dict(color='#00d4ff', width=2),
            ))
            fig_macd.add_trace(go.Scatter(
                x=chart_data_ta.index, y=signal_line,
                mode='lines', name='Signal',
                line=dict(color='#ff9500', width=1.5),
            ))
            fig_macd.update_layout(**{**PLOTLY_LAYOUT,
                                      'height': 320, 'title': 'MACD (12, 26, 9)',
                                      'yaxis_title': 'MACD'})
            st.plotly_chart(fig_macd, use_container_width=True)

        # ── Bollinger Bands ──
        with st.expander("📐 Bollinger Bands (20, ±2σ)", expanded=show_bollinger):
            bb_up, bb_mid, bb_low = compute_bollinger(close_ta)
            fig_bb = go.Figure()
            fig_bb.add_trace(go.Scatter(
                x=chart_data_ta.index, y=close_ta,
                mode='lines', name='Close',
                line=dict(color='#76b900', width=2),
            ))
            fig_bb.add_trace(go.Scatter(
                x=chart_data_ta.index, y=bb_up,
                mode='lines', name='Upper Band',
                line=dict(color='rgba(255,170,0,0.6)', width=1.2, dash='dot'),
            ))
            fig_bb.add_trace(go.Scatter(
                x=chart_data_ta.index, y=bb_low,
                mode='lines', name='Lower Band',
                line=dict(color='rgba(255,170,0,0.6)', width=1.2, dash='dot'),
                fill='tonexty', fillcolor='rgba(255,170,0,0.05)',
            ))
            fig_bb.add_trace(go.Scatter(
                x=chart_data_ta.index, y=bb_mid,
                mode='lines', name='Middle (SMA 20)',
                line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dash'),
            ))
            fig_bb.update_layout(**{**PLOTLY_LAYOUT,
                                    'height': 360, 'title': 'Bollinger Bands',
                                    'yaxis_title': 'Price (USD)'})
            st.plotly_chart(fig_bb, use_container_width=True)

        # ── Moving Averages Convergence ──
        with st.expander("📈 Moving Average Analysis", expanded=show_ma):
            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(
                x=chart_data_ta.index, y=close_ta,
                mode='lines', name='Close',
                line=dict(color='#76b900', width=1.5),
            ))
            for window, color, name in [
                (20,  '#00d4ff', 'SMA 20'),
                (50,  '#ff9500', 'SMA 50'),
                (100, '#ff4b4b', 'SMA 100'),
                (200, '#b040ff', 'SMA 200'),
            ]:
                if len(close_ta) >= window:
                    ma = close_ta.rolling(window).mean()
                    fig_ma.add_trace(go.Scatter(
                        x=chart_data_ta.index, y=ma,
                        mode='lines', name=name,
                        line=dict(color=color, width=1.5),
                    ))
            fig_ma.update_layout(**{**PLOTLY_LAYOUT,
                                    'height': 400, 'title': 'Moving Averages',
                                    'yaxis_title': 'Price (USD)'})
            st.plotly_chart(fig_ma, use_container_width=True)

    else:
        st.markdown('<div class="warn-box">⚠️ Unable to load stock data.</div>',
                    unsafe_allow_html=True)


# ╔══════════════════════════╗
# ║   TAB 4 — RAW DATA      ║
# ╚══════════════════════════╝
with tab_data:
    if stock_data is not None and not stock_data.empty:

        st.markdown('<div class="section-header">📋 Historical OHLCV Data</div>',
                    unsafe_allow_html=True)

        col_dl, col_rows = st.columns([1, 2])
        with col_rows:
            n_show = st.selectbox("Rows to display", [50, 100, 250, 500, 1000, "All"], index=1)

        display_df = stock_data.copy()
        display_df.index = display_df.index.strftime('%Y-%m-%d')
        display_df = display_df[::-1]   # newest first

        if n_show != "All":
            display_df = display_df.head(int(n_show))

        st.dataframe(display_df, use_container_width=True, height=480)

        with col_dl:
            csv = stock_data.to_csv().encode('utf-8')
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name=f"NVDA_stock_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv',
            )

        # ── Stats Summary ──
        st.markdown('<div class="section-header" style="margin-top:24px;">📐 Descriptive Statistics</div>',
                    unsafe_allow_html=True)
        st.dataframe(stock_data.describe().round(4), use_container_width=True)

    else:
        st.markdown('<div class="warn-box">⚠️ Unable to load stock data.</div>',
                    unsafe_allow_html=True)


# ╔══════════════════════════╗
# ║   TAB 5 — ABOUT         ║
# ╚══════════════════════════╝
with tab_about:
    st.markdown('<div class="section-header">ℹ️ About This Application</div>', unsafe_allow_html=True)

    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("""
        <div class="pro-card">
            <h3 style="color:#76b900; margin-top:0;">🤖 Model Architecture</h3>
            <table style="width:100%; font-size:0.88rem; border-collapse:collapse;">
                <tr><td style="color:#a0a0b0; padding:6px 0;">Model Type</td>
                    <td style="color:#e8e8e8; font-weight:600;">LSTM (Long Short-Term Memory)</td></tr>
                <tr><td style="color:#a0a0b0; padding:6px 0;">Look-Back Window</td>
                    <td style="color:#e8e8e8; font-weight:600;">5 Business Days</td></tr>
                <tr><td style="color:#a0a0b0; padding:6px 0;">LSTM Units</td>
                    <td style="color:#e8e8e8; font-weight:600;">150</td></tr>
                <tr><td style="color:#a0a0b0; padding:6px 0;">Test RMSE</td>
                    <td style="color:#e8e8e8; font-weight:600;">$1.32</td></tr>
                <tr><td style="color:#a0a0b0; padding:6px 0;">Input Feature</td>
                    <td style="color:#e8e8e8; font-weight:600;">Closing Price</td></tr>
                <tr><td style="color:#a0a0b0; padding:6px 0;">Scaler</td>
                    <td style="color:#e8e8e8; font-weight:600;">MinMax (0–1)</td></tr>
                <tr><td style="color:#a0a0b0; padding:6px 0;">Framework</td>
                    <td style="color:#e8e8e8; font-weight:600;">TensorFlow / Keras</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with c_right:
        st.markdown("""
        <div class="pro-card">
            <h3 style="color:#76b900; margin-top:0;">📚 Technology Stack</h3>
            <table style="width:100%; font-size:0.88rem; border-collapse:collapse;">
                <tr><td style="color:#a0a0b0; padding:6px 0;">Frontend</td>
                    <td style="color:#e8e8e8; font-weight:600;">Streamlit</td></tr>
                <tr><td style="color:#a0a0b0; padding:6px 0;">Data Source</td>
                    <td style="color:#e8e8e8; font-weight:600;">Yahoo Finance (yfinance)</td></tr>
                <tr><td style="color:#a0a0b0; padding:6px 0;">Charts</td>
                    <td style="color:#e8e8e8; font-weight:600;">Plotly (Interactive)</td></tr>
                <tr><td style="color:#a0a0b0; padding:6px 0;">ML Framework</td>
                    <td style="color:#e8e8e8; font-weight:600;">TensorFlow 2.x / Keras</td></tr>
                <tr><td style="color:#a0a0b0; padding:6px 0;">Data Processing</td>
                    <td style="color:#e8e8e8; font-weight:600;">NumPy · Pandas · scikit-learn</td></tr>
                <tr><td style="color:#a0a0b0; padding:6px 0;">Language</td>
                    <td style="color:#e8e8e8; font-weight:600;">Python 3.10+</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="pro-card" style="margin-top:8px;">
        <h3 style="color:#76b900; margin-top:0;">🔬 How It Works</h3>
        <ol style="font-size:0.9rem; color:#c0c0c0; line-height:1.9;">
            <li><b style="color:#e8e8e8;">Data Ingestion</b> — Historical NVDA close prices are fetched live from Yahoo Finance.</li>
            <li><b style="color:#e8e8e8;">Normalisation</b> — Prices are scaled to [0, 1] using MinMaxScaler for stable training.</li>
            <li><b style="color:#e8e8e8;">Sequence Creation</b> — A sliding window of 5 days is used as the input sequence.</li>
            <li><b style="color:#e8e8e8;">LSTM Inference</b> — The pre-trained LSTM model predicts the next closing price.</li>
            <li><b style="color:#e8e8e8;">Recursive Forecasting</b> — Each prediction becomes the next input (autoregressive loop).</li>
            <li><b style="color:#e8e8e8;">Inverse Transform</b> — Predictions are scaled back to USD for display.</li>
        </ol>
    </div>

    <div class="warn-box" style="margin-top:16px;">
        ⚠️ <b>Disclaimer:</b> This application is built purely for <b>educational and research purposes</b>.
        Stock market predictions are inherently uncertain. Past performance does not guarantee future results.
        <b>Do not make real financial decisions based on this tool.</b>
    </div>
    """, unsafe_allow_html=True)
