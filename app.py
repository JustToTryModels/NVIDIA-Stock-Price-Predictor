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
        --nvidia-green:   #76b900;
        --nvidia-dark:    #ffffff;
        --card-bg:        #f0f4e8;
        --card-border:    #c8e6a0;
        --accent:         #76b900;
        --text-primary:   #1a1a1a;
        --text-secondary: #4a5568;
        --danger:         #e53e3e;
        --warning:        #d97706;
        --success:        #38a169;
    }

    /* ── Global Base ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #ffffff !important;
        color: #1a1a1a !important;
    }

    .main {
        background: linear-gradient(135deg, #ffffff 0%, #f7faf0 50%, #eef5e0 100%);
        min-height: 100vh;
    }

    /* ── Hide Streamlit Defaults ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f7faf0 0%, #eef5e0 100%);
        border-right: 1px solid #c8e6a0;
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #4a7c00 !important;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] * {
        color: #1a1a1a !important;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0f7e0 100%);
        border: 1px solid #c8e6a0;
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #76b900, #a8e063);
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(118, 185, 0, 0.2);
    }

    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #4a5568;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1a1a1a;
        line-height: 1.2;
    }

    .metric-delta-positive { font-size: 0.85rem; color: #38a169; font-weight: 600; margin-top: 4px; }
    .metric-delta-negative { font-size: 0.85rem; color: #e53e3e; font-weight: 600; margin-top: 4px; }

    /* ── Hero Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #ffffff 0%, #f0f7e0 40%, #e6f2cc 100%);
        border: 1px solid #c8e6a0;
        border-radius: 20px;
        padding: 40px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }

    .hero-banner::after {
        content: '';
        position: absolute;
        top: -50%; right: -10%;
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(118,185,0,0.12) 0%, transparent 70%);
        pointer-events: none;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4a7c00, #76b900, #1a1a1a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 8px 0;
        line-height: 1.1;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #4a5568;
        font-weight: 400;
        margin: 0 0 20px 0;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(118,185,0,0.12);
        border: 1px solid rgba(118,185,0,0.4);
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #4a7c00;
        margin-right: 8px;
        margin-bottom: 8px;
    }

    /* ── Section Headers ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 32px 0 20px 0;
    }

    .section-header-line {
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, #c8e6a0, transparent);
    }

    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1a1a1a;
        white-space: nowrap;
    }

    /* ── Prediction Table ── */
    .pred-table-wrapper {
        background: linear-gradient(135deg, #ffffff 0%, #f0f7e0 100%);
        border: 1px solid #c8e6a0;
        border-radius: 16px;
        overflow: hidden;
    }

    .pred-row {
        display: grid;
        grid-template-columns: 50px 1fr 1fr 1fr;
        gap: 0;
        padding: 14px 20px;
        border-bottom: 1px solid rgba(0,0,0,0.06);
        align-items: center;
        transition: background 0.2s;
    }

    .pred-row:hover { background: rgba(118,185,0,0.07); }
    .pred-row.header { background: rgba(118,185,0,0.12); font-weight: 700; font-size: 0.75rem; letter-spacing: 1px; text-transform: uppercase; color: #4a7c00; }
    .pred-row:last-child { border-bottom: none; }

    .pred-cell { color: #1a1a1a; font-size: 0.9rem; }
    .pred-cell.positive { color: #38a169; font-weight: 600; }
    .pred-cell.negative { color: #e53e3e; font-weight: 600; }

    /* ── Predict Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #76b900 0%, #a8e063 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 32px !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        text-transform: uppercase !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(118,185,0,0.4) !important;
        background: linear-gradient(135deg, #a8e063 0%, #76b900 100%) !important;
    }

    .stButton > button:active { transform: translateY(0) !important; }

    /* ── Slider ── */
    .stSlider [data-baseweb="slider"] { padding: 0 4px; }
    .stSlider [data-testid="stThumbValue"] { color: #76b900 !important; }

    /* ── Info / Warning boxes ── */
    .info-box {
        background: rgba(118,185,0,0.08);
        border-left: 4px solid #76b900;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin: 16px 0;
        font-size: 0.9rem;
        color: #4a5568;
        line-height: 1.6;
    }

    .warning-box {
        background: rgba(217,119,6,0.08);
        border-left: 4px solid #d97706;
        border-radius: 0 12px 12px 0;
        padding: 16px 20px;
        margin: 16px 0;
        font-size: 0.9rem;
        color: #4a5568;
        line-height: 1.6;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #c8e6a0;
    }

    /* ── Progress & Spinner ── */
    .stProgress > div > div { background: #76b900 !important; }
    .stSpinner > div { border-top-color: #76b900 !important; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #f0f7e0;
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        border: 1px solid #c8e6a0;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #4a5568;
        font-weight: 600;
        padding: 10px 20px;
        border: none;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #76b900, #a8e063) !important;
        color: #ffffff !important;
    }

    /* ── Selectbox / Inputs ── */
    .stSelectbox [data-baseweb="select"] > div,
    .stMultiSelect [data-baseweb="select"] > div {
        background: #ffffff;
        border: 1px solid #c8e6a0;
        border-radius: 10px;
        color: #1a1a1a;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #f0f7e0; }
    ::-webkit-scrollbar-thumb { background: #c8e6a0; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #76b900; }

    /* ── General text override ── */
    p, span, div, label, li {
        color: #1a1a1a;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #1a1a1a !important;
    }

    /* Streamlit default widget labels */
    .stSlider label, .stSelectbox label, .stMultiSelect label,
    .stToggle label, [data-testid="stWidgetLabel"] {
        color: #1a1a1a !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# 🔹 Helper Utilities
# ==============================
def format_price(value):
    return f"${value:,.2f}"

def format_pct(value):
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"

def generate_business_days(start_date, num_days):
    return pd.bdate_range(start=start_date, periods=num_days).tolist()

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
        data      = yf.download(ticker, period='max', auto_adjust=True)
        info      = yf.Ticker(ticker).info
        return data, info
    except Exception as e:
        st.error(f"❌ Error fetching stock data: {e}")
        return None, {}

# ==============================
# 🔹 Prediction Function
# ==============================
def predict_next_business_days(model, data, look_back=5, days=5):
    scaler      = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    last_seq    = data_scaled[-look_back:]
    predictions = []

    for _ in range(days):
        X_input    = np.reshape(last_seq, (1, look_back, 1))
        pred       = model.predict(X_input, verbose=0)
        predictions.append(pred[0, 0])
        last_seq   = np.append(last_seq[1:], pred, axis=0)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# ==============================
# 🔹 Plotly Theme Helper
# ==============================
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(255,255,255,0)',
    plot_bgcolor='rgba(247,250,240,0.6)',
    font=dict(family='Inter', color='#1a1a1a'),
    xaxis=dict(
        gridcolor='rgba(0,0,0,0.06)',
        zerolinecolor='rgba(0,0,0,0.1)',
        showgrid=True,
        color='#1a1a1a',
        tickfont=dict(color='#1a1a1a'),
    ),
    yaxis=dict(
        gridcolor='rgba(0,0,0,0.06)',
        zerolinecolor='rgba(0,0,0,0.1)',
        showgrid=True,
        color='#1a1a1a',
        tickfont=dict(color='#1a1a1a'),
    ),
    legend=dict(
        bgcolor='rgba(240,247,224,0.9)',
        bordercolor='rgba(0,0,0,0.1)',
        borderwidth=1,
        font=dict(color='#1a1a1a'),
    ),
    margin=dict(l=0, r=0, t=40, b=0),
    hovermode='x unified',
)

# ==============================
# 🔹 Load Model at Start
# ==============================
model = load_nvidia_model()

# ══════════════════════════════
#  SIDEBAR
# ══════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px;">
        <div style="font-size:2.5rem;">🤖</div>
        <div style="font-size:1.1rem; font-weight:700; color:#4a7c00; margin-top:8px;">AI Stock Predictor</div>
        <div style="font-size:0.75rem; color:#4a5568; margin-top:4px;">LSTM Neural Network</div>
    </div>
    <hr style="border-color:#c8e6a0; margin:16px 0;">
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Forecast Settings")

    num_days = st.slider(
        "📅 Forecast Horizon (Business Days)",
        min_value=1, max_value=30, value=5,
        help="Number of future business days to predict"
    )

    st.markdown("### 📊 Chart Settings")

    history_days = st.select_slider(
        "Historical Window",
        options=[30, 60, 90, 180, 365, 730, 1825],
        value=365,
        format_func=lambda x: f"{x} days" if x < 365 else f"{x//365} yr{'s' if x//365>1 else ''}"
    )

    show_volume   = st.toggle("Show Volume Bars",  value=True)
    show_ma       = st.toggle("Show Moving Averages", value=True)

    if show_ma:
        ma_periods = st.multiselect(
            "MA Periods",
            options=[20, 50, 100, 200],
            default=[20, 50]
        )
    else:
        ma_periods = []

    st.markdown("<hr style='border-color:#c8e6a0; margin:16px 0;'>", unsafe_allow_html=True)

    # Model Info card
    st.markdown("""
    <div style="background:rgba(118,185,0,0.08); border:1px solid rgba(118,185,0,0.3); border-radius:12px; padding:16px;">
        <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:1px; color:#4a7c00; font-weight:600; margin-bottom:10px;">🧠 Model Info</div>
        <div style="font-size:0.8rem; color:#4a5568; line-height:1.8;">
            <div>Architecture &nbsp;→&nbsp; <span style="color:#1a1a1a;">LSTM</span></div>
            <div>Look-back &nbsp;&nbsp;&nbsp;&nbsp;→&nbsp; <span style="color:#1a1a1a;">5 Days</span></div>
            <div>Units &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→&nbsp; <span style="color:#1a1a1a;">150</span></div>
            <div>RMSE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→&nbsp; <span style="color:#4a7c00; font-weight:700;">$1.32</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    predict_btn = st.button(
        f"🚀 Predict Next {num_days} Days",
        key='forecast-button',
        use_container_width=True
    )

    st.markdown("""
    <div style="margin-top:24px; padding:12px; background:rgba(217,119,6,0.08); border-radius:10px; border:1px solid rgba(217,119,6,0.25);">
        <div style="font-size:0.7rem; color:#d97706; font-weight:600; margin-bottom:4px;">⚠️ DISCLAIMER</div>
        <div style="font-size:0.7rem; color:#4a5568; line-height:1.5;">
            This tool is for educational purposes only. Predictions are not financial advice. Always consult a qualified advisor before investing.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════

# ── Hero Banner ──────────────
st.markdown("""
<div class="hero-banner">
    <div style="display:flex; align-items:center; gap:24px; flex-wrap:wrap;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/NVIDIA_logo.svg/1920px-NVIDIA_logo.svg.png"
             style="height:48px;">
        <div>
            <div class="hero-title">Stock Price Predictor</div>
            <div class="hero-subtitle">AI-powered forecasting using Long Short-Term Memory neural networks</div>
            <div>
                <span class="hero-badge">🤖 LSTM Model</span>
                <span class="hero-badge">📡 Live Data</span>
                <span class="hero-badge">⚡ Real-time Prediction</span>
                <span class="hero-badge">📊 Interactive Charts</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Session State ─────────────
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

# ── Fetch Data ────────────────
stock      = 'NVDA'
stock_data, stock_info = get_stock_data(stock)

if stock_data is None or stock_data.empty:
    st.error("❌ Failed to load NVIDIA stock data. Please check your connection.")
    st.stop()

# ── Live Metrics ──────────────
close_prices_full = stock_data['Close'].values.reshape(-1, 1)
latest_close  = float(stock_data['Close'].iloc[-1])
prev_close    = float(stock_data['Close'].iloc[-2])
price_change  = latest_close - prev_close
price_pct     = (price_change / prev_close) * 100
latest_vol    = float(stock_data['Volume'].iloc[-1])
week_high     = float(stock_data['Close'].iloc[-5:].max())
week_low      = float(stock_data['Close'].iloc[-5:].min())
ytd_start     = float(stock_data[stock_data.index.year == datetime.now().year]['Close'].iloc[0])
ytd_return    = ((latest_close - ytd_start) / ytd_start) * 100
year_high     = float(stock_data['Close'].iloc[-252:].max())
year_low      = float(stock_data['Close'].iloc[-252:].min())

col1, col2, col3, col4, col5 = st.columns(5)
metrics = [
    (col1, "NVDA",          format_price(latest_close),
     f"{'▲' if price_change>=0 else '▼'} {format_price(abs(price_change))} ({format_pct(price_pct)})",
     price_change >= 0),
    (col2, "52-WEEK HIGH",  format_price(year_high),   "Past 252 Trading Days", True),
    (col3, "52-WEEK LOW",   format_price(year_low),    "Past 252 Trading Days", True),
    (col4, "YTD RETURN",    format_pct(ytd_return),    "Year-to-Date Performance", ytd_return >= 0),
    (col5, "VOLUME",        f"{latest_vol/1e6:.1f}M",  "Latest Trading Session", True),
]

for col, label, value, delta, is_positive in metrics:
    delta_class = "metric-delta-positive" if is_positive else "metric-delta-negative"
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="{delta_class}">{delta}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Predict Button Logic ──────
if predict_btn:
    if model is None:
        st.error("❌ Model not loaded. Cannot make predictions.")
    else:
        with st.spinner("🔮 Running LSTM prediction engine..."):
            predictions = predict_next_business_days(
                model, close_prices_full, look_back=5, days=num_days
            )
            last_date        = stock_data.index[-1]
            prediction_dates = generate_business_days(last_date + timedelta(days=1), num_days)

            st.session_state.prediction_results = {
                'predictions':      predictions,
                'prediction_dates': prediction_dates,
                'num_days':         num_days,
                'last_actual':      latest_close,
            }

# ══════════════════════════════
#  CHART SECTION
# ══════════════════════════════
st.markdown("""
<div class="section-header">
    <span class="section-title">📈 Price Chart &amp; Technical Analysis</span>
    <div class="section-header-line"></div>
</div>
""", unsafe_allow_html=True)

# ── Filter history window ─────
plot_data = stock_data.iloc[-history_days:].copy()
plot_data.index = pd.to_datetime(plot_data.index)

# ── Build Plotly Figure ───────
row_heights = [0.7, 0.3] if show_volume else [1.0]
n_rows      = 2 if show_volume else 1

fig = make_subplots(
    rows=n_rows, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=row_heights
)

# Candlestick
fig.add_trace(go.Candlestick(
    x=plot_data.index,
    open=plot_data['Open'].squeeze(),
    high=plot_data['High'].squeeze(),
    close=plot_data['Close'].squeeze(),
    low=plot_data['Low'].squeeze(),
    name='OHLC',
    increasing_line_color='#38a169',
    decreasing_line_color='#e53e3e',
    increasing_fillcolor='rgba(56,161,105,0.3)',
    decreasing_fillcolor='rgba(229,62,62,0.3)',
), row=1, col=1)

# Moving Averages
ma_colors = {20: '#d97706', 50: '#2563eb', 100: '#7c3aed', 200: '#db2777'}
for period in ma_periods:
    if len(plot_data) >= period:
        ma_vals = plot_data['Close'].squeeze().rolling(period).mean()
        fig.add_trace(go.Scatter(
            x=plot_data.index, y=ma_vals,
            name=f'MA{period}',
            line=dict(color=ma_colors.get(period, '#1a1a1a'), width=1.5, dash='dot'),
            opacity=0.85,
        ), row=1, col=1)

# Prediction overlay (if available)
if st.session_state.prediction_results is not None:
    res   = st.session_state.prediction_results
    pdts  = res['prediction_dates']
    pvals = res['predictions'].flatten()

    bridge_x = [plot_data.index[-1]] + list(pdts)
    bridge_y = [latest_close]        + list(pvals)

    fig.add_trace(go.Scatter(
        x=bridge_x, y=bridge_y,
        name='AI Forecast',
        line=dict(color='#76b900', width=3, dash='dash'),
        mode='lines+markers',
        marker=dict(size=8, color='#76b900', symbol='diamond',
                    line=dict(color='#1a1a1a', width=1.5)),
        fill='tonexty',
        fillcolor='rgba(118,185,0,0.07)',
    ), row=1, col=1)

    upper = [v * 1.02 for v in bridge_y]
    lower = [v * 0.98 for v in bridge_y]

    fig.add_trace(go.Scatter(
        x=bridge_x + bridge_x[::-1],
        y=upper + lower[::-1],
        fill='toself',
        fillcolor='rgba(118,185,0,0.08)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Confidence Band',
        showlegend=True,
        hoverinfo='skip',
    ), row=1, col=1)

# Volume bars
if show_volume:
    colors = [
        '#38a169' if float(plot_data['Close'].iloc[i]) >= float(plot_data['Open'].iloc[i])
        else '#e53e3e'
        for i in range(len(plot_data))
    ]
    fig.add_trace(go.Bar(
        x=plot_data.index,
        y=plot_data['Volume'].squeeze(),
        name='Volume',
        marker_color=colors,
        opacity=0.6,
    ), row=2, col=1)

fig.update_layout(
    **PLOTLY_LAYOUT,
    height=580 if show_volume else 460,
    title=dict(
        text=f"NVDA — {'Candlestick' if n_rows>1 else 'Price'} Chart  •  Last {history_days} Days",
        font=dict(size=15, color='#1a1a1a'), x=0.01
    ),
    xaxis_rangeslider_visible=False,
)
fig.update_yaxes(title_text="Price (USD)", row=1, col=1, tickprefix="$", color='#1a1a1a')
if show_volume:
    fig.update_yaxes(title_text="Volume", row=2, col=1, color='#1a1a1a')

st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════
#  PREDICTION RESULTS
# ══════════════════════════════
if st.session_state.prediction_results is not None:
    res          = st.session_state.prediction_results
    predictions  = res['predictions']
    pred_dates   = res['prediction_dates']
    stored_days  = res['num_days']
    last_actual  = res['last_actual']

    st.markdown("""
    <div class="section-header">
        <span class="section-title">🔮 Forecast Results</span>
        <div class="section-header-line"></div>
    </div>
    """, unsafe_allow_html=True)

    col_chart, col_table = st.columns([3, 2])

    # ── Prediction line chart ──
    with col_chart:
        pred_vals = predictions.flatten()
        changes   = [pred_vals[i] - (pred_vals[i-1] if i > 0 else last_actual)
                     for i in range(len(pred_vals))]
        bar_colors = ['#38a169' if c >= 0 else '#e53e3e' for c in changes]

        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=[stock_data.index[-1]] + list(pred_dates),
            y=[last_actual] + list(pred_vals),
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='#76b900', width=3),
            marker=dict(
                size=10, color='#76b900',
                symbol='diamond',
                line=dict(color='#1a1a1a', width=2)
            ),
            fill='tozeroy',
            fillcolor='rgba(118,185,0,0.1)',
        ))

        for i, (d, v) in enumerate(zip(pred_dates, pred_vals)):
            fig2.add_annotation(
                x=d, y=v,
                text=f"${v:,.2f}",
                showarrow=False,
                yshift=18,
                font=dict(size=11, color='#1a1a1a', family='Inter'),
                bgcolor='rgba(240,247,224,0.95)',
                bordercolor='#76b900',
                borderwidth=1,
                borderpad=4
            )

        fig2.add_trace(go.Scatter(
            x=[stock_data.index[-1]],
            y=[last_actual],
            mode='markers',
            name='Last Actual',
            marker=dict(size=12, color='#2563eb', symbol='circle',
                        line=dict(color='#1a1a1a', width=2))
        ))

        fig2.update_layout(
            **PLOTLY_LAYOUT,
            height=380,
            title=dict(
                text=f"Next {stored_days} Business Day Forecast — NVDA",
                font=dict(size=14, color='#1a1a1a'), x=0.01
            ),
        )
        fig2.update_yaxes(tickprefix="$", color='#1a1a1a')

        st.plotly_chart(fig2, use_container_width=True)

        fig3 = go.Figure(go.Bar(
            x=[d.strftime('%b %d') for d in pred_dates],
            y=changes,
            marker_color=bar_colors,
            text=[f"{'+'if c>=0 else ''}{c:.2f}" for c in changes],
            textposition='outside',
            textfont=dict(size=11, color='#1a1a1a'),
            name='Daily Change',
        ))
        fig3.update_layout(
            **PLOTLY_LAYOUT,
            height=240,
            title=dict(
                text="Predicted Day-over-Day Change ($)",
                font=dict(size=13, color='#1a1a1a'), x=0.01
            ),
            showlegend=False,
        )
        fig3.update_yaxes(tickprefix="$", color='#1a1a1a')
        st.plotly_chart(fig3, use_container_width=True)

    # ── Prediction Table ──────
    with col_table:
        st.markdown("<br>", unsafe_allow_html=True)

        pred_vals = predictions.flatten()
        all_prices = [last_actual] + list(pred_vals)

        table_rows_html = """
        <div class="pred-table-wrapper">
            <div class="pred-row header">
                <div>#</div>
                <div>Date</div>
                <div>Pred. Price</div>
                <div>Change</div>
            </div>
        """

        for i in range(len(pred_vals)):
            price  = pred_vals[i]
            change = price - all_prices[i]
            pct    = (change / all_prices[i]) * 100
            date_s = pred_dates[i].strftime('%b %d, %Y')
            sign   = "+" if change >= 0 else ""
            cls    = "positive" if change >= 0 else "negative"
            icon   = "▲" if change >= 0 else "▼"

            table_rows_html += f"""
            <div class="pred-row">
                <div class="pred-cell" style="color:#4a5568; font-size:0.8rem;">{i+1}</div>
                <div class="pred-cell">{date_s}</div>
                <div class="pred-cell" style="font-weight:700;">${price:,.2f}</div>
                <div class="pred-cell {cls}">{icon} {sign}{change:.2f} ({sign}{pct:.1f}%)</div>
            </div>
            """

        table_rows_html += "</div>"
        st.markdown(table_rows_html, unsafe_allow_html=True)

        total_return = pred_vals[-1] - last_actual
        total_pct    = (total_return / last_actual) * 100
        max_price    = max(pred_vals)
        min_price    = min(pred_vals)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#ffffff; border:1px solid #c8e6a0; border-radius:12px; padding:20px;">
            <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:1px; color:#4a7c00; font-weight:600; margin-bottom:14px;">
                📊 Forecast Summary
            </div>
        """, unsafe_allow_html=True)

        summary = [
            ("Current Price",    format_price(last_actual)),
            ("Target Price",     format_price(pred_vals[-1])),
            ("Total Change",     f"{'+'if total_return>=0 else ''}{format_price(total_return)}"),
            ("Total Return",     format_pct(total_pct)),
            ("Period High",      format_price(max_price)),
            ("Period Low",       format_price(min_price)),
        ]

        for label, value in summary:
            color = "#38a169" if "+" in str(value) else ("#e53e3e" if "-" in str(value) and label not in ["Current Price","Period High","Period Low"] else "#1a1a1a")
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding:8px 0; border-bottom:1px solid rgba(0,0,0,0.06);">
                <span style="color:#4a5568; font-size:0.82rem;">{label}</span>
                <span style="color:{color}; font-size:0.9rem; font-weight:600;">{value}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════
#  HISTORICAL DATA TABS
# ══════════════════════════════
st.markdown("""
<div class="section-header">
    <span class="section-title">📋 Data Explorer</span>
    <div class="section-header-line"></div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📈 Historical Prices", "📊 Return Distribution", "📉 Drawdown Analysis"])

with tab1:
    display_df = stock_data[['Open','High','Low','Close','Volume']].copy()

    if isinstance(display_df.columns, pd.MultiIndex):
        display_df.columns = [col[0] for col in display_df.columns]

    display_df = display_df.iloc[::-1]
    display_df.index = pd.to_datetime(display_df.index).strftime('%Y-%m-%d')

    for c in ['Open','High','Low','Close']:
        display_df[c] = display_df[c].squeeze().round(2)
    display_df['Volume'] = display_df['Volume'].squeeze().astype(int)

    st.dataframe(
        display_df,
        use_container_width=True,
        height=420,
    )

with tab2:
    close_series = stock_data['Close'].squeeze()
    returns = close_series.pct_change().dropna() * 100

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=returns,
        nbinsx=100,
        name='Daily Returns',
        marker_color='#76b900',
        opacity=0.75,
    ))
    fig_dist.add_vline(x=float(returns.mean()), line_dash='dash', line_color='#d97706',
                       annotation_text=f"Mean: {float(returns.mean()):.3f}%",
                       annotation_font_color='#d97706')
    fig_dist.add_vline(x=0, line_dash='solid', line_color='rgba(0,0,0,0.2)')

    fig_dist.update_layout(
        **PLOTLY_LAYOUT,
        height=400,
        title=dict(text="Distribution of Daily Returns (%)", font=dict(size=14, color='#1a1a1a'), x=0.01),
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    rc1, rc2, rc3, rc4 = st.columns(4)
    stats = [
        (rc1, "Mean Return",  f"{float(returns.mean()):.3f}%"),
        (rc2, "Std Dev",      f"{float(returns.std()):.3f}%"),
        (rc3, "Best Day",     f"+{float(returns.max()):.2f}%"),
        (rc4, "Worst Day",    f"{float(returns.min()):.2f}%"),
    ]
    for col, lbl, val in stats:
        col.markdown(f"""
        <div class="metric-card" style="padding:14px;">
            <div class="metric-label">{lbl}</div>
            <div class="metric-value" style="font-size:1.3rem;">{val}</div>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    close_series = stock_data['Close'].squeeze()
    rolling_max  = close_series.cummax()
    drawdown     = ((close_series - rolling_max) / rolling_max) * 100

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        fill='tozeroy',
        fillcolor='rgba(229,62,62,0.15)',
        line=dict(color='#e53e3e', width=1.5),
        name='Drawdown %',
    ))
    fig_dd.update_layout(
        **PLOTLY_LAYOUT,
        height=380,
        title=dict(text="Historical Drawdown from All-Time High (%)", font=dict(size=14, color='#1a1a1a'), x=0.01),
        yaxis_ticksuffix='%',
    )
    st.plotly_chart(fig_dd, use_container_width=True)

# ── Footer ───────────────────
st.markdown("""
<hr style="border-color:#c8e6a0; margin:40px 0 20px;">
<div style="text-align:center; color:#4a5568; font-size:0.8rem; padding-bottom:20px;">
    <div style="margin-bottom:6px;">
        Built with <span style="color:#4a7c00; font-weight:600;">Streamlit</span> &amp;
        <span style="color:#4a7c00; font-weight:600;">TensorFlow LSTM</span>
    </div>
    <div style="color:#718096;">
        ⚠️ For educational &amp; research purposes only — not financial advice.
    </div>
</div>
""", unsafe_allow_html=True)
