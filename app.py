# Code-1
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
    /* ── Import Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Root & Background ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0e1a 100%);
        color: #e2e8f0;
    }

    /* ── Hide Streamlit Branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b27 100%);
        border-right: 1px solid rgba(118, 185, 0, 0.2);
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #76b900;
    }

    /* ── Metric Cards ── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(22, 27, 39, 0.9), rgba(15, 20, 30, 0.9));
        border: 1px solid rgba(118, 185, 0, 0.25);
        border-radius: 16px;
        padding: 20px 24px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255,255,255,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(118, 185, 0, 0.15), inset 0 1px 0 rgba(255,255,255,0.05);
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }

    /* ── Slider ── */
    [data-testid="stSlider"] > div > div > div > div {
        background: #76b900 !important;
    }

    .stSlider [data-baseweb="slider"] {
        padding-top: 10px;
    }

    /* ── DataFrames ── */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(118, 185, 0, 0.2);
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #76b900 0%, #5a8f00 100%);
        color: #0a0a0f !important;
        border: none;
        border-radius: 12px;
        padding: 14px 32px;
        font-size: 1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.03em;
        cursor: pointer;
        transition: all 0.25s ease;
        box-shadow: 0 4px 16px rgba(118, 185, 0, 0.3);
        width: 100%;
        text-transform: uppercase;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(118, 185, 0, 0.5);
        background: linear-gradient(135deg, #8fd400 0%, #76b900 100%);
    }

    .stButton > button:active {
        transform: translateY(0px);
        box-shadow: 0 2px 8px rgba(118, 185, 0, 0.3);
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: rgba(22, 27, 39, 0.6);
        border: 1px solid rgba(118, 185, 0, 0.2);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }

    /* ── Tabs ── */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        background: rgba(15, 20, 30, 0.8);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        border: 1px solid rgba(118, 185, 0, 0.15);
    }

    [data-testid="stTabs"] [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #94a3b8 !important;
        font-weight: 500;
        padding: 10px 24px;
        transition: all 0.2s ease;
    }

    [data-testid="stTabs"] [aria-selected="true"] {
        background: linear-gradient(135deg, #76b900, #5a8f00) !important;
        color: #0a0a0f !important;
        font-weight: 700 !important;
    }

    /* ── Spinner ── */
    [data-testid="stSpinner"] {
        color: #76b900;
    }

    /* ── Select Box ── */
    [data-testid="stSelectbox"] > div > div {
        background: rgba(22, 27, 39, 0.9);
        border: 1px solid rgba(118, 185, 0, 0.3);
        border-radius: 10px;
        color: #e2e8f0;
    }

    /* ── Divider ── */
    hr {
        border: none;
        border-top: 1px solid rgba(118, 185, 0, 0.15);
        margin: 24px 0;
    }

    /* ── Custom Card ── */
    .glass-card {
        background: linear-gradient(135deg, rgba(22, 27, 39, 0.85), rgba(15, 20, 30, 0.85));
        border: 1px solid rgba(118, 185, 0, 0.2);
        border-radius: 20px;
        padding: 28px 32px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255,255,255,0.04);
        margin-bottom: 24px;
    }

    .glass-card h2, .glass-card h3 {
        color: #f1f5f9;
        margin-bottom: 8px;
    }

    /* ── Hero Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, rgba(118, 185, 0, 0.08) 0%, rgba(22, 27, 39, 0.95) 60%, rgba(15, 20, 30, 0.95) 100%);
        border: 1px solid rgba(118, 185, 0, 0.3);
        border-radius: 24px;
        padding: 40px 48px;
        margin-bottom: 32px;
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255,255,255,0.05);
        position: relative;
        overflow: hidden;
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

    /* ── Section Header ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 20px;
    }

    .section-header h3 {
        color: #f1f5f9;
        font-size: 1.25rem;
        font-weight: 700;
        margin: 0;
    }

    .section-badge {
        background: rgba(118, 185, 0, 0.15);
        border: 1px solid rgba(118, 185, 0, 0.4);
        color: #76b900;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* ── Info Box ── */
    .info-box {
        background: rgba(118, 185, 0, 0.06);
        border-left: 3px solid #76b900;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin: 12px 0;
        color: #cbd5e1;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .warning-box {
        background: rgba(251, 191, 36, 0.06);
        border-left: 3px solid #fbbf24;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin: 12px 0;
        color: #cbd5e1;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* ── Prediction Table ── */
    .pred-row-up {
        color: #4ade80;
        font-weight: 600;
    }

    .pred-row-down {
        color: #f87171;
        font-weight: 600;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(15, 20, 30, 0.5);
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(118, 185, 0, 0.4);
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(118, 185, 0, 0.7);
    }

    /* ── Sidebar items ── */
    .sidebar-stat {
        background: rgba(118, 185, 0, 0.07);
        border: 1px solid rgba(118, 185, 0, 0.18);
        border-radius: 10px;
        padding: 12px 16px;
        margin: 8px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .sidebar-stat-label {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .sidebar-stat-value {
        color: #f1f5f9;
        font-size: 0.95rem;
        font-weight: 700;
    }

    /* ── Status Indicator ── */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: #4ade80;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse-green 2s infinite;
    }

    @keyframes pulse-green {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.4); }
        50% { opacity: 0.8; box-shadow: 0 0 0 4px rgba(74, 222, 128, 0); }
    }
</style>
""", unsafe_allow_html=True)


# ==============================
# 🔹 Helper: Plotly Theme Config
# ==============================
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(13,17,23,0)',
    plot_bgcolor='rgba(13,17,23,0)',
    font=dict(family='Inter', color='#94a3b8', size=12),
    xaxis=dict(
        gridcolor='rgba(255,255,255,0.05)',
        linecolor='rgba(255,255,255,0.1)',
        tickfont=dict(color='#64748b'),
        title_font=dict(color='#94a3b8'),
        showgrid=True,
        zeroline=False
    ),
    yaxis=dict(
        gridcolor='rgba(255,255,255,0.05)',
        linecolor='rgba(255,255,255,0.1)',
        tickfont=dict(color='#64748b'),
        title_font=dict(color='#94a3b8'),
        showgrid=True,
        zeroline=False
    ),
    legend=dict(
        bgcolor='rgba(15,20,30,0.8)',
        bordercolor='rgba(118,185,0,0.3)',
        borderwidth=1,
        font=dict(color='#cbd5e1')
    ),
    margin=dict(l=16, r=16, t=48, b=16),
    hoverlabel=dict(
        bgcolor='rgba(15,20,30,0.95)',
        bordercolor='rgba(118,185,0,0.5)',
        font=dict(color='#f1f5f9', family='Inter')
    )
)


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
@st.cache_data(ttl=3600)
def get_stock_data(ticker='NVDA'):
    try:
        data = yf.download(ticker, period='max', auto_adjust=True)
        return data
    except Exception as e:
        return None


@st.cache_data(ttl=300)
def get_live_quote(ticker='NVDA'):
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        hist = t.history(period='2d')
        if len(hist) >= 2:
            prev_close = float(hist['Close'].iloc[-2])
            curr_price = float(hist['Close'].iloc[-1])
        else:
            curr_price = float(info.last_price)
            prev_close = float(info.previous_close) if hasattr(info, 'previous_close') else curr_price
        change = curr_price - prev_close
        change_pct = (change / prev_close) * 100
        return {
            'price': curr_price,
            'prev_close': prev_close,
            'change': change,
            'change_pct': change_pct,
            'market_cap': getattr(info, 'market_cap', None),
            'volume': getattr(info, 'three_month_average_volume', None),
        }
    except Exception:
        return None


# ==============================
# 🔹 Business Days
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
        pred = model.predict(X_input, verbose=0)
        predictions.append(pred[0, 0])
        last_sequence = np.append(last_sequence[1:], pred, axis=0)
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions


# ==============================
# 🔹 Chart Builders
# ==============================
def build_candlestick_chart(stock_data, predictions, prediction_dates, lookback_days=90):
    df = stock_data.tail(lookback_days).copy()

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.72, 0.28],
        subplot_titles=('', '')
    )

    # ── Candlestick ──
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'].squeeze(),
        high=df['High'].squeeze(),
        low=df['Low'].squeeze(),
        close=df['Close'].squeeze(),
        name='OHLC',
        increasing=dict(line=dict(color='#4ade80', width=1), fillcolor='rgba(74,222,128,0.8)'),
        decreasing=dict(line=dict(color='#f87171', width=1), fillcolor='rgba(248,113,113,0.8)'),
        whiskerwidth=0.5
    ), row=1, col=1)

    # ── 20-day MA ──
    close_series = df['Close'].squeeze()
    ma20 = close_series.rolling(window=20).mean()
    ma50 = close_series.rolling(window=50).mean()

    fig.add_trace(go.Scatter(
        x=df.index, y=ma20,
        name='MA 20', line=dict(color='#f59e0b', width=1.5, dash='dot'),
        opacity=0.85
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=ma50,
        name='MA 50', line=dict(color='#60a5fa', width=1.5, dash='dot'),
        opacity=0.85
    ), row=1, col=1)

    # ── Prediction Shaded Zone ──
    if predictions is not None and prediction_dates is not None:
        pred_flat = predictions.flatten()
        last_actual_price = float(df['Close'].iloc[-1])
        pred_x = [df.index[-1]] + list(prediction_dates)
        pred_y = [last_actual_price] + list(pred_flat)

        fig.add_trace(go.Scatter(
            x=pred_x + pred_x[::-1],
            y=[p * 1.015 for p in pred_y] + [p * 0.985 for p in pred_y[::-1]],
            fill='toself',
            fillcolor='rgba(118,185,0,0.07)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Forecast Band',
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=pred_x, y=pred_y,
            name='Forecast',
            line=dict(color='#76b900', width=2.5, dash='dash'),
            mode='lines+markers',
            marker=dict(size=7, color='#76b900', symbol='circle',
                        line=dict(color='#0a0a0f', width=1.5)),
        ), row=1, col=1)

    # ── Volume ──
    colors_vol = ['#4ade80' if c >= o else '#f87171'
                  for c, o in zip(close_series, df['Open'].squeeze())]

    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'].squeeze(),
        name='Volume',
        marker_color=colors_vol,
        opacity=0.65,
        showlegend=False
    ), row=2, col=1)

    layout = dict(**PLOTLY_LAYOUT)
    layout.update(dict(
        title=dict(text='<b>NVDA · Price Action & Forecast</b>',
                   font=dict(size=16, color='#f1f5f9'), x=0.02),
        xaxis2=dict(
            **PLOTLY_LAYOUT['xaxis'],
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(**PLOTLY_LAYOUT['yaxis'], title='Price (USD)'),
        yaxis2=dict(**PLOTLY_LAYOUT['yaxis'], title='Volume'),
        height=560,
        dragmode='pan',
        hovermode='x unified',
    ))
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.04)')
    return fig


def build_forecast_chart(prediction_dates, predictions, last_actual_price):
    pred_flat = predictions.flatten()
    dates_full = [pd.Timestamp(prediction_dates[0]) - timedelta(days=1)] + list(prediction_dates)
    prices_full = [last_actual_price] + list(pred_flat)
    colors = ['#f1f5f9'] + ['#4ade80' if p >= last_actual_price else '#f87171' for p in pred_flat]

    fig = go.Figure()

    # Fill area
    fig.add_trace(go.Scatter(
        x=dates_full, y=prices_full,
        fill='tozeroy',
        fillcolor='rgba(118,185,0,0.05)',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False, hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=dates_full, y=prices_full,
        name='Forecast',
        line=dict(color='#76b900', width=2.5),
        mode='lines+markers',
        marker=dict(size=10, color=colors, symbol='circle',
                    line=dict(color='#0a0a0f', width=2)),
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Price: <b>$%{y:.2f}</b><extra></extra>'
    ))

    # Reference line (last actual price)
    fig.add_hline(
        y=last_actual_price,
        line=dict(color='rgba(148,163,184,0.4)', width=1.5, dash='dot'),
        annotation_text=f'  Last Close: ${last_actual_price:.2f}',
        annotation_font=dict(color='#94a3b8', size=11)
    )

    layout = dict(**PLOTLY_LAYOUT)
    layout.update(dict(
        title=dict(text='<b>Forecast · Next Business Days</b>',
                   font=dict(size=16, color='#f1f5f9'), x=0.02),
        xaxis=dict(**PLOTLY_LAYOUT['xaxis'], tickformat='%b %d', title='Date'),
        yaxis=dict(**PLOTLY_LAYOUT['yaxis'], title='Predicted Price (USD)'),
        height=380,
        hovermode='x unified',
        showlegend=False
    ))
    fig.update_layout(**layout)
    return fig


def build_returns_chart(stock_data, days=252):
    df = stock_data.tail(days).copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    close = df['Close'].squeeze()
    returns = close.pct_change().dropna() * 100

    colors = ['#4ade80' if r >= 0 else '#f87171' for r in returns]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=returns.index, y=returns.values,
        marker_color=colors, opacity=0.8,
        name='Daily Return %',
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Return: <b>%{y:.2f}%</b><extra></extra>'
    ))

    layout = dict(**PLOTLY_LAYOUT)
    layout.update(dict(
        title=dict(text='<b>Daily Returns (1Y)</b>',
                   font=dict(size=16, color='#f1f5f9'), x=0.02),
        yaxis=dict(**PLOTLY_LAYOUT['yaxis'], title='Return (%)'),
        xaxis=dict(**PLOTLY_LAYOUT['xaxis'], title='Date'),
        height=320,
        hovermode='x unified',
    ))
    fig.update_layout(**layout)
    return fig


def build_volume_profile(stock_data, days=90):
    df = stock_data.tail(days).copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    close = df['Close'].squeeze()
    volume = df['Volume'].squeeze()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=volume,
        fill='tozeroy',
        fillcolor='rgba(96,165,250,0.15)',
        line=dict(color='#60a5fa', width=1.5),
        name='Volume',
        hovertemplate='<b>%{x|%b %d, %Y}</b><br>Vol: <b>%{y:,.0f}</b><extra></extra>'
    ))

    layout = dict(**PLOTLY_LAYOUT)
    layout.update(dict(
        title=dict(text='<b>Trading Volume (90D)</b>',
                   font=dict(size=16, color='#f1f5f9'), x=0.02),
        yaxis=dict(**PLOTLY_LAYOUT['yaxis'], title='Volume'),
        xaxis=dict(**PLOTLY_LAYOUT['xaxis'], title='Date'),
        height=280,
        hovermode='x unified',
        showlegend=False
    ))
    fig.update_layout(**layout)
    return fig


# ==============================
# 🔹 Load Model & Initial Data
# ==============================
model = load_nvidia_model()
STOCK = 'NVDA'

# ==============================
# 🔹 Sidebar
# ==============================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 8px 0 20px 0;'>
        <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/NVIDIA_logo.svg/1920px-NVIDIA_logo.svg.png'
             style='width:160px; filter: brightness(1.1);'>
        <p style='color:#64748b; font-size:0.75rem; margin-top:10px; letter-spacing:0.1em;'>
            STOCK INTELLIGENCE PLATFORM
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Model Status
    if model is not None:
        st.markdown("""
        <div style='display:flex; align-items:center; gap:8px; margin-bottom:16px;'>
            <span class='status-dot'></span>
            <span style='color:#4ade80; font-size:0.85rem; font-weight:600;'>LSTM Model Online</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='display:flex; align-items:center; gap:8px; margin-bottom:16px;'>
            <span style='display:inline-block; width:8px; height:8px; background:#f87171;
                         border-radius:50%; margin-right:6px;'></span>
            <span style='color:#f87171; font-size:0.85rem; font-weight:600;'>Model Offline</span>
        </div>
        """, unsafe_allow_html=True)

    # Model Specs
    st.markdown("#### ⚙️ Model Architecture")
    specs = [
        ("Architecture", "LSTM"),
        ("Look-Back Window", "5 Days"),
        ("Hidden Units", "150"),
        ("RMSE", "1.32"),
        ("Trained On", "Max History"),
    ]
    for label, val in specs:
        st.markdown(f"""
        <div class='sidebar-stat'>
            <span class='sidebar-stat-label'>{label}</span>
            <span class='sidebar-stat-value'>{val}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Forecast Settings
    st.markdown("#### 🎯 Forecast Settings")
    num_days = st.slider("Forecast Horizon (Days)", 1, 30, 5,
                         help="Number of business days to predict ahead")

    lookback_chart = st.selectbox(
        "Chart History Window",
        options=[30, 60, 90, 180, 365],
        index=2,
        format_func=lambda x: f"{x} Days"
    )

    st.markdown("---")
    st.markdown("""
    <div style='color:#475569; font-size:0.72rem; text-align:center; line-height:1.7;'>
        ⚠️ For educational purposes only.<br>
        Not financial advice.<br><br>
        Model predictions are based on<br>
        historical price patterns only.
    </div>
    """, unsafe_allow_html=True)


# ==============================
# 🔹 Hero Header
# ==============================
st.markdown("""
<div class='hero-banner'>
    <div style='display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:20px;'>
        <div>
            <div style='color:#76b900; font-size:0.8rem; font-weight:700; letter-spacing:0.15em;
                        text-transform:uppercase; margin-bottom:8px;'>
                AI-Powered · LSTM Neural Network
            </div>
            <h1 style='color:#f1f5f9; font-size:2.4rem; font-weight:800; margin:0; line-height:1.2;'>
                NVIDIA Stock <span style='color:#76b900;'>Predictor</span>
            </h1>
            <p style='color:#64748b; margin-top:10px; font-size:1rem; max-width:520px; line-height:1.6;'>
                Deep learning-powered price forecasting using Long Short-Term Memory networks
                trained on NVDA's complete trading history.
            </p>
        </div>
        <div style='text-align:right;'>
            <img src='https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/NVIDIA_logo.svg/1920px-NVIDIA_logo.svg.png'
                 style='width:200px; opacity:0.92;'>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================
# 🔹 Live Quote Strip
# ==============================
quote = get_live_quote(STOCK)

if quote:
    change_color = '#4ade80' if quote['change'] >= 0 else '#f87171'
    change_arrow = '▲' if quote['change'] >= 0 else '▼'
    delta_val = f"{change_arrow} {abs(quote['change']):.2f} ({abs(quote['change_pct']):.2f}%)"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("NVDA · Last Price", f"${quote['price']:.2f}",
                  delta=f"{change_arrow} {abs(quote['change']):.2f} ({abs(quote['change_pct']):.2f}%)")
    with c2:
        st.metric("Previous Close", f"${quote['prev_close']:.2f}")
    with c3:
        mktcap = quote.get('market_cap')
        if mktcap and mktcap > 0:
            if mktcap >= 1e12:
                mktcap_str = f"${mktcap/1e12:.2f}T"
            else:
                mktcap_str = f"${mktcap/1e9:.1f}B"
        else:
            mktcap_str = "N/A"
        st.metric("Market Cap", mktcap_str)
    with c4:
        vol = quote.get('volume')
        vol_str = f"{vol/1e6:.1f}M" if vol and vol > 0 else "N/A"
        st.metric("Avg Volume", vol_str)

st.markdown("---")

# ==============================
# 🔹 Session State Init
# ==============================
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None
if 'last_num_days' not in st.session_state:
    st.session_state.last_num_days = 5

# ==============================
# 🔹 Predict Button
# ==============================
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    run_prediction = st.button(
        f"🚀 Generate {num_days}-Day Forecast",
        key='forecast-button',
        use_container_width=True
    )

if run_prediction:
    if model is None:
        st.markdown("""
        <div class='warning-box'>
            <b>⚠️ Model Not Available</b><br>
            The LSTM model file could not be loaded. Please verify the model path and file integrity.
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("⚡ Running LSTM inference..."):
            stock_data = get_stock_data(STOCK)

            if stock_data is None or stock_data.empty:
                st.markdown("""
                <div class='warning-box'>
                    ❌ Failed to fetch stock data. Please check your internet connection.
                </div>
                """, unsafe_allow_html=True)
            else:
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
                    'stock': STOCK,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }

# ==============================
# 🔹 Display Results
# ==============================
if st.session_state.prediction_results is not None:
    r = st.session_state.prediction_results

    stock_data = r['stock_data']
    close_prices = r['close_prices']
    predictions = r['predictions']
    prediction_dates = r['prediction_dates']
    stored_num_days = r['num_days']
    ts = r.get('timestamp', '')
    pred_flat = predictions.flatten()

    # Flatten MultiIndex if needed
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data_display = stock_data.copy()
        stock_data_display.columns = [col[0] for col in stock_data_display.columns]
    else:
        stock_data_display = stock_data

    last_actual_price = float(stock_data_display['Close'].iloc[-1])
    final_pred_price = float(pred_flat[-1])
    pred_change = final_pred_price - last_actual_price
    pred_change_pct = (pred_change / last_actual_price) * 100

    # ── Forecast Summary Cards ──
    st.markdown(f"""
    <div class='glass-card'>
        <div class='section-header'>
            <h3>📊 Forecast Summary</h3>
            <span class='section-badge'>LSTM Prediction</span>
        </div>
        <p style='color:#64748b; font-size:0.82rem; margin:-8px 0 16px 0;'>
            Generated at {ts} · {stored_num_days}-day horizon
        </p>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Last Close", f"${last_actual_price:.2f}")
    with m2:
        st.metric("Predicted Day 1", f"${pred_flat[0]:.2f}",
                  delta=f"{pred_flat[0]-last_actual_price:+.2f}")
    with m3:
        direction = "▲" if pred_change >= 0 else "▼"
        st.metric(f"End of Forecast ({stored_num_days}D)", f"${final_pred_price:.2f}",
                  delta=f"{direction} {abs(pred_change_pct):.2f}%")
    with m4:
        avg_pred = float(np.mean(pred_flat))
        st.metric("Avg Forecast Price", f"${avg_pred:.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈  Price Action & Forecast",
        "🔮  Forecast Detail",
        "📉  Returns Analysis",
        "📋  Historical Data"
    ])

    # ─── Tab 1: Candlestick + Forecast overlay ───
    with tab1:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        fig_candle = build_candlestick_chart(
            stock_data, predictions, prediction_dates,
            lookback_days=lookback_chart
        )
        st.plotly_chart(fig_candle, use_container_width=True, config={
            'displayModeBar': True,
            'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
            'displaylogo': False,
            'scrollZoom': True
        })

        st.markdown("""
        <div class='info-box'>
            🕯️ <b>Reading the chart:</b> Green candles indicate price closed higher than open;
            red candles indicate the opposite. The dashed green line represents the LSTM model's
            forecast trajectory. MA 20 and MA 50 are moving averages overlaid for trend reference.
        </div>
        """, unsafe_allow_html=True)

    # ─── Tab 2: Forecast Detail ───
    with tab2:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        fig_forecast = build_forecast_chart(prediction_dates, predictions, last_actual_price)
        st.plotly_chart(fig_forecast, use_container_width=True, config={
            'displayModeBar': False,
            'displaylogo': False
        })

        st.markdown("<br>", unsafe_allow_html=True)

        # Prediction Table
        pred_df = pd.DataFrame({
            'Business Day': [f"Day {i+1}" for i in range(stored_num_days)],
            'Date': [d.strftime('%A, %b %d %Y') for d in prediction_dates],
            'Predicted Price': [f"${p:.2f}" for p in pred_flat],
            'Change vs Close': [f"{p - last_actual_price:+.2f}" for p in pred_flat],
            'Change %': [f"{((p - last_actual_price) / last_actual_price * 100):+.2f}%" for p in pred_flat],
            'Signal': ['🟢 BUY' if p >= last_actual_price else '🔴 SELL' for p in pred_flat]
        })

        st.markdown("""
        <div class='section-header' style='margin-bottom:12px;'>
            <h3>Detailed Forecast Table</h3>
            <span class='section-badge'>Day-by-Day</span>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(
            pred_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Signal': st.column_config.TextColumn('Signal', width='small'),
                'Predicted Price': st.column_config.TextColumn('Predicted Price', width='medium'),
            }
        )

        st.markdown("""
        <div class='warning-box'>
            ⚠️ <b>Disclaimer:</b> Signals shown are derived purely from model output relative to last
            close price. They are <b>not</b> financial advice. Past model performance does not
            guarantee future accuracy. Always consult a licensed financial advisor.
        </div>
        """, unsafe_allow_html=True)

    # ─── Tab 3: Returns Analysis ───
    with tab3:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        fig_ret = build_returns_chart(stock_data, days=252)
        st.plotly_chart(fig_ret, use_container_width=True, config={
            'displayModeBar': False, 'displaylogo': False
        })

        fig_vol = build_volume_profile(stock_data, days=lookback_chart)
        st.plotly_chart(fig_vol, use_container_width=True, config={
            'displayModeBar': False, 'displaylogo': False
        })

        # Stats row
        if isinstance(stock_data_display.columns, pd.MultiIndex):
            close_s = stock_data_display['Close'].squeeze()
        else:
            close_s = stock_data_display['Close'].squeeze()

        ret_1y = close_s.tail(252).pct_change().dropna() * 100

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class='section-header'>
            <h3>Return Statistics (1Y)</h3>
            <span class='section-badge'>Annualized</span>
        </div>
        """, unsafe_allow_html=True)

        rs1, rs2, rs3, rs4 = st.columns(4)
        with rs1:
            st.metric("Mean Daily Return", f"{ret_1y.mean():.3f}%")
        with rs2:
            st.metric("Std Deviation", f"{ret_1y.std():.3f}%")
        with rs3:
            best = ret_1y.max()
            st.metric("Best Day", f"+{best:.2f}%")
        with rs4:
            worst = ret_1y.min()
            st.metric("Worst Day", f"{worst:.2f}%")

    # ─── Tab 4: Historical Data ───
    with tab4:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='section-header'>
            <h3>Full Historical Dataset</h3>
            <span class='section-badge'>Max History</span>
        </div>
        """, unsafe_allow_html=True)

        disp = stock_data_display.sort_index(ascending=False).copy()
        # Round numeric columns
        for col in disp.select_dtypes(include=np.number).columns:
            disp[col] = disp[col].round(4)

        st.dataframe(disp, height=480, use_container_width=True)

        # Download button
        csv = disp.to_csv().encode('utf-8')
        col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 1])
        with col_dl2:
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name=f"NVDA_historical_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv',
                use_container_width=True
            )

else:
    # ── Placeholder State ──
    st.markdown("""
    <div class='glass-card' style='text-align:center; padding: 60px 40px;'>
        <div style='font-size:4rem; margin-bottom:16px;'>📡</div>
        <h2 style='color:#f1f5f9; font-size:1.6rem; margin-bottom:12px;'>Ready to Forecast</h2>
        <p style='color:#64748b; max-width:420px; margin:0 auto; line-height:1.7; font-size:0.95rem;'>
            Configure your forecast horizon in the sidebar, then click
            <b style='color:#76b900;'>Generate Forecast</b> to run the LSTM model
            and visualize predicted NVDA price movements.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Show basic chart even without prediction
    with st.spinner("Loading market data..."):
        stock_data_preview = get_stock_data(STOCK)
        if stock_data_preview is not None and not stock_data_preview.empty:
            fig_prev = build_candlestick_chart(
                stock_data_preview, None, None, lookback_days=90
            )
            st.plotly_chart(fig_prev, use_container_width=True, config={
                'displayModeBar': True,
                'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
                'displaylogo': False,
                'scrollZoom': True
            })




# Code-2
import streamlit as st
import torch
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM
)
from gliner import GLiNER
import time
import random

# =============================
# MODEL AND CONFIGURATION SETUP
# =============================

# Hugging Face model IDs
DistilGPT2_MODEL_ID = "IamPradeep/AETCSCB_OOD_IC_DistilGPT2_Fine-tuned"
CLASSIFIER_ID = "IamPradeep/Query_Classifier_DistilBERT"

# Random OOD Fallback Responses
fallback_responses = [
    "I'm sorry, but I am unable to assist with this request. If you need help regarding event tickets, I'd be happy to support you.",
    "Apologies, but I am not able to provide assistance on this matter. Please let me know if you require help with event tickets.",
    "Unfortunately, I cannot assist with this. However, I am here to help with any event ticket-related concerns you may have.",
    "Regrettably, I am unable to assist with this request. If there's anything I can do regarding event tickets, feel free to ask.",
    "I regret that I am unable to assist in this case. Please reach out if you need support related to event tickets.",
    "Apologies, but this falls outside the scope of my support. I'm here if you need any help with event ticket issues.",
    "I'm sorry, but I cannot assist with this particular topic. If you have questions about event tickets, I'd be glad to help.",
    "I regret that I'm unable to provide assistance here. Please let me know how I can support you with event ticket matters.",
    "Unfortunately, I am not equipped to assist with this. If you need help with event tickets, I am here for that.",
    "I apologize, but I cannot help with this request. However, I'd be happy to assist with anything related to event tickets.",
    "I'm sorry, but I'm unable to support this request. If it's about event tickets, I'll gladly help however I can.",
    "This matter falls outside the assistance I can offer. Please let me know if you need help with event ticket-related inquiries.",
    "Regrettably, this is not something I can assist with. I'm happy to help with any event ticket questions you may have.",
    "I'm unable to provide support for this issue. However, I can assist with concerns regarding event tickets.",
    "I apologize, but I cannot help with this matter. If your inquiry is related to event tickets, I'd be more than happy to assist.",
    "I regret that I am unable to offer help in this case. I am, however, available for any event ticket-related questions.",
    "Unfortunately, I'm not able to assist with this. Please let me know if there's anything I can do regarding event tickets.",
    "I'm sorry, but I cannot assist with this topic. However, I'm here to help with any event ticket concerns you may have.",
    "Apologies, but this request falls outside of my support scope. If you need help with event tickets, I'm happy to assist.",
    "I'm afraid I can't help with this matter. If there's anything related to event tickets you need, feel free to reach out.",
    "This is beyond what I can assist with at the moment. Let me know if there's anything I can do to help with event tickets.",
    "Sorry, I'm unable to provide support on this issue. However, I'd be glad to assist with event ticket-related topics.",
    "Apologies, but I can't assist with this. Please let me know if you have any event ticket inquiries I can help with.",
    "I'm unable to help with this matter. However, if you need assistance with event tickets, I'm here for you.",
    "Unfortunately, I can't support this request. I'd be happy to assist with anything related to event tickets instead.",
    "I'm sorry, but I can't help with this. If your concern is related to event tickets, I'll do my best to assist.",
    "Apologies, but this issue is outside of my capabilities. However, I'm available to help with event ticket-related requests.",
    "I regret that I cannot assist with this particular matter. Please let me know how I can support you regarding event tickets.",
    "I'm sorry, but I'm not able to help in this instance. I am, however, ready to assist with any questions about event tickets.",
    "Unfortunately, I'm unable to help with this topic. Let me know if there's anything event ticket-related I can support you with."
]

# =============================
# MODEL LOADING FUNCTIONS
# =============================

@st.cache_resource
def load_spell_corrector():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("oliverguhr/spelling-correction-english-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("oliverguhr/spelling-correction-english-base")
    model.to(device)
    model.eval()
    return model, tokenizer

@st.cache_resource
def load_gliner_model():
    # GLiNER handles device mapping internally if possible
    model = GLiNER.from_pretrained("gliner-community/gliner_small-v2.5")
    return model

@st.cache_resource(show_spinner=False)
def load_gpt2_model_and_tokenizer():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GPT2LMHeadModel.from_pretrained(DistilGPT2_MODEL_ID, trust_remote_code=True)
        tokenizer = GPT2Tokenizer.from_pretrained(DistilGPT2_MODEL_ID)
        model.to(device) # Move to device ONCE during load
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load GPT-2 model from Hugging Face Hub. Error: {e}")
        return None, None

@st.cache_resource(show_spinner=False)
def load_classifier_model():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_ID)
        model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_ID)
        model.to(device) # Move to device ONCE during load
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load classifier model from Hugging Face Hub. Error: {e}")
        return None, None

def preprocess_query(query: str, spell_corrector, query_tokenizer, max_tokens: int = 128):
    spell_model, spell_tokenizer = spell_corrector
    query = query.strip()
    if len(query) == 0:
        return query, None
    query = query[0].upper() + query[1:].lower()
    tokens = query_tokenizer.encode(query, add_special_tokens=True)
    token_count = len(tokens)
    if token_count > max_tokens:
        error_msg = "⚠️ Your question is too long. Try something shorter like: <b>'How do I get a refund?'</b>"
        return None, error_msg
    try:
        device = next(spell_model.parameters()).device
        inputs = spell_tokenizer(query, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = spell_model.generate(**inputs, max_length=256)
        corrected = spell_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        if corrected:
            query = corrected
    except Exception as e:
        print(f"Spell correction error: {e}")
    return query, None

def is_ood(query: str, model, tokenizer):
    # Detect which device the model is already on
    device = next(model.parameters()).device
    model.eval()
    
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pred_id = torch.argmax(outputs.logits, dim=1).item()
    return pred_id == 1 

# =============================
# ORIGINAL HELPER FUNCTIONS
# =============================

static_placeholders = {
    "{{WEBSITE_URL}}": "[website](https://github.com/MarpakaPradeepSai)",
    "{{SUPPORT_TEAM_LINK}}": "[support team](https://github.com/MarpakaPradeepSai)",
    "{{CONTACT_SUPPORT_LINK}}" : "[support team](https://github.com/MarpakaPradeepSai)",
    "{{SUPPORT_CONTACT_LINK}}" : "[support team](https://github.com/MarpakaPradeepSai)",
    "{{CANCEL_TICKET_SECTION}}": "<b>Ticket Cancellation</b>",
    "{{CANCEL_TICKET_OPTION}}": "<b>Cancel Ticket</b>",
    "{{GET_REFUND_OPTION}}": "<b>Get Refund</b>",
    "{{UPGRADE_TICKET_INFORMATION}}": "<b>Upgrade Ticket Information</b>",
    "{{TICKET_SECTION}}": "<b>Ticketing</b>",
    "{{CANCELLATION_POLICY_SECTION}}": "<b>Cancellation Policy</b>",
    "{{CHECK_CANCELLATION_POLICY_OPTION}}": "<b>Check Cancellation Policy</b>",
    "{{APP}}": "<b>App</b>",
    "{{CHECK_CANCELLATION_FEE_OPTION}}": "<b>Check Cancellation Fee</b>",
    "{{CHECK_REFUND_POLICY_OPTION}}": "<b>Check Refund Policy</b>",
    "{{CHECK_PRIVACY_POLICY_OPTION}}": "<b>Check Privacy Policy</b>",
    "{{SAVE_BUTTON}}": "<b>Save</b>",
    "{{EDIT_BUTTON}}": "<b>Edit</b>",
    "{{CANCELLATION_FEE_SECTION}}": "<b>Cancellation Fee</b>",
    "{{CHECK_CANCELLATION_FEE_INFORMATION}}": "<b>Check Cancellation Fee Information</b>",
    "{{PRIVACY_POLICY_LINK}}": "<b>Privacy Policy</b>",
    "{{REFUND_SECTION}}": "<b>Refund</b>",
    "{{REFUND_POLICY_LINK}}": "<b>Refund Policy</b>",
    "{{CUSTOMER_SERVICE_SECTION}}": "<b>Customer Service</b>",
    "{{DELIVERY_PERIOD_INFORMATION}}": "<b>Delivery Period</b>",
    "{{EVENT_ORGANIZER_OPTION}}": "<b>Event Organizer</b>",
    "{{FIND_TICKET_OPTION}}": "<b>Find Ticket</b>",
    "{{FIND_UPCOMING_EVENTS_OPTION}}": "<b>Find Upcoming Events</b>",
    "{{CONTACT_SECTION}}": "<b>Contact</b>",
    "{{SEARCH_BUTTON}}": "<b>Search</b>",
    "{{SUPPORT_SECTION}}": "<b>Support</b>",
    "{{EVENTS_SECTION}}": "<b>Events</b>",
    "{{EVENTS_PAGE}}": "<b>Events</b>",
    "{{TYPE_EVENTS_OPTION}}": "<b>Type Events</b>",
    "{{PAYMENT_SECTION}}": "<b>Payment</b>",
    "{{PAYMENT_OPTION}}": "<b>Payment</b>",
    "{{CANCELLATION_SECTION}}": "<b>Cancellation</b>",
    "{{CANCELLATION_OPTION}}": "<b>Cancellation</b>",
    "{{REFUND_OPTION}}": "<b>Refund</b>",
    "{{TRANSFER_TICKET_OPTION}}": "<b>Transfer Ticket</b>",
    "{{REFUND_STATUS_OPTION}}": "<b>Refund Status</b>",
    "{{DELIVERY_SECTION}}": "<b>Delivery</b>",
    "{{SELL_TICKET_OPTION}}": "<b>Sell Ticket</b>",
    "{{CANCELLATION_FEE_INFORMATION}}": "<b>Cancellation Fee Information</b>",
    "{{CUSTOMER_SUPPORT_PAGE}}": "<b>Customer Support</b>",
    "{{PAYMENT_METHOD}}" : "<b>Payment</b>",
    "{{VIEW_PAYMENT_METHODS}}": "<b>View Payment Methods</b>",
    "{{VIEW_CANCELLATION_POLICY}}": "<b>View Cancellation Policy</b>",
    "{{SUPPORT_ SECTION}}" : "<b>Support</b>",
    "{{CUSTOMER_SUPPORT_SECTION}}" : "<b>Customer Support</b>",
    "{{HELP_SECTION}}" : "<b>Help</b>",
    "{{TICKET_INFORMATION}}" : "<b>Ticket Information</b>",
    "{{UPGRADE_TICKET_BUTTON}}" : "<b>Upgrade Ticket</b>",
    "{{CANCEL_TICKET_BUTTON}}" : "<b>Cancel Ticket</b>",
    "{{GET_REFUND_BUTTON}}" : "<b>Get Refund</b>",
    "{{PAYMENTS_HELP_SECTION}}" : "<b>Payments Help</b>",
    "{{PAYMENTS_PAGE}}" : "<b>Payments</b>",
    "{{TICKET_DETAILS}}" : "<b>Ticket Details</b>",
    "{{TICKET_INFORMATION_PAGE}}" : "<b>Ticket Information</b>",
    "{{REPORT_PAYMENT_PROBLEM}}" : "<b>Report Payment</b>",
    "{{TICKET_OPTIONS}}" : "<b>Ticket Options</b>",
    "{{SEND_BUTTON}}" : "<b>Send</b>",
    "{{PAYMENT_ISSUE_OPTION}}" : "<b>Payment Issue</b>",
    "{{CUSTOMER_SUPPORT_PORTAL}}" : "<b>Customer Support</b>",
    "{{UPGRADE_TICKET_OPTION}}" : "<b>Upgrade Ticket</b>",
    "{{TICKET_AVAILABILITY_TAB}}" : "<b>Ticket Availability</b>",
    "{{TRANSFER_TICKET_BUTTON}}" : "<b>Transfer Ticket</b>",
    "{{TICKET_MANAGEMENT}}" : "<b>Ticket Management</b>",
    "{{TICKET_STATUS_TAB}}" : "<b>Ticket Status</b>",
    "{{TICKETING_PAGE}}" : "<b>Ticketing</b>",
    "{{TICKET_TRANSFER_TAB}}" : "<b>Ticket Transfer</b>",
    "{{CURRENT_TICKET_DETAILS}}" : "<b>Current Ticket Details</b>",
    "{{UPGRADE_OPTION}}" : "<b>Upgrade</b>",
    "{{CONNECT_WITH_ORGANIZER}}" : "<b>Connect with Organizer</b>",
    "{{TICKETS_TAB}}" : "<b>Tickets</b>",
    "{{ASSISTANCE_SECTION}}" : "<b>Assistance Section</b>",
}

def replace_placeholders(response, dynamic_placeholders, static_placeholders):
    for placeholder, value in static_placeholders.items():
        response = response.replace(placeholder, value)
    for placeholder, value in dynamic_placeholders.items():
        response = response.replace(placeholder, value)
    return response

def extract_dynamic_placeholders(user_question, gliner_model):
    labels = ["event", "city", "location", "concert", "festival", "show", "match", "game"]
    entities = gliner_model.predict_entities(user_question, labels, threshold=0.4)
    
    dynamic_placeholders = {'{{EVENT}}': "event", '{{CITY}}': "city"}
    
    for ent in entities:
        if ent["label"] in ["event", "concert", "festival", "show", "match", "game"]:
            dynamic_placeholders['{{EVENT}}'] = f"<b>{ent['text'].title()}</b>"
        elif ent["label"] in ["city", "location", "venue"]:
            dynamic_placeholders['{{CITY}}'] = f"<b>{ent['text']}</b>"
    
    return dynamic_placeholders

def generate_response(model, tokenizer, instruction, max_length=256):
    model.eval()
    # Detect which device the model is already on
    device = next(model.parameters()).device
    input_text = f"Instruction: {instruction} Response:"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.5,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_start = response.find("Response:") + len("Response:")
    return response[response_start:].strip()

# =============================
# CSS AND UI SETUP
# =============================

st.markdown(
    """
<style>
/* Change font to Tiempos for the entire app with global font size */
* { 
    font-family: 'Tiempos', 'Tiempos Text', Georgia, serif !important;
    font-size: 15px !important;
}

h1 {
    font-size: 38px !important;
}

.stButton>button { 
    background: linear-gradient(90deg, #ff8a00, #e52e71); 
    color: white !important; 
    border: none; 
    border-radius: 25px; 
    padding: 10px 20px; 
    font-size: 1.2em !important; 
    font-weight: bold; 
    cursor: pointer; 
    transition: transform 0.2s ease, box-shadow 0.2s ease; 
    display: inline-flex; 
    align-items: center; 
    justify-content: center; 
    margin-top: 5px; 
    width: auto; 
    min-width: 100px;
    font-family: 'Tiempos', 'Tiempos Text', Georgia, serif !important;
}

.stButton>button:hover { transform: scale(1.05); box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.3); color: white !important; }
.stButton>button:active { transform: scale(0.98); }

div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:nth-of-type(1) { background: linear-gradient(90deg, #29ABE2, #0077B6); color: white !important; }
.horizontal-line { border-top: 2px solid #e0e0e0; margin: 15px 0; }
div[data-testid="stChatInput"] { box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); border-radius: 5px; padding: 10px; margin: 10px 0; }

.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background: var(--streamlit-background-color);
    color: gray;
    text-align: center;
    padding: 5px 0;
    font-size: 13px !important;
    z-index: 9999;
}
.main { padding-bottom: 40px; }
</style>
    """, unsafe_allow_html=True
)

st.markdown(
    """
    <div class="footer">
        It is designed solely for <b>event ticketing</b> queries. Responses outside this scope may be inaccurate.
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1>Advanced Event Ticketing Chatbot</h1>", unsafe_allow_html=True)

# --- Initialize state variables for managing generation process ---
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "generating" not in st.session_state:
    st.session_state.generating = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

example_queries = [
    "How do I buy a ticket?", "How can I upgrade my ticket for the upcoming event in Hyderabad?",
    "How do I change my personal details on my ticket?", "How can I find details about upcoming events?",
    "How do I contact customer service?", "How do I get a refund?", "What is the ticket cancellation fee?",
    "How can I track my ticket cancellation status?", "How can I sell my ticket?"
]

if not st.session_state.models_loaded:
    with st.spinner("Loading models and resources... Please wait..."):
        try:
            spell_corrector = load_spell_corrector()
            gliner_model = load_gliner_model()
            gpt2_model, gpt2_tokenizer = load_gpt2_model_and_tokenizer()
            clf_model, clf_tokenizer = load_classifier_model()

            if all([spell_corrector, gliner_model, gpt2_model, gpt2_tokenizer, clf_model, clf_tokenizer]):
                st.session_state.models_loaded = True
                st.session_state.spell_corrector = spell_corrector
                st.session_state.gliner_model = gliner_model
                st.session_state.model = gpt2_model
                st.session_state.tokenizer = gpt2_tokenizer
                st.session_state.clf_model = clf_model
                st.session_state.clf_tokenizer = clf_tokenizer
                st.rerun()
            else:
                st.error("Failed to load one or more models. Please refresh the page.")
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")

# ==================================
# MAIN CHAT INTERFACE
# ==================================

if st.session_state.models_loaded:
    st.write("Ask me about ticket bookings, cancellations, refunds, or any event-related inquiries!")

    # Disable input widgets while generating a response
    selected_query = st.selectbox(
        "Choose a query from examples:", ["Choose your question"] + example_queries,
        key="query_selectbox", label_visibility="collapsed",
        disabled=st.session_state.generating
    )
    process_query_button = st.button(
        "Ask this question", key="query_button",
        disabled=st.session_state.generating
    )

    spell_corrector = st.session_state.spell_corrector
    gliner_model = st.session_state.gliner_model
    model = st.session_state.model
    tokenizer = st.session_state.tokenizer
    clf_model = st.session_state.clf_model
    clf_tokenizer = st.session_state.clf_tokenizer

    last_role = None

    for message in st.session_state.chat_history:
        if message["role"] == "user" and last_role == "assistant":
            st.markdown("<div class='horizontal-line'></div>", unsafe_allow_html=True)
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.markdown(message["content"], unsafe_allow_html=True)
        last_role = message["role"]

    def handle_prompt(prompt_text):
        if not prompt_text or not prompt_text.strip():
            st.toast("⚠️ Please enter or select a question.")
            return

        original_text = prompt_text
        
        # Preprocess and check token length using DistilGPT2 tokenizer
        processed_text, error_message = preprocess_query(
            prompt_text, 
            spell_corrector, 
            tokenizer,
            max_tokens=128
        )
        
        # If query is too long, add the error message as a response
        if error_message:
            st.session_state.generating = True
            st.session_state.chat_history.append({
                "role": "user", 
                "content": original_text,
                "processed_content": None,
                "avatar": "👤"
            })
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": error_message,
                "avatar": "🤖"
            })
            st.session_state.generating = False
            st.rerun()
            return

        st.session_state.generating = True

        st.session_state.chat_history.append({
            "role": "user", 
            "content": original_text,
            "processed_content": processed_text,
            "avatar": "👤"
        })

        st.rerun()


    def process_generation():
        last_message = st.session_state.chat_history[-1]
        processed_message = last_message.get("processed_content", last_message["content"])
        
        # Skip generation if processed_content is None (error case already handled)
        if processed_message is None:
            st.session_state.generating = False
            return

        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""

            # Check OOD using the DistilBERT Classifier
            if is_ood(processed_message, clf_model, clf_tokenizer):
                full_response = random.choice(fallback_responses)
            else:
                # If In-Domain, send to DistilGPT2 and GLiNER
                with st.spinner("Generating response..."):
                    dynamic_placeholders = extract_dynamic_placeholders(processed_message, gliner_model)
                    response_gpt = generate_response(model, tokenizer, processed_message)
                    full_response = replace_placeholders(response_gpt, dynamic_placeholders, static_placeholders)

            streamed_text = ""
            for word in full_response.split(" "):
                streamed_text += word + " "
                message_placeholder.markdown(streamed_text + "⬤", unsafe_allow_html=True)
                time.sleep(0.05)
            message_placeholder.markdown(full_response, unsafe_allow_html=True)

        st.session_state.chat_history.append({"role": "assistant", "content": full_response, "avatar": "🤖"})
        st.session_state.generating = False


    # Logic flow
    if process_query_button:
        if selected_query != "Choose your question":
            handle_prompt(selected_query)
        else:
            st.error("⚠️ Please select your question from the dropdown.")

    if prompt := st.chat_input("Enter your own question:", disabled=st.session_state.generating):
        handle_prompt(prompt)

    if st.session_state.generating:
        process_generation()
        st.rerun()

    if st.session_state.chat_history:
        if st.button("Clear Chat", key="reset_button", disabled=st.session_state.generating):
            st.session_state.chat_history = []
            st.session_state.generating = False
            last_role = None
            st.rerun()
