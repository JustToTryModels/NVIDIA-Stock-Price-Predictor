import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark background */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }

    /* Hide default header */
    header[data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid #30363d;
    }

    section[data-testid="stSidebar"] .stMarkdown p {
        color: #8b949e;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
    }

    .metric-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 4px;
    }

    .metric-delta-positive {
        font-size: 0.85rem;
        color: #3fb950;
        font-weight: 500;
    }

    .metric-delta-negative {
        font-size: 0.85rem;
        color: #f85149;
        font-weight: 500;
    }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 50%, #1a1a2e 100%);
        border: 1px solid #30363d;
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
    }

    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(118, 185, 0, 0.05) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 1; }
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #76b900, #a8e063, #76b900);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 10px;
        background-size: 200% auto;
        animation: shine 3s linear infinite;
    }

    @keyframes shine {
        to { background-position: 200% center; }
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #8b949e;
        font-weight: 300;
    }

    /* Prediction button */
    .stButton > button {
        background: linear-gradient(90deg, #76b900, #5a8f00);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 14px 28px;
        font-size: 1rem !important;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(118, 185, 0, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(118, 185, 0, 0.5);
        background: linear-gradient(90deg, #8fd400, #76b900);
    }

    .stButton > button:active {
        transform: translateY(0px);
    }

    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
        margin: 30px 0 15px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #76b900;
        display: inline-block;
    }

    /* Data table */
    .dataframe {
        background: #1e1e2e !important;
        color: #ffffff !important;
        border-radius: 10px !important;
    }

    /* Slider */
    .stSlider > div > div > div > div {
        background: #76b900 !important;
    }

    /* Status badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(118, 185, 0, 0.15);
        border: 1px solid rgba(118, 185, 0, 0.4);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #76b900;
    }

    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #1e1e2e, #252540);
        border: 1px solid #30363d;
        border-left: 4px solid #76b900;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 10px 0;
    }

    /* Prediction table styling */
    .pred-table-row-positive {
        color: #3fb950;
    }

    .pred-table-row-negative {
        color: #f85149;
    }

    /* Divider */
    .custom-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #30363d, transparent);
        margin: 30px 0;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #8b949e;
        border-radius: 8px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: #76b900 !important;
        color: white !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #1e1e2e;
        border-radius: 10px;
        color: #ffffff;
    }

    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #76b900, #a8e063);
    }

    /* Tooltip */
    .tooltip-text {
        font-size: 0.78rem;
        color: #8b949e;
        margin-top: 4px;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }

    ::-webkit-scrollbar-track {
        background: #1e1e2e;
    }

    ::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #76b900;
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
        st.error(f"❌ Error loading model: {e}")
        return None


# ==============================
# 🔹 Fetch Stock Data (Cached)
# ==============================
@st.cache_data(ttl=300)
def get_stock_data(ticker='NVDA'):
    try:
        data = yf.download(ticker, period='max')
        return data
    except Exception as e:
        st.error(f"❌ Error fetching stock data: {e}")
        return None


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
# 🔹 Compute Stats
# ==============================
def compute_stats(stock_data):
    close = stock_data['Close']
    latest_price = float(close.iloc[-1])
    prev_price = float(close.iloc[-2])
    daily_change = latest_price - prev_price
    daily_change_pct = (daily_change / prev_price) * 100
    high_52w = float(close.rolling(252).max().iloc[-1])
    low_52w = float(close.rolling(252).min().iloc[-1])
    vol_20d = float(close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100)
    avg_vol = float(stock_data['Volume'].rolling(20).mean().iloc[-1])
    return {
        'latest_price': latest_price,
        'daily_change': daily_change,
        'daily_change_pct': daily_change_pct,
        'high_52w': high_52w,
        'low_52w': low_52w,
        'vol_20d': vol_20d,
        'avg_vol': avg_vol,
    }


# ==============================
# 🔹 Plotly Charts
# ==============================
def plot_historical_candlestick(stock_data, days_back=180):
    df = stock_data.tail(days_back).copy()
    df.index = pd.to_datetime(df.index)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25]
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'].squeeze(),
            high=df['High'].squeeze(),
            low=df['Low'].squeeze(),
            close=df['Close'].squeeze(),
            name='Price',
            increasing_line_color='#3fb950',
            decreasing_line_color='#f85149',
            increasing_fillcolor='#3fb950',
            decreasing_fillcolor='#f85149',
        ),
        row=1, col=1
    )

    # Moving averages
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()

    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['MA20'].squeeze(),
            name='MA 20', line=dict(color='#58a6ff', width=1.5),
            opacity=0.8
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index, y=df['MA50'].squeeze(),
            name='MA 50', line=dict(color='#f0883e', width=1.5),
            opacity=0.8
        ),
        row=1, col=1
    )

    # Volume
    colors = ['#3fb950' if c >= o else '#f85149'
              for c, o in zip(df['Close'].squeeze(), df['Open'].squeeze())]

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'].squeeze(),
            name='Volume',
            marker_color=colors,
            opacity=0.6
        ),
        row=2, col=1
    )

    fig.update_layout(
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#8b949e', family='Inter'),
        xaxis_rangeslider_visible=False,
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='#30363d',
            borderwidth=1,
            font=dict(color='#ffffff')
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=500,
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#1e1e2e',
            bordercolor='#30363d',
            font=dict(color='white')
        ),
        xaxis=dict(
            gridcolor='#21262d',
            zerolinecolor='#30363d',
            showspikes=True,
            spikecolor='#30363d',
            spikethickness=1,
        ),
        yaxis=dict(
            gridcolor='#21262d',
            zerolinecolor='#30363d',
            tickprefix='$'
        ),
        xaxis2=dict(gridcolor='#21262d'),
        yaxis2=dict(gridcolor='#21262d'),
    )

    return fig


def plot_predictions(stock_data, predictions, prediction_dates, num_days, lookback_days=60):
    df = stock_data.tail(lookback_days).copy()
    df.index = pd.to_datetime(df.index)
    pred_df = pd.DataFrame({
        'Date': prediction_dates,
        'Price': predictions.flatten()
    })

    last_hist_price = float(df['Close'].iloc[-1])
    last_hist_date = df.index[-1]

    # Connect historical to prediction
    bridge_dates = [last_hist_date] + list(pred_df['Date'])
    bridge_prices = [last_hist_price] + list(pred_df['Price'])

    fig = go.Figure()

    # Historical line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'].squeeze(),
        name='Historical',
        line=dict(color='#58a6ff', width=2),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Price: $%{y:.2f}<extra></extra>'
    ))

    # Prediction fill area
    upper = pred_df['Price'] * 1.02
    lower = pred_df['Price'] * 0.98

    fig.add_trace(go.Scatter(
        x=list(pred_df['Date']) + list(pred_df['Date'])[::-1],
        y=list(upper) + list(lower)[::-1],
        fill='toself',
        fillcolor='rgba(118, 185, 0, 0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo='skip',
        name='Confidence Band'
    ))

    # Prediction line
    fig.add_trace(go.Scatter(
        x=bridge_dates,
        y=bridge_prices,
        name='Prediction',
        line=dict(color='#76b900', width=2.5, dash='dot'),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Predicted: $%{y:.2f}<extra></extra>'
    ))

    # Prediction markers
    fig.add_trace(go.Scatter(
        x=pred_df['Date'],
        y=pred_df['Price'],
        mode='markers',
        name='Predicted Points',
        marker=dict(
            color='#76b900',
            size=9,
            symbol='circle',
            line=dict(color='white', width=1.5)
        ),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Predicted: $%{y:.2f}<extra></extra>'
    ))

    # Vertical divider
    fig.add_vline(
        x=last_hist_date,
        line_dash='dash',
        line_color='#8b949e',
        opacity=0.5,
        annotation_text='Forecast Start',
        annotation_font_color='#8b949e',
        annotation_font_size=11
    )

    fig.update_layout(
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#8b949e', family='Inter'),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='#30363d',
            borderwidth=1,
            font=dict(color='#ffffff')
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=420,
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#1e1e2e',
            bordercolor='#30363d',
            font=dict(color='white')
        ),
        xaxis=dict(
            gridcolor='#21262d',
            zerolinecolor='#30363d',
        ),
        yaxis=dict(
            gridcolor='#21262d',
            zerolinecolor='#30363d',
            tickprefix='$'
        ),
    )

    return fig


def plot_forecast_only(predictions, prediction_dates):
    pred_df = pd.DataFrame({
        'Date': prediction_dates,
        'Price': predictions.flatten()
    })

    colors = []
    for i, price in enumerate(pred_df['Price']):
        if i == 0:
            colors.append('#76b900')
        elif price > pred_df['Price'].iloc[i - 1]:
            colors.append('#3fb950')
        else:
            colors.append('#f85149')

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=pred_df['Date'],
        y=pred_df['Price'],
        marker_color=colors,
        opacity=0.8,
        name='Predicted Price',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Price: $%{y:.2f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=pred_df['Date'],
        y=pred_df['Price'],
        mode='lines+markers',
        line=dict(color='white', width=1.5),
        marker=dict(color='white', size=7),
        name='Trend',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Price: $%{y:.2f}<extra></extra>'
    ))

    fig.update_layout(
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#8b949e', family='Inter'),
        legend=dict(
            bgcolor='rgba(0,0,0,0)',
            bordercolor='#30363d',
            borderwidth=1,
            font=dict(color='#ffffff')
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        height=380,
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='#1e1e2e',
            bordercolor='#30363d',
            font=dict(color='white')
        ),
        xaxis=dict(
            gridcolor='#21262d',
            tickformat='%b %d',
            tickangle=-30
        ),
        yaxis=dict(
            gridcolor='#21262d',
            tickprefix='$'
        ),
        bargap=0.25,
    )

    return fig


# ==============================
# 🔹 Load Model
# ==============================
model = load_nvidia_model()

# ==============================
# 🔹 Sidebar
# ==============================
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 10px 0 20px 0;'>
            <div style='font-size: 2rem;'>📈</div>
            <div style='font-size: 1.1rem; font-weight: 700; color: #ffffff;'>Stock Predictor</div>
            <div style='font-size: 0.75rem; color: #8b949e; margin-top: 4px;'>LSTM Neural Network</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("<p style='color:#8b949e; font-size:0.8rem; font-weight:600; text-transform:uppercase; letter-spacing:1px;'>Forecast Settings</p>", unsafe_allow_html=True)

    num_days = st.slider(
        "Forecast Horizon (Days)",
        min_value=1,
        max_value=30,
        value=5,
        help="Number of business days to forecast ahead"
    )

    st.markdown("<p style='color:#8b949e; font-size:0.8rem; font-weight:600; text-transform:uppercase; letter-spacing:1px; margin-top:20px;'>Chart Settings</p>", unsafe_allow_html=True)

    lookback_chart = st.select_slider(
        "Historical View",
        options=[30, 60, 90, 180, 365],
        value=90,
        format_func=lambda x: f"{x} days"
    )

    st.markdown("---")

    # Model info card
    st.markdown("""
        <div style='background: linear-gradient(135deg, #1e1e2e, #252540); border: 1px solid #30363d;
                    border-radius: 12px; padding: 16px; margin-top: 10px;'>
            <div style='font-size: 0.75rem; font-weight: 600; color: #8b949e;
                        text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;'>Model Info</div>
            <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                <span style='color: #8b949e; font-size: 0.8rem;'>Architecture</span>
                <span style='color: #ffffff; font-size: 0.8rem; font-weight: 600;'>LSTM</span>
            </div>
            <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                <span style='color: #8b949e; font-size: 0.8rem;'>Look-Back</span>
                <span style='color: #ffffff; font-size: 0.8rem; font-weight: 600;'>5 Days</span>
            </div>
            <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                <span style='color: #8b949e; font-size: 0.8rem;'>Units</span>
                <span style='color: #ffffff; font-size: 0.8rem; font-weight: 600;'>150</span>
            </div>
            <div style='display: flex; justify-content: space-between;'>
                <span style='color: #8b949e; font-size: 0.8rem;'>RMSE</span>
                <span style='color: #76b900; font-size: 0.8rem; font-weight: 600;'>1.32</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    model_status = "✅ Model Loaded" if model is not None else "❌ Model Error"
    status_color = "#3fb950" if model is not None else "#f85149"
    st.markdown(f"""
        <div style='text-align: center;
                    background: rgba({"63,185,80" if model is not None else "248,81,73"}, 0.1);
                    border: 1px solid {status_color};
                    border-radius: 8px; padding: 8px;
                    font-size: 0.8rem; font-weight: 600; color: {status_color};'>
            {model_status}
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; color:#8b949e; font-size:0.75rem;'>🕒 {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>", unsafe_allow_html=True)


# ==============================
# 🔹 Hero Banner
# ==============================
st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">NVIDIA Stock Predictor</div>
        <div class="hero-subtitle">
            Powered by Long Short-Term Memory Neural Networks &nbsp;•&nbsp; Real-time Market Data
        </div>
        <br>
        <div style='display: flex; justify-content: center; gap: 16px; flex-wrap: wrap;'>
            <span class="status-badge">🟢 Live Data</span>
            <span class="status-badge">🤖 LSTM Model</span>
            <span class="status-badge">📊 NASDAQ: NVDA</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# ==============================
# 🔹 Load Live Stock Data
# ==============================
with st.spinner("Fetching latest market data..."):
    stock_data = get_stock_data('NVDA')

if stock_data is None or stock_data.empty:
    st.error("❌ Unable to fetch stock data. Please check your connection.")
    st.stop()

stats = compute_stats(stock_data)

# ==============================
# 🔹 Live Metrics Row
# ==============================
col1, col2, col3, col4, col5 = st.columns(5)

delta_color_class = "metric-delta-positive" if stats['daily_change'] >= 0 else "metric-delta-negative"
delta_arrow = "▲" if stats['daily_change'] >= 0 else "▼"

metrics = [
    ("Current Price", f"${stats['latest_price']:.2f}",
     f"<span class='{delta_color_class}'>{delta_arrow} {abs(stats['daily_change_pct']):.2f}% today</span>"),
    ("52W High", f"${stats['high_52w']:.2f}",
     f"<span style='color:#8b949e;'>Annual Peak</span>"),
    ("52W Low", f"${stats['low_52w']:.2f}",
     f"<span style='color:#8b949e;'>Annual Trough</span>"),
    ("Volatility (20D)", f"{stats['vol_20d']:.1f}%",
     f"<span style='color:#8b949e;'>Annualized</span>"),
    ("Avg Volume (20D)", f"{stats['avg_vol'] / 1e6:.1f}M",
     f"<span style='color:#8b949e;'>Shares/Day</span>"),
]

for col, (label, value, delta) in zip([col1, col2, col3, col4, col5], metrics):
    with col:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                {delta}
            </div>
        """, unsafe_allow_html=True)

st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

# ==============================
# 🔹 Session State
# ==============================
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

# ==============================
# 🔹 Forecast Button
# ==============================
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    st.markdown(f"""
        <div class='info-box'>
            <span style='color:#ffffff; font-weight:600;'>🎯 Forecast Target:</span>
            <span style='color:#76b900; font-weight:700;'> Next {num_days} Business Day{'s' if num_days > 1 else ''}</span>
            <br>
            <span style='color:#8b949e; font-size:0.82rem;'>Starting from {(datetime.now() + timedelta(days=1)).strftime('%B %d, %Y')}</span>
        </div>
    """, unsafe_allow_html=True)

    predict_clicked = st.button(
        f"🚀 Generate {num_days}-Day Forecast",
        key='forecast-button'
    )

if predict_clicked:
    if model is None:
        st.error("❌ Model not loaded. Cannot make predictions.")
    else:
        progress_bar = st.progress(0, text="Initializing prediction engine...")

        close_prices = stock_data['Close'].values.reshape(-1, 1)
        dates = stock_data.index

        progress_bar.progress(30, text="Scaling data...")
        import time
        time.sleep(0.3)

        progress_bar.progress(60, text="Running LSTM inference...")

        predictions = predict_next_business_days(
            model, close_prices, look_back=5, days=num_days
        )

        progress_bar.progress(90, text="Generating forecast dates...")
        time.sleep(0.2)

        last_date = dates[-1]
        prediction_dates = generate_business_days(last_date + timedelta(days=1), num_days)

        progress_bar.progress(100, text="Done!")
        time.sleep(0.3)
        progress_bar.empty()

        st.session_state.prediction_results = {
            'stock_data': stock_data,
            'close_prices': close_prices,
            'dates': dates,
            'predictions': predictions,
            'prediction_dates': prediction_dates,
            'num_days': num_days,
            'stock': 'NVDA'
        }

        st.success(f"✅ Successfully generated {num_days}-day price forecast!")

# ==============================
# 🔹 Display Results
# ==============================
if st.session_state.prediction_results is not None:

    results = st.session_state.prediction_results
    predictions = results['predictions']
    prediction_dates = results['prediction_dates']
    stored_num_days = results['num_days']
    stored_stock = results['stock']

    first_pred = float(predictions[0])
    last_pred = float(predictions[-1])
    current_price = stats['latest_price']
    total_change = last_pred - current_price
    total_change_pct = (total_change / current_price) * 100
    max_pred = float(predictions.max())
    min_pred = float(predictions.min())

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📊 Forecast Summary</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    fc1, fc2, fc3, fc4 = st.columns(4)

    fc_color = "metric-delta-positive" if total_change >= 0 else "metric-delta-negative"
    fc_arrow = "▲" if total_change >= 0 else "▼"

    forecast_metrics = [
        ("Forecast End Price", f"${last_pred:.2f}",
         f"<span class='{fc_color}'>{fc_arrow} {abs(total_change_pct):.2f}% vs today</span>"),
        ("Forecast High", f"${max_pred:.2f}",
         f"<span style='color:#3fb950;'>Peak Predicted</span>"),
        ("Forecast Low", f"${min_pred:.2f}",
         f"<span style='color:#f85149;'>Trough Predicted</span>"),
        ("Forecast Range", f"${max_pred - min_pred:.2f}",
         f"<span style='color:#8b949e;'>High - Low Spread</span>"),
    ]

    for col, (label, value, delta) in zip([fc1, fc2, fc3, fc4], forecast_metrics):
        with col:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    {delta}
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==============================
    # 🔹 Chart Tabs
    # ==============================
    st.markdown("<div class='section-header'>📈 Price Charts</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🕯️ Historical (Candlestick)",
        "🔮 Forecast Overlay",
        "📊 Forecast Detail",
        "📋 Raw Data"
    ])

    with tab1:
        st.markdown(f"<p style='color:#8b949e; font-size:0.85rem; margin-bottom:10px;'>Showing last {lookback_chart} trading days with MA20 & MA50</p>", unsafe_allow_html=True)
        fig_candle = plot_historical_candlestick(stock_data, days_back=lookback_chart)
        st.plotly_chart(fig_candle, use_container_width=True)

    with tab2:
        st.markdown(f"<p style='color:#8b949e; font-size:0.85rem; margin-bottom:10px;'>Historical prices bridged with {stored_num_days}-day LSTM forecast and ±2% confidence band</p>", unsafe_allow_html=True)
        fig_pred = plot_predictions(stock_data, predictions, prediction_dates, stored_num_days, lookback_days=lookback_chart)
        st.plotly_chart(fig_pred, use_container_width=True)

    with tab3:
        st.markdown(f"<p style='color:#8b949e; font-size:0.85rem; margin-bottom:10px;'>Day-by-day predicted prices for the next {stored_num_days} business days</p>", unsafe_allow_html=True)
        fig_fc = plot_forecast_only(predictions, prediction_dates)
        st.plotly_chart(fig_fc, use_container_width=True)

    with tab4:
        data_tab1, data_tab2 = st.tabs(["📅 Forecast Data", "🗃️ Full Historical"])

        with data_tab1:
            pred_df = pd.DataFrame({
                'Date': [d.strftime('%Y-%m-%d') for d in prediction_dates],
                'Predicted Price ($)': [f"${p:.2f}" for p in predictions.flatten()],
                'Change vs Today ($)': [f"{'+'if p > current_price else ''}{p - current_price:.2f}" for p in predictions.flatten()],
                'Change vs Today (%)': [f"{'+'if p > current_price else ''}{((p - current_price)/current_price)*100:.2f}%" for p in predictions.flatten()],
            })
            st.dataframe(
                pred_df,
                use_container_width=True,
                hide_index=True
            )

        with data_tab2:
            st.dataframe(
                stock_data.tail(252),
                use_container_width=True,
                height=400
            )

    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)

    # ==============================
    # 🔹 Disclaimer
    # ==============================
    with st.expander("⚠️ Risk Disclaimer & Model Information"):
        st.markdown("""
            <div style='color: #8b949e; font-size: 0.85rem; line-height: 1.7;'>
            <p><strong style='color:#ffffff;'>⚠️ Important Disclaimer</strong></p>
            <p>This tool is for <strong style='color:#f0883e;'>educational and informational purposes only</strong>.
            Predictions generated by this model should <strong>not</strong> be considered as financial advice
            or a recommendation to buy, sell, or hold any security.</p>

            <p><strong style='color:#ffffff;'>Model Limitations:</strong></p>
            <ul>
                <li>LSTM models are trained on historical price patterns and cannot predict black swan events,
                earnings surprises, geopolitical events, or other market-moving news.</li>
                <li>Past performance is not indicative of future results.</li>
                <li>The ±2% confidence band shown is illustrative only and does not represent a statistically
                calibrated confidence interval.</li>
                <li>Always conduct your own research and consult a licensed financial advisor before making
                investment decisions.</li>
            </ul>
            </div>
        """, unsafe_allow_html=True)

else:
    # ==============================
    # 🔹 Empty State
    # ==============================
    st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>📈 Market Overview</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    fig_candle = plot_historical_candlestick(stock_data, days_back=lookback_chart)
    st.plotly_chart(fig_candle, use_container_width=True)

    st.markdown("""
        <div style='text-align:center; padding: 40px 20px;
                    background: linear-gradient(135deg, #1e1e2e, #252540);
                    border: 1px dashed #30363d; border-radius: 16px; margin-top: 20px;'>
            <div style='font-size: 3rem; margin-bottom: 10px;'>🔮</div>
            <div style='font-size: 1.2rem; font-weight: 700; color: #ffffff; margin-bottom: 8px;'>
                Ready to Forecast
            </div>
            <div style='color: #8b949e; font-size: 0.9rem;'>
                Set your forecast horizon in the sidebar, then click the button above to generate predictions.
            </div>
        </div>
    """, unsafe_allow_html=True)
