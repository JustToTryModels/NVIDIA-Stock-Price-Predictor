import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import plotly.graph_objects as go

# ==============================
# 🔹 Page Configuration
# ==============================
st.set_page_config(
    page_title="NVIDIA AI Stock Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# 🔹 Professional Styling
# ==============================
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #fafafa; }
    h1, h2, h3 { color: #76b900 !important; }
    .stButton > button {
        background: linear-gradient(90deg, #76b900, #4a7c00);
        color: white !important;
        border-radius: 8px;
        padding: 12px 28px;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(118, 185, 0, 0.4);
    }
    .metric-card {
        background-color: #1a1f2e;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #76b900;
    }
    .disclaimer {
        background-color: #2a2f3a;
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #ff4b4b;
        font-size: 0.95rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# 🔹 Load Model
# ==============================
@st.cache_resource
def load_nvidia_model():
    model_path = 'LSTM_Model/NVIDIA_LSTM_LB(5)_U(150)_RMSE(1.32).keras'
    try:
        return load_model(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.info(f"Make sure the model exists at: `{model_path}`")
        return None

model = load_nvidia_model()

# ==============================
# 🔹 Fetch Data
# ==============================
@st.cache_data(ttl=300)
def get_stock_data(ticker='NVDA', period='2y'):
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data.empty:
            return None
        return data[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def get_current_info():
    try:
        ticker = yf.Ticker("NVDA")
        info = ticker.info
        price = info.get('currentPrice') or info.get('regularMarketPrice')
        prev_close = info.get('regularMarketPreviousClose')
        volume = info.get('regularMarketVolume', 0)
        if price and prev_close:
            change = price - prev_close
            change_pct = (change / prev_close) * 100
            return price, change, change_pct, volume
        return None, None, None, None
    except:
        return None, None, None, None

# ==============================
# 🔹 Helper Functions
# ==============================
def generate_business_days(start_date, num_days):
    return pd.bdate_range(start=start_date, periods=num_days).tolist()

def predict_next_business_days(model, data, look_back=5, days=5):
    if model is None:
        return None
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    last_sequence = data_scaled[-look_back:]
    predictions = []

    for _ in range(days):
        X = np.reshape(last_sequence, (1, look_back, 1))
        pred = model.predict(X, verbose=0)[0, 0]
        predictions.append(pred)
        last_sequence = np.append(last_sequence[1:], [[pred]], axis=0)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# ==============================
# 🔹 Sidebar
# ==============================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/NVIDIA_logo.svg/1920px-NVIDIA_logo.svg.png", use_column_width=True)
    st.title("NVIDIA AI Forecaster")
    st.markdown("**LSTM • Lookback=5 • RMSE≈1.32**")
    st.divider()
    
    num_days = st.slider("Forecast Horizon (Business Days)", 1, 30, 10)
    
    st.divider()
    st.markdown("### Model Summary")
    st.markdown("""
    - Architecture: LSTM (150 units)
    - Lookback: 5 trading days
    - Feature: Closing price only
    - Trained on historical NVDA data
    """)
    if st.button("🔄 Refresh Market Data", use_container_width=True):
        get_stock_data.clear()
        st.rerun()

# ==============================
# 🔹 Main UI
# ==============================
st.title("📈 NVIDIA (NVDA) AI Stock Price Predictor")
st.caption("Advanced LSTM forecasting • Educational & demonstration purposes only")

price, change, change_pct, volume = get_current_info()
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Current Price", f"${price:,.2f}" if price else "—", f"{change_pct:+.2f}%" if change_pct is not None else None)
with col2:
    st.metric("Daily Change", f"${change:+.2f}" if change else "—")
with col3:
    st.metric("Volume", f"{volume:,.0f}" if volume else "—")
with col4:
    st.metric("Forecast Horizon", f"{num_days} days")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Market Overview", "📈 Historical Analysis", "🔮 AI Forecast", "ℹ️ Model & Disclaimer"])

data = get_stock_data()
if data is None or data.empty:
    st.error("Failed to load stock data.")
    st.stop()

close_prices = data['Close'].values.reshape(-1, 1)
dates = data.index

if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

# Tab 1: Overview
with tab1:
    st.subheader("Recent Performance (Last 6 Months)")
    recent = data.last('6M')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent.index, y=recent['Close'], mode='lines',
                             line=dict(color='#76b900', width=3), name='Close'))
    fig.update_layout(title="NVDA Price Trend", xaxis_title="Date", yaxis_title="Price (USD)",
                      template="plotly_dark", height=500, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key Statistics")
    cols = st.columns(4)
    cols[0].metric("All-Time High", f"${data['High'].max():,.2f}")
    cols[1].metric("52-Week High", f"${data.last('252D')['High'].max():,.2f}")
    cols[2].metric("52-Week Low", f"${data.last('252D')['Low'].min():,.2f}")
    cols[3].metric("30D Volatility", f"{data['Close'].pct_change().last('30D').std()*100:.2f}%")

# Tab 2: Historical
with tab2:
    st.subheader("Candlestick Chart")
    fig_candle = go.Figure(data=[go.Candlestick(
        x=data.index, open=data['Open'], high=data['High'],
        low=data['Low'], close=data['Close'],
        increasing_line_color='#00ff9d', decreasing_line_color='#ff4b4b')])
    fig_candle.update_layout(title="NVDA Historical Price Action", template="plotly_dark", height=600)
    st.plotly_chart(fig_candle, use_container_width=True)

    st.subheader("Historical Data Table")
    display_df = data.copy()
    display_df.index = display_df.index.strftime('%Y-%m-%d')
    st.dataframe(display_df.style.format({
        'Open': '${:,.2f}', 'High': '${:,.2f}', 'Low': '${:,.2f}',
        'Close': '${:,.2f}', 'Volume': '{:,.0f}'
    }), use_container_width=True, height=400)

# Tab 3: Forecast
with tab3:
    st.subheader("LSTM Neural Network Forecast")
    if st.button(f"🚀 Generate {num_days}-Day Forecast", type="primary", use_container_width=True):
        if model is None:
            st.error("Model not loaded.")
        else:
            with st.spinner("Running LSTM prediction..."):
                preds = predict_next_business_days(model, close_prices, days=num_days)
                last_date = dates[-1]
                pred_dates = generate_business_days(last_date + timedelta(days=1), num_days)

                st.session_state.prediction_results = {
                    'predictions': preds,
                    'dates': pred_dates,
                    'num_days': num_days,
                    'last_close': float(close_prices[-1][0])
                }

    if st.session_state.prediction_results:
        res = st.session_state.prediction_results
        pred_dates = res['dates']
        preds = res['predictions']
        last_close = res['last_close']

        # Combined Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates[-60:], y=close_prices[-60:].flatten(),
                                 name="Historical", line=dict(color='#76b900', width=3)))
        fig.add_trace(go.Scatter(x=pred_dates, y=preds, name="Predicted",
                                 line=dict(color='#00d4ff', width=3, dash='dash'), mode='lines+markers'))
        fig.update_layout(title=f"NVDA {res['num_days']}-Day Price Forecast", template="plotly_dark",
                          height=550, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # Table
        df_pred = pd.DataFrame({
            "Date": [d.strftime('%Y-%m-%d') for d in pred_dates],
            "Predicted Price": preds,
            "Change": preds - last_close,
            "% Change": ((preds - last_close) / last_close * 100)
        })

        st.dataframe(df_pred.style.format({
            "Predicted Price": "${:,.2f}",
            "Change": "${:+.2f}",
            "% Change": "{:+.2f}%"
        }).background_gradient(subset=["% Change"], cmap="RdYlGn"), use_container_width=True, hide_index=True)

        csv = df_pred.to_csv(index=False)
        st.download_button("📥 Download Forecast (CSV)", csv,
                           f"NVDA_Forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                           "text/csv", use_container_width=True)

# Tab 4: Model & Disclaimer
with tab4:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Architecture")
        st.markdown("""
        **LSTM Model Details**
        - Lookback window: **5 days**
        - LSTM units: **150**
        - Reported RMSE: **≈ 1.32**
        - Input: Daily closing prices only
        - Framework: TensorFlow/Keras
        """)
    with col2:
        st.subheader("⚠️ Critical Disclaimer")
        st.markdown("""
        <div class="disclaimer">
        <strong>This application is for educational and demonstration purposes only.</strong><br><br>
        • It does <strong>not</strong> constitute financial, investment, or trading advice.<br>
        • Stock forecasting is highly uncertain.<br>
        • Past performance does not predict future results.<br>
        • Always do your own research and consult licensed professionals before investing.
        </div>
        """, unsafe_allow_html=True)

st.caption("Built with Streamlit • TensorFlow • yfinance • Plotly • NVIDIA Green Theme")
