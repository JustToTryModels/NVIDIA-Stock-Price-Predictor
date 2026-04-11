import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="NVIDIA Stock Predictor Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# CUSTOM CSS
# ==============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main > div { padding-top: 1rem; }
.block-container { padding-top: 2rem; }

.header {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
    padding: 2rem;
    border-radius: 16px;
    border: 1px solid #2a2a2a;
    margin-bottom: 2rem;
    text-align: center;
}
.header h1 {
    background: linear-gradient(90deg, #76B900, #a8ff60);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem;
    font-weight: 700;
    margin: 0;
}
.metric-card {
    background: #111;
    border: 1px solid #2a2a2a;
    padding: 1.2rem;
    border-radius: 12px;
}
.stButton > button {
    background: linear-gradient(90deg, #76B900, #5a8f00);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-size: 1.05rem;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(118,185,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_nvidia_model():
    try:
        return load_model('LSTM_Model/NVIDIA_LSTM_LB(5)_U(150)_RMSE(1.32).keras')
    except Exception as e:
        st.warning(f"Model not found — using demo forecast. ({e})")
        return None

model = load_nvidia_model()

# ==============================
# FETCH DATA
# ==============================
@st.cache_data(ttl=3600)
def get_stock_data(ticker='NVDA', period='2y'):
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    return data

# ==============================
# PREDICTION
# ==============================
def predict_next_days(model, data, look_back=5, days=5):
    if model is None: # fallback demo
        last = data[-1][0]
        trend = np.mean(np.diff(data[-20:].flatten()))
        preds = [last + trend*(i+1) + np.random.normal(0, last*0.01) for i in range(days)]
        return np.array(preds).reshape(-1,1)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    last_seq = data_scaled[-look_back:]
    preds = []
    for _ in range(days):
        X = np.reshape(last_seq, (1, look_back, 1))
        p = model.predict(X, verbose=0)[0,0]
        preds.append(p)
        last_seq = np.append(last_seq[1:], [[p]], axis=0)
    return scaler.inverse_transform(np.array(preds).reshape(-1,1))

def business_days(start, n):
    return pd.bdate_range(start=start, periods=n)

# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/NVIDIA_logo.svg/320px-NVIDIA_logo.svg.png", width=180)
    st.markdown("### ⚙️ Configuration")
    ticker = st.selectbox("Ticker", ["NVDA", "AAPL", "MSFT", "TSLA", "AMD"], index=0)
    num_days = st.slider("Forecast days", 1, 30, 5)
    look_back = st.slider("Look-back window", 5, 60, 5)
    hist_period = st.selectbox("Historical period", ["6mo", "1y", "2y", "5y", "max"], index=2)
    st.markdown("---")
    st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

# ==============================
# HEADER
# ==============================
st.markdown('<div class="header"><h1>Stock Predictor Pro</h1><p style="color:#aaa; margin-top:0.5rem;">LSTM-powered forecasting with interactive analytics</p></div>', unsafe_allow_html=True)

# ==============================
# LOAD DATA
# ==============================
with st.spinner(f"Loading {ticker} data..."):
    stock_data = get_stock_data(ticker, hist_period)

if stock_data.empty:
    st.error("Failed to load data")
    st.stop()

close_prices = stock_data['Close'].values.reshape(-1,1)
current_price = float(close_prices[-1][0])
prev_price = float(close_prices[-2][0])
change = current_price - prev_price
pct_change = change/prev_price*100

# METRICS
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Price", f"${current_price:.2f}", f"{pct_change:+.2f}%")
c2.metric("52W High", f"${stock_data['High'].max():.2f}")
c3.metric("52W Low", f"${stock_data['Low'].min():.2f}")
c4.metric("Volume", f"{stock_data['Volume'].iloc[-1]/1e6:.1f}M")

# ==============================
# TABS
# ==============================
tab1, tab2, tab3 = st.tabs(["🔮 Forecast", "📊 Historical", "ℹ️ Model"])

with tab1:
    if st.button(f'Generate {num_days}-Day Forecast', type="primary"):
        with st.spinner("Running LSTM inference..."):
            preds = predict_next_days(model, close_prices, look_back, num_days)
            pred_dates = business_days(stock_data.index[-1] + timedelta(days=1), num_days)

            st.session_state['preds'] = preds
            st.session_state['pred_dates'] = pred_dates
            st.success("Forecast complete!")
            st.balloons()

    if 'preds' in st.session_state:
        preds = st.session_state['preds']
        pred_dates = st.session_state['pred_dates']

        # Interactive Plotly Chart
        fig = go.Figure()
        # Historical (last 180 days)
        hist_df = stock_data.tail(180)
        fig.add_trace(go.Scatter(
            x=hist_df.index, y=hist_df['Close'],
            name='Historical', line=dict(color='#76B900', width=2)
        ))
        # Forecast
        fig.add_trace(go.Scatter(
            x=pred_dates, y=preds.flatten(),
            name='Forecast', mode='lines+markers',
            line=dict(color='#e52e71', width=3, dash='dash'),
            marker=dict(size=8)
        ))
        # Confidence band (±2%)
        upper = preds.flatten()*1.02
        lower = preds.flatten()*0.98
        fig.add_trace(go.Scatter(x=pred_dates, y=upper, line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=pred_dates, y=lower, fill='tonexty', fillcolor='rgba(229,46,113,0.15)', line=dict(width=0), showlegend=False, name='Confidence'))

        fig.update_layout(
            template='plotly_dark',
            height=500,
            hovermode='x unified',
            legend=dict(orientation='h', y=1.02),
            margin=dict(l=20, r=20, t=40, b=20),
            title=f"{ticker} — {num_days} Business Day Forecast"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Prediction table
        pred_df = pd.DataFrame({
            'Date': pred_dates,
            'Predicted Close': preds.flatten(),
            'Change %': np.insert((np.diff(preds.flatten())/preds.flatten()[:-1]*100), 0, (preds[0][0]-current_price)/current_price*100)
        })
        pred_df['Predicted Close'] = pred_df['Predicted Close'].map('${:,.2f}'.format)
        pred_df['Change %'] = pred_df['Change %'].map('{:+.2f}%'.format)

        col1, col2 = st.columns([2,1])
        with col1:
            st.dataframe(pred_df, use_container_width=True, hide_index=True)
        with col2:
            final_pred = float(st.session_state['preds'][-1][0])
            st.metric("Target Price", f"${final_pred:.2f}", f"{(final_pred-current_price)/current_price*100:+.2f}%")
            csv = pd.DataFrame({'Date': pred_dates, 'Price': preds.flatten()}).to_csv(index=False).encode()
            st.download_button("📥 Download CSV", csv, f"{ticker}_forecast.csv", "text/csv", use_container_width=True)

with tab2:
    # Candlestick
    fig2 = go.Figure(data=[go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        increasing_line_color='#76B900',
        decreasing_line_color='#e52e71'
    )])
    fig2.update_layout(template='plotly_dark', height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(stock_data.tail(100).sort_index(ascending=False), use_container_width=True, height=350)

with tab3:
    st.markdown("""
    ### Model Architecture
    - **Type:** LSTM Neural Network
    - **Look-back:** 5 days → 150 units
    - **Trained on:** NVDA historical data
    - **RMSE:** 1.32
    - **Features:** MinMax scaling, business-day aware forecasting

    > This app uses `yfinance` for live data and caches results for 1 hour. Replace the model path with your trained `.keras` file for production use.
    """)
    st.code("LSTM_Model/NVIDIA_LSTM_LB(5)_U(150)_RMSE(1.32).keras", language="bash")

st.markdown("<div style='text-align:center; color:#666; padding:2rem 0;'>Built with Streamlit • Data: Yahoo Finance • Not financial advice</div>", unsafe_allow_html=True)
