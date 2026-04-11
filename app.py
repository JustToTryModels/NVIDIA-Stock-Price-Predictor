import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==============================
# 🔹 PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="NVIDIA AI Stock Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# 🔹 CUSTOM CSS
# ==============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main > div { padding-top: 1rem; }
.stButton > button {
    background: linear-gradient(90deg, #76b900 0%, #00c3ff 100%);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 12px 28px;
    font-size: 1.05em !important;
    font-weight: 600;
    transition: all 0.2s ease;
    box-shadow: 0 4px 15px rgba(0,195,255,0.2);
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,195,255,0.3); }
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 20px;
    border-radius: 16px;
    backdrop-filter: blur(10px);
}
</style>
""", unsafe_allow_html=True)

# ==============================
# 🔹 LOAD MODEL
# ==============================
@st.cache_resource
def load_nvidia_model():
    try:
        return load_model('LSTM_Model/NVIDIA_LSTM_LB(5)_U(150)_RMSE(1.32).keras')
    except Exception as e:
        st.error(f"Model not found. Place it in LSTM_Model/ folder")
        return None

model = load_nvidia_model()
RMSE = 1.32

# ==============================
# 🔹 DATA
# ==============================
@st.cache_data(ttl=3600)
def get_stock_data(ticker='NVDA', period='2y'):
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty: return None
    return data

def generate_business_days(start_date, num_days):
    return pd.bdate_range(start=start_date, periods=num_days)

def predict_next_days(model, data, look_back=5, days=5):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    last_seq = data_scaled[-look_back:]
    preds = []
    prog = st.progress(0, text="Running LSTM inference...")
    for i in range(days):
        X = last_seq.reshape(1, look_back, 1)
        pred = model.predict(X, verbose=0)[0,0]
        preds.append(pred)
        last_seq = np.append(last_seq[1:], [[pred]], axis=0)
        prog.progress((i+1)/days)
    prog.empty()
    return scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()

# ==============================
# 🔹 HEADER
# ==============================
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/NVIDIA_logo.svg/1920px-NVIDIA_logo.svg.png", width=120)
with col_title:
    st.markdown("<h1 style='margin-bottom:0'>NVIDIA AI Stock Forecaster</h1>", unsafe_allow_html=True)
    st.caption("LSTM Deep Learning • Powered by TensorFlow & yfinance")

# ==============================
# 🔹 SIDEBAR
# ==============================
with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.selectbox("Ticker", ["NVDA","AAPL","MSFT","TSLA","AMD","GOOGL"], index=0)
    if ticker != "NVDA":
        st.warning("Model trained only on NVDA. Results for other tickers will be inaccurate.")
    
    num_days = st.slider("Forecast days", 1, 30, 5)
    look_back = st.slider("Look-back window", 3, 20, 5)
    show_ci = st.toggle("Show confidence interval (±$1.32)", value=True)
    period = st.selectbox("History", ["6mo","1y","2y","5y","max"], index=2)
    
    st.divider()
    st.caption(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    run_forecast = st.button(f"🚀 Predict {num_days} Days", use_container_width=True, type="primary")

# ==============================
# 🔹 LOAD DATA
# ==============================
data = get_stock_data(ticker, period)

if data is None:
    st.stop()

close = data['Close'].values.reshape(-1,1)
dates = data.index
current_price = float(close[-1][0])
prev_price = float(close[-2][0])
change_pct = (current_price - prev_price) / prev_price * 100

# METRICS
m1,m2,m3,m4 = st.columns(4)
m1.metric("Current Price", f"${current_price:.2f}", f"{change_pct:+.2f}%")
m2.metric("Volume", f"{int(data['Volume'].iloc[-1]):,}")
m3.metric("52W High", f"${data['High'].tail(252).max():.2f}")
m4.metric("52W Low", f"${data['Low'].tail(252).min():.2f}")

# ==============================
# 🔹 TABS
# ==============================
tab1, tab2, tab3 = st.tabs(["🔮 Forecast", "📊 Historical Analysis", "ℹ️ Model Info"])

with tab1:
    if run_forecast and model is not None:
        with st.spinner("Fetching data & predicting..."):
            preds = predict_next_days(model, close, look_back, num_days)
            pred_dates = generate_business_days(dates[-1] + timedelta(days=1), num_days)
            
            st.session_state['preds'] = preds
            st.session_state['pred_dates'] = pred_dates
            st.toast("Prediction complete!", icon="✅")

    if 'preds' in st.session_state:
        preds = st.session_state['preds']
        pred_dates = st.session_state['pred_dates']

        # Interactive Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates[-90:], y=close[-90:].flatten(), name="History", line=dict(color="#00c3ff", width=2)))
        fig.add_trace(go.Scatter(x=pred_dates, y=preds, name="Forecast", line=dict(color="#76b900", width=3, dash="dash"), mode="lines+markers"))
        
        if show_ci:
            fig.add_trace(go.Scatter(x=pred_dates, y=preds+RMSE, fill=None, mode='lines', line_color='rgba(0,0,0,0)', showlegend=False))
            fig.add_trace(go.Scatter(x=pred_dates, y=preds-RMSE, fill='tonexty', mode='lines', line_color='rgba(0,0,0,0)', name='±RMSE', fillcolor='rgba(118,185,0,0.2)'))

        fig.update_layout(height=450, template="plotly_dark", hovermode="x unified", margin=dict(l=20,r=20,t=40,b=20), legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns([1.2,1])
        with col1:
            df_pred = pd.DataFrame({"Date": pred_dates, "Predicted Close": np.round(preds,2), "Lower": np.round(preds-RMSE,2), "Upper": np.round(preds+RMSE,2)})
            st.dataframe(df_pred, use_container_width=True, hide_index=True,
                column_config={"Predicted Close": st.column_config.NumberColumn(format="$%.2f")}
            )
        with col2:
            st.markdown("#### Summary")
            st.write(f"**Expected {num_days}-day move:** ${(preds[-1]-current_price):+.2f} ({(preds[-1]/current_price-1)*100:+.2f}%)")
            st.write(f"**Target Price:** ${preds[-1]:.2f}")
            st.download_button("📥 Download CSV", df_pred.to_csv(index=False), f"{ticker}_forecast_{datetime.now().date()}.csv", use_container_width=True)
    else:
        st.info("👈 Click 'Predict' in the sidebar to generate forecast")

with tab2:
    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3], vertical_spacing=0.05)
    fig2.add_trace(go.Candlestick(x=dates, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name="OHLC"), row=1, col=1)
    fig2.add_trace(go.Scatter(x=dates, y=data['Close'].rolling(20).mean(), name="MA20", line=dict(color="#ff8a00")), row=1, col=1)
    fig2.add_trace(go.Bar(x=dates, y=data['Volume'], name="Volume", marker_color="#00c3ff"), row=2, col=1)
    fig2.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=20,r=20,t=20,b=20))
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown("""
    **Model:** LSTM Sequential  
    **Architecture:** Look-back=5, Units=150  
    **Training RMSE:** 1.32  
    **Framework:** TensorFlow Keras  
    **Data Source:** Yahoo Finance (auto-adjusted close)
    
    > ⚠️ This is for educational purposes only. Not financial advice.
    """)

st.divider()
st.caption("Built with Streamlit • Model: LSTM_Model/NVIDIA_LSTM_LB(5)_U(150)_RMSE(1.32).keras")
