import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import re

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Stock Predictor Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# CUSTOM CSS
# ==============================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        border-left: 4px solid #667eea;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 12px 28px;
        font-weight: 600;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102,126,234,0.4);
    }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
</style>
""", unsafe_allow_html=True)

# ==============================
# FUNCTIONS
# ==============================
@st.cache_resource
def load_nvidia_model(model_path=None, uploaded_file=None):
    try:
        if uploaded_file is not None:
            with open("temp_model.keras", "wb") as f:
                f.write(uploaded_file.getbuffer())
            model = load_model("temp_model.keras")
            os.remove("temp_model.keras")
        else:
            model = load_model(model_path)
        return model
    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def get_stock_data(ticker='NVDA', period='5y'):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        info = stock.info
        return data, info
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None, None

def generate_business_days(start_date, num_days):
    return pd.bdate_range(start=start_date, periods=num_days)

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
    # Add simple confidence interval (±2%)
    upper = predictions * 1.02
    lower = predictions * 0.98
    return predictions.flatten(), upper.flatten(), lower.flatten()

def parse_model_info(filename):
    info = {}
    lb = re.search(r'LB\((\d+)\)', filename)
    u = re.search(r'U\((\d+)\)', filename)
    rmse = re.search(r'RMSE\(([\d.]+)\)', filename)
    info['Look Back'] = lb.group(1) if lb else '5'
    info['Units'] = u.group(1) if u else '150'
    info['RMSE'] = rmse.group(1) if rmse else '1.32'
    return info

# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/NVIDIA_logo.svg/320px-NVIDIA_logo.svg.png", width=180)
    st.markdown("## ⚙️ Configuration")
    
    ticker = st.text_input("Stock Ticker", value="NVDA").upper()
    num_days = st.slider("Forecast Days", 1, 30, 5)
    look_back = st.slider("Look Back Period", 3, 20, 5)
    
    st.markdown("### Model")
    model_file = 'LSTM_Model/NVIDIA_LSTM_LB(5)_U(150)_RMSE(1.32).keras'
    uploaded_model = st.file_uploader("Upload .keras model", type=['keras','h5'])
    
    model_info = parse_model_info(model_file)
    st.caption(f"LB: {model_info['Look Back']} | Units: {model_info['Units']} | RMSE: {model_info['RMSE']}")

# ==============================
# HEADER
# ==============================
st.markdown("""
<div class="main-header">
    <h1>Stock Predictor Pro 📈</h1>
    <p>AI-powered LSTM forecasting with interactive analytics</p>
</div>
""", unsafe_allow_html=True)

# Load model
model = load_nvidia_model(model_file, uploaded_model)

# Load data
with st.spinner(f"Loading {ticker} data..."):
    stock_data, stock_info = get_stock_data(ticker)

if stock_data is None or stock_data.empty:
    st.error("Failed to load data. Check ticker.")
    st.stop()

# KPIs
current_price = stock_data['Close'].iloc[-1]
prev_price = stock_data['Close'].iloc[-2]
change = current_price - prev_price
change_pct = (change / prev_price) * 100
volume = stock_data['Volume'].iloc[-1]
high_52 = stock_data['High'].max()
low_52 = stock_data['Low'].min()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current Price", f"${current_price:.2f}", f"{change_pct:+.2f}%")
col2.metric("Volume", f"{volume/1e6:.1f}M")
col3.metric("52W High", f"${high_52:.2f}")
col4.metric("52W Low", f"${low_52:.2f}")
col5.metric("Date", datetime.now().strftime("%Y-%m-%d"))

# ==============================
# TABS
# ==============================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔮 Forecast", "📑 Data", "ℹ️ Model"])

with tab1:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name="Price"
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(x=stock_data.index, y=stock_data['Volume'], 
                         name="Volume", marker_color='rgba(102,126,234,0.5)'), row=2, col=1)
    
    fig.update_layout(height=550, showlegend=False, 
                      xaxis_rangeslider_visible=False,
                      template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader(f"Forecast Next {num_days} Business Days")
    
    if st.button(f'🚀 Generate Forecast for {ticker}', type="primary"):
        if model is None:
            st.error("❌ Model not loaded. Upload a .keras file in sidebar.")
        else:
            with st.spinner("Running LSTM inference..."):
                close_prices = stock_data['Close'].values.reshape(-1, 1)
                preds, upper, lower = predict_next_business_days(
                    model, close_prices, look_back, num_days
                )
                last_date = stock_data.index[-1]
                pred_dates = generate_business_days(last_date + timedelta(days=1), num_days)
                
                st.session_state['forecast'] = {
                    'dates': pred_dates,
                    'preds': preds,
                    'upper': upper,
                    'lower': lower
                }
    
    if 'forecast' in st.session_state:
        f = st.session_state['forecast']
        
        # Interactive forecast chart
        fig2 = go.Figure()
        # Historical last 90 days
        hist = stock_data.tail(90)
        fig2.add_trace(go.Scatter(x=hist.index, y=hist['Close'], 
                                  mode='lines', name='Historical', 
                                  line=dict(color='#667eea', width=2)))
        # Forecast
        fig2.add_trace(go.Scatter(x=f['dates'], y=f['preds'], 
                                  mode='lines+markers', name='Forecast',
                                  line=dict(color='#e52e71', width=3, dash='dash')))
        # Confidence band
        fig2.add_trace(go.Scatter(x=list(f['dates'])+list(f['dates'][::-1]),
                                  y=list(f['upper'])+list(f['lower'][::-1]),
                                  fill='toself', fillcolor='rgba(229,46,113,0.15)',
                                  line=dict(color='rgba(255,255,255,0)'),
                                  name='±2% Confidence', showlegend=True))
        
        fig2.update_layout(template="plotly_white", height=450,
                           hovermode='x unified',
                           title=f"{ticker} Price Forecast")
        st.plotly_chart(fig2, use_container_width=True)
        
        # Prediction table
        pred_df = pd.DataFrame({
            'Date': f['dates'],
            'Predicted': f['preds'].round(2),
            'Upper Bound': f['upper'].round(2),
            'Lower Bound': f['lower'].round(2)
        })
        st.dataframe(pred_df, use_container_width=True, hide_index=True)
        
        csv = pred_df.to_csv(index=False).encode()
        st.download_button("📥 Download CSV", csv, f"{ticker}_forecast_{datetime.now().date()}.csv")

with tab3:
    st.subheader("Historical Data")
    st.dataframe(stock_data.tail(500).sort_index(ascending=False), 
                 use_container_width=True, height=400)

with tab4:
    st.subheader("Model Information")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **Architecture:** LSTM Neural Network  
        **Look Back:** {model_info['Look Back']} days  
        **LSTM Units:** {model_info['Units']}  
        **Training RMSE:** {model_info['RMSE']}  
        **Framework:** TensorFlow/Keras
        """)
    with col2:
        st.markdown(f"""
        **Ticker:** {ticker}  
        **Data Period:** 5 Years  
        **Last Updated:** {stock_data.index[-1].date()}  
        **Model Status:** {'✅ Loaded' if model else '❌ Not Loaded'}
        """)
    st.info("This model predicts closing prices based on historical sequences. Not financial advice.")

st.markdown("---")
st.caption("Built with Streamlit • LSTM forecasting • Data from Yahoo Finance")
