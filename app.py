import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import plotly.graph_objects as go

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="NVIDIA AI Stock Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# CUSTOM CSS
# ==============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    font-size: 3rem; font-weight: 800;
    background: linear-gradient(90deg, #76B900 0%, #00E5A0 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 0;
}
.sub-header { text-align: center; color: #9ca3af; margin-top: -8px; font-size: 1.1rem }

.block-container { padding-top: 1.5rem; }

div[data-testid="stMetric"] {
    background: rgba(28, 30, 33, 0.6);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 15px; border-radius: 16px;
    backdrop-filter: blur(12px);
}
.stButton > button {
    background: linear-gradient(90deg, #76B900, #1DB954);
    color: white !important; border: none; border-radius: 12px;
    padding: 12px 28px; font-size: 1.1em !important; font-weight: 700;
    width: 100%; transition: all 0.2s ease; box-shadow: 0 4px 15px rgba(118,185,0,0.3);
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(118,185,0,0.4); }
</style>
""", unsafe_allow_html=True)

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource(show_spinner="Loading AI Model...")
def load_nvidia_model():
    try:
        return load_model('LSTM_Model/NVIDIA_LSTM_LB(5)_U(150)_RMSE(1.32).keras')
    except Exception as e:
        st.error(f"Model not found: {e}")
        return None

model = load_nvidia_model()

# ==============================
# DATA FUNCTIONS
# ==============================
@st.cache_data(ttl=900)
def get_stock_data(ticker='NVDA', period='5y'):
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    return data.dropna()

@st.cache_data(ttl=900)
def get_stock_info(ticker='NVDA'):
    try:
        return yf.Ticker(ticker).info
    except:
        return {}

def generate_business_days(start_date, num_days):
    return pd.bdate_range(start=start_date, periods=num_days)

def predict_next_days(model, data, look_back=5, days=5):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    last_seq = data_scaled[-look_back:]
    preds = []
    for _ in range(days):
        X = np.reshape(last_seq, (1, look_back, 1))
        pred = model.predict(X, verbose=0)[0,0]
        preds.append(pred)
        last_seq = np.append(last_seq[1:], [[pred]], axis=0)
    return scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()

# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/NVIDIA_logo.svg/1920px-NVIDIA_logo.svg.png", width=180)
    st.markdown("## ⚙️ Forecast Controls")
    num_days = st.slider("Forecast Days", 1, 30, 5, help="Business days to predict")
    hist_period = st.selectbox("Historical View", ["1mo","3mo","6mo","1y","2y","5y","max"], index=4)
    show_confidence = st.toggle("Show Confidence Band (±2%)", value=True)
    look_back = 5
    
    st.divider()
    st.markdown("### 🧠 Model Specs")
    st.caption("• Architecture: LSTM\n• Lookback: 5 days\n• Units: 150\n• RMSE: 1.32\n• Trained on NVDA")
    st.divider()
    st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")

# ==============================
# HEADER
# ==============================
st.markdown("<h1 class='main-header'>NVIDIA Stock AI Forecaster</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>LSTM Deep Learning • Real-time Data • Professional Grade Prediction</p>", unsafe_allow_html=True)

# ==============================
# LOAD DATA
# ==============================
with st.spinner("Fetching live NVDA data..."):
    stock_data = get_stock_data('NVDA', hist_period)
    info = get_stock_info('NVDA')

if stock_data.empty:
    st.stop()

last_close = stock_data['Close'].iloc[-1]
prev_close = stock_data['Close'].iloc[-2]
day_change = last_close - prev_close
day_pct = (day_change/prev_close)*100
volume = stock_data['Volume'].iloc[-1]
market_cap = info.get('marketCap', 0)

# KPI ROW
c1,c2,c3,c4 = st.columns(4)
c1.metric("Current Price", f"${last_close:,.2f}", f"{day_pct:+.2f}%")
c2.metric("Day Change", f"${day_change:+.2f}", f"{volume/1e6:.1f}M Vol")
c3.metric("52W High", f"${stock_data['High'].max():,.2f}")
c4.metric("Market Cap", f"${market_cap/1e12:.2f}T" if market_cap else "N/A")

st.markdown("---")

# ==============================
# PREDICT BUTTON
# ==============================
col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
with col_btn2:
    predict_clicked = st.button(f'🚀 Generate {num_days}-Day AI Forecast', use_container_width=True)

if 'results' not in st.session_state:
    st.session_state.results = None

if predict_clicked:
    if model is None:
        st.error("Model failed to load")
    else:
        with st.spinner("AI is analyzing patterns..."):
            close_prices = stock_data['Close'].values.reshape(-1,1)
            preds = predict_next_days(model, close_prices, look_back, num_days)
            pred_dates = generate_business_days(stock_data.index[-1] + timedelta(days=1), num_days)
            
            st.session_state.results = {
                'preds': preds,
                'dates': pred_dates,
                'last_close': last_close
            }
        st.success(f"Forecast complete! Expected {num_days}-day return: {((preds[-1]/last_close)-1)*100:+.2f}%")
        st.balloons()

# ==============================
# DISPLAY RESULTS
# ==============================
if st.session_state.results:
    preds = st.session_state.results['preds']
    pred_dates = st.session_state.results['dates']
    
    tab1, tab2, tab3 = st.tabs(["📊 Interactive Dashboard", "🔮 Forecast Details", "📁 Data"])
    
    with tab1:
        # MAIN CHART
        fig = go.Figure()
        
        # Historical
        fig.add_trace(go.Scatter(
            x=stock_data.index, y=stock_data['Close'],
            name='Historical', line=dict(color='#76B900', width=2),
            hovertemplate='Date: %{x|%b %d}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=pred_dates, y=preds,
            name='AI Forecast', line=dict(color='#FF4B4B', width=3, dash='dash'),
            mode='lines+markers', marker=dict(size=8),
            hovertemplate='Forecast: %{x|%b %d}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        
        # Confidence band
        if show_confidence:
            upper = preds * 1.02
            lower = preds * 0.98
            fig.add_trace(go.Scatter(
                x=list(pred_dates)+list(pred_dates[::-1]),
                y=list(upper)+list(lower[::-1]),
                fill='toself', fillcolor='rgba(255,75,75,0.15)',
                line=dict(color='rgba(0,0,0,0)'), name='Confidence', showlegend=False
            ))
        
        fig.update_layout(
            template='plotly_dark', height=550,
            title=f"NVDA Price History + {num_days}-Day LSTM Forecast",
            xaxis_title="", yaxis_title="Price (USD)",
            hovermode='x unified', legend=dict(orientation='h', y=1.02),
            margin=dict(l=20,r=20,t=60,b=20)
        )
        fig.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast metrics
        m1,m2,m3 = st.columns(3)
        m1.metric("Predicted Price (Final)", f"${preds[-1]:,.2f}", f"{((preds[-1]/last_close)-1)*100:+.2f}%")
        m2.metric("Predicted High", f"${preds.max():,.2f}")
        m3.metric("Predicted Low", f"${preds.min():,.2f}")
    
    with tab2:
        fig2 = go.Figure()
        colors = ['#1DB954' if p > last_close else '#FF4B4B' for p in preds]
        fig2.add_trace(go.Bar(x=pred_dates, y=preds, marker_color=colors, 
                              text=[f"${p:.2f}" for p in preds], textposition='outside'))
        fig2.add_hline(y=last_close, line_dash="dot", line_color="white",
                       annotation_text=f"Current: ${last_close:.2f}")
        fig2.update_layout(template='plotly_dark', height=400, title="Daily Forecast Breakdown",
                           yaxis_title="Price USD", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Table
        df_pred = pd.DataFrame({
            'Date': pred_dates,
            'Predicted Close': preds,
            'vs Current (%)': ((preds/last_close)-1)*100,
            'Day Change (%)': np.insert(np.diff(preds)/preds[:-1]*100, 0, (preds[0]/last_close-1)*100)
        })
        
        st.dataframe(
            df_pred.style.format({
                'Predicted Close': '${:.2f}',
                'vs Current (%)': '{:+.2f}%',
                'Day Change (%)': '{:+.2f}%'
            }).background_gradient(subset=['Predicted Close'], cmap='RdYlGn'),
            use_container_width=True, hide_index=True
        )
        
        csv = df_pred.to_csv(index=False).encode()
        st.download_button("📥 Download Forecast CSV", csv, 
                          f"NVDA_{num_days}day_forecast_{datetime.now().date()}.csv",
                          mime="text/csv", use_container_width=True)
    
    with tab3:
        st.dataframe(stock_data.tail(200).sort_index(ascending=False).style.format({
            'Open':'${:.2f}', 'High':'${:.2f}', 'Low':'${:.2f}', 'Close':'${:.2f}', 'Volume':'{:,.0f}'
        }), use_container_width=True, height=400)

else:
    # Show placeholder chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], line=dict(color='#76B900', width=2)))
    fig.update_layout(template='plotly_dark', height=500, title="NVDA Historical Price - Click 'Generate Forecast' to start",
                      xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)
    st.info("👈 Adjust settings in the sidebar, then click the forecast button to generate AI predictions.")

# FOOTER
st.markdown("---")
st.caption("Disclaimer: This is an AI model demonstration for educational purposes only. Not financial advice. • Built with Streamlit + TensorFlow LSTM")
