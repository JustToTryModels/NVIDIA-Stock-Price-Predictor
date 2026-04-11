import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# ==============================
# 🎨 Page Configuration
# ==============================
st.set_page_config(
    page_title="NVIDIA Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# 🎨 Custom CSS Styling
# ==============================
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main Container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Header Styling */
    h1 {
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        text-align: center;
        padding: 1rem 0;
        font-size: 3.5rem !important;
    }
    
    h2, h3 {
        color: #667eea;
        font-weight: 600;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 600;
        color: #555;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.9rem;
    }
    
    /* Custom Button */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem !important;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 10px 10px 0 0;
        padding: 1rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Info Cards */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    /* Dataframe Styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        color: white !important;
        border: none;
        border-radius: 50px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.4);
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Success/Error Messages */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #667eea;
        font-size: 0.9rem;
        border-top: 2px solid #667eea;
        margin-top: 3rem;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animated {
        animation: fadeIn 0.6s ease-out;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# 🔹 Load Model (Cached)
# ==============================
@st.cache_resource(show_spinner=False)
def load_nvidia_model():
    model_file = 'LSTM_Model/NVIDIA_LSTM_LB(5)_U(150)_RMSE(1.32).keras'
    try:
        model = load_model(model_file)
        return model, None
    except Exception as e:
        return None, str(e)

# ==============================
# 🔹 Fetch Stock Data (Cached)
# ==============================
@st.cache_data(show_spinner=False)
def get_stock_data(ticker='NVDA', period='max'):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        info = stock.info
        return data, info, None
    except Exception as e:
        return None, None, str(e)

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
# 🔹 Calculate Technical Indicators
# ==============================
def calculate_indicators(data):
    # Moving Averages
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    return data

# ==============================
# 🎯 Main App
# ==============================

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/NVIDIA_logo.svg/1920px-NVIDIA_logo.svg.png", width=200)
    st.markdown("---")
    
    st.markdown("### ⚙️ Configuration")
    stock_ticker = st.text_input("Stock Ticker", value="NVDA", help="Enter stock ticker symbol")
    num_days = st.slider("Forecast Days", 1, 30, 5, help="Number of business days to predict")
    historical_period = st.selectbox(
        "Historical Data Period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
        index=6
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.info("""
    **LSTM Model**
    - Look-back: 5 days
    - Units: 150
    - RMSE: 1.32
    """)
    
    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
    This app uses a deep learning LSTM model to predict stock prices based on historical data.
    
    **⚠️ Disclaimer:** This is for educational purposes only. Not financial advice.
    """)

# Header
st.markdown("<h1>📈 AI Stock Price Predictor</h1>", unsafe_allow_html=True)

# Load Model
with st.spinner("🔄 Loading AI Model..."):
    model, model_error = load_nvidia_model()

if model_error:
    st.error(f"❌ Error loading model: {model_error}")
    st.stop()

# Success message
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.success("✅ Model loaded successfully!")

st.markdown("---")

# Current Date Display
col1, col2, col3 = st.columns(3)
with col2:
    st.markdown(f"""
    <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    border-radius: 15px; color: white; font-size: 1.2rem; font-weight: 600;'>
        📅 Current Date: {datetime.now().strftime('%Y-%m-%d')}
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Predict Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button(
        f"🚀 Predict Next {num_days} Days for {stock_ticker}",
        key='forecast-button',
        use_container_width=True
    )

# Session State Init
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

# ==============================
# 🔹 Prediction Logic
# ==============================
if predict_button:
    with st.spinner("🔍 Fetching stock data..."):
        stock_data, stock_info, data_error = get_stock_data(stock_ticker, historical_period)
        time.sleep(0.5)  # For effect
    
    if data_error or stock_data is None:
        st.error(f"❌ Error fetching data: {data_error}")
        st.stop()
    
    if stock_data.empty:
        st.error("❌ No data available for this ticker.")
        st.stop()
    
    # Calculate indicators
    stock_data = calculate_indicators(stock_data)
    
    with st.spinner("🤖 AI is analyzing patterns..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        close_prices = stock_data['Close'].values.reshape(-1, 1)
        dates = stock_data.index
        
        predictions = predict_next_business_days(
            model,
            close_prices,
            look_back=5,
            days=num_days
        )
        
        last_date = dates[-1]
        prediction_dates = generate_business_days(last_date + timedelta(days=1), num_days)
        
        st.session_state.prediction_results = {
            'stock_data': stock_data,
            'stock_info': stock_info,
            'close_prices': close_prices,
            'dates': dates,
            'predictions': predictions,
            'prediction_dates': prediction_dates,
            'num_days': num_days,
            'stock': stock_ticker
        }
    
    st.success("✅ Prediction completed successfully!")
    time.sleep(0.5)
    st.rerun()

# ==============================
# 🔹 Display Results
# ==============================
if st.session_state.prediction_results is not None:
    results = st.session_state.prediction_results
    
    stock_data = results['stock_data']
    stock_info = results['stock_info']
    close_prices = results['close_prices']
    dates = results['dates']
    predictions = results['predictions']
    prediction_dates = results['prediction_dates']
    stored_num_days = results['num_days']
    stored_stock = results['stock']
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ==============================
    # 📊 Key Metrics Section
    # ==============================
    st.markdown("### 📊 Key Metrics")
    
    current_price = close_prices[-1][0]
    predicted_price = predictions[-1][0]
    price_change = predicted_price - current_price
    price_change_pct = (price_change / current_price) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Current Price",
            value=f"${current_price:.2f}",
            delta=None
        )
    
    with col2:
        st.metric(
            label=f"Predicted ({stored_num_days}d)",
            value=f"${predicted_price:.2f}",
            delta=f"{price_change:.2f} ({price_change_pct:+.2f}%)"
        )
    
    with col3:
        avg_prediction = np.mean(predictions)
        st.metric(
            label="Avg Predicted",
            value=f"${avg_prediction:.2f}",
            delta=f"{((avg_prediction - current_price) / current_price * 100):+.2f}%"
        )
    
    with col4:
        volatility = np.std(stock_data['Close'].tail(30))
        st.metric(
            label="30d Volatility",
            value=f"${volatility:.2f}",
            delta=None
        )
    
    st.markdown("---")
    
    # ==============================
    # 📈 Tabs for Different Views
    # ==============================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Price Prediction",
        "📊 Technical Analysis",
        "📋 Data Tables",
        "ℹ️ Stock Info"
    ])
    
    # Tab 1: Price Prediction
    with tab1:
        st.markdown("### 📈 Stock Price Prediction")
        
        # Combined Chart
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price Chart', 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # Historical prices
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=close_prices.flatten(),
                name='Historical Price',
                line=dict(color='#667eea', width=2),
                hovertemplate='<b>Date</b>: %{x}<br><b>Price</b>: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Predicted prices
        fig.add_trace(
            go.Scatter(
                x=prediction_dates,
                y=predictions.flatten(),
                name='Predicted Price',
                line=dict(color='#ff6b6b', width=2, dash='dash'),
                marker=dict(size=8, symbol='diamond'),
                hovertemplate='<b>Date</b>: %{x}<br><b>Predicted</b>: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Volume
        fig.add_trace(
            go.Bar(
                x=dates,
                y=stock_data['Volume'],
                name='Volume',
                marker_color='rgba(102, 126, 234, 0.5)',
                hovertemplate='<b>Date</b>: %{x}<br><b>Volume</b>: %{y:,.0f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=700,
            showlegend=True,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Poppins', size=12),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction Only Chart
        st.markdown("### 🎯 Detailed Forecast")
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=prediction_dates,
            y=predictions.flatten(),
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='#667eea', width=3),
            marker=dict(size=10, color='#764ba2', line=dict(width=2, color='white')),
            hovertemplate='<b>Date</b>: %{x}<br><b>Price</b>: $%{y:.2f}<extra></extra>'
        ))
        
        # Add confidence band (simple estimation)
        std_pred = np.std(predictions) * 1.5
        upper_bound = predictions.flatten() + std_pred
        lower_bound = predictions.flatten() - std_pred
        
        fig2.add_trace(go.Scatter(
            x=prediction_dates + prediction_dates[::-1],
            y=np.concatenate([upper_bound, lower_bound[::-1]]),
            fill='toself',
            fillcolor='rgba(102, 126, 234, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Band',
            hoverinfo='skip'
        ))
        
        fig2.update_layout(
            title=f'{stored_stock} - {stored_num_days} Day Forecast',
            xaxis_title='Date',
            yaxis_title='Predicted Price ($)',
            height=500,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Poppins', size=12)
        )
        
        fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # Tab 2: Technical Analysis
    with tab2:
        st.markdown("### 📊 Technical Indicators")
        
        # Moving Averages Chart
        fig3 = go.Figure()
        
        fig3.add_trace(go.Scatter(
            x=dates,
            y=close_prices.flatten(),
            name='Close Price',
            line=dict(color='#667eea', width=2)
        ))
        
        fig3.add_trace(go.Scatter(
            x=dates,
            y=stock_data['SMA_20'],
            name='SMA 20',
            line=dict(color='#ff6b6b', width=1.5)
        ))
        
        fig3.add_trace(go.Scatter(
            x=dates,
            y=stock_data['SMA_50'],
            name='SMA 50',
            line=dict(color='#4ecdc4', width=1.5)
        ))
        
        fig3.add_trace(go.Scatter(
            x=dates,
            y=stock_data['EMA_20'],
            name='EMA 20',
            line=dict(color='#ffe66d', width=1.5, dash='dot')
        ))
        
        fig3.update_layout(
            title='Moving Averages',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=500,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Poppins')
        )
        
        fig3.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig3.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # RSI Chart
        fig4 = go.Figure()
        
        fig4.add_trace(go.Scatter(
            x=dates,
            y=stock_data['RSI'],
            name='RSI',
            line=dict(color='#667eea', width=2),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.3)'
        ))
        
        # Overbought/Oversold lines
        fig4.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig4.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        
        fig4.update_layout(
            title='Relative Strength Index (RSI)',
            xaxis_title='Date',
            yaxis_title='RSI',
            height=400,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Poppins')
        )
        
        fig4.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        fig4.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
        
        st.plotly_chart(fig4, use_container_width=True)
        
        # Technical Indicators Summary
        col1, col2, col3 = st.columns(3)
        
        current_rsi = stock_data['RSI'].iloc[-1]
        current_sma20 = stock_data['SMA_20'].iloc[-1]
        current_price = close_prices[-1][0]
        
        with col1:
            rsi_signal = "🔴 Overbought" if current_rsi > 70 else "🟢 Oversold" if current_rsi < 30 else "🟡 Neutral"
            st.metric("RSI Signal", rsi_signal, f"{current_rsi:.2f}")
        
        with col2:
            ma_signal = "🟢 Bullish" if current_price > current_sma20 else "🔴 Bearish"
            st.metric("MA Signal", ma_signal, f"${current_sma20:.2f}")
        
        with col3:
            trend = "📈 Upward" if predictions[-1][0] > current_price else "📉 Downward"
            st.metric("Predicted Trend", trend)
    
    # Tab 3: Data Tables
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 Predicted Prices")
            prediction_df = pd.DataFrame({
                'Date': [d.strftime('%Y-%m-%d') for d in prediction_dates],
                'Predicted Price': [f"${p[0]:.2f}" for p in predictions],
                'Change from Current': [f"{((p[0] - current_price) / current_price * 100):+.2f}%" for p in predictions]
            })
            st.dataframe(prediction_df, use_container_width=True, height=400)
            
            # Download button
            csv = prediction_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions",
                data=csv,
                file_name=f"{stored_stock}_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            st.markdown("### 📊 Recent Historical Data")
            recent_data = stock_data[['Close', 'Volume', 'SMA_20', 'RSI']].tail(10)
            recent_data = recent_data.round(2)
            st.dataframe(recent_data, use_container_width=True, height=400)
            
            # Download button
            csv_historical = stock_data.to_csv()
            st.download_button(
                label="📥 Download Historical Data",
                data=csv_historical,
                file_name=f"{stored_stock}_historical_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    # Tab 4: Stock Info
    with tab4:
        st.markdown("### ℹ️ Company Information")
        
        if stock_info:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class='info-card'>
                    <h3>{stock_info.get('longName', 'N/A')}</h3>
                    <p><b>Sector:</b> {stock_info.get('sector', 'N/A')}</p>
                    <p><b>Industry:</b> {stock_info.get('industry', 'N/A')}</p>
                    <p><b>Country:</b> {stock_info.get('country', 'N/A')}</p>
                    <p><b>Website:</b> <a href="{stock_info.get('website', '#')}" target="_blank" style="color: white;">{stock_info.get('website', 'N/A')}</a></p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 15px; margin: 1rem 0;'>
                    <h4 style='color: #667eea;'>Market Data</h4>
                    <p><b>Market Cap:</b> ${stock_info.get('marketCap', 0):,.0f}</p>
                    <p><b>52 Week High:</b> ${stock_info.get('fiftyTwoWeekHigh', 0):.2f}</p>
                    <p><b>52 Week Low:</b> ${stock_info.get('fiftyTwoWeekLow', 0):.2f}</p>
                    <p><b>Average Volume:</b> {stock_info.get('averageVolume', 0):,}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 15px;'>
                    <h4 style='color: #667eea;'>Valuation Metrics</h4>
                    <p><b>P/E Ratio:</b> {stock_info.get('trailingPE', 'N/A')}</p>
                    <p><b>Forward P/E:</b> {stock_info.get('forwardPE', 'N/A')}</p>
                    <p><b>PEG Ratio:</b> {stock_info.get('pegRatio', 'N/A')}</p>
                    <p><b>Price to Book:</b> {stock_info.get('priceToBook', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 15px; margin: 1rem 0;'>
                    <h4 style='color: #667eea;'>Dividend Info</h4>
                    <p><b>Dividend Rate:</b> ${stock_info.get('dividendRate', 0):.2f}</p>
                    <p><b>Dividend Yield:</b> {stock_info.get('dividendYield', 0) * 100:.2f}%</p>
                    <p><b>Payout Ratio:</b> {stock_info.get('payoutRatio', 0) * 100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Company Description
            if 'longBusinessSummary' in stock_info:
                st.markdown("### 📝 Company Overview")
                st.markdown(f"""
                <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 15px;'>
                    {stock_info['longBusinessSummary']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Stock information not available.")

# Footer
st.markdown("---")
st.markdown("""
<div class='footer'>
    <p><b>🤖 AI Stock Price Predictor</b></p>
    <p>Powered by LSTM Deep Learning | Built with Streamlit</p>
    <p style='font-size: 0.8rem; color: #999;'>
        ⚠️ <b>Disclaimer:</b> This application is for educational and informational purposes only. 
        It should not be considered as financial advice. Always do your own research and consult with a 
        qualified financial advisor before making investment decisions.
    </p>
    <p style='margin-top: 1rem;'>Made with ❤️ by AI Enthusiasts</p>
</div>
""", unsafe_allow_html=True)
