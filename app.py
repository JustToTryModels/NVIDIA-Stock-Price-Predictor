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
from datetime import date
import io
import base64

# ==============================
# 🔹 Page Configuration
# ==============================
st.set_page_config(
    page_title="Stock Price Predictor Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# 🔹 Custom CSS Styling
# ==============================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Custom title */
    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 3rem !important;
        font-weight: 800;
        background: linear-gradient(90deg, #00d4ff, #7c3aed, #ff0080);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        padding: 20px 0;
        margin-bottom: 10px;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #8892b0;
        font-size: 1.1rem;
        margin-bottom: 30px;
        font-weight: 400;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        border-color: rgba(0, 212, 255, 0.3);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    .metric-change {
        font-size: 1rem;
        margin-top: 5px;
    }
    
    .positive { color: #00ff88; }
    .negative { color: #ff4757; }
    
    /* Sidebar styling */
    .css-sidebar {
        background: linear-gradient(180deg, rgba(26, 26, 46, 0.95) 0%, rgba(22, 33, 62, 0.95) 100%);
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #00d4ff, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 20px 0;
    }
    
    /* Prediction cards */
    .prediction-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid;
        transition: all 0.3s ease;
    }
    
    .prediction-card:hover {
        background: rgba(255, 255, 255, 0.08);
    }
    
    .prediction-up { border-left-color: #00ff88; }
    .prediction-down { border-left-color: #ff4757; }
    
    /* Info box */
    .info-box {
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Warning box */
    .warning-box {
        background: rgba(255, 193, 7, 0.1);
        border: 1px solid rgba(255, 193, 7, 0.3);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Hide default streamlit elements */
    .stDeployButton { display: none !important; }
    #MainMenu { visibility: hidden; }
    
    /* Custom button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 30px;
        padding: 15px 40px;
        font-size: 1.1rem !important;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Download button */
    .download-btn {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
    }
    
    /* Table styling */
    .dataframe {
        border: none !important;
    }
    
    .dataframe th {
        background: rgba(0, 212, 255, 0.2) !important;
        color: #00d4ff !important;
        font-weight: 600 !important;
        border: none !important;
    }
    
    .dataframe td {
        background: rgba(255, 255, 255, 0.05) !important;
        border: none !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background: rgba(255, 255, 255, 0.05);
        padding: 10px;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        padding: 10px 20px;
        border-radius: 10px;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #00d4ff !important;
    }
    
    /* Success message */
    .stSuccess {
        background: rgba(0, 255, 136, 0.1);
        border: 1px solid rgba(0, 255, 136, 0.3);
        border-radius: 10px;
    }
    
    /* Info message */
    .stInfo {
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 10px;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        margin: 30px 0;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: #8892b0;
        padding: 20px;
        margin-top: 50px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# 🔹 Load Model (Cached)
# ==============================
@st.cache_resource
def load_nvidia_model():
    """Load the LSTM model for prediction"""
    model_file = 'LSTM_Model/NVIDIA_LSTM_LB(5)_U(150)_RMSE(1.32).keras'
    try:
        model = load_model(model_file)
        return model, None
    except FileNotFoundError:
        return None, "Model file not found. Please ensure the model is in the correct directory."
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

# ==============================
# 🔹 Fetch Stock Data (Cached)
# ==============================
@st.cache_data(ttl=3600)
def get_stock_data(ticker, period='2y'):
    """Fetch stock data from Yahoo Finance"""
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data.empty:
            return None, "No data found for the specified ticker."
        return data, None
    except Exception as e:
        return None, f"Error fetching stock data: {str(e)}"

# ==============================
# 🔹 Prediction Function
# ==============================
def predict_next_business_days(model, data, look_back=5, days=5):
    """Make predictions for future business days"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    
    last_sequence = data_scaled[-look_back:].copy()
    predictions = []
    
    for _ in range(days):
        X_input = last_sequence.reshape(1, look_back, 1)
        prediction = model.predict(X_input, verbose=0)
        predictions.append(prediction[0, 0])
        
        last_sequence = np.append(last_sequence[1:], prediction.reshape(1, 1), axis=0)
    
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

# ==============================
# 🔹 Generate Business Days
# ==============================
def generate_business_days(start_date, num_days):
    """Generate list of business days starting from start_date"""
    return pd.bdate_range(start=start_date + timedelta(days=1), periods=num_days).tolist()

# ==============================
# 🔹 Create Download Link
# ==============================
def create_download_link(df, filename, link_text):
    """Create a download link for dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="text-decoration: none;"><button style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; border: none; border-radius: 25px; padding: 10px 25px; font-weight: 600; cursor: pointer; font-size: 1rem;">{link_text}</button></a>'
    return href

# ==============================
# 🔹 Create Plotly Chart
# ==============================
def create_candlestick_chart(data, predictions, prediction_dates):
    """Create an interactive candlestick chart with predictions"""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Historical Close',
        line=dict(color='#00d4ff', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 255, 0.1)'
    ))
    
    # Add moving average
    ma_20 = data['Close'].rolling(window=20).mean()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=ma_20,
        mode='lines',
        name='20-Day MA',
        line=dict(color='#ff0080', width=1.5, dash='dot')
    ))
    
    # Predictions
    fig.add_trace(go.Scatter(
        x=prediction_dates,
        y=predictions,
        mode='lines+markers',
        name='Predicted Price',
        line=dict(color='#00ff88', width=2, dash='dash'),
        marker=dict(size=10, symbol='diamond', color='#00ff88')
    ))
    
    # Confidence interval (simulated)
    upper_bound = predictions * 1.02
    lower_bound = predictions * 0.98
    
    fig.add_trace(go.Scatter(
        x=prediction_dates + prediction_dates[::-1],
        y=list(upper_bound) + list(lower_bound)[::-1],
        fill='toself',
        fillcolor='rgba(0, 255, 136, 0.2)',
        line=dict(color='rgba(0, 255, 136, 0.2)'),
        name='Confidence Interval'
    ))
    
    fig.update_layout(
        title=dict(
            text='📊 Stock Price Analysis & Prediction',
            font=dict(size=20, color='white'),
            x=0.5
        ),
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(0,0,0,0)'
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='white')
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            tickfont=dict(color='white')
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def create_prediction_chart(prediction_dates, predictions, last_price):
    """Create a focused prediction chart"""
    colors = ['#00ff88' if pred > last_price else '#ff4757' for pred in predictions]
    
    fig = go.Figure(data=[
        go.Bar(
            x=[d.strftime('%Y-%m-%d') for d in prediction_dates],
            y=predictions,
            marker_color=colors,
            marker_line_color='white',
            marker_line_width=1
        )
    ])
    
    # Add trend line
    fig.add_trace(go.Scatter(
        x=[d.strftime('%Y-%m-%d') for d in prediction_dates],
        y=predictions,
        mode='lines+markers',
        line=dict(color='white', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title=dict(
            text='🎯 Future Price Predictions',
            font=dict(size=18, color='white'),
            x=0.5
        ),
        xaxis_title='Date',
        yaxis_title='Predicted Price (USD)',
        template='plotly_dark',
        xaxis=dict(
            tickfont=dict(color='white'),
            tickangle=-45
        ),
        yaxis=dict(
            tickfont=dict(color='white'),
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=False
    )
    
    return fig

def create_volume_chart(data):
    """Create volume chart"""
    fig = go.Figure(data=[
        go.Bar(
            x=data.index,
            y=data['Volume'],
            marker=dict(
                color=data['Volume'],
                colorscale='Viridis',
                showscale=False
            ),
            name='Volume'
        )
    ])
    
    fig.update_layout(
        title=dict(
            text='📊 Trading Volume',
            font=dict(size=16, color='white'),
            x=0.5
        ),
        xaxis_title='Date',
        yaxis_title='Volume',
        template='plotly_dark',
        xaxis=dict(
            tickfont=dict(color='white'),
            showgrid=False
        ),
        yaxis=dict(
            tickfont=dict(color='white'),
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=False,
        height=250
    )
    
    return fig

# ==============================
# 🔹 Sidebar Configuration
# ==============================
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    
    # Stock ticker input
    ticker_input = st.text_input(
        "📈 Stock Ticker Symbol",
        value="NVDA",
        help="Enter the stock ticker symbol (e.g., NVDA, AAPL, MSFT)"
    ).upper()
    
    # Date range
    st.markdown("### 📅 Historical Period")
    period_options = {
        '1 Month': '1mo',
        '3 Months': '3mo',
        '6 Months': '6mo',
        '1 Year': '1y',
        '2 Years': '2y',
        '5 Years': '5y',
        'Max': 'max'
    }
    selected_period = st.select_slider(
        "Select period",
        options=list(period_options.keys()),
        value='2 Years'
    )
    
    st.markdown("---")
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        look_back = st.slider(
            "Look Back Period",
            min_value=3,
            max_value=30,
            value=5,
            help="Number of previous days used for prediction"
        )
        
        show_volume = st.checkbox("Show Volume Chart", value=True)
        show_ma = st.checkbox("Show Moving Averages", value=True)
        confidence_level = st.slider(
            "Confidence Interval (%)",
            min_value=90,
            max_value=99,
            value=95
        )
    
    st.markdown("---")
    
    # Model info
    st.markdown("### 🤖 Model Information")
    st.info(f"""
    **Model Type:** LSTM
    **Architecture:** Deep Learning
    **Look Back:** {look_back} days
    **Status:** ✅ Loaded
    """)
    
    # Help section
    with st.expander("❓ Help & Info"):
        st.markdown("""
        **How to use:**
        1. Select your stock ticker
        2. Choose historical period
        3. Set number of prediction days
        4. Click Predict!
        
        **Interpretation:**
        - 🟢 Green = Price expected to go UP
        - 🔴 Red = Price expected to go DOWN
        """)

# ==============================
# 🔹 Main Content
# ==============================

# Title
st.markdown('<h1 class="main-title">Stock Price Predictor Pro 📈</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced AI-Powered Stock Price Forecasting with Deep Learning</p>', unsafe_allow_html=True)

# Load model
model, model_error = load_nvidia_model()

if model_error:
    st.error(f"⚠️ {model_error}")
    st.info("💡 Please ensure your LSTM model file is in the correct directory path.")
else:
    st.success("✅ LSTM Model loaded successfully!")

# Fetch stock data
stock_data, data_error = get_stock_data(ticker_input, period_options[selected_period])

if data_error:
    st.error(f"⚠️ {data_error}")
    st.info("💡 Please check the ticker symbol and try again.")

elif stock_data is not None:
    
    # ==============================
    # 🔹 Top Metrics Row
    # ==============================
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = stock_data['Close'].iloc[-1]
    previous_price = stock_data['Close'].iloc[-2]
    price_change = current_price - previous_price
    price_change_pct = (price_change / previous_price) * 100
    high_price = stock_data['High'].iloc[-30:].max()
    low_price = stock_data['Low'].iloc[-30:].min()
    avg_volume = stock_data['Volume'].iloc[-30:].mean()
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Current Price</div>
            <div class="metric-value">${current_price:.2f}</div>
            <div class="metric-change {'positive' if price_change >= 0 else 'negative'}">
                {'▲' if price_change >= 0 else '▼'} ${abs(price_change):.2f} ({abs(price_change_pct):.2f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">30-Day High</div>
            <div class="metric-value">${high_price:.2f}</div>
            <div class="metric-change positive">All Time High Range</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">30-Day Low</div>
            <div class="metric-value">${low_price:.2f}</div>
            <div class="metric-change negative">All Time Low Range</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Volume (30D)</div>
            <div class="metric-value">{avg_volume/1e6:.2f}M</div>
            <div class="metric-change">Daily Avg</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # ==============================
    # 🔹 Prediction Configuration
    # ==============================
    st.markdown("### 🎯 Configure Prediction")
    
    col_pred1, col_pred2 = st.columns([1, 3])
    
    with col_pred1:
        num_days = st.number_input(
            "Business Days to Predict",
            min_value=1,
            max_value=30,
            value=5,
            help="Number of future business days to forecast"
        )
    
    with col_pred2:
        st.markdown("&nbsp;")  # Spacer
        predict_button = st.button(f"🔮 Predict Next {num_days} Days for {ticker_input}", use_container_width=True)
    
    # ==============================
    # 🔹 Prediction Results
    # ==============================
    if predict_button:
        with st.spinner("🤖 AI Model is predicting... Please wait..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate progress
            for i in range(100):
                progress_bar.progress(i + 1)
                if i < 30:
                    status_text.text("📊 Loading data...")
                elif i < 60:
                    status_text.text("🧠 Processing with LSTM model...")
                elif i < 90:
                    status_text.text("✨ Finalizing predictions...")
                else:
                    status_text.text("✅ Prediction complete!")
                import time
                time.sleep(0.02)
            
            # Make predictions
            close_prices = stock_data['Close'].values
            predictions = predict_next_business_days(model, close_prices, look_back, num_days)
            last_date = stock_data.index[-1]
            prediction_dates = generate_business_days(last_date, num_days)
            
            st.success("🎉 Prediction complete!")
            
            # Store results in session state
            st.session_state.predictions = predictions
            st.session_state.prediction_dates = prediction_dates
            st.session_state.last_price = current_price
            st.session_state.ticker = ticker_input
    
    # Display results if available
    if 'predictions' in st.session_state and st.session_state.predictions is not None:
        predictions = st.session_state.predictions
        prediction_dates = st.session_state.prediction_dates
        last_price = st.session_state.last_price
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # ==============================
        # 🔹 Prediction Summary Cards
        # ==============================
        st.markdown("### 📊 Prediction Summary")
        
        pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)
        
        avg_prediction = np.mean(predictions)
        min_prediction = np.min(predictions)
        max_prediction = np.max(predictions)
        overall_change = predictions[-1] - last_price
        overall_change_pct = (overall_change / last_price) * 100
        
        with pred_col1:
            trend = "📈 BULLISH" if overall_change >= 0 else "📉 BEARISH"
            trend_color = "positive" if overall_change >= 0 else "negative"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Market Trend</div>
                <div class="metric-value {trend_color}">{trend}</div>
                <div class="metric-change {trend_color}">{overall_change_pct:+.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with pred_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Predicted Price</div>
                <div class="metric-value">${avg_prediction:.2f}</div>
                <div class="metric-change">Next {num_days} Days</div>
            </div>
            """, unsafe_allow_html=True)
        
        with pred_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Maximum Predicted</div>
                <div class="metric-value positive">${max_prediction:.2f}</div>
                <div class="metric-change">Peak Price</div>
            </div>
            """, unsafe_allow_html=True)
        
        with pred_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Minimum Predicted</div>
                <div class="metric-value negative">${min_prediction:.2f}</div>
                <div class="metric-change">Lowest Price</div>
            </div>
            """, unsafe_allow_html=True)
        
        # ==============================
        # 🔹 Tabs for Different Views
        # ==============================
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Combined Chart",
            "🎯 Predictions Only",
            "📋 Data Table",
            "📊 Statistics"
        ])
        
        with tab1:
            st.plotly_chart(
                create_candlestick_chart(stock_data, predictions, prediction_dates),
                use_container_width=True
            )
        
        with tab2:
            st.plotly_chart(
                create_prediction_chart(prediction_dates, predictions, last_price),
                use_container_width=True
            )
            
            # Detailed prediction cards
            st.markdown("### 🎯 Day-by-Day Predictions")
            
            for i, (date, pred) in enumerate(zip(prediction_dates, predictions)):
                change_from_last = pred - last_price
                change_pct = (change_from_last / last_price) * 100
                trend_class = "prediction-up" if pred >= last_price else "prediction-down"
                trend_icon = "📈" if pred >= last_price else "📉"
                
                st.markdown(f"""
                <div class="prediction-card {trend_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h3 style="margin: 0; color: white;">{trend_icon} Day {i+1}</h3>
                            <p style="margin: 5px 0 0 0; color: #8892b0; font-size: 0.9rem;">{date.strftime('%A, %B %d, %Y')}</p>
                        </div>
                        <div style="text-align: right;">
                            <h2 style="margin: 0; color: {'#00ff88' if pred >= last_price else '#ff4757'};">${pred:.2f}</h2>
                            <p style="margin: 5px 0 0 0; color: {'#00ff88' if change_from_last >= 0 else '#ff4757'};">
                                {'+' if change_from_last >= 0 else ''}{change_from_last:.2f} ({change_pct:+.2f}%)
                            </p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            # Create prediction dataframe
            prediction_df = pd.DataFrame({
                'Day': range(1, num_days + 1),
                'Date': [d.strftime('%Y-%m-%d') for d in prediction_dates],
                'Day Name': [d.strftime('%A') for d in prediction_dates],
                'Predicted Price': [f"${p:.2f}" for p in predictions],
                'Change from Today': [f"${p - last_price:.2f}" for p in predictions],
                'Change %': [f"{((p - last_price) / last_price) * 100:+.2f}%" for p in predictions],
                'Trend': ['📈 UP' if p >= last_price else '📉 DOWN' for p in predictions]
            })
            
            st.dataframe(
                prediction_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Download button
            st.markdown("---")
            st.markdown("### 📥 Export Results")
            
            # Prepare downloadable CSV
            csv_df = pd.DataFrame({
                'Day': range(1, num_days + 1),
                'Date': [d.strftime('%Y-%m-%d') for d in prediction_dates],
                'Predicted_Price': predictions,
                'Change_from_Today': predictions - last_price,
                'Change_Percentage': ((predictions - last_price) / last_price) * 100
            })
            
            st.markdown(create_download_link(csv_df, f'{ticker_input}_predictions.csv', '📥 Download CSV'), unsafe_allow_html=True)
        
        with tab4:
            # Statistical analysis
            st.markdown("### 📊 Prediction Statistics")
            
            stat_col1, stat_col2 = st.columns(2)
            
            with stat_col1:
                st.markdown("**📉 Statistical Measures**")
                st.markdown(f"""
                - **Mean Price:** ${np.mean(predictions):.2f}
                - **Median Price:** ${np.median(predictions):.2f}
                - **Standard Deviation:** ${np.std(predictions):.2f}
                - **Range:** ${np.ptp(predictions):.2f}
                """)
            
            with stat_col2:
                st.markdown("**📈 Price Change Analysis**")
                st.markdown(f"""
                - **Starting Price:** ${last_price:.2f}
                - **Final Predicted:** ${predictions[-1]:.2f}
                - **Total Change:** ${predictions[-1] - last_price:.2f}
                - **Total Change %:** {((predictions[-1] - last_price) / last_price) * 100:+.2f}%
                """)
            
            # Price volatility visualization
            volatility_data = pd.DataFrame({
                'Day': range(1, num_days + 1),
                'Price': predictions,
                'MA': pd.Series(predictions).rolling(window=min(3, num_days)).mean()
            })
            
            fig_volatility = px.line(
                volatility_data,
                x='Day',
                y='Price',
                title='📊 Price Volatility Over Prediction Period',
                markers=True
            )
            fig_volatility.update_layout(
                template='plotly_dark',
                xaxis_title='Day',
                yaxis_title='Predicted Price (USD)',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig_volatility, use_container_width=True)
    
    # ==============================
    # 🔹 Historical Data Section
    # ==============================
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 📜 Historical Data")
    
    with st.expander("Click to view historical stock data"):
        st.dataframe(
            stock_data.tail(30),
            use_container_width=True,
            height=400
        )
        
        # Download historical data
        hist_csv = stock_data.to_csv()
        b64_hist = base64.b64encode(hist_csv.encode()).decode()
        href_hist = f'<a href="data:file/csv;base64,{b64_hist}" download="{ticker_input}_historical.csv"><button style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 25px; padding: 10px 25px; font-weight: 600; cursor: pointer; font-size: 1rem;">📥 Download Historical Data</button></a>'
        st.markdown(href_hist, unsafe_allow_html=True)
    
    # ==============================
    # 🔹 Volume Chart
    # ==============================
    if show_volume:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.plotly_chart(create_volume_chart(stock_data), use_container_width=True)

# ==============================
# 🔹 Footer
# ==============================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    <p>📈 <strong>Stock Price Predictor Pro</strong> - Powered by LSTM Deep Learning</p>
    <p>⚠️ <em>Disclaimer: This tool is for educational purposes only. Past predictions do not guarantee future results. Always do your own research before making investment decisions.</em></p>
    <p style="margin-top: 15px;">Built with Streamlit | TensorFlow | Plotly</p>
</div>
""", unsafe_allow_html=True)
