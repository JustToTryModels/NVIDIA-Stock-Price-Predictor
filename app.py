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
    page_title="AI Stock Price Predictor",
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
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0rem;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem 1rem;
    }
    
    [data-testid="stSidebar"] .element-container {
        color: white;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    /* Card Styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(90deg, #ff6b6b, #ee5a6f);
        color: white !important;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 2.5rem;
        font-size: 1.1rem !important;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(238, 90, 111, 0.4);
        width: 100%;
        margin: 1rem 0;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #ee5a6f, #ff6b6b);
        transform: translateY(-2px);
        box-shadow: 0 7px 20px rgba(238, 90, 111, 0.6);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #4facfe, #00f2fe);
        color: white !important;
        border: none;
        border-radius: 50px;
        padding: 0.5rem 2rem;
        font-size: 0.95rem !important;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(79, 172, 254, 0.4);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 20px rgba(79, 172, 254, 0.6);
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 10px 10px 0 0;
        padding: 1rem 2rem;
        font-weight: 600;
        color: #667eea;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 1rem;
        font-weight: 600;
    }
    
    /* Dataframe Styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Success Box */
    .success-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Loading Animation */
    .loading {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
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
        data = yf.download(ticker, period=period, progress=False)
        if data.empty:
            return None, "No data received"
        return data, None
    except Exception as e:
        return None, str(e)

# ==============================
# 🔹 Get Real-time Stock Info
# ==============================
@st.cache_data(ttl=300, show_spinner=False)
def get_stock_info(ticker='NVDA'):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info, None
    except Exception as e:
        return None, str(e)

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
def calculate_indicators(data, window=20):
    df = data.copy()
    df['SMA_20'] = df['Close'].rolling(window=window).mean()
    df['EMA_20'] = df['Close'].ewm(span=window, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=window).mean()
    df['BB_Std'] = df['Close'].rolling(window=window).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
    
    return df

# ==============================
# 🎯 MAIN APP
# ==============================

# Header
st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Stock Price Predictor</h1>
        <p>Advanced LSTM Neural Network for Stock Market Forecasting</p>
    </div>
""", unsafe_allow_html=True)

# ==============================
# 📊 SIDEBAR
# ==============================
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Stock Selection
    stock_options = {
        'NVIDIA': 'NVDA',
        'Apple': 'AAPL',
        'Microsoft': 'MSFT',
        'Tesla': 'TSLA',
        'Amazon': 'AMZN',
        'Google': 'GOOGL'
    }
    
    selected_stock_name = st.selectbox(
        "🎯 Select Stock",
        options=list(stock_options.keys()),
        index=0
    )
    stock = stock_options[selected_stock_name]
    
    st.markdown("---")
    
    # Prediction Settings
    st.markdown("### 📅 Prediction Settings")
    num_days = st.slider(
        "Forecast Days",
        min_value=1,
        max_value=30,
        value=5,
        help="Number of business days to forecast"
    )
    
    st.markdown("---")
    
    # Historical Data Period
    st.markdown("### 📈 Historical Data")
    period_options = {
        '1 Month': '1mo',
        '3 Months': '3mo',
        '6 Months': '6mo',
        '1 Year': '1y',
        '2 Years': '2y',
        '5 Years': '5y',
        'Max': 'max'
    }
    
    selected_period = st.selectbox(
        "Time Period",
        options=list(period_options.keys()),
        index=6
    )
    period = period_options[selected_period]
    
    st.markdown("---")
    
    # Model Info
    st.markdown("### 🧠 Model Information")
    st.info("""
        **LSTM Neural Network**
        - Look-back: 5 days
        - Units: 150
        - RMSE: 1.32
        - Framework: TensorFlow/Keras
    """)
    
    st.markdown("---")
    
    # Current Date
    current_date = datetime.now().strftime('%Y-%m-%d')
    st.markdown(f"**📅 Current Date**  \n`{current_date}`")

# ==============================
# 🔹 Load Model
# ==============================
with st.spinner('🔄 Loading AI Model...'):
    model, model_error = load_nvidia_model()

if model_error:
    st.error(f"❌ **Model Loading Error:** {model_error}")
    st.stop()

# ==============================
# 🔹 Session State Init
# ==============================
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

# ==============================
# 📊 STOCK INFO SECTION
# ==============================
st.markdown("## 📊 Real-Time Stock Information")

with st.spinner(f'📡 Fetching real-time data for {stock}...'):
    stock_info, info_error = get_stock_info(stock)

if stock_info and not info_error:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = stock_info.get('currentPrice', stock_info.get('regularMarketPrice', 'N/A'))
        previous_close = stock_info.get('previousClose', 0)
        change = current_price - previous_close if isinstance(current_price, (int, float)) else 0
        change_pct = (change / previous_close * 100) if previous_close else 0
        
        st.metric(
            label="💰 Current Price",
            value=f"${current_price:.2f}" if isinstance(current_price, (int, float)) else current_price,
            delta=f"{change_pct:+.2f}%" if change_pct else None
        )
    
    with col2:
        market_cap = stock_info.get('marketCap', 'N/A')
        if isinstance(market_cap, (int, float)):
            market_cap_b = market_cap / 1e9
            st.metric(label="🏢 Market Cap", value=f"${market_cap_b:.2f}B")
        else:
            st.metric(label="🏢 Market Cap", value=market_cap)
    
    with col3:
        volume = stock_info.get('volume', 'N/A')
        if isinstance(volume, (int, float)):
            volume_m = volume / 1e6
            st.metric(label="📊 Volume", value=f"{volume_m:.2f}M")
        else:
            st.metric(label="📊 Volume", value=volume)
    
    with col4:
        pe_ratio = stock_info.get('trailingPE', 'N/A')
        if isinstance(pe_ratio, (int, float)):
            st.metric(label="📈 P/E Ratio", value=f"{pe_ratio:.2f}")
        else:
            st.metric(label="📈 P/E Ratio", value=pe_ratio)
    
    st.markdown("---")

# ==============================
# 🎯 PREDICTION BUTTON
# ==============================
st.markdown("## 🎯 Generate Predictions")

predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])

with predict_col2:
    predict_button = st.button(
        f'🚀 Predict Next {num_days} Days for {stock}',
        key='forecast-button',
        use_container_width=True
    )

if predict_button:
    # Progress Bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Fetch Data
    status_text.text('📡 Fetching historical stock data...')
    progress_bar.progress(20)
    time.sleep(0.3)
    
    stock_data, data_error = get_stock_data(stock, period)
    
    if data_error or stock_data is None or stock_data.empty:
        st.error(f"❌ **Data Error:** {data_error or 'Failed to load stock data'}")
        progress_bar.empty()
        status_text.empty()
        st.stop()
    
    # Step 2: Prepare Data
    status_text.text('🔧 Preparing data for prediction...')
    progress_bar.progress(40)
    time.sleep(0.3)
    
    close_prices = stock_data['Close'].values.reshape(-1, 1)
    dates = stock_data.index
    
    # Step 3: Generate Predictions
    status_text.text('🤖 Running AI prediction model...')
    progress_bar.progress(60)
    time.sleep(0.5)
    
    predictions = predict_next_business_days(
        model,
        close_prices,
        look_back=5,
        days=num_days
    )
    
    # Step 4: Generate Future Dates
    status_text.text('📅 Generating forecast dates...')
    progress_bar.progress(80)
    time.sleep(0.3)
    
    last_date = dates[-1]
    prediction_dates = generate_business_days(last_date + timedelta(days=1), num_days)
    
    # Step 5: Calculate Indicators
    status_text.text('📊 Calculating technical indicators...')
    progress_bar.progress(90)
    time.sleep(0.3)
    
    stock_data_with_indicators = calculate_indicators(stock_data)
    
    # Complete
    progress_bar.progress(100)
    status_text.text('✅ Prediction completed successfully!')
    time.sleep(0.5)
    
    progress_bar.empty()
    status_text.empty()
    
    # Store Results
    st.session_state.prediction_results = {
        'stock_data': stock_data,
        'stock_data_with_indicators': stock_data_with_indicators,
        'close_prices': close_prices,
        'dates': dates,
        'predictions': predictions,
        'prediction_dates': prediction_dates,
        'num_days': num_days,
        'stock': stock,
        'selected_stock_name': selected_stock_name
    }
    
    st.success(f"✅ Successfully generated {num_days}-day forecast for {selected_stock_name} ({stock})!")

# ==============================
# 📈 DISPLAY RESULTS
# ==============================
if st.session_state.prediction_results is not None:
    results = st.session_state.prediction_results
    
    stock_data = results['stock_data']
    stock_data_with_indicators = results['stock_data_with_indicators']
    close_prices = results['close_prices']
    dates = results['dates']
    predictions = results['predictions']
    prediction_dates = results['prediction_dates']
    stored_num_days = results['num_days']
    stored_stock = results['stock']
    stored_stock_name = results['selected_stock_name']
    
    st.markdown("---")
    
    # Prediction Summary
    st.markdown("## 📊 Prediction Summary")
    
    pred_col1, pred_col2, pred_col3, pred_col4 = st.columns(4)
    
    current_price_val = close_prices[-1][0]
    predicted_price = predictions[-1][0]
    price_change = predicted_price - current_price_val
    price_change_pct = (price_change / current_price_val) * 100
    
    with pred_col1:
        st.metric(
            label="📍 Current Price",
            value=f"${current_price_val:.2f}"
        )
    
    with pred_col2:
        st.metric(
            label=f"🎯 Predicted Price (Day {stored_num_days})",
            value=f"${predicted_price:.2f}",
            delta=f"{price_change_pct:+.2f}%"
        )
    
    with pred_col3:
        st.metric(
            label="📈 Expected Change",
            value=f"${price_change:+.2f}"
        )
    
    with pred_col4:
        trend = "📈 Bullish" if price_change > 0 else "📉 Bearish"
        st.metric(
            label="🔮 Trend",
            value=trend
        )
    
    st.markdown("---")
    
    # Tabs for Different Views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Interactive Charts",
        "📊 Prediction Table",
        "📉 Technical Analysis",
        "📋 Historical Data"
    ])
    
    # ==============================
    # TAB 1: Interactive Charts
    # ==============================
    with tab1:
        st.markdown("### 📈 Historical vs Predicted Prices")
        
        # Create Combined Chart
        fig = go.Figure()
        
        # Historical Data (last 90 days for better visibility)
        display_days = min(90, len(dates))
        
        fig.add_trace(go.Scatter(
            x=dates[-display_days:],
            y=close_prices[-display_days:].flatten(),
            mode='lines',
            name='Historical Prices',
            line=dict(color='#667eea', width=2),
            hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Price</b>: $%{y:.2f}<extra></extra>'
        ))
        
        # Predicted Data
        fig.add_trace(go.Scatter(
            x=prediction_dates,
            y=predictions.flatten(),
            mode='lines+markers',
            name='Predicted Prices',
            line=dict(color='#ff6b6b', width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond'),
            hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Predicted Price</b>: $%{y:.2f}<extra></extra>'
        ))
        
        # Add connecting line
        fig.add_trace(go.Scatter(
            x=[dates[-1], prediction_dates[0]],
            y=[close_prices[-1][0], predictions[0][0]],
            mode='lines',
            name='Transition',
            line=dict(color='gray', width=1, dash='dot'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=f'{stored_stock_name} ({stored_stock}) - Stock Price Forecast',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            font=dict(family='Poppins', size=12),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Prediction Only Chart
        st.markdown(f"### 🎯 {stored_num_days}-Day Forecast Detailed View")
        
        fig2 = go.Figure()
        
        colors = ['#4facfe' if i == 0 else '#00f2fe' if i == len(predictions)-1 else '#43e97b' 
                  for i in range(len(predictions))]
        
        fig2.add_trace(go.Bar(
            x=[d.strftime('%Y-%m-%d') for d in prediction_dates],
            y=predictions.flatten(),
            marker=dict(
                color=colors,
                line=dict(color='#667eea', width=2)
            ),
            text=[f'${p[0]:.2f}' for p in predictions],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Predicted Price: $%{y:.2f}<extra></extra>'
        ))
        
        fig2.update_layout(
            title=f'Next {stored_num_days} Business Days Prediction',
            xaxis_title='Date',
            yaxis_title='Predicted Price (USD)',
            template='plotly_white',
            height=450,
            font=dict(family='Poppins', size=12),
            showlegend=False
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # ==============================
    # TAB 2: Prediction Table
    # ==============================
    with tab2:
        st.markdown(f"### 📊 Detailed {stored_num_days}-Day Forecast")
        
        # Create prediction dataframe with more details
        prediction_df = pd.DataFrame({
            'Date': [d.strftime('%Y-%m-%d') for d in prediction_dates],
            'Day': [f'Day {i+1}' for i in range(len(predictions))],
            'Predicted Price': [f'${p[0]:.2f}' for p in predictions],
            'Change from Current': [f'${(p[0] - current_price_val):+.2f}' for p in predictions],
            'Change %': [f'{((p[0] - current_price_val) / current_price_val * 100):+.2f}%' for p in predictions],
        })
        
        st.dataframe(
            prediction_df,
            use_container_width=True,
            height=400,
            hide_index=True
        )
        
        # Download Button
        csv = prediction_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name=f'{stored_stock}_predictions_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
        )
        
        # Summary Statistics
        st.markdown("### 📊 Forecast Statistics")
        
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        
        with stat_col1:
            avg_predicted = np.mean(predictions)
            st.metric("Average Predicted Price", f"${avg_predicted:.2f}")
        
        with stat_col2:
            max_predicted = np.max(predictions)
            st.metric("Maximum Predicted Price", f"${max_predicted:.2f}")
        
        with stat_col3:
            min_predicted = np.min(predictions)
            st.metric("Minimum Predicted Price", f"${min_predicted:.2f}")
    
    # ==============================
    # TAB 3: Technical Analysis
    # ==============================
    with tab3:
        st.markdown("### 📉 Technical Indicators")
        
        # Candlestick Chart with Bollinger Bands
        fig3 = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(f'{stored_stock_name} Price with Bollinger Bands', 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # Display last 60 days
        display_days = min(60, len(stock_data_with_indicators))
        data_display = stock_data_with_indicators.iloc[-display_days:]
        
        # Candlestick
        fig3.add_trace(
            go.Candlestick(
                x=data_display.index,
                open=data_display['Open'],
                high=data_display['High'],
                low=data_display['Low'],
                close=data_display['Close'],
                name='OHLC',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )
        
        # Bollinger Bands
        fig3.add_trace(
            go.Scatter(
                x=data_display.index,
                y=data_display['BB_Upper'],
                name='BB Upper',
                line=dict(color='rgba(102, 126, 234, 0.3)', width=1)
            ),
            row=1, col=1
        )
        
        fig3.add_trace(
            go.Scatter(
                x=data_display.index,
                y=data_display['BB_Lower'],
                name='BB Lower',
                line=dict(color='rgba(102, 126, 234, 0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(102, 126, 234, 0.1)'
            ),
            row=1, col=1
        )
        
        # SMA
        fig3.add_trace(
            go.Scatter(
                x=data_display.index,
                y=data_display['SMA_20'],
                name='SMA 20',
                line=dict(color='orange', width=1.5)
            ),
            row=1, col=1
        )
        
        # Volume
        colors_volume = ['red' if row['Close'] < row['Open'] else 'green' 
                        for idx, row in data_display.iterrows()]
        
        fig3.add_trace(
            go.Bar(
                x=data_display.index,
                y=data_display['Volume'],
                name='Volume',
                marker_color=colors_volume,
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig3.update_layout(
            height=700,
            template='plotly_white',
            xaxis_rangeslider_visible=False,
            font=dict(family='Poppins', size=11),
            hovermode='x unified'
        )
        
        fig3.update_xaxes(title_text="Date", row=2, col=1)
        fig3.update_yaxes(title_text="Price (USD)", row=1, col=1)
        fig3.update_yaxes(title_text="Volume", row=2, col=1)
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # Technical Indicators Summary
        st.markdown("### 📊 Current Technical Indicators")
        
        tech_col1, tech_col2, tech_col3 = st.columns(3)
        
        with tech_col1:
            latest_sma = stock_data_with_indicators['SMA_20'].iloc[-1]
            st.metric("SMA (20)", f"${latest_sma:.2f}")
        
        with tech_col2:
            latest_ema = stock_data_with_indicators['EMA_20'].iloc[-1]
            st.metric("EMA (20)", f"${latest_ema:.2f}")
        
        with tech_col3:
            latest_bb_upper = stock_data_with_indicators['BB_Upper'].iloc[-1]
            latest_bb_lower = stock_data_with_indicators['BB_Lower'].iloc[-1]
            st.metric("BB Range", f"${latest_bb_lower:.2f} - ${latest_bb_upper:.2f}")
    
    # ==============================
    # TAB 4: Historical Data
    # ==============================
    with tab4:
        st.markdown(f"### 📋 Historical Data for {stored_stock_name} ({stored_stock})")
        
        # Display controls
        show_col1, show_col2 = st.columns(2)
        
        with show_col1:
            show_rows = st.selectbox(
                "Rows to display",
                options=[50, 100, 200, 500, 'All'],
                index=1
            )
        
        with show_col2:
            sort_order = st.selectbox(
                "Sort order",
                options=['Newest First', 'Oldest First'],
                index=0
            )
        
        # Prepare data
        display_data = stock_data.copy()
        
        if sort_order == 'Newest First':
            display_data = display_data.sort_index(ascending=False)
        
        if show_rows != 'All':
            display_data = display_data.head(show_rows)
        
        # Format and display
        st.dataframe(
            display_data.style.format({
                'Open': '${:.2f}',
                'High': '${:.2f}',
                'Low': '${:.2f}',
                'Close': '${:.2f}',
                'Volume': '{:,.0f}'
            }),
            use_container_width=True,
            height=500
        )
        
        # Download Historical Data
        csv_historical = stock_data.to_csv()
        st.download_button(
            label="📥 Download Historical Data as CSV",
            data=csv_historical,
            file_name=f'{stored_stock}_historical_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
        )

# ==============================
# 📚 Footer
# ==============================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p style='font-size: 0.9rem;'>
            <strong>⚠️ Disclaimer:</strong> This tool is for educational purposes only. 
            Stock predictions are not financial advice. Always do your own research before making investment decisions.
        </p>
        <p style='font-size: 0.85rem; margin-top: 1rem;'>
            Powered by <strong>TensorFlow</strong> & <strong>Streamlit</strong> | 
            Data provided by <strong>Yahoo Finance</strong>
        </p>
    </div>
""", unsafe_allow_html=True)
