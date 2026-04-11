import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the NVIDIA model with caching
@st.cache_resource
def load_nvidia_model():
    model_file = 'LSTM_Model/NVIDIA_LSTM_LB(5)_U(150)_RMSE(1.32).keras'
    try:
        model = load_model(model_file)
        print("NVIDIA model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading NVIDIA model: {e}")
        return None

model = load_nvidia_model()

# Function to get the stock data
def get_stock_data(ticker='NVDA'):
    data = yf.download(ticker, period='max')
    return data

# Function to generate a list of business days
def generate_business_days(start_date, num_days):
    """
    Generate a list of business days starting from start_date for num_days.
    """
    return pd.bdate_range(start=start_date, periods=num_days).tolist()

# Function to make predictions for business days
def predict_next_business_days(model, data, look_back=5, days=5):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    last_sequence = data_scaled[-look_back:]
    predictions = []

    for _ in range(days):
        X_input = np.reshape(last_sequence, (1, look_back, 1))
        prediction = model.predict(X_input)
        predictions.append(prediction[0, 0])
        
        # Update the sequence for the next prediction
        last_sequence = np.append(last_sequence[1:], prediction, axis=0)
    
    # Inverse transform the predictions to the original scale
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

# Apply custom CSS to style the forecast button like the "Ask this question" button
st.markdown(
    """
    <style>
    .stButton > button {
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
    }

    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.3);
        color: white !important;
    }

    .stButton > button:active {
        transform: scale(0.98);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit app layout
st.markdown("<h1 style='text-align: center; font-size: 49px;'>Stock-Price-Predictor 📈📉💰</h1>", unsafe_allow_html=True)

# Center the NVIDIA logo image
st.markdown(
    """
    <div style="display: flex; justify-content: center;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/NVIDIA_logo.svg/1920px-NVIDIA_logo.svg.png" width="560">
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)  # Add a gap between rows

# Set NVIDIA as the selected stock
stock = 'NVDA'

# User input for number of business days to forecast
num_days = st.slider("Select number of business days to forecast", min_value=1, max_value=30, value=5)

# Display current date
current_date = datetime.now().strftime('%Y-%m-%d')
st.write(f"Current Date: {current_date}")

# Initialize session state to store prediction results
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

# Use unique key for the "Forecast" button
if st.button(f'Predict Next {num_days} Days Stock Prices for {stock}', key='forecast-button'):
    # Load stock data
    stock_data = get_stock_data(stock)
    close_prices = stock_data['Close'].values.reshape(-1, 1)
    dates = stock_data.index

    # Predict the next num_days business days
    look_back = 5
    predictions = predict_next_business_days(model, close_prices, look_back=look_back, days=num_days)
    
    # Create dates for the predictions
    last_date = dates[-1]
    prediction_dates = generate_business_days(last_date + timedelta(days=1), num_days)

    # Store results in session state
    st.session_state.prediction_results = {
        'stock_data': stock_data,
        'close_prices': close_prices,
        'dates': dates,
        'predictions': predictions,
        'prediction_dates': prediction_dates,
        'num_days': num_days,
        'stock': stock
    }

# Display results from session state (persists across slider changes)
if st.session_state.prediction_results is not None:
    results = st.session_state.prediction_results

    stock_data = results['stock_data']
    close_prices = results['close_prices']
    dates = results['dates']
    predictions = results['predictions']
    prediction_dates = results['prediction_dates']
    stored_num_days = results['num_days']
    stored_stock = results['stock']

    # Display the historical data
    st.markdown(f"### Historical Data for NVIDIA")
    st.dataframe(stock_data, height=400, width=1000)

    st.write(" ")
    
    # Prepare data for plotting the historical and predicted prices
    fig, ax = plt.subplots()
    ax.plot(dates, close_prices, label='Historical Prices')
    ax.plot(prediction_dates, predictions, label='Predicted Prices', linestyle='--', color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(f'{stored_stock} Stock Prices', fontsize=10, fontweight='bold')
    ax.legend()

    st.pyplot(fig)

    st.write(" ")
    
    # Plot only the predicted stock prices
    fig2, ax2 = plt.subplots()
    ax2.plot(prediction_dates, predictions, marker='o', color='blue')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Predicted Price')
    ax2.set_title(f'Predicted Stock Prices for the Next {stored_num_days} Business Days ({stored_stock})', fontsize=10, fontweight='bold')
    
    # Use DayLocator to specify spacing of tick marks and set the format for the date labels
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.xticks(rotation=45)
    
    st.pyplot(fig2)
    
    st.write(" ")
    
    # Show predictions in a table format
    prediction_df = pd.DataFrame({
        'Date': prediction_dates,
        'Predicted Price': predictions.flatten()
    })
    st.markdown(f"##### Predicted Stock Prices for the Next {stored_num_days} Business Days ({stored_stock})")
    st.dataframe(prediction_df, width=600)
