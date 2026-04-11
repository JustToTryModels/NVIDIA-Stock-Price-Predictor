import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- 1 & 2. Load the NEW NVIDIA model IN CACHE ---
@st.cache_resource
def load_nvidia_model():
    model_file = 'LSTM_Model/NVIDIA_LSTM_LB(5)_U(150)_RMSE(1.32).keras'
    try:
        model = load_model(model_file)
        print("NVIDIA model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading NVIDIA model: {e}")
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
        prediction = model.predict(X_input, verbose=0)
        predictions.append(prediction[0, 0])
        
        # Update the sequence for the next prediction
        last_sequence = np.append(last_sequence[1:], prediction, axis=0)
    
    # Inverse transform the predictions to the original scale
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

# --- 3. SESSION STATE to keep predictions ---
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
    st.session_state.prediction_dates = None
    st.session_state.stock_data = None
    st.session_state.dates = None
    st.session_state.close_prices = None
    st.session_state.num_days_predicted = None

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

st.markdown("<br>", unsafe_allow_html=True)

# Set NVIDIA as the selected stock
stock = 'NVDA'

# User input for number of business days to forecast
num_days = st.slider("Select number of business days to forecast", min_value=1, max_value=30, value=5)

# Display current date
current_date = datetime.now().strftime('%Y-%m-%d')
st.write(f"Current Date: {current_date}")

# Apply custom CSS
st.markdown(
    """
    <style>
    div.stButton > button#forecast-button {
        background-color: green;
        color: white;
    }
    div.stButton > button#forecast-button:focus,
    div.stButton > button#forecast-button:hover,
    div.stButton > button#forecast-button:active {
        color: white;
        outline: 2px solid green;
    }
    div.stButton > button:not(#forecast-button) {
        background-color: red;
        color: white;
    }
    div.stButton > button:not(#forecast-button):focus,
    div.stButton > button:not(#forecast-button):hover,
    div.stButton > button:not(#forecast-button):active {
        color: white;
        outline: 2px solid green;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Use unique key for the "Forecast" button
if st.button(f'Predict Next {num_days} Days Stock Prices for {stock}', key='forecast-button'):
    if model is not None:
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

        # --- SAVE TO SESSION STATE ---
        st.session_state.predictions = predictions
        st.session_state.prediction_dates = prediction_dates
        st.session_state.stock_data = stock_data
        st.session_state.close_prices = close_prices
        st.session_state.dates = dates
        st.session_state.num_days_predicted = num_days
    else:
        st.error("Model could not be loaded. Predictions unavailable.")

# --- DISPLAY FROM SESSION STATE (persists when slider moves) ---
if st.session_state.predictions is not None:
    stock_data = st.session_state.stock_data
    close_prices = st.session_state.close_prices
    dates = st.session_state.dates
    predictions = st.session_state.predictions
    prediction_dates = st.session_state.prediction_dates
    num_days_predicted = st.session_state.num_days_predicted

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
    ax.set_title(f'{stock} Stock Prices', fontsize=10, fontweight='bold')
    ax.legend()
    st.pyplot(fig)

    st.write(" ")
    
    # Plot only the predicted stock prices
    fig2, ax2 = plt.subplots()
    ax2.plot(prediction_dates, predictions, marker='o', color='blue')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Predicted Price')
    ax2.set_title(f'Predicted Stock Prices for the Next {num_days_predicted} Business Days ({stock})', fontsize=10, fontweight='bold')
    
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
    st.markdown(f"##### Predicted Stock Prices for the Next {num_days_predicted} Business Days ({stock})")
    st.dataframe(prediction_df, width=600)
