import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import sklearn
import tensorflow as tf
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# =========================
# Load the NVIDIA model
# =========================
model_file = 'NVIDIA_LSTM_5(1.38).h5'

try:
    model = load_model(model_file)
    st.success("NVIDIA model loaded successfully.")
except Exception as e:
    st.error(f"Error loading NVIDIA model: {e}")
    model = None

# =========================
# Functions
# =========================
def get_stock_data(ticker='NVDA'):
    data = yf.download(ticker, period='max')
    return data

def generate_business_days(start_date, num_days):
    return pd.bdate_range(start=start_date, periods=num_days).tolist()

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

# =========================
# Streamlit UI
# =========================
st.markdown("<h1 style='text-align: center; font-size: 49px;'>Stock-Price-Predictor 📈📉💰</h1>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="display: flex; justify-content: center;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/NVIDIA_logo.svg/1920px-NVIDIA_logo.svg.png" width="560">
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

stock = 'NVDA'

num_days = st.slider("Select number of business days to forecast", min_value=1, max_value=30, value=5)

current_date = datetime.now().strftime('%Y-%m-%d')
st.write(f"Current Date: {current_date}")

# =========================
# Button Styling
# =========================
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
        outline: 2px solid green;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Prediction Button
# =========================
if st.button(f'Predict Next {num_days} Days Stock Prices for {stock}', key='forecast-button'):

    if model is None:
        st.error("Model not loaded. Cannot proceed.")
    else:
        stock_data = get_stock_data(stock)
        close_prices = stock_data['Close'].values.reshape(-1, 1)
        dates = stock_data.index

        st.markdown("### Historical Data for NVIDIA")
        st.dataframe(stock_data, height=400, width=1000)

        look_back = 5
        predictions = predict_next_business_days(model, close_prices, look_back, num_days)

        last_date = dates[-1]
        prediction_dates = generate_business_days(last_date + timedelta(days=1), num_days)

        # =========================
        # Plot 1: Historical + Prediction
        # =========================
        fig, ax = plt.subplots()
        ax.plot(dates, close_prices, label='Historical Prices')
        ax.plot(prediction_dates, predictions, '--', color='red', label='Predicted Prices')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.set_title(f'{stock} Stock Prices')
        ax.legend()
        st.pyplot(fig)

        # =========================
        # Plot 2: Prediction Only
        # =========================
        fig2, ax2 = plt.subplots()
        ax2.plot(prediction_dates, predictions, marker='o', color='blue')
        ax2.set_title(f'Predicted Prices ({num_days} Days)')
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        st.pyplot(fig2)

        # =========================
        # Table
        # =========================
        prediction_df = pd.DataFrame({
            'Date': prediction_dates,
            'Predicted Price': predictions.flatten()
        })
        st.dataframe(prediction_df)

# =========================
# EXACT VERSIONS USED
# =========================
st.markdown("## 📦 Package Versions (Exact Environment)")

versions = {
    "streamlit": st.__version__,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "yfinance": yf.__version__,
    "scikit-learn": sklearn.__version__,
    "tensorflow": tf.__version__,
    "matplotlib": matplotlib.__version__,
}

st.json(versions)
