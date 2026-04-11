import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==============================
# 🔹 Load Model (Cached)
# ==============================
@st.cache_resource
def load_nvidia_model():
    model_file = 'LSTM_Model/NVIDIA_LSTM_LB(5)_U(150)_RMSE(1.32).keras'
    try:
        model = load_model(model_file)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_nvidia_model()

# ==============================
# 🔹 Fetch Stock Data (Cached)
# ==============================
@st.cache_data
def get_stock_data(ticker='NVDA'):
    try:
        data = yf.download(ticker, period='max')
        return data
    except Exception as e:
        st.error(f"❌ Error fetching stock data: {e}")
        return None

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
# 🔹 Custom Button Styling
# ==============================
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
    }

    .stButton > button:active {
        transform: scale(0.98);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================
# 🔹 UI Layout
# ==============================
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

# ==============================
# 🔹 User Input
# ==============================
num_days = st.slider("Select number of business days to forecast", 1, 30, 5)

current_date = datetime.now().strftime('%Y-%m-%d')
st.write(f"📅 Current Date: {current_date}")

# ==============================
# 🔹 Session State Init
# ==============================
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

# ==============================
# 🔹 Predict Button
# ==============================
if st.button(f'Predict Next {num_days} Days Stock Prices for {stock}', key='forecast-button'):

    if model is None:
        st.error("❌ Model not loaded. Cannot make predictions.")
    else:
        stock_data = get_stock_data(stock)

        if stock_data is None or stock_data.empty:
            st.error("❌ Failed to load stock data.")
        else:
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
                'close_prices': close_prices,
                'dates': dates,
                'predictions': predictions,
                'prediction_dates': prediction_dates,
                'num_days': num_days,
                'stock': stock
            }

# ==============================
# 🔹 Display Results
# ==============================
if st.session_state.prediction_results is not None:

    results = st.session_state.prediction_results

    stock_data = results['stock_data']
    close_prices = results['close_prices']
    dates = results['dates']
    predictions = results['predictions']
    prediction_dates = results['prediction_dates']
    stored_num_days = results['num_days']
    stored_stock = results['stock']

    # 📊 Historical Data
    st.markdown(f"### Historical Data for {stored_stock}")
    st.dataframe(stock_data, height=400, width=1000)

    st.write(" ")

    # 📈 Combined Plot
    fig, ax = plt.subplots()
    ax.plot(dates, close_prices, label='Historical Prices')
    ax.plot(prediction_dates, predictions, label='Predicted Prices', linestyle='--', color='red')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title(f'{stored_stock} Stock Prices', fontsize=10, fontweight='bold')
    ax.legend()
    st.pyplot(fig)

    st.write(" ")

    # 📉 Prediction Only Plot
    fig2, ax2 = plt.subplots()
    ax2.plot(prediction_dates, predictions, marker='o', color='blue')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Predicted Price')
    ax2.set_title(f'Next {stored_num_days} Business Days Prediction ({stored_stock})')

    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.xticks(rotation=45)
    st.pyplot(fig2)

    st.write(" ")

    # 📋 Table Output
    prediction_df = pd.DataFrame({
        'Date': prediction_dates,
        'Predicted Price': predictions.flatten()
    })

    st.markdown(f"### Predicted Prices ({stored_num_days} Days)")
    st.dataframe(prediction_df, width=600)
