import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import plotly.graph_objects as go

# ==============================
# ⚙️ PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Stock Predictor AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# 🎨 CUSTOM CSS
# ==============================
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    .stButton > button {
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        color: white;
        border-radius: 12px;
        padding: 0.6em 1.2em;
        font-weight: bold;
        border: none;
    }
    .stDataFrame {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# 📦 LOAD MODEL (CACHED)
# ==============================
@st.cache_resource
def load_nvidia_model():
    model_path = 'LSTM_Model/NVIDIA_LSTM_LB(5)_U(150)_RMSE(1.32).keras'
    return load_model(model_path)

# ==============================
# 📊 FETCH DATA (CACHED)
# ==============================
@st.cache_data
def get_stock_data(ticker):
    return yf.download(ticker, period="max")

# ==============================
# 📅 BUSINESS DAYS
# ==============================
def generate_business_days(start_date, n_days):
    return pd.bdate_range(start=start_date, periods=n_days)

# ==============================
# 🤖 PREDICTION FUNCTION
# ==============================
def predict_future(model, data, look_back=5, days=5):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    last_seq = scaled[-look_back:]
    preds = []

    for _ in range(days):
        x = np.reshape(last_seq, (1, look_back, 1))
        pred = model.predict(x, verbose=0)[0][0]
        preds.append(pred)
        last_seq = np.append(last_seq[1:], [[pred]], axis=0)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
    return preds

# ==============================
# 📌 LOAD MODEL
# ==============================
with st.spinner("Loading AI model..."):
    model = load_nvidia_model()

# ==============================
# 🧭 SIDEBAR CONTROLS
# ==============================
st.sidebar.title("⚙️ Controls")

ticker = st.sidebar.text_input("Stock Ticker", "NVDA")
days = st.sidebar.slider("Forecast Days", 1, 30, 5)
run = st.sidebar.button("🚀 Run Prediction")

st.title("📈 AI Stock Price Predictor")
st.caption("LSTM-based forecasting powered by TensorFlow + Streamlit")

# ==============================
# 📊 LOAD DATA
# ==============================
if run:
    if model is None:
        st.error("Model failed to load.")
        st.stop()

    with st.spinner("Fetching stock data..."):
        data = get_stock_data(ticker)

    if data is None or data.empty:
        st.error("Failed to load stock data.")
        st.stop()

    close = data["Close"].values.reshape(-1, 1)
    dates = data.index

    with st.spinner("Running predictions..."):
        preds = predict_future(model, close, look_back=5, days=days)

    pred_dates = generate_business_days(dates[-1] + timedelta(days=1), days)

    # ==============================
    # 📦 STORE STATE
    # ==============================
    st.session_state["results"] = {
        "data": data,
        "close": close,
        "dates": dates,
        "preds": preds,
        "pred_dates": pred_dates
    }

# ==============================
# 📊 DISPLAY RESULTS
# ==============================
if "results" in st.session_state:

    r = st.session_state["results"]

    data = r["data"]
    close = r["close"]
    dates = r["dates"]
    preds = r["preds"]
    pred_dates = r["pred_dates"]

    last_price = float(close[-1])
    next_price = float(preds[-1][0])
    change = next_price - last_price

    # ==============================
    # 📌 METRICS
    # ==============================
    col1, col2, col3 = st.columns(3)

    col1.metric("Last Close Price", f"${last_price:.2f}")
    col2.metric("Predicted Last Price", f"${next_price:.2f}")
    col3.metric("Expected Change", f"${change:.2f}")

    st.divider()

    # ==============================
    # 📊 TABS
    # ==============================
    tab1, tab2, tab3 = st.tabs(["📁 Data", "📈 Forecast", "📉 Charts"])

    # ==============================
    # TAB 1 - DATA
    # ==============================
    with tab1:
        st.subheader(f"{ticker} Historical Data")
        st.dataframe(data, use_container_width=True)

        csv = data.to_csv().encode("utf-8")
        st.download_button("⬇️ Download Data", csv, f"{ticker}_data.csv", "text/csv")

    # ==============================
    # TAB 2 - FORECAST TABLE
    # ==============================
    with tab2:
        forecast_df = pd.DataFrame({
            "Date": pred_dates,
            "Predicted Price": preds.flatten()
        })

        st.subheader("Future Predictions")
        st.dataframe(forecast_df, use_container_width=True)

        csv2 = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Forecast", csv2, "forecast.csv", "text/csv")

    # ==============================
    # TAB 3 - CHARTS (INTERACTIVE)
    # ==============================
    with tab3:

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dates,
            y=close.flatten(),
            name="Historical",
            line=dict(color="blue")
        ))

        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=preds.flatten(),
            name="Forecast",
            line=dict(color="red", dash="dash")
        ))

        fig.update_layout(
            title=f"{ticker} Stock Price Prediction",
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Zoomed forecast chart
        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=pred_dates,
            y=preds.flatten(),
            mode="lines+markers",
            name="Forecast"
        ))

        fig2.update_layout(
            title="Zoomed Forecast",
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Price"
        )

        st.plotly_chart(fig2, use_container_width=True)

# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.caption("⚠️ This tool is for educational purposes only. Not financial advice.")
