import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import plotly.graph_objects as go

# ==============================
# 🔷 PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Stock Forecast Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# 🔷 MODEL LOADING (CACHED)
# ==============================
@st.cache_resource
def load_nvidia_model():
    model_file = "LSTM_Model/NVIDIA_LSTM_LB(5)_U(150)_RMSE(1.32).keras"
    try:
        return load_model(model_file)
    except Exception as e:
        st.error(f"Model load failed: {e}")
        return None

model = load_nvidia_model()

# ==============================
# 🔷 DATA LOADING (CACHED)
# ==============================
@st.cache_data(ttl=3600)
def get_stock_data(ticker="NVDA"):
    return yf.download(ticker, period="max")

# ==============================
# 🔷 BUSINESS DAYS
# ==============================
def generate_business_days(start_date, num_days):
    return pd.bdate_range(start=start_date, periods=num_days)

# ==============================
# 🔷 PREDICTION FUNCTION
# ==============================
def predict_next_days(model, data, look_back=5, days=5):
    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(data)

    last_seq = scaled[-look_back:]
    preds = []

    for _ in range(days):
        x = np.reshape(last_seq, (1, look_back, 1))
        yhat = model.predict(x, verbose=0)

        preds.append(yhat[0, 0])
        last_seq = np.append(last_seq[1:], yhat, axis=0)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
    return preds

# ==============================
# 🔷 SIDEBAR UI
# ==============================
st.sidebar.title("⚙️ Controls")

stock = st.sidebar.text_input("Stock Ticker", value="NVDA")
num_days = st.sidebar.slider("Forecast Days", 1, 30, 5)

run = st.sidebar.button("🚀 Run Forecast")

st.title("📊 Stock Price Forecast Dashboard")
st.caption("LSTM-powered predictive analytics for stock prices")

# ==============================
# 🔷 SESSION STATE
# ==============================
if "results" not in st.session_state:
    st.session_state.results = None

# ==============================
# 🔷 RUN PIPELINE
# ==============================
if run:

    if model is None:
        st.error("Model not loaded.")
        st.stop()

    with st.spinner("Fetching stock data..."):
        df = get_stock_data(stock)

    if df is None or df.empty:
        st.error("No data found.")
        st.stop()

    close = df["Close"].values.reshape(-1, 1)
    dates = df.index

    with st.spinner("Running prediction..."):
        preds = predict_next_days(model, close, look_back=5, days=num_days)

    pred_dates = generate_business_days(dates[-1] + timedelta(days=1), num_days)

    st.session_state.results = {
        "df": df,
        "dates": dates,
        "close": close,
        "preds": preds,
        "pred_dates": pred_dates,
        "stock": stock,
        "num_days": num_days
    }

# ==============================
# 🔷 DASHBOARD DISPLAY
# ==============================
if st.session_state.results:

    r = st.session_state.results

    df = r["df"]
    dates = r["dates"]
    close = r["close"]
    preds = r["preds"]
    pred_dates = r["pred_dates"]
    stock = r["stock"]

    last_price = float(close[-1])

    # ==========================
    # KPI CARDS
    # ==========================
    col1, col2, col3 = st.columns(3)

    col1.metric("📌 Stock", stock)
    col2.metric("💰 Last Close Price", f"${last_price:.2f}")
    col3.metric("📅 Forecast Days", len(preds))

    st.divider()

    # ==========================
    # TABS
    # ==========================
    tab1, tab2, tab3 = st.tabs(["📈 Overview", "🔮 Forecast", "📋 Data"])

    # --------------------------
    # TAB 1 - OVERVIEW
    # --------------------------
    with tab1:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=dates,
            y=close.flatten(),
            name="Historical",
            line=dict(color="royalblue")
        ))

        fig.update_layout(
            title=f"{stock} Historical Prices",
            template="plotly_dark",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    # --------------------------
    # TAB 2 - FORECAST
    # --------------------------
    with tab2:
        fig2 = go.Figure()

        fig2.add_trace(go.Scatter(
            x=pred_dates,
            y=preds.flatten(),
            name="Forecast",
            line=dict(color="orange", dash="dash"),
            marker=dict(size=8)
        ))

        fig2.update_layout(
            title=f"{stock} Forecast ({len(preds)} Days)",
            template="plotly_dark",
            height=500
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.success("Prediction completed successfully 🎯")

    # --------------------------
    # TAB 3 - DATA
    # --------------------------
    with tab3:
        st.subheader("📊 Historical Data")
        st.dataframe(df, use_container_width=True)

        st.subheader("🔮 Predicted Values")

        pred_df = pd.DataFrame({
            "Date": pred_dates,
            "Predicted Price": preds.flatten()
        })

        st.dataframe(pred_df, use_container_width=True)
