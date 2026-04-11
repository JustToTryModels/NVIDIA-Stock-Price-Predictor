import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import plotly.graph_objects as go

# ==============================================================================
# 🔹 Page Configuration
# ==============================================================================
# Must be the first Streamlit command.
st.set_page_config(
    page_title="NVIDIA Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==============================================================================
# 🔹 Custom Styling
# ==============================================================================
def load_css():
    """Applies custom CSS for a professional look and feel."""
    st.markdown(
        """
        <style>
        /* Main page background */
        .main {
            background-color: #f0f2f6;
        }

        /* Title styling */
        h1 {
            font-size: 3rem !important;
            color: #0e1117;
            text-shadow: 2px 2px 4px #d0d0d0;
        }

        /* Center the button */
        .stButton {
            display: flex;
            justify-content: center;
        }

        /* Custom button styling */
        .stButton > button {
            background: linear-gradient(90deg, #4CAF50, #2E7D32); /* NVIDIA Green Gradient */
            color: white !important;
            border: none;
            border-radius: 25px;
            padding: 12px 24px;
            font-size: 1.25em !important;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            width: auto;
            min-width: 300px;
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
        }

        .stButton > button:hover {
            transform: scale(1.05);
            box-shadow: 0px 6px 20px rgba(0, 0, 0, 0.3);
        }

        .stButton > button:active {
            transform: scale(0.98);
        }

        /* Style for st.metric */
        div[data-testid="metric-container"] {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# 🔹 Cached Helper Functions
# ==============================================================================
@st.cache_resource
def load_nvidia_model():
    """Loads the pre-trained Keras model from disk, cached for performance."""
    model_file = 'LSTM_Model/NVIDIA_LSTM_LB(5)_U(150)_RMSE(1.32).keras'
    try:
        model = load_model(model_file)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.error("Please ensure the model file `LSTM_Model/NVIDIA_LSTM_LB(5)_U(150)_RMSE(1.32).keras` exists.")
        return None

@st.cache_data
def get_stock_data(ticker='NVDA'):
    """Fetches historical stock data from Yahoo Finance, cached for performance."""
    try:
        # Fetch data up to today
        data = yf.download(ticker, start="2010-01-01", end=datetime.now())
        if data.empty:
            st.error(f"❌ No data found for ticker {ticker}. It might be delisted or a typo.")
            return None
        return data
    except Exception as e:
        st.error(f"❌ Error fetching stock data: {e}")
        return None

# ==============================================================================
# 🔹 Core Logic Functions
# ==============================================================================
def generate_business_days(start_date, num_days):
    """Generates a list of future business days."""
    return pd.bdate_range(start=start_date, periods=num_days).tolist()

def predict_next_days(model, data, look_back=5, days=5):
    """Predicts future stock prices using the LSTM model."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit scaler on the entire dataset to ensure consistency
    data_scaled = scaler.fit_transform(data)

    # Get the last 'look_back' days for the initial prediction
    last_sequence = data_scaled[-look_back:]
    predictions_scaled = []

    current_sequence = last_sequence.reshape((1, look_back, 1))

    for _ in range(days):
        # Predict the next value
        prediction = model.predict(current_sequence, verbose=0)
        predictions_scaled.append(prediction[0, 0])

        # Update the sequence: remove the first element and append the prediction
        current_sequence = np.append(current_sequence[:, 1:, :], [[prediction]], axis=1)

    # Inverse transform the scaled predictions to get actual price values
    predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))
    return predictions

def create_plotly_chart(dates, close_prices, prediction_dates, predictions, stock):
    """Creates an interactive Plotly chart of historical and predicted prices."""
    fig = go.Figure()

    # Historical Prices
    fig.add_trace(go.Scatter(
        x=dates,
        y=close_prices.flatten(),
        mode='lines',
        name='Historical Close Price',
        line=dict(color='#1f77b4', width=2)
    ))

    # Predicted Prices
    fig.add_trace(go.Scatter(
        x=prediction_dates,
        y=predictions.flatten(),
        mode='lines+markers',
        name='Predicted Price',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        marker=dict(color='#ff7f0e', size=6)
    ))
    
    last_hist_date = dates[-1]
    last_hist_price = close_prices[-1]
    
    # Add a point to connect the historical and prediction lines
    fig.add_trace(go.Scatter(
        x=[last_hist_date, prediction_dates[0]],
        y=[last_hist_price, predictions[0]],
        mode='lines',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        showlegend=False
    ))


    # Update layout for a professional look
    fig.update_layout(
        title=f'{stock} Stock Price: Historical and Forecasted',
        xaxis_title='Date',
        yaxis_title='Stock Price (USD)',
        xaxis_rangeslider_visible=True,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    return fig

# ==============================================================================
# 🔹 Main Application UI
# ==============================================================================
def main():
    load_css()

    st.markdown("<h1 style='text-align: center;'>NVIDIA Stock Price Predictor 🤖</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="display: flex; justify-content: center; margin-bottom: 20px;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a4/NVIDIA_logo.svg/600px-NVIDIA_logo.svg.png" width="300">
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")

    stock = 'NVDA'
    model = load_nvidia_model()

    if model is None:
        return # Stop execution if model failed to load

    # --- User Controls ---
    st.subheader("⚙️ Controls")
    num_days = st.slider("Select Number of Business Days to Forecast:", 1, 30, 5, key='num_days_slider')

    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None

    # --- Prediction Button ---
    if st.button(f'Forecast Next {num_days} Days for {stock}', key='forecast_button'):
        with st.spinner(f'🧠 Performing market analysis for {stock}...'):
            stock_data = get_stock_data(stock)

            if stock_data is None:
                st.error("Data fetching failed. Cannot proceed with prediction.")
            else:
                close_prices = stock_data['Close'].values.reshape(-1, 1)
                predictions = predict_next_days(model, close_prices, look_back=5, days=num_days)

                last_date = stock_data.index[-1]
                prediction_dates = generate_business_days(last_date + timedelta(days=1), num_days)

                # Store results in session state to persist them across reruns
                st.session_state.prediction_results = {
                    'stock_data': stock_data,
                    'predictions': predictions,
                    'prediction_dates': prediction_dates,
                    'num_days': num_days,
                }
    st.markdown("---")

    # --- Display Results ---
    if st.session_state.prediction_results:
        results = st.session_state.prediction_results
        stock_data = results['stock_data']
        predictions = results['predictions']
        prediction_dates = results['prediction_dates']
        stored_num_days = results['num_days']

        last_close_price = stock_data['Close'].iloc[-1]
        next_day_prediction = predictions[0][0]
        final_day_prediction = predictions[-1][0]

        # --- Dashboard Metrics ---
        st.subheader("📊 Prediction Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label=f"Next Day's Predicted Close ({prediction_dates[0].strftime('%Y-%m-%d')})",
                value=f"${next_day_prediction:,.2f}",
                delta=f"${next_day_prediction - last_close_price:,.2f}"
            )
        with col2:
            st.metric(
                label=f"End of Forecast Prediction ({prediction_dates[-1].strftime('%Y-%m-%d')})",
                value=f"${final_day_prediction:,.2f}",
                delta=f"${final_day_prediction - last_close_price:,.2f}"
            )
        
        st.info(f"Last available close price on {stock_data.index[-1].strftime('%Y-%m-%d')} was **${last_close_price:,.2f}**.")


        # --- Tabs for Detailed Views ---
        tab1, tab2, tab3 = st.tabs(["📈 Combined Chart", "📅 Prediction Details", "📚 Historical Data"])

        with tab1:
            fig = create_plotly_chart(
                stock_data.index,
                stock_data['Close'].values,
                prediction_dates,
                predictions,
                stock
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader(f"Predicted Prices for the Next {stored_num_days} Business Days")
            prediction_df = pd.DataFrame({
                'Date': prediction_dates,
                'Predicted Price': predictions.flatten()
            })
            st.dataframe(
                prediction_df.style.format({'Predicted Price': "${:,.2f}"}),
                use_container_width=True,
                hide_index=True
            )

        with tab3:
            st.subheader(f"Full Historical Data for {stock}")
            st.dataframe(stock_data.style.format("{:,.2f}"), use_container_width=True)

if __name__ == '__main__':
    main()
