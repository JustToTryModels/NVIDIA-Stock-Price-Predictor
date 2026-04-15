<div align="center">

# 📈 NVIDIA Stock Price Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

</div>

A comprehensive time series forecasting project that analyzes and predicts **NVIDIA Corporation's (NVDA)** stock closing prices using multiple forecasting models — including LSTM, ARIMA, Facebook Prophet, and Exponential Smoothing techniques — on over **25 years of historical market data** (January 1999 – August 2024).

<br>

---

## 📋 Table of Contents

- [What is Stock Price Prediction?](#-what-is-stock-price-prediction)
- [Why NVIDIA?](#-why-nvidia)
- [Project Overview](#-project-overview)
- [Demo](#-demo)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Exploratory Data Analysis](#-exploratory-data-analysis-eda)
- [Methodology](#-methodology)
- [Model Comparison & Selection](#-model-comparison--selection)
- [Final Model Results](#-final-model-results)
- [10-Day Price Forecast](#-10-day-price-forecast)
- [Installation & Usage](#-installation--usage)

<br>

---

## ❓ What is Stock Price Prediction?

<br>

**Stock price prediction** is the practice of forecasting future stock values using historical data, statistical methods, and machine learning algorithms. Key approaches include:

- 📊 **Technical Analysis:** Examines historical price patterns, trading volumes, and chart indicators
- 💼 **Fundamental Analysis:** Evaluates company financials, industry trends, and economic indicators
- 🤖 **Machine Learning & AI:** Utilizes advanced algorithms (LSTM, ARIMA, Prophet, etc.) to capture complex patterns
- 📰 **Sentiment Analysis:** Analyzes news articles and social media to gauge investor psychology

### ⚠️ Important Limitations
| Limitation | Description |
|------------|-------------|
| **Market Unpredictability** | Markets are influenced by unforeseen events (geopolitical issues, natural disasters) |
| **No Guarantees** | Past performance doesn't guarantee future results |
| **Complementary Tool** | Predictions should be used as one tool among many |
| **Accuracy Limits** | No model can predict stock prices with 100% accuracy |

<br>

---

## 🎯 Why NVIDIA?

<div align="center">
  <picture>
    <source srcset="https://raw.githubusercontent.com/MarpakaPradeepSai/NVIDIA-Stock-Price-Predictor/c26001d4afcc32fb695c14e5685610afa8534427/Data/Images%20%26%20GIF/NVIDIA_logo_white.svg" media="(prefers-color-scheme: dark)">
    <img src="https://raw.githubusercontent.com/MarpakaPradeepSai/NVIDIA-Stock-Price-Predictor/28d9cf153ca898289f7d3c793fa372634f581d87/Data/Images%20%26%20GIF/NVIDIA_logo_black.svg" alt="NVIDIA Logo" width="700"/>
  </picture>
</div>

<br>

NVIDIA presents a compelling subject for stock price analysis due to several key factors:

### 🏆 Industry Leadership & Innovation
- **Dominant Market Share:** NVIDIA holds a commanding lead in the GPU market
- **Technological Innovation:** Continuously advances GPU technology, driving progress in AI, gaming, and data centers

### 🚀 Growth Potential in Expanding Markets
- **AI & Deep Learning:** NVIDIA's GPUs are critical for training and deploying Large Language Models (LLMs)
- **Major Clients:** Google, Microsoft, OpenAI, Meta, Amazon Web Services, IBM, Tesla, and more
- **Data Centers & Cloud Computing:** Rapidly growing demand for cloud infrastructure

### 💰 Financial Performance
- **Strong Revenue Growth:** Consistently delivered strong revenue driven by AI demand
- **Market Volatility:** NVIDIA's stock shows significant price movements — ideal for time series modeling

<br>

---
