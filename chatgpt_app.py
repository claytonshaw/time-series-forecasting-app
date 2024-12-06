import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Helper functions
def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = np.where(y_true == 0, 1e-6, y_true)
    y_pred = np.where(y_pred == 0, 1e-6, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))  
    return {"RMSE": rmse, "MAE": mae, "SMAPE": smape}

# Streamlit App
st.title("Univariate Time Series Forecasting App")

# Sidebar for user inputs
st.sidebar.header("Model Parameters")

# ETS parameters
st.sidebar.subheader("ETS Model")
seasonal = st.sidebar.selectbox("Seasonality", ["add", "mul", "none"], index=0)

# ARIMA parameters
st.sidebar.subheader("ARIMA Model")
p = st.sidebar.slider("AR(p)", 0, 5, 1)
d = st.sidebar.slider("I(d)", 0, 2, 1)
q = st.sidebar.slider("MA(q)", 0, 5, 1)

# XGBoost parameters
st.sidebar.subheader("XGBoost Model")
lags = st.sidebar.slider("Number of Lags", 1, 20, 5)

# LSTM parameters
st.sidebar.subheader("LSTM Model")
lstm_units = st.sidebar.slider("Number of LSTM Units", 10, 100, 50)
epochs = st.sidebar.slider("Epochs", 1, 50, 10)

# Upload data
uploaded_file = st.file_uploader("Upload your time series CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    time_series = df.iloc[:, -1]
    st.write("Uploaded Data")
    st.line_chart(time_series)
else:
    st.warning("Please upload a time series dataset.")

# Run forecast
if st.button("Run Forecast"):
    results = {}

    # ETS Model
    ets_model = ExponentialSmoothing(time_series, seasonal=seasonal, seasonal_periods=12).fit()
    ets_forecast = ets_model.forecast(steps=12)
    results["ETS"] = calculate_metrics(time_series[-12:], ets_forecast)

    # ARIMA Model
    arima_model = ARIMA(time_series, order=(p, d, q)).fit()
    arima_forecast = arima_model.forecast(steps=12)
    results["ARIMA"] = calculate_metrics(time_series[-12:], arima_forecast)

    # XGBoost Model
    X = np.array([time_series.shift(i) for i in range(1, lags + 1)]).T[lags:]
    y = time_series[lags:]
    xgb_model = xgb.XGBRegressor().fit(X[:-12], y[:-12])
    xgb_forecast = xgb_model.predict(X[-12:])
    results["XGBoost"] = calculate_metrics(y[-12:], xgb_forecast)

    # LSTM Model
    lstm_model = Sequential([
        LSTM(lstm_units, activation="relu", input_shape=(lags, 1)),
        Dense(1)
    ])
    lstm_model.compile(optimizer="adam", loss="mse")
    X_lstm = np.array([time_series.shift(i) for i in range(1, lags + 1)]).T[lags:]
    y_lstm = time_series[lags:]
    X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))
    lstm_model.fit(X_lstm[:-12], y_lstm[:-12], epochs=epochs, verbose=0)
    lstm_forecast = lstm_model.predict(X_lstm[-12:])
    results["LSTM"] = calculate_metrics(y_lstm[-12:], lstm_forecast.flatten())

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(time_series, label="Original Data")
    plt.plot(range(len(time_series), len(time_series) + 12), ets_forecast, label="ETS Forecast")
    plt.plot(range(len(time_series), len(time_series) + 12), arima_forecast, label="ARIMA Forecast")
    plt.plot(range(len(time_series), len(time_series) + 12), xgb_forecast, label="XGBoost Forecast")
    plt.plot(range(len(time_series), len(time_series) + 12), lstm_forecast, label="LSTM Forecast")
    plt.legend()
    st.pyplot(plt)

    # Metrics Table
    metrics_df = pd.DataFrame(results).T
    st.write("Performance Metrics")
    st.dataframe(metrics_df)
