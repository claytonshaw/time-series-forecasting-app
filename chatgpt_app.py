import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

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
trend = st.sidebar.selectbox("Trend", ["add", "mul", "none"], index=0)
damped_trend = st.sidebar.checkbox("Damped Trend", value=False)
seasonal_periods = st.sidebar.number_input("Seasonal Periods", min_value=1, max_value=365, value=12)

# ARIMA parameters
st.sidebar.subheader("ARIMA Model")
p = st.sidebar.slider("AR(p)", 0, 5, 1)
d = st.sidebar.slider("I(d)", 0, 2, 1)
q = st.sidebar.slider("MA(q)", 0, 5, 1)
use_seasonality = st.sidebar.checkbox("Seasonal ARIMA (SARIMA)", value=False)
if use_seasonality:
    P = st.sidebar.slider("SAR(P)", 0, 5, 1)
    D = st.sidebar.slider("SAR(I)", 0, 2, 1)
    Q = st.sidebar.slider("SAR(Q)", 0, 5, 1)
    m = st.sidebar.number_input("Seasonal Period (m)", min_value=1, max_value=365, value=12)

# XGBoost parameters
st.sidebar.subheader("XGBoost Model")
lags = st.sidebar.slider("Number of Lags", 1, 50, 5)
learning_rate = st.sidebar.number_input("Learning Rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
n_estimators = st.sidebar.slider("Number of Estimators", 10, 500, 100)
max_depth = st.sidebar.slider("Max Depth", 1, 20, 6)

# LSTM parameters
st.sidebar.subheader("LSTM Model")
lstm_units = st.sidebar.slider("Number of LSTM Units", 10, 200, 50)
lstm_layers = st.sidebar.slider("Number of LSTM Layers", 1, 5, 1)
dropout = st.sidebar.number_input("Dropout Rate", min_value=0.0, max_value=0.5, value=0.2, step=0.1)
epochs = st.sidebar.slider("Epochs", 1, 100, 10)
batch_size = st.sidebar.slider("Batch Size", 1, 128, 32)

# Upload data
uploaded_file = st.file_uploader("Upload your time series CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df.groupby(df.columns[0]).sum().reset_index()
    time_series = df.iloc[:, -1]  # Assuming last column is the time series data
    
    # Display Uploaded Data
    st.write("Uploaded Data")
    st.line_chart(time_series)
    
    # Add Summary Statistics and Decomposition
    st.write("### Time Series Exploration")

    # Display Summary Statistics
    st.subheader("Summary Statistics")
    st.write("Here are the basic statistics of your time series data:")
    summary_stats = {
        "Mean": time_series.mean(),
        "Median": time_series.median(),
        "Standard Deviation": time_series.std(),
        "Min": time_series.min(),
        "Max": time_series.max(),
        "Range": time_series.max() - time_series.min(),
    }
    st.table(pd.DataFrame.from_dict(summary_stats, orient="index", columns=["Value"]))

    # Time Series Decomposition
    st.subheader("Time Series Decomposition")
    seasonal_period = st.sidebar.number_input(
        "Seasonal Period for Decomposition", min_value=1, max_value=365, value=12
    )

    decomposition = seasonal_decompose(time_series, model="additive", period=seasonal_period)

    # Plot Decomposition Results
    fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(time_series, label="Original", color="blue")
    ax[0].set_title("Original Time Series")
    ax[1].plot(decomposition.trend, label="Trend", color="orange")
    ax[1].set_title("Trend Component")
    ax[2].plot(decomposition.seasonal, label="Seasonal", color="green")
    ax[2].set_title("Seasonal Component")
    ax[3].plot(decomposition.resid, label="Residuals", color="red")
    ax[3].set_title("Residual Component")
    for a in ax:
        a.legend(loc="upper left")
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.warning("Please upload a time series dataset.")

# Run forecast
if st.button("Run Forecast"):
    results = {}

    # ETS Model
    ets_model = ExponentialSmoothing(
        time_series,
        trend=trend if trend != "none" else None,
        seasonal=seasonal if seasonal != "none" else None,
        seasonal_periods=seasonal_periods,
        damped_trend=damped_trend,
    ).fit()
    ets_forecast = ets_model.forecast(steps=12)
    results["ETS"] = calculate_metrics(time_series[-12:], ets_forecast)

    # ARIMA/SARIMA Model
    if use_seasonality:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        arima_model = SARIMAX(
            time_series,
            order=(p, d, q),
            seasonal_order=(P, D, Q, m),
        ).fit(disp=False)
    else:
        arima_model = ARIMA(time_series, order=(p, d, q)).fit()
    arima_forecast = arima_model.forecast(steps=12)
    results["ARIMA"] = calculate_metrics(time_series[-12:], arima_forecast)

    # XGBoost Model
    X = np.array([time_series.shift(i) for i in range(1, lags + 1)]).T[lags:]
    y = time_series[lags:]
    xgb_model = xgb.XGBRegressor(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    xgb_model.fit(X[:-12], y[:-12])
    xgb_forecast = xgb_model.predict(X[-12:])
    results["XGBoost"] = calculate_metrics(y[-12:], xgb_forecast)

    # LSTM Model
    lstm_model = Sequential()
    for _ in range(lstm_layers):
        lstm_model.add(LSTM(lstm_units, activation="relu", return_sequences=True))
    lstm_model.add(LSTM(lstm_units, activation="relu"))
    lstm_model.add(Dense(1))
    lstm_model.add(Dropout(dropout))
    lstm_model.compile(optimizer="adam", loss="mse")
    X_lstm = np.array([time_series.shift(i) for i in range(1, lags + 1)]).T[lags:]
    y_lstm = time_series[lags:]
    X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))
    lstm_model.fit(X_lstm[:-12], y_lstm[:-12], epochs=epochs, batch_size=batch_size, verbose=0)
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
