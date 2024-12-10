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
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import time
from dateutil.parser import ParserError

# Set theme for Matplotlib
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={"axes.facecolor": "#1a1a1a", "grid.color": "#333333"})

# Helper Functions
@st.cache_data
def calculate_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = np.where(y_true == 0, 1e-6, y_true)
    y_pred = np.where(y_pred == 0, 1e-6, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
    return {"RMSE": rmse, "MAE": mae, "SMAPE": smape}

@st.cache_data
def try_multiple_formats(value):
    formats = [
    '%d/%m/%Y', '%Y/%m/%d', '%m/%d/%Y',
    '%d-%m-%Y', '%Y-%m-%d', '%m-%d-%Y',
    '%d-%b-%Y', '%d/%b/%Y', '%Y-%b-%d',
    '%d-%B-%Y', '%B %d, %Y', '%Y-%B-%d',
    '%Y-%m', '%Y-%b', '%y-%b',
    '%d-%b', '%d/%m', '%b %d',
    '%d-%m-%Y %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%d/%m/%Y %I:%M %p',
    '%s'
]
    for fmt in formats:
        try:
            parsed = pd.to_datetime(value, format=fmt, errors='coerce')
            if parsed is not pd.NaT:
                return parsed
        except ValueError:
            continue
    return pd.NaT

def preprocess_file(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        if len(df.columns) < 2:
            st.error("The uploaded file must have at least two columns.")
            return None
        try:
            # Attempt to parse dates using multiple formats
            df[df.columns[0]] = df[df.columns[0]].apply(try_multiple_formats)

            # Check for invalid rows (NaT values)
            invalid_rows = df[df[df.columns[0]].isna()]
            if not invalid_rows.empty:
                st.warning("Some dates could not be parsed and were set to NaT.")
        except Exception as e:
            st.error(f"Error processing dates: {e}")
        df = df.groupby(df.columns[0]).sum().reset_index()
        time_series = df.iloc[:, -1]  # Assuming last column is the time series data
        return time_series
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        return None

def plot_decomposition(decomposition, time_series):
    """Plots the decomposition components."""
    fig, ax = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    components = ["Original", "Trend", "Seasonal", "Residual"]
    colors = ["#39ff14", "#00ccff", "#ffcc00", "#ff5050"]
    data = [time_series, decomposition.trend, decomposition.seasonal, decomposition.resid]

    for i, (comp, color) in enumerate(zip(components, colors)):
        ax[i].plot(data[i], label=comp, color=color)
        ax[i].set_title(comp, fontsize=12, color="white")
        legend = ax[i].legend(loc="upper left", fontsize=10, facecolor="#1a1a1a", edgecolor="#333333")
        plt.setp(legend.get_texts(), color="white")
        ax[i].set_facecolor("#1a1a1a")
        ax[i].tick_params(axis="x", colors="white")
        ax[i].tick_params(axis="y", colors="white")

    plt.tight_layout()
    return fig

def time_model_execution(model_function, *args, **kwargs):
    """Times the execution of a model."""
    start_time = time.time()
    result = model_function(*args, **kwargs)
    execution_time = time.time() - start_time
    st.write(f"{model_function.__name__} executed in {execution_time:.2f} seconds.")
    return result

# Model Functions
# Define ETS model
def run_ets_model(time_series, trend, seasonal, damped_trend, seasonal_periods, steps=12):
    ets_model = ExponentialSmoothing(
        time_series,
        trend=trend if trend != "none" else None,
        seasonal=seasonal if seasonal != "none" else None,
        seasonal_periods=seasonal_periods,
        damped_trend=damped_trend,
    ).fit()
    return ets_model.forecast(steps=steps)

# Define ARIMA/SARIMA model
def run_arima_model(time_series, p, d, q, P, D, Q, m, use_seasonality, steps=12):
    if use_seasonality:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        model = SARIMAX(
            time_series,
            order=(p, d, q),
            seasonal_order=(P, D, Q, m),
        ).fit(disp=False)
    else:
        model = ARIMA(time_series, order=(p, d, q)).fit()
    return model.forecast(steps=steps)

# Define XGBoost model
def run_xgboost_model(time_series, lags, learning_rate, n_estimators, max_depth, steps=12):
    X = np.array([time_series.shift(i) for i in range(1, lags + 1)]).T[lags:]
    y = time_series[lags:]
    model = xgb.XGBRegressor(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    model.fit(X[:-steps], y[:-steps])
    return model.predict(X[-steps:])

# Define LSTM model
def run_lstm_model(time_series, lags, lstm_units, lstm_layers, dropout, epochs, batch_size, steps=12):
    X = np.array([time_series.shift(i) for i in range(1, lags + 1)]).T[lags:]
    y = time_series[lags:]
    X = X.reshape((X.shape[0], X.shape[1], 1))
    model = Sequential()
    for _ in range(lstm_layers - 1):
        model.add(LSTM(lstm_units, activation="relu", return_sequences=True))
    model.add(LSTM(lstm_units, activation="relu"))
    model.add(Dense(1))
    model.add(Dropout(dropout))
    model.compile(optimizer="adam", loss="mse")
    model.fit(X[:-steps], y[:-steps], epochs=epochs, batch_size=batch_size, verbose=0)
    return model.predict(X[-steps:]).flatten()

# Streamlit App
st.title("Univariate Time Series Forecasting App")
st.write("""
Upload a CSV file containing a univariate time series, configure the models in the sidebar, 
and generate forecasts with performance metrics.
""")

# Sidebar for user inputs
st.sidebar.header("Model Parameters")

# ETS parameters
with st.sidebar.expander("ETS Parameters", expanded=True):
    seasonal = st.selectbox("Seasonality", ["add", "mul", "none"], index=0)
    trend = st.selectbox("Trend", ["add", "mul", "none"], index=0)
    damped_trend = st.checkbox("Damped Trend", value=False)
    seasonal_periods = st.number_input("Seasonal Periods", min_value=1, max_value=365, value=12)

# ARIMA Parameters
with st.sidebar.expander("ARIMA Parameters", expanded=False):
    p = st.slider("AR(p)", 0, 5, 1)
    d = st.slider("I(d)", 0, 2, 1)
    q = st.slider("MA(q)", 0, 5, 1)
    use_seasonality = st.checkbox("Seasonal ARIMA (SARIMA)", value=False)
    if use_seasonality:
        P = st.slider("SAR(P)", 0, 5, 1)
        D = st.slider("SAR(I)", 0, 2, 1)
        Q = st.slider("SAR(Q)", 0, 5, 1)
        m = st.number_input("Seasonal Period (m)", min_value=1, max_value=365, value=12)
    else:
        P, D, Q, m = 0, 0, 0, 1  # Default values when seasonality is not used

# XGBoost parameters
with st.sidebar.expander("XGBoost Parameters", expanded=False):
    lags = st.slider("Number of Lags", 1, 50, 5)
    learning_rate = st.number_input("Learning Rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    n_estimators = st.slider("Number of Estimators", 10, 500, 100)
    max_depth = st.slider("Max Depth", 1, 20, 6)

# LSTM parameters
with st.sidebar.expander("LSTM Parameters", expanded=False):
    lstm_units = st.slider("Number of LSTM Units", 10, 200, 50)
    lstm_layers = st.slider("Number of LSTM Layers", 1, 5, 1)
    dropout = st.number_input("Dropout Rate", min_value=0.0, max_value=0.5, value=0.2, step=0.1)
    epochs = st.slider("Epochs", 1, 100, 10)
    batch_size = st.slider("Batch Size", 1, 128, 32)

# Upload and process data
uploaded_file = st.file_uploader("Upload your time series CSV file", type="csv")
time_series = preprocess_file(uploaded_file) if uploaded_file else None

if time_series is not None:
    # Display uploaded data
    st.write("Uploaded Data")
    st.line_chart(time_series)

    # Decomposition and visualization
    st.subheader("Time Series Decomposition")
    seasonal_period = st.sidebar.number_input("Seasonal Period for Decomposition", min_value=1, max_value=365, value=12)

    # Add an expander for decomposition plots
    with st.expander("Show Decomposition Plots"):
        try:
            decomposition = seasonal_decompose(time_series, model="additive", period=seasonal_period)
            st.pyplot(plot_decomposition(decomposition, time_series))
        except Exception as e:
            st.error(f"Decomposition failed: {e}")

    # Run forecast
    if st.button("Run Forecast"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Run models in parallel
        results = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                "ETS": executor.submit(time_model_execution, run_ets_model, time_series, trend, seasonal, damped_trend, seasonal_periods),
                "ARIMA": executor.submit(time_model_execution, run_arima_model, time_series, p, d, q, P, D, Q, m, use_seasonality),
                "XGBoost": executor.submit(time_model_execution, run_xgboost_model, time_series, lags, learning_rate, n_estimators, max_depth),
                "LSTM": executor.submit(time_model_execution, run_lstm_model, time_series, lags, lstm_units, lstm_layers, dropout, epochs, batch_size)
            }

            for i, (model, future) in enumerate(futures.items(), 1):
                try:
                    results[model] = calculate_metrics(time_series[-12:], future.result())
                except Exception as e:
                    st.error(f"{model} failed: {e}")
                progress_bar.progress(i / len(futures))

        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot original time series
        ax.plot(time_series, label="Actual", color="#39ff14")

        forecasts = {
            "ETS Forecast": run_ets_model(time_series, trend, seasonal, damped_trend, seasonal_periods),
            "ARIMA Forecast": run_arima_model(time_series, p, d, q, P, D, Q, m, use_seasonality),
            "XGBoost Forecast": run_xgboost_model(time_series, lags, learning_rate, n_estimators, max_depth),
            "LSTM Forecast": run_lstm_model(time_series, lags, lstm_units, lstm_layers, dropout, epochs, batch_size)
        }

        for label, forecast in forecasts.items():
            ax.plot(range(len(time_series), len(time_series) + 12), forecast, label=label, linestyle="--")

        ax.set_title("Forecast Comparison", fontsize=12, color="white")
        legend = ax.legend(loc="upper left", fontsize=10, facecolor="#1a1a1a", edgecolor="#333333")
        plt.setp(legend.get_texts(), color="white")
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")

        plt.tight_layout()
        st.pyplot(fig)

        # Metrics Table
        col1, col2 = st.columns(2)
        with col1:
            st.write("Performance Metrics")
            metrics_df = pd.DataFrame(results).T
            st.dataframe(metrics_df)

        progress_bar.progress(100)
        status_text.text("Forecasting Completed!")
