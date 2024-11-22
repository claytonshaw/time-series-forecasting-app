import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


# Utility functions
def try_multiple_formats(value):
    formats = [
        '%m/%d/%Y',
        '%Y-%b',
        '%y-%b',
    ]
    for fmt in formats:
        try:
            parsed = pd.to_datetime(value, format=fmt, errors='coerce')
            if parsed is not pd.NaT:
                return parsed
        except ValueError:
            continue
    return pd.NaT

def manual_train_test_split(y, train_size):
    split_point = int(len(y) * train_size)
    return y[:split_point], y[split_point:]


def run_forecast(y_train, model, fh, **kwargs):
    forecaster = AutoETS(**kwargs) if model == "ETS" else AutoARIMA(**kwargs)
    forecaster.fit(y_train)
    return forecaster.predict(fh=fh)

def plot_interactive_forecast(y_train, y_test, y_pred, y_forecast, title, metrics=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_train.index, y=y_train.values, mode='lines', name="Train", line=dict(color="green")))
    if y_test is not None:
        fig.add_trace(go.Scatter(x=y_test.index, y=y_test.values, mode='lines', name="Test", line=dict(color="orange")))
        fig.add_trace(go.Scatter(x=y_test.index, y=y_pred.values, mode='lines', name="Test Predictions", line=dict(color="red")))
    if y_forecast is not None:
        fig.add_trace(go.Scatter(x=y_forecast.index, y=y_forecast.values, mode='lines', name="Forecast", line=dict(color="purple")))
    if metrics:
        mae, rmse, smape = metrics
        fig.add_annotation(
            xref="paper", yref="paper", x=0.95, y=1.1, showarrow=False,
            text=f"<b>MAE:</b> {mae:,.0f}<br><b>RMSE:</b> {rmse:,.0f}<br><b>sMAPE:</b> {smape:.1f}%",
            align="right",
            font=dict(color="white"),
            bgcolor="rgba(0, 0, 0, 0.8)",
            bordercolor="white",
            borderwidth=1
        )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10))
    return mae, rmse, smape


def ensure_frequency(df):
    if df.index.freq is None:
        inferred_freq = pd.infer_freq(df.index)
        if inferred_freq == "MS":
            inferred_freq = "M"
        if inferred_freq in ["D", "W", "M", "Q", "Y", "W-SUN"]:
            df.index = pd.date_range(start=df.index[0], periods=len(df), freq=inferred_freq)
        else:
            st.warning("Could not infer a valid frequency. Please select it manually.")
            freq_options = ["D", "W", "M", "Q", "Y", "W-SUN"]
            selected_freq = st.selectbox("Select the frequency of your data:", freq_options, index=2)
            df.index = pd.date_range(start=df.index[0], periods=len(df), freq=selected_freq)
            st.info(f"Frequency set to: {selected_freq}")
    return df


def lstm_forecast(df, target_variable, train_size=0.8, look_back=3, epochs=200, batch_size=32, lstm_params=None):
    # Data Preprocessing
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[target_variable] = scaler.fit_transform(df[target_variable].values.reshape(-1, 1))

    def create_sequences(data, look_back):
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:i + look_back])
            y.append(data[i + look_back])
        return np.array(X), np.array(y)

    data = df[target_variable].values
    X, y = create_sequences(data, look_back)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Train-test split
    train_size = int(len(X) * train_size)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # LSTM Model with Advanced Parameters
    if lstm_params is None:
        lstm_params = {}
    num_layers = lstm_params.get("num_layers", 1)
    num_units = lstm_params.get("num_units", 16)
    activation = lstm_params.get("activation", "tanh")
    optimizer = lstm_params.get("optimizer", "adam")
    learning_rate = lstm_params.get("learning_rate", None)

    # Build LSTM Model
    model = models.Sequential()
    for _ in range(num_layers):
        model.add(layers.LSTM(num_units, activation=activation, return_sequences=_ < num_layers - 1, input_shape=(look_back, 1)))
    model.add(layers.Dense(1))

    if learning_rate:
        opt = tf.keras.optimizers.get(optimizer)
        opt.learning_rate = learning_rate
        model.compile(optimizer=opt, loss='mae')
    else:
        model.compile(optimizer=optimizer, loss='mae')

    # Train Model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    return y_pred, y_test, model

def main():
    st.set_page_config(layout="wide")
    st.title("Time Series Forecasting")

    with st.sidebar:
        st.header("Settings")
        model_choice = st.selectbox("Model", ["ETS", "ARIMA", "LSTM"])
        train_size = st.slider("Train size (%)", 50, 95, 80) / 100
        forecast_periods = st.number_input("Forecast periods", min_value=1, value=12)
        advanced_settings = st.checkbox("Advanced Settings", value=False)
        # Advanced Model Parameters
        model_params = {}
        lstm_params = {}
        if advanced_settings:
            if model_choice == "ETS":
                st.subheader("Advanced ETS Settings")
                model_params.update({
                    "error": st.selectbox("Error type", ["add", "mul"]),
                    "trend": st.selectbox("Trend type", ["add", "mul", None]),
                    "seasonal": st.selectbox("Seasonal type", ["add", "mul", None]),
                    "damped_trend": st.checkbox("Damped trend", False),
                    "sp": st.number_input("Seasonal periods", min_value=1, value=1)
                })
            elif model_choice == "ARIMA":
                st.subheader("Advanced ARIMA Settings")
                model_params.update({
                    "start_p": st.number_input("Min p", 0, value=0),
                    "max_p": st.number_input("Max p", 0, value=5),
                    "start_q": st.number_input("Min q", 0, value=0),
                    "max_q": st.number_input("Max q", 0, value=5),
                    "d": st.number_input("Differencing (d)", 0, value=1),
                    "seasonal": st.checkbox("Seasonal", True)
                })
                if model_params["seasonal"]:
                    model_params.update({
                        "start_P": st.number_input("Min P", 0, value=0),
                        "max_P": st.number_input("Max P", 0, value=2),
                        "start_Q": st.number_input("Min Q", 0, value=0),
                        "max_Q": st.number_input("Max Q", 0, value=2),
                        "D": st.number_input("Seasonal differencing (D)", 0, value=1),
                        "sp": st.number_input("Seasonal periods (sp)", 1, value=12)
                    })
            elif model_choice == "LSTM":
                st.subheader("Advanced LSTM Settings")
                lstm_params.update({
                    "num_layers": st.number_input("Number of LSTM layers", 1, 10, value=1),
                    "num_units": st.number_input("Number of units per layer", 1, 128, value=16),
                    "activation": st.selectbox("Activation function", ["tanh", "relu", "sigmoid"]),
                    "optimizer": st.selectbox("Optimizer", ["adam", "sgd", "rmsprop"]),
                    "learning_rate": st.number_input("Learning rate (optional)", min_value=0.0001, max_value=1.0, step=0.0001, value=0.001),
                    "look_back": st.number_input("Look-back window size", 1, 365, value=3),
                    "epochs": st.number_input("Number of epochs", 1, 1000, value=200),
                    "batch_size": st.number_input("Batch size", 1, 128, value=32)
                })

    st.header("Upload and Explore Data")
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    if uploaded_file:
        try:
            # Load and process the data
            df = pd.read_csv(uploaded_file)
            df = df.groupby(df.columns[0]).sum().reset_index()
            df['date'] = df.iloc[:, 0].apply(try_multiple_formats)
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)

            # Ensure frequency
            df = ensure_frequency(df)

            target_variable = st.selectbox("Select target variable", df.select_dtypes(np.number).columns)
            y = df[target_variable]

            if st.button("Run Forecast"):
                if model_choice == "LSTM":
                    with st.spinner("Running LSTM Forecast..."):
                        y_pred, y_test, _ = lstm_forecast(
                            df, target_variable, train_size=train_size,
                            look_back=lstm_params.get("look_back", 3),
                            epochs=lstm_params.get("epochs", 200),
                            batch_size=lstm_params.get("batch_size", 32),
                            lstm_params=lstm_params
                        )
                        y_test_series = pd.Series(y_test.flatten(), index=y.iloc[-len(y_test):].index)
                        y_pred_series = pd.Series(y_pred.flatten(), index=y.iloc[-len(y_test):].index)

                        # Calculate metrics
                        mae, rmse, smape = calculate_metrics(y_test_series, y_pred_series)
                        metrics = (mae, rmse, smape)

                        # Plot results
                        st.subheader(f"LSTM Forecast for {target_variable}")
                        fig = plot_interactive_forecast(
                            y_train=y[:int(len(y) * train_size)],
                            y_test=y_test_series,
                            y_pred=y_pred_series,
                            y_forecast=None,
                            title="LSTM Forecast",
                            metrics=metrics
                        )
                        st.plotly_chart(fig, use_container_width=True)
                elif model_choice in ["ETS", "ARIMA"]:
                    with st.spinner(f"Running {model_choice} Forecast..."):
                        y_train, y_test = manual_train_test_split(y, train_size)
                        fh_test = ForecastingHorizon(y_test.index, is_relative=False)
                        fh_forecast = ForecastingHorizon(
                            pd.date_range(start=y.index[-1], periods=forecast_periods + 1, freq=y.index.freq)[1:],
                            is_relative=False,
                        )
                        y_pred = run_forecast(y_train, model_choice, fh_test, **model_params)
                        y_forecast = run_forecast(y_train, model_choice, fh_forecast, **model_params)

                        # Calculate metrics
                        mae, rmse, smape = calculate_metrics(y_test, y_pred)
                        metrics = (mae, rmse, smape)

                        # Plot results
                        st.subheader(f"{model_choice} Forecast for {target_variable}")
                        fig = plot_interactive_forecast(
                            y_train, y_test, y_pred, y_forecast,
                            f"{model_choice} Forecast for {target_variable}",
                            metrics=metrics
                        )
                        st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
