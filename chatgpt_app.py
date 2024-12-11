import streamlit as st
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from helpers import (
    calculate_metrics,
    preprocess_file,
    plot_decomposition,
    plot_autocorrelation_heatmaps,
    time_model_execution,
)
from models import (
    run_ets_model,
    run_arima_model,
    run_xgboost_model,
    run_lstm_model,
)

# Streamlit App
st.title("Univariate Time Series Forecasting App")
st.write("""
Upload a CSV file containing a univariate time series, configure the models in the sidebar, 
and generate forecasts with performance metrics.
""")

# Sidebar for user inputs
st.sidebar.header("Model Parameters")

# Train-test split parameter
train_size = st.sidebar.slider("Train-Test Split Ratio (Train %)", min_value=50, max_value=95, value=80, step=5) / 100

# ETS parameters
with st.sidebar.expander("ETS Parameters", expanded=False):
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
    # Train-test split
    train_data, test_data = train_test_split(time_series, train_size=train_size, shuffle=False)

    # Align data for lag-based models
    if len(test_data) < lags:
        st.error("Test dataset is too small for the specified number of lags.")
    else:
        fig_original = go.Figure()
        fig_original.add_trace(go.Scatter(
            x=train_data.index,
            y=train_data.values,
            mode='lines',
            name='Train Data',
            line=dict(color='#39ff14')
        ))
        fig_original.add_trace(go.Scatter(
            x=test_data.index,
            y=test_data.values,
            mode='lines',
            name='Test Data',
            line=dict(color='#bababa')
        ))
        fig_original.update_layout(
            title="Train and Test Data",
            xaxis_title="Time",
            yaxis_title="Value",
            template="plotly_dark",
            legend=dict(title="Legend", x=0.01, y=0.99),
            height=400
        )
        st.plotly_chart(fig_original, use_container_width=True)

    # Decomposition and visualization
    st.subheader("Time Series Decomposition")
    seasonal_period = st.sidebar.number_input("Seasonal Period for Decomposition", min_value=1, max_value=365, value=12)
    with st.expander("Show Decomposition Plots"):
        try:
            decomposition = seasonal_decompose(train_data, model="additive", period=seasonal_period)
            st.pyplot(plot_decomposition(decomposition, train_data))
        except Exception as e:
            st.error(f"Decomposition failed: {e}")

    with st.expander("Show Autocorrelation and Partial Autocorrelation Plots"):
        try:
            st.pyplot(plot_autocorrelation_heatmaps(train_data, seasonal_period=seasonal_period))
            st.markdown("Number of lags for ACF and PACF plots are set to 2 * Seasonal Period")
        except Exception as e:
            st.error(f"Autocorrelation plots failed: {e}")

    if st.button("Run Forecast"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Run models in parallel
        results = {}
        losses = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                "ETS": executor.submit(time_model_execution, run_ets_model, train_data, trend, seasonal, damped_trend, seasonal_periods, len(test_data)),
                "ARIMA": executor.submit(time_model_execution, run_arima_model, train_data, p, d, q, P, D, Q, m, use_seasonality, len(test_data)),
                "XGBoost": executor.submit(time_model_execution, run_xgboost_model, train_data, lags, learning_rate, n_estimators, max_depth, len(test_data)),
                "LSTM": executor.submit(time_model_execution, run_lstm_model, train_data, lags, lstm_units, lstm_layers, dropout, epochs, batch_size, len(test_data))
            }

            for i, (model, future) in enumerate(futures.items(), 1):
                try:
                    if model == "LSTM":
                        result, train_loss, val_loss = future.result()
                        results[model] = calculate_metrics(test_data, result)
                        losses[model] = (train_loss, val_loss)
                    else:
                        results[model] = calculate_metrics(test_data, future.result())
                except Exception as e:
                    st.error(f"{model} failed: {e}")
                progress_bar.progress(i / len(futures))

            # Create a forecast comparison plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=train_data.index,
                y=train_data.values,
                mode='lines',
                name='Actual',
                line=dict(color='#39ff14')
            ))
            fig.add_trace(go.Scatter(
                x=test_data.index,
                y=test_data.values,
                mode='lines',
                name='Test Data',
                line=dict(color='#bababa')
            ))

            forecasts = {
                "ETS Forecast": run_ets_model(train_data, trend, seasonal, damped_trend, seasonal_periods, steps=len(test_data)),
                "ARIMA Forecast": run_arima_model(train_data, p, d, q, P, D, Q, m, use_seasonality, steps=len(test_data)),
                "XGBoost Forecast": run_xgboost_model(train_data, lags, learning_rate, n_estimators, max_depth, steps=len(test_data)),
                "LSTM Forecast": run_lstm_model(train_data, lags, lstm_units, lstm_layers, dropout, epochs, batch_size, steps=len(test_data))
            }

            for label, forecast in forecasts.items():
                future_index = list(range(len(train_data), len(train_data) + len(test_data)))
                fig.add_trace(go.Scatter(
                    x=future_index,
                    y=forecast,
                    mode='lines',
                    name=label,
                    line=dict(dash='dash')
                ))

            fig.update_layout(
                title="Forecast Comparison",
                xaxis_title="Time",
                yaxis_title="Value",
                template="plotly_dark",
                legend=dict(title="Legend",
                            x=0.01,
                            y=0.99),
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Metrics Table and Loss Plot
            col1, col2 = st.columns(2)
            with col1:
                st.write("Performance Metrics")
                metrics_df = pd.DataFrame(results).T
                st.dataframe(metrics_df)

            if "LSTM" in losses:
                train_loss, val_loss = losses["LSTM"]

                # Plot training and validation loss
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=list(range(len(train_loss))),
                    y=train_loss,
                    mode='lines',
                    name='Training Loss',
                    line=dict(color='#1f77b4')
                ))
                fig_loss.add_trace(go.Scatter(
                    x=list(range(len(val_loss))),
                    y=val_loss,
                    mode='lines',
                    name='Validation Loss',
                    line=dict(color='#ff7f0e')
                ))
                fig_loss.update_layout(
                    title="LSTM Training and Validation Loss",
                    xaxis_title="Epochs",
                    yaxis_title="Loss",
                    template="plotly_dark",
                    legend=dict(title="Legend"),
                    height=250
                )
                with col2:
                    st.plotly_chart(fig_loss, use_container_width=True)

            progress_bar.progress(100)
            status_text.text("Forecasting Completed!")