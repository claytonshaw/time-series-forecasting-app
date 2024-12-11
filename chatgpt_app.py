import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from concurrent.futures import ThreadPoolExecutor
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
    # Display uploaded data
    st.write("Uploaded Data")
        # Create an interactive Plotly figure for the original time series
    fig_original = go.Figure()
    
    # Add the original time series
    fig_original.add_trace(go.Scatter(
        x=time_series.index,
        y=time_series.values,
        mode='lines',
        name='Original Time Series',
        line=dict(color='#39ff14')
    ))
    
    # Customize layout
    fig_original.update_layout(
        title="Original Time Series",
        xaxis_title="Time",
        yaxis_title="Value",
        template="plotly_dark",
        legend=dict(title="Legend"),
        height=400
    )
    
    # Render Plotly chart in Streamlit
    st.plotly_chart(fig_original, use_container_width=True)

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

    # Add an expander for ACF and PACF plots
    with st.expander("Show Autocorrelation and Partial Autocorrelation Plots"):
        
        try:
            st.pyplot(plot_autocorrelation_heatmaps(time_series, seasonal_period=seasonal_period))
            st.markdown("Number of lags for ACF and PACF plots are set to 2 * Seasonal Period")
        except Exception as e:
            st.error(f"Autocorrelation plots failed: {e}")

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

            # Create an interactive Plotly figure
            fig = go.Figure()

            # Add the original time series
            fig.add_trace(go.Scatter(
                x=time_series.index,
                y=time_series.values,
                mode='lines',
                name='Actual',
                line=dict(color='#39ff14')
            ))

            # Add forecasts
            forecasts = {
                "ETS Forecast": run_ets_model(time_series, trend, seasonal, damped_trend, seasonal_periods),
                "ARIMA Forecast": run_arima_model(time_series, p, d, q, P, D, Q, m, use_seasonality),
                "XGBoost Forecast": run_xgboost_model(time_series, lags, learning_rate, n_estimators, max_depth),
                "LSTM Forecast": run_lstm_model(time_series, lags, lstm_units, lstm_layers, dropout, epochs, batch_size)
            }

            for label, forecast in forecasts.items():
                # Convert range to a list for Plotly
                future_index = list(range(len(time_series), len(time_series) + len(forecast)))
                fig.add_trace(go.Scatter(
                    x=future_index,
                    y=forecast,
                    mode='lines',
                    name=label,
                    line=dict(dash='dash')
                ))

            # Customize layout
            fig.update_layout(
                title="Forecast Comparison",
                xaxis_title="Time",
                yaxis_title="Value",
                template="plotly_dark",
                legend=dict(title="Legend",
                            x=0.01,
                            y=0.99,
                            bgcolor='rgba(0,0,0,0.5)',
                            bordercolor='black',
                            borderwidth=1),
                height=600
            )

            # Render Plotly chart in Streamlit
            st.plotly_chart(fig, use_container_width=True)

        # Metrics Table
        col1, col2 = st.columns(2)
        with col1:
            st.write("Performance Metrics")
            metrics_df = pd.DataFrame(results).T
            st.dataframe(metrics_df)

        progress_bar.progress(100)
        status_text.text("Forecasting Completed!")