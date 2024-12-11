import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import time
import streamlit as st

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
    rmse = round(rmse)
    mae = round(mae)
    smape = f"{round(smape, 1)}%"
    return {"RMSE": rmse, "MAE": mae, "SMAPE": smape}

@st.cache_data
def try_multiple_formats(value):
    formats = ['%Y-%m', '%Y-%b', '%y-%b',
    '%d/%m/%Y', '%Y/%m/%d', '%m/%d/%Y',
    '%d-%m-%Y', '%Y-%m-%d', '%m-%d-%Y',
    '%d-%b-%Y', '%d/%b/%Y', '%Y-%b-%d',
    '%d-%B-%Y', '%B %d, %Y', '%Y-%B-%d', '%d-%b', '%d/%m', '%b %d',
    '%d-%m-%Y %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%d/%m/%Y %I:%M %p', '%s'
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

@st.cache_data
def plot_autocorrelation_heatmaps(time_series, seasonal_period):
    """Plots autocorrelation and partial autocorrelation heatmaps with dynamic lags."""
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    lags = seasonal_period * 2
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_acf(time_series, ax=axes[0], lags=lags, color="#00ccff")
    plot_pacf(time_series, ax=axes[1], lags=lags, color="#ff5050")
    axes[0].set_title(f"Autocorrelation (ACF) - {lags} Lags", color="white")
    axes[1].set_title(f"Partial Autocorrelation (PACF) - {lags} Lags", color="white")
    for ax in axes:
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")
        ax.title.set_color("white")
    plt.tight_layout()
    return fig

def time_model_execution(model_function, *args, **kwargs):
    """Times the execution of a model."""
    start_time = time.time()
    result = model_function(*args, **kwargs)
    execution_time = time.time() - start_time
    st.write(f"{model_function.__name__} executed in {execution_time:.2f} seconds.")
    return result