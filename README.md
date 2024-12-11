# Time Series Forecasting Application

## Overview
This Streamlit application provides a comprehensive platform for time series forecasting using various models, including ETS, ARIMA, LSTM, and XGBoost. It allows users to upload time series data, preprocess it, and evaluate forecast accuracy using interactive visualizations and performance metrics. It is the final project for my deep forecasting class.

## Features
- **Data Upload and Exploration**: Upload a CSV file, select the target variable, and explore the data.
- **Preprocessing**: Automatically infer or set the frequency of the time series data.
- **Model Selection**: Choose from multiple forecasting models:
  - **ETS (Exponential Smoothing)**
  - **ARIMA (Auto-Regressive Integrated Moving Average)**
  - **LSTM (Long Short-Term Memory Networks)**
  - **XGBoost (Extreme Gradient Boosting)**
- **Customizable Settings**:
  - Adjust train-test split ratios.
  - Set forecasting horizons.
  - Modify advanced model parameters for optimal performance.
- **Interactive Visualizations**: View training data, test predictions, and forecasts using interactive Plotly graphs.
- **Performance Metrics**: Evaluate model accuracy with MAE, RMSE, and sMAPE metrics.
