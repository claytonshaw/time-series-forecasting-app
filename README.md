Overview

This Streamlit application provides a comprehensive platform for time series forecasting using various models, including ETS, ARIMA, LSTM, and XGBoost. It allows users to upload time series data, preprocess it, and evaluate forecast accuracy using interactive visualizations and performance metrics.

Features
	•	Data Upload and Exploration: Upload a CSV file, select the target variable, and explore the data.
	•	Preprocessing: Automatically infer or set the frequency of the time series data.
	•	Model Selection: Choose from multiple forecasting models:
	•	ETS (Exponential Smoothing)
	•	ARIMA (Auto-Regressive Integrated Moving Average)
	•	LSTM (Long Short-Term Memory Networks)
	•	XGBoost (Extreme Gradient Boosting)
	•	Customizable Settings:
	•	Adjust train-test split ratios.
	•	Set forecasting horizons.
	•	Modify advanced model parameters for optimal performance.
	•	Interactive Visualizations: View training data, test predictions, and forecasts using interactive Plotly graphs.
	•	Performance Metrics: Evaluate model accuracy with MAE, RMSE, and sMAPE metrics.

 How to Use
	1.	Launch the application and upload a CSV file containing time series data.
	2.	Select the target variable for forecasting.
	3.	Choose a forecasting model and configure the settings (if needed).
	4.	Click “Run Forecast” to generate predictions.
	5.	Review the visualizations and metrics for model evaluation.
