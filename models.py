import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

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