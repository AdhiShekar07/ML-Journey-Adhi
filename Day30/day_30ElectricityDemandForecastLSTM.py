# File: ElectricityDemandForecastLSTM.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error

# Simulate hourly electricity consumption data for 3 years
np.random.seed(42)
hours = 24 * 365 * 3
time = pd.date_range(start="2020-01-01", periods=hours, freq='H')
consumption = np.sin(np.linspace(0, 100 * np.pi, hours)) * 10 + 50 + np.random.normal(0, 5, hours)

df = pd.DataFrame({'Datetime': time, 'Consumption': consumption})
df.set_index('Datetime', inplace=True)

# Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Consumption']])

# Create sequences
sequence_length = 24
X, y = [], []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i - sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Train/test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=5, batch_size=64)

# Predict
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate and Plot
mae = mean_absolute_error(y_test_actual, predictions)

plt.figure(figsize=(12, 6))
plt.plot(y_test_actual[:168], label='Actual')
plt.plot(predictions[:168], label='Predicted')
plt.title(f'7-Day Electricity Demand Forecast (MAE: {mae:.2f} kWh)')
plt.xlabel('Hour')
plt.ylabel('Consumption (kWh)')
plt.legend()
plt.tight_layout()
plt.show()
