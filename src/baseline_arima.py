import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# LOAD DATA - using Intermediate Job for prediction baseline
df = pd.read_csv('week03_updated_cleaned_data.csv') # --- ENSURE CORRECT FILE PATH ON PERSONAL MACHINE --
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# Filter for the target job
target_data = df[df['job_type'] == 'Intermediate Job']['cpu_cores']

# SPLIT DATA (80/20 Train/test split)
train_size = int(len(target_data) * 0.8)
train, test = target_data.iloc[:train_size], target_data.iloc[train_size:]

print(f"Training Samples: {len(train)} | Testing Samples: {len(test)}")

# TRAIN ARIMA MODEL
# (p,d,q) = (5,1,0) as a standard starting point for time series
print("Training ARIMA Model (this might take a moment)...")
model = ARIMA(train, order=(5,1,0))
model_fit = model.fit()

print(model_fit.summary())

# FORECAST
print("Generating Forecast...")
forecast = model_fit.forecast(steps=len(test))
forecast = pd.Series(forecast, index=test.index)

# EVALUATE ACCURACY
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))

print(f"\n--- ARIMA BASELINE RESULTS ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} Cores")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} Cores")

# VISUALIZATION
plt.figure(figsize=(12, 6))
plt.plot(train.index[-200:], train[-200:], label='Training Data (End)', color='green', alpha=0.5)
plt.plot(test.index[:200], test[:200], label='Actual Demand', color='blue')
plt.plot(test.index[:200], forecast[:200], label='ARIMA Prediction', color='red', linestyle='--')

plt.title(f"Baseline Prediction: ARIMA Model\nMAE: {mae:.2f} | RMSE: {rmse:.2f}")
plt.xlabel("Time")
plt.ylabel("CPU Cores")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig('Baseline_ARIMA.png')
print("Graph saved as 'Baseline_ARIMA.png'")
plt.show()
