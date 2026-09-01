"""
Dataset generator script for Smart Air Quality Monitoring System.
Generates realistic historical data reflecting nonlinear relationships between
PM2.5, CO, NO2, Temperature, Humidity, Wind Speed, and AQI.
Includes sample duplicates and missing values to demonstrate data cleaning.
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 1500

# 1. Simulate environmental and weather conditions
# PM2.5 (ug/m3) log-normal distribution typical of urban & suburban zones
pm25 = np.random.lognormal(mean=3.2, sigma=0.75, size=n_samples).clip(2.0, 450.0)

# NO2 (ug/m3) correlated with PM2.5 (traffic & combustion) with variance
no2 = (0.35 * pm25 + np.random.normal(25, 12, size=n_samples)).clip(3.0, 250.0)

# CO (mg/m3) correlated with vehicle exhaust & incomplete combustion
co = (0.015 * pm25 + 0.01 * no2 + np.random.normal(0.4, 0.25, size=n_samples)).clip(0.1, 15.0)

# Temperature (Celsius) typical ranges (-5 to 45 C)
temperature = np.random.normal(24, 8, size=n_samples).clip(-5.0, 45.0)

# Humidity (%) (10% to 98%)
humidity = np.random.normal(55, 18, size=n_samples).clip(10.0, 98.0)

# Wind Speed (km/h) (0.5 to 45 km/h) - higher wind accelerates dispersion
wind_speed = np.random.exponential(scale=7.5, size=n_samples).clip(0.5, 45.0)

# 2. Compute Ground Truth AQI based on EPA break-points and nonlinear physical dynamics
def calculate_aqi(p, c, n, temp, hum, wind):
    # Primary PM2.5 AQI breakpoint approximation (EPA Standard scale)
    if p <= 12.0:
        base_aqi = (50.0 / 12.0) * p
    elif p <= 35.4:
        base_aqi = 50.0 + ((100.0 - 51.0) / (35.4 - 12.1)) * (p - 12.1)
    elif p <= 55.4:
        base_aqi = 101.0 + ((150.0 - 101.0) / (55.4 - 35.5)) * (p - 35.5)
    elif p <= 150.4:
        base_aqi = 151.0 + ((200.0 - 151.0) / (150.4 - 55.5)) * (p - 55.5)
    elif p <= 250.4:
        base_aqi = 201.0 + ((300.0 - 201.0) / (250.4 - 150.5)) * (p - 150.5)
    else:
        base_aqi = 301.0 + ((500.0 - 301.0) / (500.0 - 250.5)) * (min(p, 500.0) - 250.5)

    # Multi-pollutant & meteorological non-linear contribution
    no2_factor = (n / 100.0) * 8.0
    co_factor = (c / 4.0) * 6.0
    
    # Inversion effect: calm, humid air traps pollutants
    dispersion_damping = max(0.0, (15.0 - wind) / 15.0) * (hum / 100.0) * 12.0
    
    noise = np.random.normal(0, 2.5)
    aqi_val = base_aqi + no2_factor + co_factor + dispersion_damping + noise
    return max(5.0, round(aqi_val, 1))

aqi = [calculate_aqi(p, c, n, t, h, w) for p, c, n, t, h, w in zip(pm25, co, no2, temperature, humidity, wind_speed)]

df = pd.DataFrame({
    'PM2.5': np.round(pm25, 2),
    'CO': np.round(co, 2),
    'NO2': np.round(no2, 2),
    'Temperature': np.round(temperature, 1),
    'Humidity': np.round(humidity, 1),
    'Wind_Speed': np.round(wind_speed, 1),
    'AQI': aqi
})

# 3. Add intentional anomalies to demonstrate real-world data cleaning:
# a) A few duplicate rows
duplicates = df.iloc[[12, 45, 88, 120, 230]].copy()
df = pd.concat([df, duplicates], ignore_index=True)

# b) A few missing values in environmental sensor columns (simulating packet drops)
null_indices_pm = [15, 72, 180, 410]
null_indices_wind = [33, 99, 210, 505]
null_indices_hum = [60, 142]

df.loc[null_indices_pm, 'PM2.5'] = np.nan
df.loc[null_indices_wind, 'Wind_Speed'] = np.nan
df.loc[null_indices_hum, 'Humidity'] = np.nan

# 4. Save to CSV
base_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(os.path.dirname(base_dir), 'data')
os.makedirs(out_dir, exist_ok=True)
csv_path = os.path.join(out_dir, 'air_quality_data.csv')
df.to_csv(csv_path, index=False)

print(f"Generated {len(df)} records saved to {csv_path}")
