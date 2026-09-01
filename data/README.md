# Air Quality Dataset Documentation

## Overview
This dataset contains historical environmental observations calibrated to standard atmospheric physics and EPA AQI calculation breakpoints.

> **Note on Data Provenance:**
> This dataset provides representative environmental conditions (including seasonal temperature shifts, humidity inversions, and particulate dispersion dynamics). While structurally modeled on real-world urban air quality distributions, it is constructed for this monitoring system to enable complete, reproducible end-to-end preprocessing, training, hyperparameter tuning, and model evaluation.

## Dataset Structure & Schema

| Column Name | Units | Physical Interpretation | Target / Feature | Expected Normal Range |
|---|---|---|---|---|
| `PM2.5` | $\mu g/m^3$ | Fine inhalable particulate matter ($\le 2.5\,\mu m$) | Feature | $2.0 - 500.0$ |
| `CO` | $mg/m^3$ | Carbon monoxide concentration from combustion | Feature | $0.1 - 15.0$ |
| `NO2` | $\mu g/m^3$ | Nitrogen dioxide from vehicle emissions & energy | Feature | $3.0 - 250.0$ |
| `Temperature` | $^\circ C$ | Ambient air temperature at 2 meters | Feature | $-10.0 - 45.0$ |
| `Humidity` | $\%$ | Relative ambient humidity | Feature | $10.0 - 100.0$ |
| `Wind_Speed` | $km/h$ | Wind velocity at 10 meters (dispersion driver) | Feature (Optional) | $0.0 - 50.0$ |
| `AQI` | Index (0-500) | Overall Air Quality Index | **Target Variable** | $0.0 - 500.0$ |

## Quality & Anomaly Handling
The dataset intentionally contains simulated packet drops / sensor dropouts (`NaN` values in `PM2.5`, `Humidity`, and `Wind_Speed`) and duplicate sensor transmissions to test and demonstrate automated data cleaning.
