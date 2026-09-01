"""
Smart Air Quality Monitoring and Prediction System - Flask Backend.
Supports multi-algorithm AQI prediction across 7 trained models:
Linear Regression, Decision Tree, Random Forest, Gradient Boosting,
Extra Trees, SVR (with StandardScaler), and KNN (with StandardScaler).
Integrates automated best model routing, Open-Meteo live telemetry,
EPA AQI classification, health advice, and SQLite history persistence.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import requests
from flask import Flask, render_template, request, jsonify

from ml.classifier import classify_aqi
from database import init_db, save_prediction, get_recent_predictions

app = Flask(__name__)

# Initialize database schema
init_db()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
METADATA_PATH = os.path.join(MODELS_DIR, 'model_metadata.json')

# ========================================================
# LOAD MODELS & METADATA
# ========================================================

LOADED_MODELS = {}
metadata = {}

MODEL_DISPLAY_NAMES = {
    'linear_regression': 'Linear Regression',
    'decision_tree': 'Decision Tree Regressor',
    'random_forest': 'Random Forest Regressor',
    'gradient_boosting': 'Gradient Boosting Regressor',
    'extra_trees': 'Extra Trees Regressor',
    'svr': 'Support Vector Regression (SVR)',
    'knn': 'K-Nearest Neighbors Regressor (KNN)'
}

try:
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
except Exception as e:
    print(f"Error loading model_metadata.json: {e}")

BEST_MODEL_KEY = metadata.get('best_model', 'gradient_boosting')
MODEL_FEATURES = metadata.get('features', ['PM2.5', 'CO', 'NO2', 'Temperature', 'Humidity', 'Wind_Speed'])

# Load each model from models/
for key in MODEL_DISPLAY_NAMES.keys():
    pkl_file = os.path.join(MODELS_DIR, f'{key}.pkl')
    if os.path.exists(pkl_file):
        try:
            with open(pkl_file, 'rb') as f:
                LOADED_MODELS[key] = pickle.load(f)
        except Exception as e:
            print(f"Failed to load model {key}: {e}")

# Fallback to root model.pkl if specific files not yet present
if not LOADED_MODELS:
    root_pkl = os.path.join(BASE_DIR, 'model.pkl')
    if os.path.exists(root_pkl):
        with open(root_pkl, 'rb') as f:
            fb_model = pickle.load(f)
            LOADED_MODELS['best_model'] = fb_model
            LOADED_MODELS['decision_tree'] = fb_model

FEATURE_DEFAULTS = {
    'PM2.5': 25.0,
    'CO': 0.8,
    'NO2': 30.0,
    'Temperature': 24.0,
    'Humidity': 55.0,
    'Wind_Speed': 5.0
}


# ========================================================
# HELPER: FEATURE PARSER & VALIDATION
# ========================================================

def extract_and_validate_features(data_dict):
    """
    Extracts, validates, and orders input features into a Pandas DataFrame.
    Strictly checks that pollutant concentrations and wind speed are non-negative,
    and humidity is between 0 and 100.
    """
    errors = []
    features_ordered = []
    parsed_dict = {}

    alias_map = {
        'PM2.5': ['pm25', 'pm2_5', 'pm_25', 'PM2.5'],
        'CO': ['co', 'carbon_monoxide', 'CO'],
        'NO2': ['no2', 'nitrogen_dioxide', 'NO2'],
        'Temperature': ['temperature', 'temp', 'Temperature'],
        'Humidity': ['humidity', 'Humidity'],
        'Wind_Speed': ['wind_speed', 'wind', 'Wind_Speed']
    }

    for feat in MODEL_FEATURES:
        aliases = alias_map.get(feat, [feat.lower()])
        val = None
        for alias in aliases:
            if alias in data_dict and data_dict[alias] not in (None, ''):
                val = data_dict[alias]
                break

        if val is None:
            if feat in ['Temperature', 'Humidity', 'Wind_Speed']:
                val = FEATURE_DEFAULTS.get(feat, 0.0)
            else:
                errors.append(f"Missing required parameter: {feat}")
                continue

        try:
            val_float = float(val)
        except (ValueError, TypeError):
            errors.append(f"Parameter '{feat}' must be a valid numeric number.")
            continue

        if feat in ['PM2.5', 'CO', 'NO2'] and val_float < 0:
            errors.append(f"Pollutant parameter '{feat}' cannot be negative.")
        if feat == 'Wind_Speed' and val_float < 0:
            errors.append("Wind Speed cannot be negative.")
        if feat == 'Humidity' and not (0 <= val_float <= 100):
            errors.append("Humidity must be between 0% and 100%.")

        parsed_dict[feat] = val_float
        features_ordered.append(val_float)

    if errors:
        return None, None, errors

    feature_df = pd.DataFrame([parsed_dict])[MODEL_FEATURES]
    return feature_df, parsed_dict, []


def resolve_model(requested_key: str):
    """
    Resolves requested model string to (model_key, model_display_name, model_instance).
    Defaults to automatically selected BEST_MODEL_KEY if 'best_model' is requested.
    """
    if not requested_key or requested_key.lower() in ['best_model', 'best', 'default']:
        key = BEST_MODEL_KEY
    else:
        key = requested_key.lower()

    if key in LOADED_MODELS:
        return key, MODEL_DISPLAY_NAMES.get(key, key), LOADED_MODELS[key]

    # Fallback to whatever is available
    first_key = list(LOADED_MODELS.keys())[0] if LOADED_MODELS else 'best_model'
    return first_key, MODEL_DISPLAY_NAMES.get(first_key, first_key), LOADED_MODELS.get(first_key)


# ========================================================
# DASHBOARD ROUTE
# ========================================================

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None

    if request.method == "POST":
        form_data = request.form.to_dict()
        req_model = form_data.get("model", "best_model")
        model_key, model_name, active_model = resolve_model(req_model)
        feature_matrix, parsed_dict, errors = extract_and_validate_features(form_data)

        if not errors and active_model is not None:
            raw_aqi = float(active_model.predict(feature_matrix)[0])
            classification = classify_aqi(raw_aqi)
            prediction_result = {
                "predicted_aqi": round(raw_aqi, 2),
                "category": classification["category"],
                "badge": classification["badge"],
                "recommendation": classification["recommendation"],
                "color": classification["color"],
                "model": model_name,
                "model_key": model_key,
                "is_best_model": (model_key == BEST_MODEL_KEY),
                "inputs": parsed_dict
            }
            save_prediction(
                location=form_data.get("location", "Custom Input"),
                pm25=parsed_dict.get("PM2.5", 0.0),
                co=parsed_dict.get("CO", 0.0),
                no2=parsed_dict.get("NO2", 0.0),
                temperature=parsed_dict.get("Temperature", 0.0),
                humidity=parsed_dict.get("Humidity", 0.0),
                wind_speed=parsed_dict.get("Wind_Speed", 0.0),
                predicted_aqi=prediction_result["predicted_aqi"],
                category=prediction_result["category"],
                recommendation=prediction_result["recommendation"],
                model_used=model_name
            )

    recent_history = get_recent_predictions(limit=10)
    return render_template(
        "index.html",
        prediction_result=prediction_result,
        metadata=metadata,
        best_model_key=BEST_MODEL_KEY,
        model_names=MODEL_DISPLAY_NAMES,
        history=recent_history
    )


# ========================================================
# PREDICTION API ENDPOINTS
# ========================================================

@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict API Endpoint.
    Accepts JSON payload with environmental features and optional 'model' selector.
    Supports 'best_model' or individual algorithm keys.
    """
    payload = request.get_json(silent=True) or request.form.to_dict()
    if not payload:
        return jsonify({
            "success": False,
            "error": "Empty or invalid request payload. Please supply environmental metrics."
        }), 400

    requested_model = payload.get("model", "best_model")
    model_key, model_name, active_model = resolve_model(requested_model)

    if active_model is None:
        return jsonify({
            "success": False,
            "error": f"Model '{requested_model}' is not loaded on the server."
        }), 503

    feature_matrix, parsed_dict, errors = extract_and_validate_features(payload)
    if errors:
        return jsonify({
            "success": False,
            "error": "; ".join(errors)
        }), 400

    try:
        raw_aqi = float(active_model.predict(feature_matrix)[0])
        classification = classify_aqi(raw_aqi)
        location = str(payload.get("location") or "Custom Telemetry / Input")

        record_id = save_prediction(
            location=location,
            pm25=parsed_dict.get("PM2.5", 0.0),
            co=parsed_dict.get("CO", 0.0),
            no2=parsed_dict.get("NO2", 0.0),
            temperature=parsed_dict.get("Temperature", 0.0),
            humidity=parsed_dict.get("Humidity", 0.0),
            wind_speed=parsed_dict.get("Wind_Speed", 0.0),
            predicted_aqi=round(raw_aqi, 2),
            category=classification["category"],
            recommendation=classification["recommendation"],
            model_used=model_name
        )

        return jsonify({
            "success": True,
            "id": record_id,
            "predicted_aqi": round(raw_aqi, 2),
            "category": classification["category"],
            "badge": classification["badge"],
            "color": classification["color"],
            "recommendation": classification["recommendation"],
            "tips": classification["tips"],
            "model": model_name,
            "model_key": model_key,
            "is_best_model": (model_key == BEST_MODEL_KEY),
            "features_used": MODEL_FEATURES,
            "inputs": parsed_dict
        })

    except Exception as e:
        print(f"Prediction inference error: {e}")
        return jsonify({
            "success": False,
            "error": "An error occurred while evaluating the prediction model."
        }), 500


@app.route("/predict-best", methods=["POST"])
def predict_best():
    """
    POST /predict-best Endpoint.
    Automatically forwards to the best-performing model determined during training.
    """
    payload = request.get_json(silent=True) or request.form.to_dict()
    if not payload:
        return jsonify({
            "success": False,
            "error": "Empty or invalid request payload."
        }), 400

    payload['model'] = 'best_model'
    return predict()


# ========================================================
# MODEL COMPARISON & METADATA API
# ========================================================

@app.route("/api/model-comparison", methods=["GET"])
@app.route("/api/model-info", methods=["GET"])
def model_comparison():
    """Returns comparative metrics for all 7 models, best model details, and plot manifests."""
    return jsonify({
        "success": True,
        "best_model": BEST_MODEL_KEY,
        "best_model_name": MODEL_DISPLAY_NAMES.get(BEST_MODEL_KEY, BEST_MODEL_KEY),
        "features": MODEL_FEATURES,
        "available_models": list(MODEL_DISPLAY_NAMES.keys()),
        "model_display_names": MODEL_DISPLAY_NAMES,
        "metadata": metadata
    })


# ========================================================
# PREDICTION HISTORY API
# ========================================================

@app.route("/api/history", methods=["GET"])
def history():
    """Returns latest predictions from the SQLite database."""
    limit = request.args.get("limit", 15, type=int)
    records = get_recent_predictions(limit=min(limit, 50))
    return jsonify({
        "success": True,
        "count": len(records),
        "history": records
    })


# ========================================================
# REALTIME AQI & WEATHER TELEMETRY
# ========================================================

@app.route("/realtime")
def realtime():
    lat = request.args.get("lat")
    lon = request.args.get("lon")

    if not lat or not lon:
        return jsonify({
            "aqi": "Error",
            "city": "Location not provided"
        })

    try:
        lat = float(lat)
        lon = float(lon)

        # 1. Open-Meteo Air Quality (current + hourly trend)
        air_url = (
            "https://air-quality-api.open-meteo.com/v1/air-quality"
            f"?latitude={lat}&longitude={lon}"
            "&current=us_aqi,pm2_5,carbon_monoxide,nitrogen_dioxide"
            "&hourly=us_aqi"
            "&past_days=1"
            "&forecast_days=1"
            "&timezone=auto"
        )
        air_data = requests.get(air_url, timeout=10).json()

        # 2. Open-Meteo Weather
        weather_url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
            "&timezone=auto"
        )
        weather_data = requests.get(weather_url, timeout=10).json()

        current_air = air_data.get("current", {})
        aqi = current_air.get("us_aqi")
        pm25 = current_air.get("pm2_5", 0)
        co_raw = current_air.get("carbon_monoxide", 0)
        # Open-Meteo returns CO in ug/m3; convert to standard mg/m3 if > 20
        co = round(float(co_raw) / 1000.0, 2) if float(co_raw) > 20.0 else round(float(co_raw), 2)
        no2 = current_air.get("nitrogen_dioxide", 0)

        current_weather = weather_data.get("current", {})
        temperature = current_weather.get("temperature_2m", 0)
        humidity = current_weather.get("relative_humidity_2m", 0)
        wind = current_weather.get("wind_speed_10m", 0)

        # 3. Reverse Geocoding
        city = "Selected Location"
        try:
            reverse_url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json&zoom=10"
            reverse_data = requests.get(
                reverse_url,
                headers={"User-Agent": "Smart-AQI-Dashboard"},
                timeout=10
            ).json()
            address = reverse_data.get("address", {})
            city = (
                address.get("city")
                or address.get("town")
                or address.get("municipality")
                or address.get("village")
                or address.get("county")
                or "Selected Location"
            )
        except Exception:
            city = "Selected Location"

        if aqi is None:
            return jsonify({
                "aqi": "Error",
                "city": city,
                "temp": temperature,
                "humidity": humidity,
                "wind": wind,
                "pm25": pm25,
                "co": co,
                "no2": no2
            })

        aqi_numeric = round(float(aqi), 0)

        # Predict using automatically selected best model
        predicted_aqi = None
        ml_classification = None
        _, best_name, best_model_inst = resolve_model("best_model")

        if best_model_inst is not None:
            feature_matrix, _, errs = extract_and_validate_features({
                'PM2.5': pm25,
                'CO': co,
                'NO2': no2,
                'Temperature': temperature,
                'Humidity': humidity,
                'Wind_Speed': wind
            })
            if not errs:
                pred_val = float(best_model_inst.predict(feature_matrix)[0])
                predicted_aqi = round(pred_val, 2)
                ml_classification = classify_aqi(predicted_aqi)

        if predicted_aqi is None:
            predicted_aqi = round(aqi_numeric * 1.05, 2)
            ml_classification = classify_aqi(predicted_aqi)

        current_classification = classify_aqi(aqi_numeric)

        # Extract hourly AQI trend for Chart.js
        hourly_raw = air_data.get("hourly", {})
        hourly_times = hourly_raw.get("time", [])
        hourly_aqi = hourly_raw.get("us_aqi", [])

        trend_labels = []
        trend_values = []
        if hourly_times and hourly_aqi:
            valid_pairs = [(t, a) for t, a in zip(hourly_times, hourly_aqi) if a is not None]
            recent_pairs = valid_pairs[-16:] if len(valid_pairs) > 16 else valid_pairs
            for t_str, val in recent_pairs:
                time_label = t_str.split('T')[-1] if 'T' in t_str else t_str
                trend_labels.append(time_label)
                trend_values.append(round(float(val), 1))

        return jsonify({
            "aqi": aqi_numeric,
            "city": city,
            "temp": round(float(temperature), 1),
            "humidity": round(float(humidity), 0),
            "wind": round(float(wind), 1),
            "pm25": round(float(pm25), 2),
            "co": round(float(co), 2),
            "no2": round(float(no2), 2),
            "status": current_classification["badge"],
            "category": current_classification["category"],
            "color": current_classification["color"],
            "recommendation": current_classification["recommendation"],
            "tips": current_classification["tips"],
            "predicted_aqi": predicted_aqi,
            "predicted_category": ml_classification["category"],
            "predicted_badge": ml_classification["badge"],
            "predicted_recommendation": ml_classification["recommendation"],
            "best_model_name": best_name,
            "trend_labels": trend_labels,
            "trend_values": trend_values
        })

    except Exception as e:
        print("ERROR in realtime:", e)
        return jsonify({
            "aqi": "Error",
            "city": "Location Error",
            "message": "Unable to fetch live environmental telemetry."
        })


# ========================================================
# RUN APP
# ========================================================

if __name__ == "__main__":
    app.run(
        debug=True,
        port=5001
    )
