from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import requests

app = Flask(__name__)

# =========================
# LOAD ML MODEL
# =========================

model = pickle.load(open("model.pkl", "rb"))


# =========================
# AQI STATUS
# =========================

def get_aqi_status(aqi):

    aqi = float(aqi)

    if aqi <= 50:
        return "Good 😊"

    elif aqi <= 100:
        return "Moderate 😐"

    elif aqi <= 150:
        return "Poor 😷"

    elif aqi <= 200:
        return "Unhealthy 🚨"

    else:
        return "Severe ☠"


# =========================
# HOME PAGE
# =========================

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    status = None

    if request.method == "POST":

        pm25 = float(request.form["pm25"])
        co = float(request.form["co"])
        no2 = float(request.form["no2"])

        data = np.array([[pm25, co, no2]])

        prediction = round(float(model.predict(data)[0]), 2)

        status = get_aqi_status(prediction)

    return render_template(
        "index.html",
        prediction=prediction,
        status=status
    )


# =========================
# REALTIME AQI + WEATHER
# =========================

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

        # =========================
        # OPEN-METEO AIR QUALITY
        # =========================

        air_url = (
            "https://air-quality-api.open-meteo.com/v1/air-quality"
            f"?latitude={lat}"
            f"&longitude={lon}"
            "&current=us_aqi,pm2_5,carbon_monoxide,nitrogen_dioxide"
            "&timezone=auto"
        )

        air_response = requests.get(
            air_url,
            timeout=10
        )

        air_data = air_response.json()

        # =========================
        # OPEN-METEO WEATHER
        # =========================

        weather_url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}"
            f"&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
            "&timezone=auto"
        )

        weather_response = requests.get(
            weather_url,
            timeout=10
        )

        weather_data = weather_response.json()

        # =========================
        # AIR DATA
        # =========================

        current_air = air_data.get("current", {})

        aqi = current_air.get("us_aqi")

        pm25 = current_air.get("pm2_5", 0)

        co = current_air.get("carbon_monoxide", 0)

        no2 = current_air.get("nitrogen_dioxide", 0)

        # =========================
        # WEATHER DATA
        # =========================

        current_weather = weather_data.get("current", {})

        temperature = current_weather.get(
            "temperature_2m",
            0
        )

        humidity = current_weather.get(
            "relative_humidity_2m",
            0
        )

        wind = current_weather.get(
            "wind_speed_10m",
            0
        )

        # =========================
        # CITY NAME
        # =========================

        city = "Selected Location"

        try:

            reverse_url = (
                "https://nominatim.openstreetmap.org/reverse"
                f"?lat={lat}"
                f"&lon={lon}"
                "&format=json"
                "&zoom=10"
            )

            reverse_response = requests.get(
                reverse_url,
                headers={
                    "User-Agent": "Smart-AQI-Dashboard"
                },
                timeout=10
            )

            reverse_data = reverse_response.json()

            address = reverse_data.get(
                "address",
                {}
            )

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

        # =========================
        # CHECK AQI
        # =========================

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

        aqi = round(float(aqi), 0)

        # =========================
        # FUTURE AQI PREDICTION
        # =========================

        predicted_aqi = round(
            aqi * 1.15,
            2
        )

        return jsonify({

            "aqi": aqi,

            "city": city,

            "temp": round(
                float(temperature),
                1
            ),

            "humidity": round(
                float(humidity),
                0
            ),

            "wind": round(
                float(wind),
                1
            ),

            "pm25": round(
                float(pm25),
                2
            ),

            "co": round(
                float(co),
                2
            ),

            "no2": round(
                float(no2),
                2
            ),

            "status": get_aqi_status(aqi),

            "predicted_aqi": predicted_aqi

        })

    except Exception as e:

        print("ERROR:", e)

        return jsonify({

            "aqi": "Error",

            "city": "Location Error",

            "message": str(e)

        })


# =========================
# RUN APP
# =========================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5001
    )
