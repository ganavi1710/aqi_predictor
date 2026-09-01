"""
AQI Severity Classification and Health Recommendations Module.
Decouples numerical regression output from domain-specific categorization and advice.
Thresholds are based on EPA standards and are centrally configurable.
"""

AQI_CATEGORIES = [
    {
        "min": 0,
        "max": 50,
        "category": "Good",
        "badge": "Good 😊",
        "color": "#00ff88",
        "recommendation": "Air quality is satisfactory. Normal outdoor activities are suitable.",
        "tips": [
            "Ideal conditions for outdoor exercise and recreation",
            "Keep home and office windows open for ventilation",
            "Enjoy outdoor walks without health restrictions"
        ]
    },
    {
        "min": 51,
        "max": 100,
        "category": "Moderate",
        "badge": "Moderate 😐",
        "color": "#ffee00",
        "recommendation": "Air quality is acceptable, but unusually sensitive individuals should consider reducing prolonged outdoor exposure.",
        "tips": [
            "Unusually sensitive individuals should take breaks during outdoor exertion",
            "Stay well hydrated throughout the day",
            "Monitor local air quality if you have asthma or allergies"
        ]
    },
    {
        "min": 101,
        "max": 150,
        "category": "Unhealthy for Sensitive Groups",
        "badge": "Unhealthy for Sensitive Groups 😷",
        "color": "#ff9900",
        "recommendation": "Sensitive individuals should reduce prolonged outdoor activity.",
        "tips": [
            "Children, elderly, and individuals with respiratory issues should limit time outside",
            "Wear an N95 mask during prolonged exposure",
            "Keep indoor air filtered and clean"
        ]
    },
    {
        "min": 151,
        "max": 200,
        "category": "Unhealthy",
        "badge": "Unhealthy 🚨",
        "color": "#ff4d6d",
        "recommendation": "Everyone should consider reducing prolonged outdoor activity.",
        "tips": [
            "Avoid strenuous outdoor activities",
            "Wear a protective mask if traveling outdoors",
            "Run indoor HEPA air purifiers and keep windows closed"
        ]
    },
    {
        "min": 201,
        "max": 300,
        "category": "Very Unhealthy",
        "badge": "Very Unhealthy 🛑",
        "color": "#9d4edd",
        "recommendation": "Avoid prolonged outdoor exposure and consider staying indoors.",
        "tips": [
            "Remain indoors as much as possible",
            "Seal doors and windows to minimize infiltration of polluted air",
            "Seek medical attention if experiencing breathing difficulty"
        ]
    },
    {
        "min": 301,
        "max": 1000,
        "category": "Hazardous",
        "badge": "Hazardous ☠️",
        "color": "#7b2cbf",
        "recommendation": "Avoid outdoor exposure and follow local health advisories.",
        "tips": [
            "Emergency condition: Avoid all outdoor physical activity",
            "Keep air purifiers running continuously in closed rooms",
            "Follow all local civil and health emergency guidelines"
        ]
    }
]

def classify_aqi(aqi: float) -> dict:
    """
    Classifies continuous numerical AQI into its corresponding severity category
    and generates appropriate health advice.
    """
    try:
        aqi_val = float(aqi)
    except (ValueError, TypeError):
        aqi_val = 0.0

    aqi_rounded = round(aqi_val, 1)

    for item in AQI_CATEGORIES:
        if item["min"] <= aqi_rounded <= item["max"]:
            return {
                "aqi": aqi_rounded,
                "category": item["category"],
                "badge": item["badge"],
                "color": item["color"],
                "recommendation": item["recommendation"],
                "tips": item["tips"]
            }

    # Fallback for extreme values above 300
    highest = AQI_CATEGORIES[-1]
    return {
        "aqi": aqi_rounded,
        "category": highest["category"],
        "badge": highest["badge"],
        "color": highest["color"],
        "recommendation": highest["recommendation"],
        "tips": highest["tips"]
    }
