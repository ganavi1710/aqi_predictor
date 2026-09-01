"""
Comprehensive Test Suite for Smart AQI System.
Tests Model Loading, Preprocessing, Decision Tree Inference,
AQI Classification, Recommendations, Flask /predict API, Validation, and SQLite History.
"""

import os
import sys
import unittest
import json
import numpy as np

# Ensure root directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app, model, metadata, extract_and_validate_features
from ml.classifier import classify_aqi
from database import init_db, save_prediction, get_recent_predictions

class TestMLPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_01_model_and_metadata_loaded(self):
        """Verify Decision Tree model and metadata exist and are valid."""
        self.assertIsNotNone(model, "Decision Tree model should be loaded successfully.")
        self.assertIn("features", metadata, "Metadata must contain 'features' list.")
        self.assertIn("evaluation_metrics", metadata, "Metadata must contain 'evaluation_metrics'.")
        self.assertEqual(metadata.get("model_name"), "DecisionTreeRegressor")

    def test_02_aqi_classification_boundaries(self):
        """Verify continuous AQI values map to correct EPA categories and badges."""
        test_cases = [
            (25.0, "Good"),
            (50.0, "Good"),
            (51.0, "Moderate"),
            (100.0, "Moderate"),
            (101.0, "Unhealthy for Sensitive Groups"),
            (150.0, "Unhealthy for Sensitive Groups"),
            (151.0, "Unhealthy"),
            (200.0, "Unhealthy"),
            (201.0, "Very Unhealthy"),
            (300.0, "Very Unhealthy"),
            (350.0, "Hazardous"),
            (500.0, "Hazardous")
        ]
        for aqi_val, expected_cat in test_cases:
            res = classify_aqi(aqi_val)
            self.assertEqual(res["category"], expected_cat, f"Failed for AQI {aqi_val}")
            self.assertTrue(len(res["recommendation"]) > 10)
            self.assertTrue(len(res["tips"]) >= 3)

    def test_03_extract_and_validate_valid_input(self):
        """Verify valid payload is parsed into ordered feature matrix."""
        payload = {
            "pm25": 45.5,
            "co": 1.1,
            "no2": 32.0,
            "temperature": 28.5,
            "humidity": 60.0,
            "wind_speed": 4.2
        }
        matrix, parsed, errors = extract_and_validate_features(payload)
        self.assertEqual(len(errors), 0)
        self.assertIsNotNone(matrix)
        self.assertEqual(matrix.shape, (1, 6))
        self.assertEqual(parsed["PM2.5"], 45.5)

    def test_04_extract_and_validate_negative_input(self):
        """Verify negative pollutant values are strictly rejected."""
        payload = {
            "pm25": -10.0,
            "co": 1.1,
            "no2": 32.0,
            "temperature": 25.0,
            "humidity": 50.0,
            "wind_speed": 5.0
        }
        matrix, parsed, errors = extract_and_validate_features(payload)
        self.assertTrue(len(errors) > 0, "Negative PM2.5 must produce validation error.")
        self.assertTrue(any("cannot be negative" in e for e in errors))

    def test_05_extract_and_validate_invalid_numeric(self):
        """Verify non-numeric values are caught and reported."""
        payload = {
            "pm25": "abc",
            "co": 1.1,
            "no2": 32.0
        }
        matrix, parsed, errors = extract_and_validate_features(payload)
        self.assertTrue(len(errors) > 0)
        self.assertTrue(any("must be a valid numeric" in e for e in errors))

    def test_06_predict_endpoint_success(self):
        """Verify POST /predict returns expected JSON structure."""
        payload = {
            "pm25": 85.0,
            "co": 1.8,
            "no2": 45.0,
            "temperature": 29.0,
            "humidity": 68.0,
            "wind_speed": 3.5,
            "location": "Central Station"
        }
        response = self.client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn("predicted_aqi", data)
        self.assertIn("category", data)
        self.assertIn("recommendation", data)
        self.assertIn("model", data)
        self.assertEqual(data["model"], "DecisionTreeRegressor")
        self.assertIsInstance(data["predicted_aqi"], float)

    def test_07_predict_endpoint_invalid_payload(self):
        """Verify POST /predict with invalid data returns 400 and user-friendly error."""
        response = self.client.post(
            "/predict",
            data=json.dumps({"pm25": -50.0}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("error", data)

    def test_08_model_info_endpoint(self):
        """Verify GET /api/model-info returns metadata and evaluation metrics."""
        response = self.client.get("/api/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn("metadata", data)
        meta = data["metadata"]
        self.assertIn("evaluation_metrics", meta)
        self.assertIn("R2_Score", meta["evaluation_metrics"])
        self.assertIn("feature_importances", meta)

    def test_09_database_history(self):
        """Verify prediction records are written to SQLite and can be retrieved."""
        rec_id = save_prediction(
            location="Test Park",
            pm25=22.0,
            co=0.6,
            no2=15.0,
            temperature=21.0,
            humidity=50.0,
            wind_speed=8.0,
            predicted_aqi=42.0,
            category="Good",
            recommendation="Air quality is satisfactory."
        )
        self.assertGreater(rec_id, 0)
        history = get_recent_predictions(limit=5)
        self.assertTrue(len(history) > 0)
        self.assertEqual(history[0]["location"], "Test Park")
        self.assertEqual(history[0]["predicted_aqi"], 42.0)

        # Also test API endpoint
        api_res = self.client.get("/api/history")
        self.assertEqual(api_res.status_code, 200)
        self.assertTrue(api_res.get_json()["count"] > 0)

if __name__ == '__main__':
    unittest.main()
