"""
Automated Test Suite for Multi-Model AQI Prediction System.
Tests all 7 models (Linear Regression, Decision Tree, Random Forest,
Gradient Boosting, Extra Trees, SVR, KNN), Best Model Routing,
Prediction Endpoints, Validation, and Model Comparison APIs.
"""

import os
import sys
import unittest
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app, LOADED_MODELS, BEST_MODEL_KEY, MODEL_DISPLAY_NAMES, metadata
from ml.classifier import classify_aqi
from database import init_db, save_prediction, get_recent_predictions

class TestMultiModelPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        init_db()
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_01_all_seven_models_loaded(self):
        """Verify that all 7 regression models are trained and loaded in memory."""
        expected_keys = [
            'linear_regression', 'decision_tree', 'random_forest',
            'gradient_boosting', 'extra_trees', 'svr', 'knn'
        ]
        for k in expected_keys:
            self.assertIn(k, LOADED_MODELS, f"Model '{k}' must be loaded.")
            self.assertIsNotNone(LOADED_MODELS[k], f"Model '{k}' must not be None.")

    def test_02_best_model_metadata_valid(self):
        """Verify that the best model was dynamically selected and stored in metadata."""
        self.assertIsNotNone(BEST_MODEL_KEY)
        self.assertIn(BEST_MODEL_KEY, LOADED_MODELS)
        self.assertIn("models", metadata)
        self.assertEqual(len(metadata["models"]), 7, "Metadata must contain all 7 models.")
        best_info = metadata["models"][BEST_MODEL_KEY]
        self.assertIn("test_metrics", best_info)
        self.assertIn("RMSE", best_info["test_metrics"])
        self.assertIn("R2_Score", best_info["test_metrics"])

    def test_03_predict_each_individual_model(self):
        """Verify that prediction works for every single model individually."""
        sample_payload = {
            "pm25": 65.0,
            "co": 1.4,
            "no2": 38.0,
            "temperature": 27.0,
            "humidity": 60.0,
            "wind_speed": 4.5,
            "location": "Sensor Station"
        }

        for model_key in LOADED_MODELS.keys():
            payload = dict(sample_payload)
            payload["model"] = model_key
            res = self.client.post("/predict", json=payload)
            self.assertEqual(res.status_code, 200, f"Predict failed for model: {model_key}")
            data = res.get_json()
            self.assertTrue(data["success"])
            self.assertIsInstance(data["predicted_aqi"], float)
            self.assertGreater(data["predicted_aqi"], 0)
            self.assertIn("category", data)
            self.assertIn("recommendation", data)
            self.assertEqual(data["model_key"], model_key)

    def test_04_predict_best_model_routing(self):
        """Verify that 'best_model' automatically routes to the winning algorithm."""
        payload = {
            "model": "best_model",
            "pm25": 70.0,
            "co": 1.5,
            "no2": 40.0,
            "temperature": 25.0,
            "humidity": 55.0,
            "wind_speed": 5.0
        }
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["model_key"], BEST_MODEL_KEY)
        self.assertTrue(data["is_best_model"])

    def test_05_predict_best_dedicated_endpoint(self):
        """Verify POST /predict-best invokes the best model directly."""
        payload = {
            "pm25": 50.0,
            "co": 1.0,
            "no2": 30.0,
            "temperature": 24.0,
            "humidity": 50.0,
            "wind_speed": 6.0
        }
        res = self.client.post("/predict-best", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["model_key"], BEST_MODEL_KEY)

    def test_06_model_comparison_api(self):
        """Verify GET /api/model-comparison returns all 7 models' benchmarks."""
        res = self.client.get("/api/model-comparison")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["best_model"], BEST_MODEL_KEY)
        self.assertIn("metadata", data)
        self.assertEqual(len(data["metadata"]["models"]), 7)

    def test_07_validation_rejects_negative_pollutants(self):
        """Verify that negative concentrations are strictly rejected with HTTP 400."""
        payload = {
            "pm25": -15.0,
            "co": 1.2,
            "no2": 30.0
        }
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertFalse(data["success"])
        self.assertIn("cannot be negative", data["error"])

    def test_08_validation_rejects_invalid_humidity(self):
        """Verify humidity outside 0-100% is rejected."""
        payload = {
            "pm25": 40.0,
            "co": 1.0,
            "no2": 25.0,
            "humidity": 150.0
        }
        res = self.client.post("/predict", json=payload)
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertFalse(data["success"])
        self.assertIn("Humidity must be between 0% and 100%", data["error"])

    def test_09_database_logs_model_used(self):
        """Verify SQLite prediction history stores the specific model name."""
        res = self.client.post("/predict", json={
            "model": "extra_trees",
            "pm25": 80.0,
            "co": 1.6,
            "no2": 45.0,
            "location": "Airport Terminal"
        })
        self.assertEqual(res.status_code, 200)
        history = get_recent_predictions(limit=1)
        self.assertTrue(len(history) > 0)
        self.assertEqual(history[0]["location"], "Airport Terminal")
        self.assertEqual(history[0]["model_used"], "Extra Trees Regressor")

if __name__ == '__main__':
    unittest.main()
