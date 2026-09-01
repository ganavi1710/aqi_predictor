"""
SQLite Persistent Storage for AQI Prediction History.
Provides lightweight, reliable logging of prediction requests and environmental observations.
"""

import os
import sqlite3
import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database.db')

def get_db_connection():
    """Returns a SQLite connection with dict-like row access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database tables if they do not already exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            location TEXT NOT NULL,
            pm25 REAL,
            co REAL,
            no2 REAL,
            temperature REAL,
            humidity REAL,
            wind_speed REAL,
            predicted_aqi REAL NOT NULL,
            category TEXT NOT NULL,
            recommendation TEXT NOT NULL,
            model_used TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_prediction(
    location: str,
    pm25: float,
    co: float,
    no2: float,
    temperature: float,
    humidity: float,
    wind_speed: float,
    predicted_aqi: float,
    category: str,
    recommendation: str,
    model_used: str = "DecisionTreeRegressor"
) -> int:
    """Saves a new prediction record into SQLite database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO predictions (
            timestamp, location, pm25, co, no2,
            temperature, humidity, wind_speed,
            predicted_aqi, category, recommendation, model_used
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp, location, pm25, co, no2,
        temperature, humidity, wind_speed,
        predicted_aqi, category, recommendation, model_used
    ))
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id

def get_recent_predictions(limit: int = 10) -> list:
    """Fetches the latest prediction records ordered by timestamp descending."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM predictions
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        results.append({
            "id": row["id"],
            "timestamp": row["timestamp"],
            "location": row["location"],
            "pm25": row["pm25"],
            "co": row["co"],
            "no2": row["no2"],
            "temperature": row["temperature"],
            "humidity": row["humidity"],
            "wind_speed": row["wind_speed"],
            "predicted_aqi": row["predicted_aqi"],
            "category": row["category"],
            "recommendation": row["recommendation"],
            "model_used": row["model_used"]
        })
    return results
