"""
Data Preprocessing Pipeline for Smart Air Quality Monitoring System.
Handles missing values, duplicate removal, invalid value filtering,
type casting, feature selection, and train/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DEFAULT_FEATURES = ['PM2.5', 'CO', 'NO2', 'Temperature', 'Humidity', 'Wind_Speed']
TARGET_COLUMN = 'AQI'

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw dataset by removing duplicates, coercing types,
    and filtering out physically invalid or corrupt records.
    """
    initial_count = len(df)
    df = df.copy()

    # 1. Remove exact duplicate rows
    df = df.drop_duplicates()
    dedup_count = initial_count - len(df)
    if dedup_count > 0:
        print(f"[-] Removed {dedup_count} duplicate row(s).")

    # 2. Enforce numeric conversion (coercing non-numeric to NaN)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. Handle invalid physical values:
    # Concentrations cannot be negative; Humidity is [0, 100]; Wind >= 0
    pollutant_cols = [c for c in ['PM2.5', 'CO', 'NO2', 'Wind_Speed', 'AQI'] if c in df.columns]
    for col in pollutant_cols:
        invalid_mask = df[col] < 0
        if invalid_mask.any():
            print(f"[-] Filtered {invalid_mask.sum()} negative values in {col}.")
            df.loc[invalid_mask, col] = np.nan

    if 'Humidity' in df.columns:
        invalid_hum = (df['Humidity'] < 0) | (df['Humidity'] > 100)
        if invalid_hum.any():
            df.loc[invalid_hum, 'Humidity'] = np.nan

    return df

def impute_missing_values(df: pd.DataFrame, strategy='median') -> pd.DataFrame:
    """
    Imputes missing values in continuous environmental measurements.
    Uses median by default to avoid skew from extreme outlier spikes.
    """
    df = df.copy()
    null_counts = df.isnull().sum()
    if null_counts.any():
        print("[-] Handling missing values:")
        for col, count in null_counts.items():
            if count > 0:
                fill_val = df[col].median() if strategy == 'median' else df[col].mean()
                df[col] = df[col].fillna(fill_val)
                print(f"    • {col}: imputed {count} missing value(s) with {strategy} ({fill_val:.2f})")
    return df

def prepare_features(df: pd.DataFrame, target_col: str = TARGET_COLUMN, feature_cols: list = None):
    """
    Selects valid features and target. Enforces that target (AQI) is NEVER an input feature.
    Gracefully handles presence or absence of Wind_Speed.
    """
    if feature_cols is None:
        # Detect available features in df matching DEFAULT_FEATURES
        feature_cols = [col for col in DEFAULT_FEATURES if col in df.columns and col != target_col]

    # Strict safety check: AQI must NEVER be in feature_cols
    feature_cols = [col for col in feature_cols if col.upper() != target_col.upper()]

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    return X, y, feature_cols

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
    Splits dataset into 80% train and 20% test partitions.
    Features are NOT scaled since Decision Trees are invariant to monotonic scale.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"[+] Dataset split: {len(X_train)} train samples ({(1-test_size)*100:.0f}%), "
          f"{len(X_test)} test samples ({test_size*100:.0f}%).")
    return X_train, X_test, y_train, y_test
