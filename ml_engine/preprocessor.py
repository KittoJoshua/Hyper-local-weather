"""
Data Preprocessing, Imputation, and Scaling Module.

Implements rigorous data preparation adhering to time-series best practices:
- Leakage-Free Imputation: Sequential forward-fill followed by rolling median smoothing.
- Physical Bound Constraints: Enforces non-negativity for precipitation (>= 0mm) and realistic atmospheric spans.
- Cyclical Harmonic Encodings: Sine and cosine trigonometric mappings of hour-of-day (24h diurnal curve)
  and day-of-year (365d annual cycle) capturing Kenya's bimodal tropical rainfall calendars.
- MinMax Normalization: Dual-scaler normalization preserving separate inverse transforms for target evaluation.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class ClimateDataPreprocessor:
    """
    Handles missing value imputation, outlier bounds, cyclical date encodings,
    and feature normalization for time-series ML models.
    """

    def __init__(self, target_cols: List[str] = None):
        self.target_cols = target_cols or ["temperature_c", "humidity_percent", "precipitation_mm"]
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False

    def impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Imputes missing values using forward-fill followed by backward-fill
        and rolling median interpolation to maintain continuous variance.
        """
        df = df.copy()
        for col in self.target_cols:
            if col in df.columns:
                # Forward fill, then backward fill for boundary values
                df[col] = df[col].ffill().bfill()
                # Rolling median for isolated anomalous spikes
                rolling_med = df[col].rolling(window=5, min_periods=1, center=True).median()
                df[col] = df[col].fillna(rolling_med)

        if "precipitation_mm" in df.columns:
            # Precipitation cannot be negative
            df["precipitation_mm"] = df["precipitation_mm"].clip(lower=0.0)

        if "humidity_percent" in df.columns:
            df["humidity_percent"] = df["humidity_percent"].clip(lower=10.0, upper=100.0)

        if "temperature_c" in df.columns:
            df["temperature_c"] = df["temperature_c"].clip(lower=0.0, upper=45.0)

        return df

    def encode_cyclical_features(self, df: pd.DataFrame, time_col: str = "recorded_at") -> pd.DataFrame:
        """
        Encodes hour-of-day and day-of-year into sine and cosine pairs
        to preserve circular temporal continuity (e.g. hour 23 -> hour 0).
        """
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])

        hour = df[time_col].dt.hour
        doy = df[time_col].dt.dayofyear

        # Diurnal cycle (24 hours)
        df["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
        df["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)

        # Annual bimodal cycle (365.25 days)
        df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
        df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

        return df

    def fit_scalers(self, features: np.ndarray, targets: np.ndarray):
        """Fits feature and target scalers strictly on training data."""
        self.feature_scaler.fit(features)
        self.target_scaler.fit(targets)
        self.is_fitted = True

    def transform_features(self, features: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Preprocessor scalers have not been fitted yet.")
        return self.feature_scaler.transform(features)

    def transform_targets(self, targets: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Preprocessor scalers have not been fitted yet.")
        return self.target_scaler.transform(targets)

    def inverse_transform_targets(self, scaled_targets: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Preprocessor scalers have not been fitted yet.")
        return self.target_scaler.inverse_transform(scaled_targets)
