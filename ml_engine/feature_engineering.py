"""
Feature Engineering for Time-Series Regression & Sequential Deep Learning.
Generates lagged features, rolling statistical windows, topographical covariates,
and 3D sequential sliding windows for LSTM networks.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from ml_engine.config import LAG_STEPS, ROLLING_WINDOWS, TARGET_VARIABLES, SEQUENCE_LENGTH


def build_tabular_features(
    df: pd.DataFrame,
    target_cols: List[str] = None,
    lag_steps: List[int] = None,
    rolling_windows: List[int] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Constructs rich tabular features for gradient-boosted decision trees (XGBoost).
    Includes lags, rolling averages, rolling standard deviations, and topographical interactions.
    Strictly avoids look-ahead data leakage by shifting all window aggregations by >= 1 step.
    """
    target_cols = target_cols or TARGET_VARIABLES
    lag_steps = lag_steps or LAG_STEPS
    rolling_windows = rolling_windows or ROLLING_WINDOWS

    df_feat = df.copy()
    feature_cols = []

    # 1. Base meteorological & topographical features
    base_cols = ["latitude", "longitude", "elevation_m", "sin_hour", "cos_hour", "sin_doy", "cos_doy"]
    for col in ["pressure_hpa", "wind_speed_ms"]:
        if col in df_feat.columns:
            base_cols.append(col)

    feature_cols.extend([c for c in base_cols if c in df_feat.columns])

    # Topographical lapse rate proxy (cooler at high altitudes)
    if "elevation_m" in df_feat.columns:
        df_feat["lapse_rate_offset"] = -(df_feat["elevation_m"] / 1000.0) * 6.5
        feature_cols.append("lapse_rate_offset")

    # 2. Lagged features (t - k)
    for col in target_cols:
        for lag in lag_steps:
            col_name = f"{col}_lag_{lag}"
            df_feat[col_name] = df_feat[col].shift(lag)
            feature_cols.append(col_name)

    # 3. Rolling window statistics (shifted by 1 to prevent current-step leakage)
    for col in target_cols:
        for window in rolling_windows:
            mean_col = f"{col}_rolling_mean_{window}"
            std_col = f"{col}_rolling_std_{window}"
            max_col = f"{col}_rolling_max_{window}"

            shifted_series = df_feat[col].shift(1)
            df_feat[mean_col] = shifted_series.rolling(window=window, min_periods=1).mean()
            df_feat[std_col] = shifted_series.rolling(window=window, min_periods=1).std().fillna(0.0)
            df_feat[max_col] = shifted_series.rolling(window=window, min_periods=1).max()

            feature_cols.extend([mean_col, std_col, max_col])

    # Drop rows with NaN resulting from initial lag shifts
    max_lag = max(lag_steps)
    df_feat = df_feat.iloc[max_lag:].reset_index(drop=True)

    return df_feat, feature_cols


def create_sequential_windows(
    data: np.ndarray,
    target: np.ndarray,
    sequence_length: int = SEQUENCE_LENGTH,
    forecast_horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates 3D input tensors (samples, sequence_length, features) and corresponding
    multi-step targets for LSTM sequence models.
    """
    num_samples = len(data) - sequence_length - forecast_horizon + 1
    if num_samples <= 0:
        raise ValueError(
            f"Dataset length ({len(data)}) is too short for sequence_length={sequence_length} "
            f"and horizon={forecast_horizon}."
        )

    X, y = [], []
    for i in range(num_samples):
        X.append(data[i : i + sequence_length])
        # Target at t + sequence_length + forecast_horizon - 1
        y.append(target[i + sequence_length + forecast_horizon - 1])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
