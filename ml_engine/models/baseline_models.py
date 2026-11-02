"""
Statistical Baseline Models for Micro-Climate Forecasting Benchmarking.

Implements standard benchmark models fulfilling Strathmore Capstone Specific Objective (v):
- Naive Persistence Baseline: Assumes atmospheric conditions remain static over horizon h:
  y_{t+h} = y_t
- Historical Rolling Average Baseline: Calculates moving average over trailing window W:
  \hat{y}_{t+h} = \frac{1}{W} \sum_{i=0}^{W-1} y_{t-i}
Establishes minimum performance baselines (RMSE, MAE, R²) against which advanced LSTM,
XGBoost, and Hybrid architectures are benchmarked during walk-forward validation.
"""
import numpy as np


class PersistenceModel:
    """
    Naive Persistence Baseline Forecaster.
    Projects the most recent observed meteorological vector forward across all forecast horizons.
    Serves as the primary null hypothesis benchmark in time-series evaluation.
    """

    def __init__(self):
        self.last_observed = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """No parameters to train for persistence baseline."""
        if len(y) > 0:
            self.last_observed = y[-1]
        return self

    def predict(self, X_last: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Given the most recent observed target vector X_last,
        projects it forward across `steps` forecast horizons.
        """
        if X_last is None and self.last_observed is not None:
            X_last = self.last_observed

        if X_last.ndim == 1:
            return np.tile(X_last, (steps, 1))
        elif X_last.ndim == 2:
            return np.repeat(X_last[-1:], steps, axis=0)
        return np.full((steps, 1), X_last)


class RollingAverageModel:
    """
    Historical Rolling Average Baseline: Predicts future values as the mean
    of the most recent W historical observations (e.g. 24h, 72h).
    """

    def __init__(self, window_size: int = 24):
        self.window_size = window_size
        self.mean_values = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Computes mean across target array or trailing window."""
        if len(y) >= self.window_size:
            self.mean_values = np.mean(y[-self.window_size:], axis=0)
        else:
            self.mean_values = np.mean(y, axis=0)
        return self

    def predict(self, historical_seq: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Given a historical sequence of length >= window_size,
        computes the window average and projects it forward.
        """
        if historical_seq is not None and len(historical_seq) > 0:
            window = historical_seq[-self.window_size:]
            avg = np.mean(window, axis=0)
        elif self.mean_values is not None:
            avg = self.mean_values
        else:
            raise ValueError("No historical sequence or fitted mean available.")

        return np.tile(avg, (steps, 1))
