"""
Hybrid Residual Stacking Architecture for Hyper-Local Micro-Climate Forecasting.

Overview
--------
This module implements a two-stage ensemble that fuses deep recurrent learning
(LSTM) with gradient-boosted tabular regression (XGBoost) via residual stacking.
The architecture is specifically designed for hyper-local agro-meteorological
time-series forecasting at farm-plot granularity.

Stage 1 — LSTM Temporal Baseline
---------------------------------
A multi-layer, bidirectional-ready PyTorch LSTM captures long-range temporal
dependencies in sequential climate windows (default lookback = 24 hourly steps).
It produces a continuous baseline prediction vector over all TARGET_VARIABLES.

Stage 2 — XGBoost Residual Corrector
--------------------------------------
Training residuals (actual − LSTM prediction) encode the systematic error
patterns the LSTM cannot capture — micro-topographic effects, land-surface
anomalies, and local crop-canopy heat-flux offsets.  XGBoost learns a
non-linear tabular mapping from hand-crafted meteorological features to
these residuals, acting as a spatially-aware bias corrector.

Mathematical Formulation
------------------------
Let X_seq ∈ R^{T×F_seq} be the sequential feature window and
X_tab ∈ R^{F_tab} be the contemporaneous tabular feature vector.

    y_base  = LSTM(X_seq)          # Stage 1: temporal baseline
    e       = y_true − y_base      # Residual error (training only)
    e_pred  = XGBoost(X_tab)       # Stage 2: residual correction
    y_final = y_base + e_pred      # Final calibrated forecast

Domain boundary constraints are applied post-hoc:
    precipitation_mm  → clipped to [0, ∞)
    humidity_percent  → clipped to [10, 100]

Serialisation
-------------
Model weights are persisted in WEIGHTS_DIR:
    <name>_lstm.pt        — PyTorch state-dict (LSTM)
    <name>_xgb_<target>.ubj — XGBoost booster files (one per target variable)

Usage Example
-------------
    forecaster = HybridResidualForecaster()
    forecaster.fit(X_seq_train, y_train, X_tab_train,
                   X_seq_val, y_val, X_tab_val)
    preds = forecaster.predict(X_seq_test, X_tab_test)
    forecaster.save("hybrid_model")
"""
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from ml_engine.models.lstm_model import MicroClimateLSTM
from ml_engine.models.xgboost_model import MicroClimateXGBoost
from ml_engine.config import WEIGHTS_DIR, TARGET_VARIABLES


class HybridResidualForecaster:
    """
    Two-stage ensemble combining Recurrent Deep Learning and Gradient Boosted Trees.

    The hybrid architecture addresses two complementary weaknesses:
    - LSTMs excel at capturing temporal autocorrelation but can be insensitive
      to abrupt local perturbations (wind gusts, soil moisture spikes).
    - XGBoost excels at learning non-linear tabular interactions but cannot
      model sequential dependencies across time steps.

    By training XGBoost exclusively on the LSTM's residual errors, we ensure
    the corrector only models what the LSTM missed — preventing redundant
    capacity and minimising overfitting risk.

    Attributes
    ----------
    target_cols : list[str]
        Climate variables to forecast (e.g. ["temperature_c", "humidity_percent"]).
    lstm_model : MicroClimateLSTM
        Stage-1 recurrent baseline trained on sliding windows.
    xgb_residual_model : MicroClimateXGBoost
        Stage-2 gradient-boosted residual corrector.
    is_fitted : bool
        True once both stages have been fitted via ``fit()``.
    """

    def __init__(
        self,
        lstm_input_size: int = 7,
        target_cols: List[str] = None,
        lstm_params: Dict = None,
        xgb_params: Dict = None
    ):
        self.target_cols = target_cols or TARGET_VARIABLES
        self.lstm_model = MicroClimateLSTM(
            input_size=lstm_input_size,
            output_size=len(self.target_cols),
            params=lstm_params
        )
        self.xgb_residual_model = MicroClimateXGBoost(
            target_cols=self.target_cols,
            params=xgb_params
        )
        self.is_fitted = False

    def fit(
        self,
        X_seq_train: np.ndarray,
        y_train: np.ndarray,
        X_tab_train: pd.DataFrame,
        X_seq_val: np.ndarray = None,
        y_val: np.ndarray = None,
        X_tab_val: pd.DataFrame = None
    ) -> Dict:
        """
        Execute the two-stage stacked training procedure.

        Stage 1 trains the LSTM baseline on sequential windows.
        Stage 2 computes residuals (y_true − y_lstm) on the training set
        and trains XGBoost to map X_tab → residuals.

        Parameters
        ----------
        X_seq_train : np.ndarray, shape (N, T, F_seq)
            Sequential feature windows for LSTM training.
        y_train : np.ndarray, shape (N, n_targets)
            Ground-truth target values aligned with the last step of each window.
        X_tab_train : pd.DataFrame, shape (N, F_tab)
            Tabular features for XGBoost residual training.
        X_seq_val : np.ndarray or None
            Optional validation sequential windows (used for LSTM early stopping).
        y_val : np.ndarray or None
            Optional validation targets.
        X_tab_val : pd.DataFrame or None
            Optional validation tabular features (used for XGBoost eval_set).

        Returns
        -------
        dict
            ``{"lstm_history": ..., "feature_importances": ...}``
        """
        # Step 1: Train LSTM Baseline
        lstm_history = self.lstm_model.fit(
            X_train=X_seq_train,
            y_train=y_train,
            X_val=X_seq_val,
            y_val=y_val
        )

        # Step 2: Generate LSTM predictions on training set
        lstm_preds_train = self.lstm_model.predict(X_seq_train)

        # Step 3: Compute residual error matrix
        # e = actual - predicted_baseline
        residuals_train = y_train - lstm_preds_train
        residual_df_train = pd.DataFrame(residuals_train, columns=self.target_cols)

        # Step 4: Train XGBoost to learn residual error patterns
        eval_set = None
        if X_seq_val is not None and y_val is not None and X_tab_val is not None:
            lstm_preds_val = self.lstm_model.predict(X_seq_val)
            residuals_val = y_val - lstm_preds_val
            eval_set = (X_tab_val.values, pd.DataFrame(residuals_val, columns=self.target_cols))

        self.xgb_residual_model.fit(X=X_tab_train, y=residual_df_train, eval_set=eval_set)
        self.is_fitted = True

        return {
            "lstm_history": lstm_history,
            "feature_importances": self.xgb_residual_model.get_feature_importances()
        }

    def predict(self, X_seq: np.ndarray, X_tab: pd.DataFrame) -> np.ndarray:
        """
        Generate final calibrated micro-climate predictions.

        Applies the full two-stage forward pass:
            y_final = LSTM(X_seq) + XGBoost(X_tab)

        Domain constraints are enforced post-hoc to ensure physically
        plausible output values (e.g. non-negative precipitation).

        Parameters
        ----------
        X_seq : np.ndarray, shape (N, T, F_seq)
            Sequential feature windows (same format as training).
        X_tab : pd.DataFrame, shape (N, F_tab)
            Contemporaneous tabular features for the residual corrector.

        Returns
        -------
        np.ndarray, shape (N, n_targets)
            Calibrated forecast array ordered by ``self.target_cols``.
        """
        # Baseline from LSTM
        lstm_baseline = self.lstm_model.predict(X_seq)

        # Residual correction from XGBoost (residual values can be negative)
        predicted_residuals = self.xgb_residual_model.predict(X_tab, is_residual=True)

        # Additive combination
        final_preds = lstm_baseline + predicted_residuals

        # Domain boundary constraints
        for i, col in enumerate(self.target_cols):
            if col == "precipitation_mm":
                final_preds[:, i] = np.clip(final_preds[:, i], 0.0, None)
            elif col == "humidity_percent":
                final_preds[:, i] = np.clip(final_preds[:, i], 10.0, 100.0)

        return final_preds

    def save(self, name: str = "hybrid_model"):
        """
        Persist both model stages to WEIGHTS_DIR.

        Writes:
            <name>_lstm.pt            — PyTorch state-dict
            <name>_xgb_<target>.ubj   — XGBoost booster (one per target)
        """
        lstm_path = str(WEIGHTS_DIR / f"{name}_lstm.pt")
        xgb_prefix = str(WEIGHTS_DIR / f"{name}_xgb")
        self.lstm_model.save(lstm_path)
        self.xgb_residual_model.save(xgb_prefix)

    def load(self, name: str = "hybrid_model"):
        """
        Load previously saved model weights for both stages.

        Reads:
            <name>_lstm.pt            — PyTorch state-dict
            <name>_xgb_<target>.ubj   — XGBoost booster (one per target)

        Sets ``is_fitted = True`` and returns self for method chaining.
        """
        lstm_path = str(WEIGHTS_DIR / f"{name}_lstm.pt")
        xgb_prefix = str(WEIGHTS_DIR / f"{name}_xgb")
        self.lstm_model.load(lstm_path)
        self.xgb_residual_model.load(xgb_prefix)
        self.is_fitted = True
        return self
