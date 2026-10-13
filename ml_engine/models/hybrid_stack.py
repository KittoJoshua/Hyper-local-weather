"""
Hybrid Residual Stacking Architecture.
Combines LSTM (Long-Term Temporal Sequence Baseline) with
XGBoost (Localized Tabular Residual Error Corrector).

Mathematical formulation:
1. Baseline forecast: y_base = LSTM(X_seq)
2. Residual error: e = y - y_base
3. Residual prediction: e_pred = XGBoost(X_tabular)
4. Final calibrated prediction: y_final = y_base + e_pred
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
        Fits LSTM on sequential data, computes residuals on training set,
        and trains XGBoost to predict the residual corrections.
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
        Generates final calibrated micro-climate predictions.
        Combines LSTM baseline + XGBoost residual correction.
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
        """Saves both LSTM and XGBoost components."""
        lstm_path = str(WEIGHTS_DIR / f"{name}_lstm.pt")
        xgb_prefix = str(WEIGHTS_DIR / f"{name}_xgb")
        self.lstm_model.save(lstm_path)
        self.xgb_residual_model.save(xgb_prefix)

    def load(self, name: str = "hybrid_model"):
        """Loads both LSTM and XGBoost components."""
        lstm_path = str(WEIGHTS_DIR / f"{name}_lstm.pt")
        xgb_prefix = str(WEIGHTS_DIR / f"{name}_xgb")
        self.lstm_model.load(lstm_path)
        self.xgb_residual_model.load(xgb_prefix)
        self.is_fitted = True
        return self
