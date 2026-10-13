"""
XGBoost Regression Model for Micro-Climate Weather Variables.
Handles multi-variable prediction (temperature, humidity, precipitation)
using gradient-boosted decision trees with optimized objective functions.
"""
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import xgboost as xgb
from ml_engine.config import XGBOOST_PARAMS, WEIGHTS_DIR, TARGET_VARIABLES


class MicroClimateXGBoost:
    """
    Gradient-boosted decision tree regressor for tabular meteorological features.
    Maintains separate specialized regressors for temperature, humidity, and rainfall
    to accommodate their distinct statistical distributions (e.g. Tweedie for zero-inflated precipitation).
    """

    def __init__(self, target_cols: List[str] = None, params: Dict = None):
        self.target_cols = target_cols or TARGET_VARIABLES
        self.params = params or XGBOOST_PARAMS.copy()
        self.models: Dict[str, xgb.XGBRegressor] = {}
        self.feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, eval_set=None):
        """
        Fits an XGBoost regressor for each target column.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_mat = X.values
        else:
            X_mat = X

        for col in self.target_cols:
            y_col = y[col].values if isinstance(y, pd.DataFrame) else y

            # Customize objective function per variable
            col_params = self.params.copy()
            # Tweedie deviance requires non-negative labels (>= 0).
            # For residual errors in hybrid models, negative residuals occur, so use squarederror.
            if col == "precipitation_mm" and np.all(y_col >= 0):
                col_params["objective"] = "reg:tweedie"
                col_params["tweedie_variance_power"] = 1.3
            else:
                col_params["objective"] = "reg:squarederror"

            model = xgb.XGBRegressor(**col_params)

            if eval_set:
                X_val, y_val = eval_set
                eval_target = y_val[col].values if isinstance(y_val, pd.DataFrame) else y_val
                model.fit(X_mat, y_col, eval_set=[(X_val, eval_target)], verbose=False)
            else:
                model.fit(X_mat, y_col, verbose=False)

            self.models[col] = model

        return self

    def predict(self, X: pd.DataFrame, is_residual: bool = False) -> np.ndarray:
        """
        Predicts target variables for input features X.
        Returns 2D array of shape (n_samples, n_targets).
        """
        X_mat = X.values if isinstance(X, pd.DataFrame) else X
        preds = []

        for col in self.target_cols:
            if col in self.models:
                pred_col = self.models[col].predict(X_mat)
                if not is_residual:
                    if col == "precipitation_mm":
                        pred_col = np.clip(pred_col, 0.0, None)
                    elif col == "humidity_percent":
                        pred_col = np.clip(pred_col, 10.0, 100.0)
                preds.append(pred_col)
            else:
                raise ValueError(f"Model for {col} has not been fitted.")

        return np.column_stack(preds)

    def get_feature_importances(self) -> Dict[str, Dict[str, float]]:
        """Returns feature importance ranking per target."""
        importances = {}
        for col, model in self.models.items():
            if self.feature_names and hasattr(model, "feature_importances_"):
                imp_dict = dict(zip(self.feature_names, model.feature_importances_.tolist()))
                # Sort descending
                importances[col] = dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True))
        return importances

    def save(self, filepath_prefix: str = None):
        """Persists trained XGBoost models to disk."""
        prefix = filepath_prefix or str(WEIGHTS_DIR / "xgboost_model")
        for col, model in self.models.items():
            path = f"{prefix}_{col}.json"
            model.save_model(path)
        # Save metadata
        meta_path = f"{prefix}_metadata.joblib"
        joblib.dump({"target_cols": self.target_cols, "feature_names": self.feature_names}, meta_path)

    def load(self, filepath_prefix: str = None):
        """Loads trained XGBoost models from disk."""
        prefix = filepath_prefix or str(WEIGHTS_DIR / "xgboost_model")
        meta_path = f"{prefix}_metadata.joblib"
        if os.path.exists(meta_path):
            meta = joblib.load(meta_path)
            self.target_cols = meta["target_cols"]
            self.feature_names = meta["feature_names"]

        for col in self.target_cols:
            path = f"{prefix}_{col}.json"
            if os.path.exists(path):
                model = xgb.XGBRegressor()
                model.load_model(path)
                self.models[col] = model
        return self
