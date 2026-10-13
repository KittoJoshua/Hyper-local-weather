"""
Evaluation Metrics for Continuous Micro-Climate Regression.
Calculates RMSE, MAE, and R² as stipulated in Capstone Objective (v).
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_regression_metrics(
    y_true: Union[np.ndarray, pd.Series, pd.DataFrame],
    y_pred: Union[np.ndarray, pd.Series, pd.DataFrame],
    target_names: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Computes RMSE, MAE, and R² across all target variables.
    """
    if isinstance(y_true, (pd.DataFrame, pd.Series)):
        y_true = y_true.values
    if isinstance(y_pred, (pd.DataFrame, pd.Series)):
        y_pred = y_pred.values

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    target_names = target_names or [f"target_{i}" for i in range(y_true.shape[1])]
    metrics = {}

    for i, name in enumerate(target_names):
        true_col = y_true[:, i]
        pred_col = y_pred[:, i]

        rmse = float(np.sqrt(mean_squared_error(true_col, pred_col)))
        mae = float(mean_absolute_error(true_col, pred_col))
        r2 = float(r2_score(true_col, pred_col))

        metrics[name] = {
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "R2": round(r2, 4)
        }

    return metrics


def format_metrics_table(results: Dict[str, Dict[str, Dict[str, float]]]) -> pd.DataFrame:
    """
    Flattens a dictionary of model comparison metrics into a clean summary DataFrame
    ideal for display in the admin panel and capstone documentation.
    
    Structure of results:
    {
       "Persistence Baseline": {"temperature_c": {"RMSE": ..., "MAE": ..., "R2": ...}},
       "Rolling Average": {...},
       "Standalone XGBoost": {...},
       "Standalone LSTM": {...},
       "Hybrid (LSTM + XGBoost)": {...}
    }
    """
    records = []
    for model_name, target_dict in results.items():
        for target_name, metric_dict in target_dict.items():
            records.append({
                "Model Architecture": model_name,
                "Target Variable": target_name,
                "RMSE": metric_dict["RMSE"],
                "MAE": metric_dict["MAE"],
                "R² Score": metric_dict["R2"]
            })
    return pd.DataFrame(records)
