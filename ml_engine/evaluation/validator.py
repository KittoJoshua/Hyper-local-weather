"""
Time-Series Walk-Forward (Rolling Origin) Validation Strategy.
Enforces strict chronological sequence partitioning without look-ahead bias,
benchmarking Machine Learning models against statistical baselines.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Generator
from ml_engine.evaluation.metrics import calculate_regression_metrics
from ml_engine.models.baseline_models import PersistenceModel, RollingAverageModel


class TimeSeriesWalkForwardValidator:
    """
    Generates expanding or rolling window chronological splits for backtesting
    meteorological time-series models.
    """

    def __init__(self, n_splits: int = 4, test_size_ratio: float = 0.15, min_train_ratio: float = 0.50):
        self.n_splits = n_splits
        self.test_size_ratio = test_size_ratio
        self.min_train_ratio = min_train_ratio

    def split(self, df: pd.DataFrame) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Yields (train_df, test_df) pairs along a strict forward chronological axis.
        """
        n = len(df)
        total_test_samples = int(n * self.test_size_ratio)
        test_step_size = total_test_samples // self.n_splits
        min_train_samples = int(n * self.min_train_ratio)

        for i in range(self.n_splits):
            train_end_idx = min_train_samples + (i * test_step_size)
            test_end_idx = train_end_idx + test_step_size

            if test_end_idx > n:
                test_end_idx = n

            train_df = df.iloc[:train_end_idx]
            test_df = df.iloc[train_end_idx:test_end_idx]

            yield train_df, test_df

    def evaluate_baselines(
        self,
        df: pd.DataFrame,
        target_cols: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluates Persistence and Rolling Average baselines across walk-forward splits.
        """
        persistence_scores = {col: {"RMSE": [], "MAE": [], "R2": []} for col in target_cols}
        rolling_avg_scores = {col: {"RMSE": [], "MAE": [], "R2": []} for col in target_cols}

        for train_df, test_df in self.split(df):
            y_test = test_df[target_cols].values
            y_train = train_df[target_cols].values

            # 1. Persistence
            pers_model = PersistenceModel()
            pers_model.fit(None, y_train)
            pers_pred = pers_model.predict(y_train[-1], steps=len(y_test))
            p_metrics = calculate_regression_metrics(y_test, pers_pred, target_cols)
            for col in target_cols:
                for k in ["RMSE", "MAE", "R2"]:
                    persistence_scores[col][k].append(p_metrics[col][k])

            # 2. Rolling Average (24h)
            roll_model = RollingAverageModel(window_size=24)
            roll_model.fit(None, y_train)
            roll_pred = roll_model.predict(y_train, steps=len(y_test))
            r_metrics = calculate_regression_metrics(y_test, roll_pred, target_cols)
            for col in target_cols:
                for k in ["RMSE", "MAE", "R2"]:
                    rolling_avg_scores[col][k].append(r_metrics[col][k])

        # Aggregate means across folds
        def _avg_scores(score_dict):
            return {
                col: {k: round(float(np.mean(vals)), 4) for k, vals in metrics.items()}
                for col, metrics in score_dict.items()
            }

        return {
            "Persistence Baseline": _avg_scores(persistence_scores),
            "Historical Rolling Average (24h)": _avg_scores(rolling_avg_scores)
        }
