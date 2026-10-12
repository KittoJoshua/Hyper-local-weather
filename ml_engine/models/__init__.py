"""
Predictive models subpackage: Baselines, LSTM, XGBoost, and Hybrid Stacking.
"""
from ml_engine.models.baseline_models import PersistenceModel, RollingAverageModel
from ml_engine.models.xgboost_model import MicroClimateXGBoost
from ml_engine.models.lstm_model import MicroClimateLSTM
from ml_engine.models.hybrid_stack import HybridResidualForecaster

__all__ = [
    "PersistenceModel",
    "RollingAverageModel",
    "MicroClimateXGBoost",
    "MicroClimateLSTM",
    "HybridResidualForecaster"
]
