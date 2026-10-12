"""
Unit and Integration Tests for Machine Learning Components.
Validates Preprocessor, Tabular Feature Engineering, Baselines, and Walk-Forward Splits.
"""
from django.test import TestCase
import numpy as np
import pandas as pd
from ml_engine.preprocessor import ClimateDataPreprocessor
from ml_engine.feature_engineering import build_tabular_features
from ml_engine.models.baseline_models import PersistenceModel, RollingAverageModel
from ml_engine.evaluation.validator import TimeSeriesWalkForwardValidator
from ml_engine.data_loader import generate_synthetic_historical_dataset


class MachineLearningPipelineTestCase(TestCase):
    def setUp(self):
        # Generate 1 month of test hourly climate data
        self.df = generate_synthetic_historical_dataset(
            region_key="Nakuru_Njoro",
            start_date="2024-01-01",
            end_date="2024-02-01",
            freq="1h"
        )

    def test_preprocessor_imputation_and_cyclical_encoding(self):
        # Inject missing value
        self.df.loc[5, "temperature_c"] = np.nan
        preprocessor = ClimateDataPreprocessor()
        df_clean = preprocessor.impute_missing(self.df)
        self.assertFalse(df_clean["temperature_c"].isnull().any())

        df_enc = preprocessor.encode_cyclical_features(df_clean)
        self.assertIn("sin_hour", df_enc.columns)
        self.assertIn("cos_hour", df_enc.columns)
        self.assertIn("sin_doy", df_enc.columns)
        self.assertIn("cos_doy", df_enc.columns)

    def test_feature_engineering_lags(self):
        preprocessor = ClimateDataPreprocessor()
        df_enc = preprocessor.encode_cyclical_features(self.df)
        df_feat, feature_cols = build_tabular_features(df_enc, lag_steps=[1, 2], rolling_windows=[6])
        
        self.assertIn("temperature_c_lag_1", feature_cols)
        self.assertIn("temperature_c_rolling_mean_6", feature_cols)
        self.assertFalse(df_feat.isnull().any().any())

    def test_persistence_and_rolling_average_baselines(self):
        y_train = np.array([[20.0, 60.0, 0.0], [21.0, 58.0, 2.5]])
        pers = PersistenceModel()
        pers.fit(None, y_train)
        pred = pers.predict(y_train[-1], steps=3)
        self.assertEqual(pred.shape, (3, 3))
        self.assertEqual(pred[0, 0], 21.0)
        self.assertEqual(pred[0, 2], 2.5)

        roll = RollingAverageModel(window_size=2)
        roll.fit(None, y_train)
        roll_pred = roll.predict(y_train, steps=2)
        self.assertEqual(roll_pred.shape, (2, 3))
        self.assertAlmostEqual(roll_pred[0, 0], 20.5)

    def test_walk_forward_validator_chronological_splits(self):
        validator = TimeSeriesWalkForwardValidator(n_splits=3, test_size_ratio=0.15)
        splits = list(validator.split(self.df))
        self.assertEqual(len(splits), 3)
        for train_df, test_df in splits:
            # Enforce that max train timestamp < min test timestamp (no data leakage)
            self.assertLess(train_df["recorded_at"].max(), test_df["recorded_at"].min())
