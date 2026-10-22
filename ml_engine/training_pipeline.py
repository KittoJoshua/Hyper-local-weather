"""
End-to-End Training & Model Comparison Pipeline.
Executes the full machine learning lifecycle:
1. Data ingestion & cleaning
2. Feature engineering & cyclical transformations
3. Walk-forward backtesting across baselines, standalone models, and hybrid stack
4. Production model training and checkpoint serialization
"""
import os
import json
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime
from ml_engine.config import (
    TARGET_VARIABLES, WEIGHTS_DIR, SEQUENCE_LENGTH,
    LSTM_PARAMS, XGBOOST_PARAMS, DEFAULT_TRAINING_RECORDS
)
from ml_engine.data_loader import load_dataset
from ml_engine.preprocessor import ClimateDataPreprocessor
from ml_engine.feature_engineering import build_tabular_features, create_sequential_windows
from ml_engine.models.baseline_models import PersistenceModel, RollingAverageModel
from ml_engine.models.xgboost_model import MicroClimateXGBoost
from ml_engine.models.lstm_model import MicroClimateLSTM
from ml_engine.models.hybrid_stack import HybridResidualForecaster
from ml_engine.evaluation.metrics import calculate_regression_metrics, format_metrics_table
from ml_engine.evaluation.validator import TimeSeriesWalkForwardValidator


def run_training_and_benchmarking(
    region_key: str = "Nakuru_Njoro",
    max_records: Optional[int] = DEFAULT_TRAINING_RECORDS
) -> Dict:
    """
    Executes walk-forward comparative benchmark and saves final trained model weights.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Step 1: Loading climate dataset for {region_key} (records limit: {max_records})...")
    df_raw = load_dataset(region_key=region_key, max_records=max_records)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Step 2: Preprocessing and cyclical encoding...")
    preprocessor = ClimateDataPreprocessor(target_cols=TARGET_VARIABLES)
    df_clean = preprocessor.impute_missing(df_raw)
    df_encoded = preprocessor.encode_cyclical_features(df_clean)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Step 3: Feature engineering (lags, rolling stats, elevation offset)...")
    df_feat, feature_cols = build_tabular_features(df_encoded, target_cols=TARGET_VARIABLES)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Step 4: Setting up Walk-Forward Validation splits...")
    validator = TimeSeriesWalkForwardValidator(n_splits=3, test_size_ratio=0.15)
    baseline_results = validator.evaluate_baselines(df_feat, TARGET_VARIABLES)

    # Perform walk-forward evaluation on XGBoost, LSTM, and Hybrid Stacking
    model_scores = {
        "Persistence Baseline": baseline_results["Persistence Baseline"],
        "Historical Rolling Average (24h)": baseline_results["Historical Rolling Average (24h)"],
        "Standalone XGBoost": {col: {"RMSE": [], "MAE": [], "R2": []} for col in TARGET_VARIABLES},
        "Standalone LSTM": {col: {"RMSE": [], "MAE": [], "R2": []} for col in TARGET_VARIABLES},
        "Hybrid (LSTM + XGBoost)": {col: {"RMSE": [], "MAE": [], "R2": []} for col in TARGET_VARIABLES},
    }

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Step 5: Training and evaluating models across folds...")
    fold_idx = 1
    for train_df, test_df in validator.split(df_feat):
        print(f"  --> Processing Fold {fold_idx}/3 (Train samples: {len(train_df)}, Test samples: {len(test_df)})...")
        
        # Prepare tabular matrices
        X_tab_train = train_df[feature_cols]
        y_train_df = train_df[TARGET_VARIABLES]
        X_tab_test = test_df[feature_cols]
        y_test_df = test_df[TARGET_VARIABLES]
        y_test_mat = y_test_df.values

        # 5a. Standalone XGBoost
        xgb_standalone = MicroClimateXGBoost(target_cols=TARGET_VARIABLES)
        xgb_standalone.fit(X_tab_train, y_train_df)
        xgb_preds = xgb_standalone.predict(X_tab_test)
        xgb_metrics = calculate_regression_metrics(y_test_mat, xgb_preds, TARGET_VARIABLES)
        for col in TARGET_VARIABLES:
            for k in ["RMSE", "MAE", "R2"]:
                model_scores["Standalone XGBoost"][col][k].append(xgb_metrics[col][k])

        # Prepare sequential windows for LSTM (using smaller seq_len for rapid fold evaluation)
        eval_seq_len = min(72, max(4, len(train_df) // 10))
        base_features = ["temperature_c", "humidity_percent", "precipitation_mm", "sin_hour", "cos_hour", "sin_doy", "cos_doy"]
        
        train_raw_seq = train_df[base_features].values
        train_raw_targets = train_df[TARGET_VARIABLES].values
        
        # Prepend the tail of train_df so test set has sufficient sequence history
        combined_test_df = pd.concat([train_df.tail(eval_seq_len), test_df])
        test_raw_seq = combined_test_df[base_features].values
        test_raw_targets = combined_test_df[TARGET_VARIABLES].values

        X_seq_train, y_seq_train = create_sequential_windows(train_raw_seq, train_raw_targets, sequence_length=eval_seq_len)
        X_seq_test, y_seq_test = create_sequential_windows(test_raw_seq, test_raw_targets, sequence_length=eval_seq_len)

        # 5b. Standalone LSTM
        fast_lstm_params = LSTM_PARAMS.copy()
        fast_lstm_params["epochs"] = 3 if (max_records and max_records <= 4000) else 12
        lstm_standalone = MicroClimateLSTM(input_size=len(base_features), output_size=len(TARGET_VARIABLES), params=fast_lstm_params)
        lstm_standalone.fit(X_seq_train, y_seq_train)
        lstm_preds = lstm_standalone.predict(X_seq_test)
        lstm_metrics = calculate_regression_metrics(y_seq_test, lstm_preds, TARGET_VARIABLES)
        for col in TARGET_VARIABLES:
            for k in ["RMSE", "MAE", "R2"]:
                model_scores["Standalone LSTM"][col][k].append(lstm_metrics[col][k])

        # 5c. Hybrid Stacking (LSTM baseline + XGBoost residual)
        hybrid_forecaster = HybridResidualForecaster(
            lstm_input_size=len(base_features),
            target_cols=TARGET_VARIABLES,
            lstm_params=fast_lstm_params
        )
        # Tabular features aligned with sequence target indices
        X_tab_train_aligned = X_tab_train.iloc[eval_seq_len:].reset_index(drop=True)
        X_tab_test_aligned = X_tab_test.reset_index(drop=True)

        hybrid_forecaster.fit(
            X_seq_train=X_seq_train,
            y_train=y_seq_train,
            X_tab_train=X_tab_train_aligned
        )
        hybrid_preds = hybrid_forecaster.predict(X_seq_test, X_tab_test_aligned)
        hybrid_metrics = calculate_regression_metrics(y_seq_test, hybrid_preds, TARGET_VARIABLES)
        for col in TARGET_VARIABLES:
            for k in ["RMSE", "MAE", "R2"]:
                model_scores["Hybrid (LSTM + XGBoost)"][col][k].append(hybrid_metrics[col][k])

        fold_idx += 1

    # Aggregate results across folds
    aggregated_benchmark = {}
    for model_name, target_dict in model_scores.items():
        aggregated_benchmark[model_name] = {}
        for target_name, metric_dict in target_dict.items():
            aggregated_benchmark[model_name][target_name] = {
                k: round(float(np.mean(vals)), 4) if isinstance(vals, list) else vals
                for k, vals in metric_dict.items()
            }

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Step 6: Compiling comparative metrics table...")
    df_metrics = format_metrics_table(aggregated_benchmark)

    # Save benchmark report to JSON and CSV
    metrics_json_path = WEIGHTS_DIR / "model_benchmark_report.json"
    with open(metrics_json_path, "w") as f:
        json.dump(aggregated_benchmark, f, indent=2)

    df_metrics.to_csv(WEIGHTS_DIR / "model_benchmark_report.csv", index=False)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Step 7: Training production Hybrid and XGBoost models on complete data...")
    prod_xgb = MicroClimateXGBoost(target_cols=TARGET_VARIABLES)
    prod_xgb.fit(df_feat[feature_cols], df_feat[TARGET_VARIABLES])
    prod_xgb.save(str(WEIGHTS_DIR / "production_xgboost"))

    # Train production Hybrid
    prod_seq_len = 72
    base_features = ["temperature_c", "humidity_percent", "precipitation_mm", "sin_hour", "cos_hour", "sin_doy", "cos_doy"]
    X_seq_full, y_seq_full = create_sequential_windows(
        df_feat[base_features].values,
        df_feat[TARGET_VARIABLES].values,
        sequence_length=prod_seq_len
    )
    X_tab_full = df_feat[feature_cols].iloc[prod_seq_len:].reset_index(drop=True)

    prod_hybrid = HybridResidualForecaster(
        lstm_input_size=len(base_features),
        target_cols=TARGET_VARIABLES,
        lstm_params=fast_lstm_params
    )
    prod_hybrid.fit(X_seq_full, y_seq_full, X_tab_full)
    prod_hybrid.save("production_hybrid")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Model training and validation complete! Artifacts saved to {WEIGHTS_DIR}")

    return {
        "benchmark_summary": aggregated_benchmark,
        "metrics_table": df_metrics,
        "feature_cols": feature_cols
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Micro-Climate model training and benchmarking")
    parser.add_argument("--region", type=str, default="Nakuru_Njoro", help="Target agro-climatic region")
    parser.add_argument("--records", type=int, default=DEFAULT_TRAINING_RECORDS, help="Number of records to train on (default: 8760 = 1 year)")
    args = parser.parse_args()

    run_training_and_benchmarking(region_key=args.region, max_records=args.records)
