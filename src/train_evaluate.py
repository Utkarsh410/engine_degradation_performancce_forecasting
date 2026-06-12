"""
Engine Performance Degradation Forecasting — Main Pipeline
Utkarsh Chaudhari | Portfolio Project

Run:
    python src/train_evaluate.py

Outputs:
    outputs/results_FD001.csv     — per-engine predictions with uncertainty
    outputs/metrics_summary.txt   — RMSE, PHM score, late-flag rate
    outputs/fig_sensor_degradation.png
    outputs/fig_egt_margin.png
    outputs/fig_predictions.png
    outputs/fig_shap.png
    outputs/fig_inspection_schedule.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loader import (
    load_dataset, preprocess, SELECTED_SENSORS, COLUMNS
)
from src.models import (
    train_lgb_cv, train_xgb_cv,
    predict_with_uncertainty,
    phm_score, rmse
)
from src.visualisation import (
    plot_sensor_degradation,
    plot_egt_margin,
    plot_predictions,
    plot_shap_analysis,
    plot_inspection_schedule
)

DATA_DIR    = Path('data')
OUTPUT_DIR  = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)


def get_test_features(test_df: pd.DataFrame, scaler, feature_cols: list) -> np.ndarray:
    """
    For the test set, we only want the LAST row per engine
    (the cycle at which the test snapshot was taken, and from
    which we predict how many cycles remain).
    """
    last_cycles = test_df.groupby('engine_id').last().reset_index()
    X_test, _, _, _ = preprocess(test_df, is_train=False, scaler=scaler)
    # Pick rows corresponding to last cycle per engine
    last_row_mask = test_df.groupby('engine_id').tail(1).index
    X_test_last = X_test[test_df.index.isin(last_row_mask)]
    return X_test_last


def main():
    print("=" * 60)
    print("ENGINE RUL PREDICTION PIPELINE")
    print("NASA C-MAPSS FD001 | LightGBM + XGBoost Ensemble")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────
    print("\n[1/6] Loading CMAPSS FD001 dataset...")
    train_df, test_df, rul_true = load_dataset(DATA_DIR, subset='FD001')
    print(f"  Train: {len(train_df):,} rows | {train_df.engine_id.nunique()} engines")
    print(f"  Test:  {len(test_df):,} rows  | {test_df.engine_id.nunique()} engines")

    # ── 2. EDA plots ──────────────────────────────────────────
    print("\n[2/6] Generating EDA plots...")
    plot_sensor_degradation(
        train_df, sensors=['s2', 's3', 's4', 's7'],
        engine_ids=[1, 5, 10, 25, 50],
        out_path=OUTPUT_DIR / 'fig_sensor_degradation.png'
    )
    print("  Saved: fig_sensor_degradation.png")

    plot_egt_margin(
        train_df,
        engine_ids=[1, 3, 7, 15, 30],
        out_path=OUTPUT_DIR / 'fig_egt_margin.png'
    )
    print("  Saved: fig_egt_margin.png")

    # ── 3. Preprocessing ──────────────────────────────────────
    print("\n[3/6] Feature engineering & preprocessing...")
    X_train, y_train, feature_cols, scaler = preprocess(
        train_df, is_train=True
    )
    print(f"  Feature matrix: {X_train.shape} | {len(feature_cols)} features")
    print(f"  RUL label range: [{y_train.min():.0f}, {y_train.max():.0f}] cycles")

    # Test set — last observation per engine
    X_test_full, _, _, _ = preprocess(test_df, is_train=False, scaler=scaler)
    # Select only the final row per engine (the prediction snapshot)
    last_indices = test_df.groupby('engine_id').apply(lambda g: g.index[-1])
    test_df_reset = test_df.reset_index(drop=True)
    last_positions = test_df_reset.groupby('engine_id').apply(lambda g: g.index[-1]).values
    X_test = X_test_full[last_positions]
    print(f"  Test feature matrix: {X_test.shape}")

    # ── 4. Model training ─────────────────────────────────────
    print("\n[4/6] Training LightGBM (5-fold CV)...")
    lgb_model, lgb_oof, lgb_cv = train_lgb_cv(X_train, y_train, n_splits=5)
    print(f"  LGB OOF RMSE: {rmse(y_train, lgb_oof):.2f} cycles")
    print(f"  LGB CV RMSE:  {np.mean(lgb_cv):.2f} ± {np.std(lgb_cv):.2f} cycles")

    print("\n  Training XGBoost (5-fold CV)...")
    xgb_model, xgb_oof, xgb_cv = train_xgb_cv(X_train, y_train, n_splits=5)
    print(f"  XGB OOF RMSE: {rmse(y_train, xgb_oof):.2f} cycles")
    print(f"  XGB CV RMSE:  {np.mean(xgb_cv):.2f} ± {np.std(xgb_cv):.2f} cycles")

    # ── 5. Test evaluation ────────────────────────────────────
    print("\n[5/6] Evaluating on test set...")
    results_df = predict_with_uncertainty(
        lgb_model, xgb_model, X_test,
        rul_true=rul_true.values,
        n_bootstrap=50
    )

    # Metrics
    ens_rmse   = rmse(results_df.rul_true, results_df.ensemble_pred)
    lgb_rmse   = rmse(results_df.rul_true, results_df.lgb_pred)
    xgb_rmse   = rmse(results_df.rul_true, results_df.xgb_pred)
    ens_score  = phm_score(results_df.rul_true.values, results_df.ensemble_pred.values)
    late_rate  = results_df.late_flag.mean() * 100
    mae        = results_df.error.abs().mean()

    metrics_text = f"""
ENGINE RUL PREDICTION — TEST SET METRICS
Dataset:      NASA C-MAPSS FD001 (100 test engines)
Model:        LightGBM + XGBoost Ensemble (50/50 average)

PERFORMANCE METRICS
───────────────────────────────────────────
RMSE — LightGBM:          {lgb_rmse:.2f} cycles
RMSE — XGBoost:           {xgb_rmse:.2f} cycles
RMSE — Ensemble:          {ens_rmse:.2f} cycles  ← primary metric
MAE  — Ensemble:          {mae:.2f} cycles
NASA PHM Score:           {ens_score:.1f}  ← lower is better

SAFETY-CRITICAL FLAGS
───────────────────────────────────────────
Late prediction rate:     {late_rate:.1f}%
  (error > +15 cycles — unsafe for MRO scheduling)
Mean prediction error:    {results_df.error.mean():.1f} cycles
  (negative = conservative/early — operationally safer)

FEATURE ENGINEERING
───────────────────────────────────────────
Total features:           {X_train.shape[1]}
  - 14 sensor readings (raw)
  - 42 rolling statistics (mean + std, windows 5/10/20)
  - EGT margin proxy
  - Normalised cycle position
  - Absolute cycle count

TRAINING CONFIGURATION
───────────────────────────────────────────
RUL clip (piece-wise linear):   125 cycles
CV folds:                        5-fold
Ensemble method:                 Simple average
Uncertainty bands:               Bootstrap feature dropout (n=50)
"""
    print(metrics_text)
    with open(OUTPUT_DIR / 'metrics_summary.txt', 'w') as f:
        f.write(metrics_text)

    results_df.to_csv(OUTPUT_DIR / 'results_FD001.csv', index=False)
    print(f"  Results saved: results_FD001.csv")

    # ── 6. Visualisations ─────────────────────────────────────
    print("\n[6/6] Generating result plots...")
    plot_predictions(
        results_df,
        out_path=OUTPUT_DIR / 'fig_predictions.png'
    )
    print("  Saved: fig_predictions.png")

    plot_shap_analysis(
        lgb_model, X_test, feature_cols,
        top_n=15,
        out_path=OUTPUT_DIR / 'fig_shap.png'
    )
    print("  Saved: fig_shap.png")

    plot_inspection_schedule(
        results_df,
        inspection_buffer=20,
        out_path=OUTPUT_DIR / 'fig_inspection_schedule.png'
    )
    print("  Saved: fig_inspection_schedule.png")

    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETE")
    print(f"All outputs in: {OUTPUT_DIR.resolve()}")
    print(f"Ensemble RMSE: {ens_rmse:.2f} cycles | PHM Score: {ens_score:.1f}")
    print("=" * 60)

    return results_df, lgb_model, xgb_model, feature_cols


if __name__ == '__main__':
    main()
