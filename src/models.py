"""
Engine RUL Prediction — Model Training & Evaluation
Utkarsh Chaudhari | Engine Performance Degradation Forecasting

Models implemented:
    1. LightGBM (gradient boosting, fast)
    2. XGBoost  (gradient boosting, robust)
    3. Ensemble (average of both — standard in PHM competition winning solutions)

Metrics:
    - RMSE   : standard regression error in cycles
    - Score  : NASA PHM competition asymmetric score
                s = sum(exp(-d/13) - 1) for d < 0 (early prediction)
                s = sum(exp( d/10) - 1) for d >= 0 (late prediction)
              Late predictions penalised ~2x harder than early predictions
              (a missed maintenance window is worse than a premature one)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


# ─── Hyperparameters ──────────────────────────────────────────────────────────
LGB_PARAMS = {
    'objective':       'regression_l1',   # MAE loss — more robust to outliers
    'metric':          'rmse',
    'learning_rate':   0.03,
    'n_estimators':    2000,
    'num_leaves':      63,
    'max_depth':       -1,
    'min_child_samples': 20,
    'subsample':       0.8,
    'colsample_bytree': 0.8,
    'reg_alpha':       0.1,
    'reg_lambda':      1.0,
    'n_jobs':          -1,
    'random_state':    42,
    'verbose':         -1,
}

XGB_PARAMS = {
    'objective':       'reg:squarederror',
    'eval_metric':     'rmse',
    'learning_rate':   0.03,
    'n_estimators':    2000,
    'max_depth':       6,
    'min_child_weight': 5,
    'subsample':       0.8,
    'colsample_bytree': 0.8,
    'reg_alpha':       0.1,
    'reg_lambda':      1.0,
    'n_jobs':          -1,
    'random_state':    42,
    'verbosity':       0,
}


# ─── NASA PHM Score ───────────────────────────────────────────────────────────
def phm_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA PHM 2008 competition asymmetric health score.

    Penalises late predictions (positive error) more heavily than early ones.
    This reflects real MRO scheduling risk: predicting failure later than
    actual causes missed maintenance windows; predicting early wastes runway.

    Lower score is better. Score = 0 means perfect prediction.
    """
    d = y_pred - y_true
    s = np.where(d < 0,
                 np.exp(-d / 13.0) - 1,
                 np.exp(d  / 10.0) - 1)
    return float(np.sum(s))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ─── Cross-Validated Training ─────────────────────────────────────────────────
def train_lgb_cv(X: np.ndarray, y: np.ndarray,
                 n_splits: int = 5) -> tuple:
    """
    Train LightGBM with K-fold cross-validation.

    Returns
    -------
    model      : final model trained on full training data
    oof_preds  : out-of-fold predictions (shape: n_samples)
    cv_rmse    : list of per-fold RMSE
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    cv_rmse_list = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(-1)]
        )
        preds = model.predict(X_val)
        oof[val_idx] = preds
        fold_rmse = rmse(y_val, preds)
        cv_rmse_list.append(fold_rmse)
        print(f"  LGB Fold {fold+1}/{n_splits}  RMSE={fold_rmse:.2f}")

    # Final model on all data
    final_model = lgb.LGBMRegressor(**LGB_PARAMS)
    final_model.fit(X, y,
                    callbacks=[lgb.log_evaluation(-1)])

    return final_model, oof, cv_rmse_list


def train_xgb_cv(X: np.ndarray, y: np.ndarray,
                 n_splits: int = 5) -> tuple:
    """
    Train XGBoost with K-fold cross-validation.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    cv_rmse_list = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        preds = model.predict(X_val)
        oof[val_idx] = preds
        fold_rmse = rmse(y_val, preds)
        cv_rmse_list.append(fold_rmse)
        print(f"  XGB Fold {fold+1}/{n_splits}  RMSE={fold_rmse:.2f}")

    # Final model on all data
    final_model = xgb.XGBRegressor(**XGB_PARAMS)
    final_model.fit(X, y, verbose=False)

    return final_model, oof, cv_rmse_list


def predict_with_uncertainty(lgb_model, xgb_model,
                              X_test: np.ndarray,
                              rul_true: np.ndarray,
                              n_bootstrap: int = 50,
                              seed: int = 42) -> pd.DataFrame:
    """
    Generate point predictions, ensemble mean, and bootstrap uncertainty bands.

    Bootstrap: re-train n_bootstrap LGB models on resampled training data
    is expensive; here we approximate uncertainty via feature subsampling
    of the trained model (faster, good proxy for epistemic uncertainty).

    For a production deployment, use quantile regression or conformal prediction.

    Returns
    -------
    DataFrame with columns:
        engine_id, rul_true, lgb_pred, xgb_pred, ensemble_pred,
        lower_80, upper_80, error, late_flag
    """
    rng = np.random.default_rng(seed)
    lgb_pred = lgb_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)
    ensemble  = 0.5 * lgb_pred + 0.5 * xgb_pred

    # Bootstrap uncertainty via perturbation of feature matrix
    boot_preds = np.zeros((n_bootstrap, len(X_test)))
    n_features = X_test.shape[1]
    for i in range(n_bootstrap):
        # Randomly zero out 15% of features (dropout approximation)
        mask = rng.binomial(1, 0.85, size=X_test.shape).astype(float)
        X_perturbed = X_test * mask
        boot_preds[i] = 0.5 * lgb_model.predict(X_perturbed) + \
                        0.5 * xgb_model.predict(X_perturbed)

    lower = np.percentile(boot_preds, 10, axis=0)
    upper = np.percentile(boot_preds, 90, axis=0)

    df = pd.DataFrame({
        'engine_id':    np.arange(1, len(rul_true) + 1),
        'rul_true':     rul_true,
        'lgb_pred':     np.round(lgb_pred, 1),
        'xgb_pred':     np.round(xgb_pred, 1),
        'ensemble_pred': np.round(ensemble, 1),
        'lower_80':     np.round(lower, 1),
        'upper_80':     np.round(upper, 1),
        'error':        np.round(ensemble - rul_true, 1),
    })
    # Late prediction flag: model says engine has MORE life than it does
    # This is the dangerous case from an MRO scheduling perspective
    df['late_flag'] = df['error'] > 15

    return df
