"""
Unit tests — Engine RUL Pipeline
Run: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from src.data_loader import compute_rul, add_egt_proxy, add_rolling_features, SELECTED_SENSORS
from src.models import phm_score, rmse


# ─── Fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture
def toy_engine_df():
    """Minimal two-engine DataFrame for unit tests."""
    rows = []
    for eng in [1, 2]:
        for cycle in range(1, 51):
            row = {'engine_id': eng, 'cycle': float(cycle)}
            row.update({f's{i}': float(cycle + i) for i in range(1, 22)})
            row.update({'op1': 0.0, 'op2': 0.0, 'op3': 100.0})
            rows.append(row)
    return pd.DataFrame(rows)


# ─── data_loader tests ────────────────────────────────────────────────────────
def test_compute_rul_shape(toy_engine_df):
    rul = compute_rul(toy_engine_df, clip=125)
    assert len(rul) == len(toy_engine_df), "RUL length must match DataFrame"


def test_compute_rul_last_cycle_is_zero(toy_engine_df):
    rul = compute_rul(toy_engine_df, clip=125)
    last_rows = toy_engine_df.groupby('engine_id').tail(1).index
    assert (rul[last_rows] == 0).all(), "RUL at last cycle must be 0"


def test_compute_rul_clip(toy_engine_df):
    clip_val = 10
    rul = compute_rul(toy_engine_df, clip=clip_val)
    assert rul.max() == clip_val, f"RUL should be clipped at {clip_val}"


def test_compute_rul_decreasing_per_engine(toy_engine_df):
    rul = compute_rul(toy_engine_df, clip=200)
    for eng_id in toy_engine_df.engine_id.unique():
        eng_rul = rul[toy_engine_df.engine_id == eng_id]
        diffs = eng_rul.diff().dropna()
        assert (diffs <= 0).all(), f"RUL must be non-increasing for engine {eng_id}"


def test_add_egt_proxy_columns(toy_engine_df):
    result = add_egt_proxy(toy_engine_df)
    assert 'egt_margin' in result.columns, "egt_margin column must be created"


def test_add_egt_proxy_nonnegative(toy_engine_df):
    result = add_egt_proxy(toy_engine_df)
    assert (result['egt_margin'] >= 0).all(), "EGT margin must be non-negative (max - current >= 0)"


def test_add_egt_proxy_last_cycle_zero(toy_engine_df):
    """At the cycle where s7 is maximum, margin should be 0."""
    result = add_egt_proxy(toy_engine_df)
    # s7 = cycle + 7, so max per engine is at last cycle
    last = result.groupby('engine_id').tail(1)
    assert (last['egt_margin'] == 0).all(), "EGT margin should be 0 at peak s7 cycle"


def test_rolling_features_shape(toy_engine_df):
    sensors = ['s2', 's4', 's7']
    windows = [5, 10]
    result = add_rolling_features(toy_engine_df, sensors, windows)
    expected_new_cols = len(sensors) * len(windows) * 2  # mean + std per sensor per window
    assert result.shape[1] == toy_engine_df.shape[1] + expected_new_cols


def test_rolling_features_no_nan_at_cycle1(toy_engine_df):
    """min_periods=1 ensures no NaN at first cycle."""
    result = add_rolling_features(toy_engine_df, ['s2'], windows=[5])
    assert result['s2_mean_5'].isna().sum() == 0, "Rolling mean should have no NaN (min_periods=1)"


# ─── models tests ─────────────────────────────────────────────────────────────
def test_rmse_perfect():
    y = np.array([10.0, 20.0, 30.0])
    assert rmse(y, y) == pytest.approx(0.0, abs=1e-9), "RMSE of perfect predictions must be 0"


def test_rmse_known_value():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    # RMSE = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.536
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(12.5), rel=1e-6)


def test_phm_score_perfect():
    y = np.array([50.0, 75.0, 100.0])
    score = phm_score(y, y)
    assert score == pytest.approx(0.0, abs=1e-6), "PHM score of perfect prediction must be 0"


def test_phm_score_late_worse_than_early():
    """Late predictions (positive error) must score worse than symmetric early ones."""
    y_true = np.array([50.0])
    late_score  = phm_score(y_true, y_true + 20)   # predict 20 cycles too late
    early_score = phm_score(y_true, y_true - 20)   # predict 20 cycles too early
    assert late_score > early_score, "Late prediction must score worse than equal early prediction"


def test_phm_score_nonnegative():
    y_true = np.array([10.0, 50.0, 100.0])
    y_pred = np.array([15.0, 40.0, 120.0])
    assert phm_score(y_true, y_pred) >= 0, "PHM score must be non-negative"


# ─── Integration smoke test ───────────────────────────────────────────────────
def test_preprocess_returns_correct_shape(toy_engine_df):
    """preprocess should return scaled numpy arrays without NaN."""
    from src.data_loader import preprocess
    X, y, feature_cols, scaler = preprocess(toy_engine_df, is_train=True)
    assert X.shape[0] == len(toy_engine_df), "Row count must match DataFrame"
    assert len(feature_cols) == X.shape[1], "Feature col count must match array width"
    assert not np.isnan(X).any(), "No NaN values in feature matrix"
    assert not np.isnan(y).any(), "No NaN values in RUL labels"


def test_preprocess_scale_range(toy_engine_df):
    """MinMaxScaler should produce values in [0, 1]."""
    from src.data_loader import preprocess
    X, _, _, _ = preprocess(toy_engine_df, is_train=True)
    assert X.min() >= -1e-9, "Scaled features should be >= 0"
    assert X.max() <= 1 + 1e-9, "Scaled features should be <= 1"
