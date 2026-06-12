"""
NASA CMAPSS Turbofan Engine Degradation - Data Loader & Feature Engineering
Utkarsh Chaudhari | Engine Performance Degradation Forecasting Portfolio Project

Dataset: NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)
Reference: Saxena et al. (2008), PHM08 Challenge
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

# Column schema: 26 space-separated columns per row
COLUMNS = (
    ['engine_id', 'cycle', 'op1', 'op2', 'op3'] +
    [f's{i}' for i in range(1, 22)]
)

# Sensors with near-zero variance in FD001 (constant across all cycles) — dropped
# Identified via std < 0.01 threshold during EDA
CONSTANT_SENSORS = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']

# Sensors selected for modelling (14 of 21 — standard literature selection)
SELECTED_SENSORS = [s for s in [f's{i}' for i in range(1, 22)]
                    if s not in CONSTANT_SENSORS]

# Piece-wise linear RUL cap: engines assumed healthy until cap cycles before failure
# Standard in literature (Heimes 2008); prevents over-penalising early predictions
RUL_CLIP = 125


def load_dataset(data_dir: str, subset: str = 'FD001') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load train, test and ground-truth RUL for a given CMAPSS subset.

    Parameters
    ----------
    data_dir : path to directory containing CMAPSS .txt files
    subset   : one of 'FD001', 'FD002', 'FD003', 'FD004'

    Returns
    -------
    train_df  : raw training DataFrame
    test_df   : raw test DataFrame
    rul_true  : Series of true RUL values for test engines (one per engine)
    """
    base = Path(data_dir)
    train_df = pd.read_csv(
        base / f'train_{subset}.txt', sep=r'\s+', header=None, names=COLUMNS
    )
    test_df = pd.read_csv(
        base / f'test_{subset}.txt', sep=r'\s+', header=None, names=COLUMNS
    )
    rul_true = pd.read_csv(
        base / f'RUL_{subset}.txt', sep=r'\s+', header=None, names=['RUL']
    )['RUL']

    return train_df, test_df, rul_true


def compute_rul(df: pd.DataFrame, clip: int = RUL_CLIP) -> pd.Series:
    """
    Compute piece-wise linear Remaining Useful Life label for training data.

    For each engine: RUL = (max_cycle - current_cycle), clipped at `clip`.
    This assumes engines are fully healthy in early life — avoids penalising
    the model for imprecise early-life predictions.

    Parameters
    ----------
    df   : training DataFrame with engine_id and cycle columns
    clip : upper cap on RUL label (default 125 cycles)

    Returns
    -------
    rul : pd.Series aligned with df index
    """
    max_cycles = df.groupby('engine_id')['cycle'].transform('max')
    rul = (max_cycles - df['cycle']).clip(upper=clip)
    return rul.rename('RUL')


def add_rolling_features(df: pd.DataFrame,
                          sensors: list,
                          windows: list = [5, 10, 20]) -> pd.DataFrame:
    """
    Append rolling mean and std features for each sensor over given windows.

    Rolling statistics capture degradation trend — a single-cycle snapshot
    is noisy; the rolling mean smooths the sensor trajectory and the rolling
    std captures variability increase (a known early-degradation signal).

    Parameters
    ----------
    df      : DataFrame with engine_id, cycle, and sensor columns
    sensors : list of sensor column names to roll
    windows : list of window sizes in cycles

    Returns
    -------
    df : copy with additional rolling feature columns appended
    """
    df = df.copy()
    for w in windows:
        for s in sensors:
            grp = df.groupby('engine_id')[s]
            df[f'{s}_mean_{w}'] = grp.transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
            df[f'{s}_std_{w}'] = grp.transform(
                lambda x: x.rolling(w, min_periods=1).std().fillna(0)
            )
    return df


def add_egt_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer an EGT (Exhaust Gas Temperature) margin proxy.

    In real MRO operations, EGT margin (difference between observed EGT
    and engine's rated limit) is the primary health indicator for engine
    life planning. s7 (Total temperature at HPC outlet) is the closest
    CMAPSS analogue.

    We define:
        egt_margin = max(s7 per engine) - s7
    A decreasing margin signals thermal degradation.
    """
    df = df.copy()
    max_s7 = df.groupby('engine_id')['s7'].transform('max')
    df['egt_margin'] = max_s7 - df['s7']
    return df


def add_cycle_ratio(df: pd.DataFrame, max_rul: int = RUL_CLIP) -> pd.DataFrame:
    """
    Add normalised cycle position (0 = start of known life, 1 = failure).

    Helps the model understand where in the engine's lifetime it is,
    independent of absolute cycle count.
    """
    df = df.copy()
    max_cycle = df.groupby('engine_id')['cycle'].transform('max')
    df['cycle_ratio'] = df['cycle'] / max_cycle
    return df


def preprocess(df: pd.DataFrame,
               sensors: Optional[list] = None,
               is_train: bool = True,
               scaler=None,
               rul_clip: int = RUL_CLIP):
    """
    Full preprocessing pipeline:
        1. Drop constant sensors
        2. Add EGT margin proxy
        3. Add rolling features
        4. Add cycle ratio
        5. Min-max normalise sensor readings
        6. Compute RUL label (train only)

    Parameters
    ----------
    df       : raw DataFrame
    sensors  : sensor list (default: SELECTED_SENSORS)
    is_train : if True, fit scaler and compute RUL labels
    scaler   : pre-fitted MinMaxScaler (for test set); if None, fit from df
    rul_clip : RUL cap

    Returns
    -------
    features : np.ndarray of shape (n_samples, n_features)
    labels   : np.ndarray of shape (n_samples,) — only for train; else None
    feature_names : list of column names (for SHAP)
    scaler   : fitted scaler (return to use on test set)
    """
    from sklearn.preprocessing import MinMaxScaler

    if sensors is None:
        sensors = SELECTED_SENSORS

    df = df.copy()
    df = add_egt_proxy(df)
    df = add_rolling_features(df, sensors=sensors, windows=[5, 10, 20])
    df = add_cycle_ratio(df)

    # Collect all feature columns
    roll_cols = [c for c in df.columns if any(f'mean_{w}' in c or f'std_{w}' in c
                                               for w in [5, 10, 20])]
    extra_cols = ['egt_margin', 'cycle_ratio', 'cycle']
    feature_cols = sensors + roll_cols + extra_cols

    X = df[feature_cols].values

    if is_train:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        labels = compute_rul(df, clip=rul_clip).values
    else:
        if scaler is None:
            raise ValueError("scaler must be provided for test set preprocessing")
        X = scaler.transform(X)
        labels = None

    return X, labels, feature_cols, scaler
