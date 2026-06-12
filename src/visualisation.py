"""
Engine RUL Prediction — Visualisation Module
Utkarsh Chaudhari | Engine Performance Degradation Forecasting

Produces publication-quality figures for:
    1. Sensor degradation trajectories
    2. EGT margin decline
    3. Prediction vs actual scatter with uncertainty bands
    4. SHAP feature importance (global + local)
    5. Late-prediction risk flag summary
    6. Inspection interval scheduling calendar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Consistent colour palette (aviation-safety inspired)
COLOURS = {
    'safe':     '#2E86AB',   # blue
    'warning':  '#F6AE2D',   # amber
    'danger':   '#E63946',   # red
    'neutral':  '#6C757D',   # grey
    'lgb':      '#2E86AB',
    'xgb':      '#43AA8B',
    'ensemble': '#264653',
    'band':     '#A8DADC',
}

plt.rcParams.update({
    'font.family':    'DejaVu Sans',
    'font.size':      11,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'figure.dpi':     150,
    'savefig.bbox':   'tight',
    'savefig.dpi':    150,
})


def plot_sensor_degradation(train_df: pd.DataFrame,
                             sensors: list,
                             engine_ids: list = [1, 5, 10, 25, 50],
                             out_path: str = None) -> plt.Figure:
    """
    Plot raw sensor trajectories for selected engines.
    Shows the characteristic degradation trend (signal drift toward failure).
    """
    n = min(4, len(sensors))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for ax, sensor in zip(axes, sensors[:4]):
        for eng_id in engine_ids:
            eng_df = train_df[train_df.engine_id == eng_id]
            ax.plot(eng_df['cycle'], eng_df[sensor],
                    alpha=0.6, linewidth=0.9)
        ax.set_title(f'{sensor} — signal trajectory', fontsize=11, fontweight='bold')
        ax.set_xlabel('Flight cycle')
        ax.set_ylabel('Sensor reading (normalised)')
        ax.grid(True, alpha=0.3, linestyle='--')

    fig.suptitle('CMAPSS Sensor Degradation Trajectories\n'
                 'Each line = one engine (colour = engine ID)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    return fig


def plot_egt_margin(train_df: pd.DataFrame,
                    engine_ids: list = [1, 3, 7, 15, 30],
                    out_path: str = None) -> plt.Figure:
    """
    Plot EGT margin decline for selected engines.

    EGT margin = (peak s7 over engine life) - current s7.
    In real MRO ops, EGT margin triggers shop visit scheduling.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.cm.plasma
    colors = [cmap(i / len(engine_ids)) for i in range(len(engine_ids))]

    for color, eng_id in zip(colors, engine_ids):
        eng_df = train_df[train_df.engine_id == eng_id].copy()
        max_s7 = eng_df['s7'].max()
        eng_df['egt_margin'] = max_s7 - eng_df['s7']
        ax.plot(eng_df['cycle'], eng_df['egt_margin'],
                color=color, linewidth=1.5, alpha=0.85,
                label=f'Engine {eng_id}')

    ax.axhline(0, color=COLOURS['danger'], linestyle='--', linewidth=1.5,
               label='Zero margin (shop visit trigger)')
    ax.fill_between([0, train_df.cycle.max()], 0, -5,
                    alpha=0.1, color=COLOURS['danger'])

    ax.set_xlabel('Flight cycle', fontsize=12)
    ax.set_ylabel('EGT margin proxy (s7 units)', fontsize=12)
    ax.set_title('EGT Margin Decline — Analogue of Real MRO Trigger\n'
                 'Decreasing margin indicates thermal degradation in HPC',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.8)
    ax.grid(True, alpha=0.3, linestyle='--')

    if out_path:
        plt.savefig(out_path)
    return fig


def plot_predictions(results_df: pd.DataFrame,
                     out_path: str = None) -> plt.Figure:
    """
    Four-panel prediction analysis figure:
        A) Predicted vs actual RUL scatter
        B) Error distribution histogram
        C) Uncertainty band plot (engines sorted by true RUL)
        D) Late-flag rate by RUL bin
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    df = results_df.copy()

    # A: Scatter — predicted vs actual
    ax = axes[0, 0]
    late = df[df.late_flag]
    on_time = df[~df.late_flag]
    ax.scatter(on_time.rul_true, on_time.ensemble_pred,
               alpha=0.6, s=25, color=COLOURS['safe'], label='On-time / early')
    ax.scatter(late.rul_true, late.ensemble_pred,
               alpha=0.8, s=35, color=COLOURS['danger'], marker='^',
               label=f'Late prediction (>{15} cycle error)')
    lim = max(df.rul_true.max(), df.ensemble_pred.max()) + 5
    ax.plot([0, lim], [0, lim], 'k--', linewidth=1, alpha=0.5, label='Perfect prediction')
    ax.plot([0, lim], [15, lim + 15], color=COLOURS['warning'],
            linewidth=1, linestyle=':', label='+15 cycle bound')
    ax.set_xlabel('True RUL (cycles)')
    ax.set_ylabel('Predicted RUL (cycles)')
    ax.set_title('A — Predicted vs True RUL\nEnsemble model (LGB + XGB average)',
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # B: Error distribution
    ax = axes[0, 1]
    errors = df.error
    ax.hist(errors[errors < 0],  bins=25, color=COLOURS['safe'],
            alpha=0.7, label='Early prediction (conservative)')
    ax.hist(errors[errors >= 0], bins=25, color=COLOURS['danger'],
            alpha=0.7, label='Late prediction (risky)')
    ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
    ax.axvline(errors.mean(), color=COLOURS['warning'], linewidth=2,
               linestyle='-', label=f'Mean error = {errors.mean():.1f} cyc')
    ax.set_xlabel('Prediction error (pred - true, cycles)')
    ax.set_ylabel('Count')
    ax.set_title('B — Error Distribution\nNegative = conservative, Positive = late (risky)',
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # C: Uncertainty band (sorted by true RUL)
    ax = axes[1, 0]
    df_sorted = df.sort_values('rul_true').reset_index(drop=True)
    x = np.arange(len(df_sorted))
    ax.fill_between(x, df_sorted.lower_80, df_sorted.upper_80,
                    alpha=0.25, color=COLOURS['band'], label='80% uncertainty band')
    ax.plot(x, df_sorted.rul_true,    color='black',           linewidth=1.5,
            label='True RUL', zorder=3)
    ax.plot(x, df_sorted.ensemble_pred, color=COLOURS['ensemble'],
            linewidth=1.5, linestyle='--', label='Ensemble prediction', zorder=3)
    ax.set_xlabel('Engine (sorted by true RUL)')
    ax.set_ylabel('RUL (cycles)')
    ax.set_title('C — Predictions with 80% Uncertainty Band\nSorted by true RUL',
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # D: Late flag rate by RUL bin
    ax = axes[1, 1]
    bins = [0, 25, 50, 75, 100, 125]
    labels_bin = ['0-25', '25-50', '50-75', '75-100', '100-125']
    df['rul_bin'] = pd.cut(df.rul_true, bins=bins, labels=labels_bin)
    late_rate = df.groupby('rul_bin')['late_flag'].mean() * 100
    bars = ax.bar(late_rate.index, late_rate.values,
                  color=[COLOURS['danger'] if v > 20 else COLOURS['warning']
                         if v > 10 else COLOURS['safe'] for v in late_rate.values])
    ax.set_xlabel('True RUL bin (cycles)')
    ax.set_ylabel('Late prediction rate (%)')
    ax.set_title('D — Late Prediction Rate by RUL Bin\nRed = >20% (MRO scheduling risk)',
                 fontweight='bold')
    ax.axhline(20, color=COLOURS['danger'], linestyle='--',
               linewidth=1, alpha=0.7, label='20% threshold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, late_rate.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.0f}%', ha='center', va='bottom', fontsize=10)

    plt.suptitle('Engine RUL Prediction — Full Test-Set Analysis\n'
                 'NASA C-MAPSS FD001 | LightGBM + XGBoost Ensemble',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    return fig


def plot_shap_analysis(model, X_test: np.ndarray,
                        feature_names: list,
                        top_n: int = 15,
                        out_path: str = None) -> plt.Figure:
    """
    SHAP global feature importance bar chart + beeswarm summary.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Bar chart — mean |SHAP|
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[-top_n:]
    feat_labels = [feature_names[i] for i in sorted_idx]
    feat_vals   = mean_abs_shap[sorted_idx]

    axes[0].barh(range(top_n), feat_vals, color=COLOURS['safe'], alpha=0.8)
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels(feat_labels, fontsize=9)
    axes[0].set_xlabel('Mean |SHAP value| (impact on RUL prediction)')
    axes[0].set_title(f'Top {top_n} Features — Global Importance\nMean absolute SHAP value',
                      fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')

    # Beeswarm / scatter for top 10
    plt.sca(axes[1])
    shap.summary_plot(shap_values, X_test,
                      feature_names=feature_names,
                      max_display=10,
                      show=False,
                      plot_type='dot',
                      color_bar=True)
    axes[1].set_title('SHAP Beeswarm — Feature Effect Direction\n'
                      'Red = high feature value, Blue = low',
                      fontweight='bold')

    plt.suptitle('SHAP Feature Importance Analysis\n'
                 'LightGBM model — NASA C-MAPSS FD001',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
    return fig


def plot_inspection_schedule(results_df: pd.DataFrame,
                              inspection_buffer: int = 20,
                              out_path: str = None) -> plt.Figure:
    """
    Maintenance scheduling Gantt-style chart.

    Converts RUL prediction into actionable inspection triggers:
        - Green  : predicted RUL > 50 cycles (safe)
        - Amber  : predicted RUL 20-50 cycles (schedule inspection)
        - Red    : predicted RUL < 20 cycles (immediate inspection required)

    The `inspection_buffer` adds a safety margin to the raw prediction,
    mirroring how MRO operators build in scheduling margin.
    """
    df = results_df.copy()
    df['scheduled_rul'] = df['ensemble_pred'] - inspection_buffer
    df['status'] = pd.cut(df['ensemble_pred'],
                          bins=[-1, 20, 50, 999],
                          labels=['IMMEDIATE', 'SCHEDULE', 'SAFE'])

    status_colors = {'SAFE': COLOURS['safe'],
                     'SCHEDULE': COLOURS['warning'],
                     'IMMEDIATE': COLOURS['danger']}

    fig, ax = plt.subplots(figsize=(14, 6))

    counts = df.status.value_counts()
    bars = ax.bar(counts.index,
                  counts.values,
                  color=[status_colors[s] for s in counts.index],
                  width=0.5, edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f'{val} engines\n({val/len(df)*100:.0f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_xlabel('Maintenance Status', fontsize=12)
    ax.set_ylabel('Number of Engines', fontsize=12)
    ax.set_title(f'Engine Fleet — Inspection Scheduling Dashboard\n'
                 f'{inspection_buffer}-cycle safety buffer applied to ensemble prediction',
                 fontsize=13, fontweight='bold')

    # Legend patches
    patches = [mpatches.Patch(color=v, label=k) for k, v in status_colors.items()]
    legend_labels = {
        'SAFE':      'SAFE: RUL > 50 cycles',
        'SCHEDULE':  'SCHEDULE: RUL 20–50 cycles',
        'IMMEDIATE': 'IMMEDIATE: RUL < 20 cycles'
    }
    patches = [mpatches.Patch(color=status_colors[k], label=legend_labels[k])
               for k in ['SAFE', 'SCHEDULE', 'IMMEDIATE']]
    ax.legend(handles=patches, loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, counts.max() * 1.25)

    if out_path:
        plt.savefig(out_path)
    return fig
