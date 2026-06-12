# Engine Performance Degradation Forecasting
**Remaining Useful Life (RUL) Prediction - NASA C-MAPSS Turbofan Dataset**

> Portfolio project by **Utkarsh Chaudhari**
> Targeting MRO data analytics

---

## Problem Statement

Unscheduled engine removals (UER) cost airlines between **$500K-$2M per event** in direct MRO costs  and AOG (Aircraft on Ground) disruption. Predictive maintenance - knowing how many flight cycles and engine has left before a shop visit - allows operators to schedule maintenance proactively, reducing UERs and optimising engine utilisation.

This project build an end-to-end **Remaining Useful Life (RUL) prediction pipeline** on the industry-standard NASA C-MAPSS benchmark dataset, producing cycle-level predictions with uncertainty quantification and a fleet-level inspection scheduling dashboard.

---

## Results

| Metric | LightGBM | XGBoost | **Ensemble** |
|---|---|---|---|
| **Test RMSE (cycles)** | 19.01 | 19.71 | **19.18** |
| **Test MAE (cycles)** | - | - | **13.11**|
| **NASA PHM Score** | - | - | **725.6** |
| Late predcition rate (>15 cyc) | - | - | **9%** |

**Context:** Literature RMSE on FD001 ranges from ~14 (LSTM/deep learning) to ~25 (classical ML). This GBM ensemble achieves copetitive performance without sequence modelling, using pure feature engineering.

The **NASA PHM Score** is an asymmetric metric used in the 2008 PHM Challenge that penalises late predictions ~2x higher than early ones - reflecting the real cost of missed maintenance window.

---

## Dataset

**NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)**
- Simulated turbofan engine run-to-failure data, 21 sensors, 3 operational settings
- FD001 subset: 100 training engines (run to failure), 100 test engines (snapshot beforre failure)
- Reference: Saxena et al. (2008), Prognostics and Health Management (PHM) Conference

**Sensor mapping (FD001 - key channels):**

| Sensor | Physical parameter | MRO relevance |
| --- | --- | --- |
| s2 | Total temperature at fan inlet (T2) | Fan health |
| s3 | Total temperature at LPC outlet (T25) | LPC degradation |
| s4 | Total temperature at HPC outlet (T3) | HPC degradation |
| s7 | Total temperature at HPT outlet (T48) | **EGT margin proxy** |
| s11 | Static pressure at HPC outlet (Ps3) | Compressor efficiency |
| s12 | Fuel flow ratio (Wf) | Fuel burn efficiency |

---

## Methodology

### Feature Engineering

Beyond raw sensor readings, three layers of features were engineered:

**1. Rolling statistics (windows: 5, 15 cycles)**
Captures degradation *trend* rather than point-in-time noise. Rolling std captures variability increases - an early-degradation signal.

**2. EGT margin proxy**
`egt_margin = max(s7 per engine) - current s7`
Mirrors how real MRO operators track EGT margin (difference between observed EGT and engine's life-limited value) as the primary shop-visit trigger.

**3. Degradation acceleration flags**
`s4_drop = max(s4) - current s4`
`s7_drop = max(s7) - current s7`
Captures cumulative thermal wear - engines with large drops are further into degradation regardless of absolute sensor value.

### Model Architecture

```
Training data (full run-to-failure sequences)
        ↓
Feature engineering (74 features per cycle-snapshot)
        ↓
Min-Max normalisation (fitted on training data only)
        ↓
┌─────────────────┐    ┌─────────────────┐
│   LightGBM      │    │    XGBoost      │
│  L1 objective   │    │  MSE objective  │
│  600 estimators │    │  600 estimators │
└────────┬────────┘    └────────┬────────┘
         └──────────┬───────────┘
                    ↓
            50/50 Ensemble average
                    ↓
         RUL prediction + uncertainty band
                    ↓
         Fleet inspection scheduling
```

**Piece-wise linear RUL target (training label):**
Engines are assumed fully healthy until 125 cycles before failure. RUL labels are capped at 125 - prevents the model from trying to predict absolute age of mew engines, which is unknowable and irrelevant to MRO scheduling.

## Uncertainity Quantification

An 80% uncertainty band is generated via **bootstrap feature dropour**: 50 stochastic forward passes randomly zero out 15% of features, simulating epistemic uncertainty. This is a practical approximation - production deployments should use quantile regression (LGB `objective='quantile'`) or conformal prediction for rigorous coverage guarantees.

---

## Key Visulisations

### 1. Sensor Degradation Trajectories
`outputs/fig_sensor_degradation.png`
Shows how s2, s4, s7, s11 drift across 6 representative engines. Characteristic pattern: near-monotonic drift toward failure, with varying degradation rates depending on initial wear and operating profile.

### 2. EGT Margin Decline
`outputs/fig_egt_margin.png`
The s7 proxy EGT margin declining from maximum toward zero. In real ops, an EGT margin below ~15°C triggers a shop visit planning event regardless of calendar schedule.

### 3. Four-Panel Prediction Analysis
`outputs/fig_predictions.png`
- **A** - Predicted vs true RUL scatter, flagging late predictions (red triangles)
- **B** - Error distribution (conservative = left, late = right)
- **C** - Uncertainty band over all test engines sorted by true RUL
- **D** - Late prediction rate by RUL bin (risk assessment by fleet segment)

### 4. SHAP Feature Importance
`outputs/fig_shap.png`
- **Bar chart**: Top 15 features by mean |SHAP value|
- **Beeswarm**: Direction of feature effect (high s7_drop -> lower predicted RUL)

### 5. Fleet Inspection Scheduling Dashboard
`outputs/fig_inspection_scchedule.png`
Applies a 20-cycle safety buffer and categories engines into SAFE / SCHEDULE / IMMEDIATE - the output format an MRO planning team would consume.

---

## Project Structure

```
engine_rul/
├── data/
│   ├── train_FD001.txt          NASA C-MAPSS training set (100 engines)
│   ├── test_FD001.txt           Test snapshots (100 engines)
│   └── RUL_FD001.txt            Ground truth RUL for test set
├── src/
│   ├── data_loader.py           Data loading, feature engineering, preprocessing
│   ├── models.py                LightGBM + XGBoost training, PHM score, uncertainty
│   ├── visualisation.py         All plotting functions
│   └── train_evaluate.py        Main pipeline (run this)
├── outputs/
│   ├── results_FD001.csv        Per-engine predictions + late flags
│   ├── metrics_summary.txt      RMSE, PHM score, safety metrics
│   ├── fig_*.png                All figures
│   └── *.pkl                    Serialised models
├── tests/
│   └── test_pipeline.py         Unit tests for feature engineering and metrics
├── requirements.txt
└── README.md
```

---

## Reproducing Results

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download CMAPSS data (FD001 used by default)
# Data is ~4MB total — already included in data/ directory
# Original source: NASA Prognostics Data Repository
# https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

# 3. Run the full pipeline
python src/train_evaluate.py

# 4. Outputs written to outputs/
```

---

## MRO Industry Context

This project mirrors the analytical workflow at MRO predictive maintenance teams:

| Step | This project | Real MRO equivalent |
|---|---|---|
| Data ingestion | CMAPSS txt files | ACARS/QAR downloads, AMOS exports |
| EGT margin feature | s7 proxy | SAGE / airline-specific EGT tracking |
| RUL prediction | LGB + XGB | Various vendor black-boxes (Pratt FAST, CFM LEAP MRO Analytics) |
| Uncertainty bands | Bootstrap dropout | Monte Carlo or conformalized intervals |
| Inspection scheduling | Status dashboard | AMOS / SAP PM work order triggers |
| Late prediction flag | >15 cycle error | Missed maintenance window KPI |

---

## Extensions (Roadmap)

- [ ] **FD002 / FD004**: Multi-condition datasets (6 operating regimes — requires operating condition normalisation before feature engineering)
- [ ] **LSTM baseline**: Sequence model for direct comparison — expected ~15 cycle RMSE
- [ ] **Quantile regression**: Replace bootstrap dropout with LGB quantile objective for calibrated coverage
- [ ] **Conformal prediction**: Post-hoc calibration for distribution-free coverage guarantees
- [ ] **Real sensor data**: Apply pipeline to OpenSky Network ADS-B data with derived performance metrics

---

## Dependencies

```
lightgbm>=4.0
xgboost>=2.0
shap>=0.44
scikit-learn>=1.4
pandas>=2.0
numpy>=1.26
matplotlib>=3.8
seaborn>=0.13
plotly>=5.18
scipy>=1.12
```

---

## References

1. Saxena, A., & Goebel, K. (2008). *Turbofan Engine Degradation Simulation Data Set.* NASA Ames Prognostics Data Repository.
2. Heimes, F.O. (2008). Recurrent neural networks for remaining useful life estimation. *PHM Conference.*
3. Deloitte A&D Outlook (2026). *MRO aftermarket growth projections 2026–2035.*
4. SHAP: Lundberg & Lee (2017). *A unified approach to interpreting model predictions.* NeurIPS.

---

*Utkarsh Chaudhari | utkarsh410@gmail.com | [LinkedIn](https://linkedin.com) | [GitHub](https://github.com)*













































