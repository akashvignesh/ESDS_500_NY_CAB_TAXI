# 🚕 NYC Yellow Taxi Cab Demand Predictor

A machine learning system that predicts **next-hour taxi ride demand** across all NYC taxi zones. The project covers the full MLOps lifecycle — from raw data ingestion and feature engineering to model training, a live feature/model store, and a real-time Streamlit dashboard.

---

## 📌 Project Overview

| Detail | Value |
|---|---|
| **Course** | ESDS 500 |
| **Dataset** | NYC TLC Yellow Taxi Trip Records |
| **Prediction target** | Number of rides in the next hour per pickup zone |
| **Primary model** | LightGBM (LGBMRegressor) |
| **Feature store / Model registry** | Hopsworks |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     DATA LAYER                          │
│  NYC TLC Parquet Files  ──►  Raw Data  ──►  Processed  │
└─────────────────────────────────┬───────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────┐
│                  FEATURE PIPELINE                       │
│  Time-Series Transformation  ──►  Hopsworks Feature     │
│  (hourly ride counts per zone)      Group / View        │
└─────────────────────────────────┬───────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────┐
│              MODEL TRAINING PIPELINE                    │
│  Feature Engineering  ──►  LightGBM  ──►  Hopsworks    │
│  (lag features, temporal)              Model Registry   │
└─────────────────────────────────┬───────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────┐
│               INFERENCE PIPELINE                        │
│  Load Features  ──►  Load Model  ──►  Predictions  ──► │
│                                     Hopsworks FG        │
└─────────────────────────────────┬───────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────┐
│                  STREAMLIT FRONTEND                     │
│  Interactive NYC Map  │  Zone Selector  │  Top-10 Chart │
└─────────────────────────────────────────────────────────┘
```

---

## 🗂️ Directory Structure

```
ESDS_500_NY_CAB_TAXI/
├── notebooks/              # Step-by-step Jupyter notebooks (EDA → modelling → inference)
│   ├── 01_fetch_data.ipynb
│   ├── 02_validate_and_save.ipynb
│   ├── 03_transform_processed_data_into_ts_data.ipynb
│   ├── 04_transform_ts_data_into_features_and_targets.ipynb
│   ├── 07_baseline_models.ipynb
│   ├── 08_xgboost_model.ipynb
│   ├── 09_lightgbm_model.ipynb
│   ├── 12_load_features_hopsworks.ipynb
│   ├── 13_feature_pipeline.ipynb
│   ├── 14_model_training_pipeline.ipynb
│   ├── 16_inference_pipeline.ipynb
│   ├── 20_Arima_Model.ipynb
│   ├── 22_Prophet_Model.ipynb
│   └── ...
├── src/                    # Core Python modules
│   ├── config.py           # Paths, env vars, Hopsworks config
│   ├── data_utils.py       # Data fetching and transformation helpers
│   ├── feature_pipeline.py # Standalone feature pipeline script
│   ├── inference.py        # Model loading and prediction helpers
│   ├── pipeline_utils.py   # Sklearn pipeline & feature engineering
│   ├── plot_utils.py       # Plotly chart helpers
│   └── experiment_utils.py # MLflow experiment helpers
├── pipelines/              # Production pipeline entry-points
│   ├── feature_pipeline.py (via src)
│   ├── model_training_pipeline.py
│   └── inference_pipeline.py
├── frontend/               # Streamlit dashboard
│   ├── frontend_v2.py      # Main interactive app
│   └── frontend_monitor.py # Monitoring view
├── data/                   # Auto-created at runtime
│   ├── raw/
│   ├── processed/
│   └── transformed/
├── models/                 # Saved model artefacts
├── requirements.txt
└── requirements_feature_pipeline.txt
```

---

## 🧰 Tech Stack

### Data & Storage
| Tool | Purpose |
|---|---|
| **pandas / NumPy** | Data manipulation and numerical computation |
| **PyArrow** | Parquet file I/O (NYC TLC trip records) |
| **GeoPandas** | Geospatial operations on NYC taxi zone shapefiles |
| **Hopsworks** | Managed feature store (feature groups & views) and model registry |
| **Confluent Kafka** | Streaming data ingestion for the feature pipeline |

### Machine Learning
| Tool | Purpose |
|---|---|
| **LightGBM** | Primary gradient-boosting model (`LGBMRegressor`) |
| **XGBoost** | Alternative boosting model (explored in notebooks) |
| **scikit-learn** | Pipeline, `FunctionTransformer`, `BaseEstimator`, MAE evaluation |
| **statsmodels (ARIMA / ARMA)** | Time-series baseline models |
| **Prophet** | Facebook's forecasting library (baseline comparison) |
| **MLflow** | Experiment tracking and run logging |
| **joblib** | Model serialization (`.pkl`) |

### Feature Engineering
- **Lag features** — ride counts from the same zone at 1 – 28 days in the past
- **Rolling average** — average rides over the last 4 weeks (same weekday/hour)
- **Temporal features** — hour of day, day of week

### Frontend & Visualization
| Tool | Purpose |
|---|---|
| **Streamlit** | Interactive web dashboard |
| **Folium / streamlit-folium** | Interactive choropleth map of NYC taxi zones |
| **Plotly** | Time-series prediction charts |
| **Matplotlib / Seaborn** | Static plots in notebooks |
| **Branca** | Color scales for the Folium map |

### Dev & Config
| Tool | Purpose |
|---|---|
| **python-dotenv** | Environment variable management (`.env`) |
| **Jupyter Notebook** | Exploratory data analysis and prototyping |

---

## ⚙️ Setup

### Prerequisites
- Python 3.10+
- A [Hopsworks](https://www.hopsworks.ai/) account (free tier available)

### 1. Clone the repository
```bash
git clone https://github.com/akashvignesh/ESDS_500_NY_CAB_TAXI.git
cd ESDS_500_NY_CAB_TAXI
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
Create a `.env` file in the project root:
```dotenv
HOPSWORKS_API_KEY=your_hopsworks_api_key
HOPSWORKS_PROJECT_NAME=your_project_name
```

---

## 🚀 Running the Pipelines

### Feature Pipeline
Fetches the latest 28 days of NYC TLC data and upserts it into the Hopsworks feature group:
```bash
python src/feature_pipeline.py
```

### Model Training Pipeline
Fetches features from the store, trains/evaluates a LightGBM model, and registers it if it beats the current champion:
```bash
python pipelines/model_training_pipeline.py
```

### Inference Pipeline
Loads the latest model from the registry, generates next-hour predictions for all zones, and writes them back to Hopsworks:
```bash
python pipelines/inference_pipeline.py
```

---

## 🖥️ Running the Dashboard

```bash
streamlit run frontend/frontend_v2.py
```

The dashboard shows:
- 🗺️ **Interactive choropleth map** — colour-coded predicted demand per NYC taxi zone
- 📊 **Prediction statistics** — average, max, and minimum predicted rides
- 🔍 **Zone selector** — filter historical features and predictions for any zone
- 🏆 **Top 10 busiest zones** — ranked by predicted demand with time-series charts

---

## 📓 Notebook Walkthrough

| Notebook | Description |
|---|---|
| `01_fetch_data` | Download raw TLC parquet files |
| `02_validate_and_save` | Clean outliers, validate date ranges, save processed data |
| `03` | Convert processed data to hourly time-series |
| `04–05` | Create lag features and target variable |
| `06` | Exploratory visualizations |
| `07` | Baseline model benchmarks |
| `08–11` | XGBoost and LightGBM modelling & hyperparameter tuning |
| `12–16` | Hopsworks integration — feature loading, training, inference pipelines |
| `18` | MAE tracking over time |
| `19` | Model retraining workflow |
| `20–22` | ARIMA, ARMA, and Prophet time-series baselines |

---

## 🔑 Key Design Decisions

- **Lag-based feature engineering** — the model uses ride counts from the same zone over the past 28 days (672 hourly lags) as input features, capturing weekly seasonality.
- **sklearn Pipeline** — feature engineering (rolling average, temporal features) is wrapped in an sklearn `Pipeline` alongside the LGBMRegressor, ensuring reproducible transforms at inference time.
- **Continuous retraining** — the training pipeline compares the new model's MAE against the registered champion and only promotes the new model if it improves.
- **Hopsworks as the single source of truth** — raw features, transformed features, and predictions all live in Hopsworks, decoupling each pipeline stage.
