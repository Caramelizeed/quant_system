from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.mixture import GaussianMixture

from sklearn.preprocessing import StandardScaler

from loguru import logger

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = (
    PROJECT_ROOT / "data/panel/ranked"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/feature_store/regimes"
)

LOG_DIR = (
    PROJECT_ROOT / "logs"
)

# Create directories
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =========================================================
# LOGGER CONFIGURATION
# =========================================================

logger.add(
    LOG_DIR / "hmm_model.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# =========================================================
# CONFIGURATION
# =========================================================

N_REGIMES = 3

RANDOM_STATE = 42

MAX_ITERATIONS = 500

# =========================================================
# LOAD PANEL DATA
# =========================================================

logger.info(
    "Loading ranked panel dataset"
)

df = pd.read_parquet(
    DATA_DIR
    / "train_ranked.parquet"
)

logger.info(
    f"Loaded shape -> {df.shape}"
)

# =========================================================
# TIMESTAMP STANDARDIZATION
# =========================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df.sort_values(
    by="timestamp",
    inplace=True
)

# =========================================================
# MARKET AGGREGATION
# =========================================================

logger.info(
    "Building market-level features"
)

market_df = (

    df.groupby("timestamp")

    .agg({

        "return_5d": "mean",

        "return_21d": "mean",

        "volatility_21d": "mean",

        "volatility_63d": "mean",

        "daily_range": "mean",

        "volume_zscore_20": "mean"

    })

    .reset_index()
)

logger.info(
    f"Market dataframe shape -> "
    f"{market_df.shape}"
)

# =========================================================
# FEATURE MATRIX
# =========================================================

FEATURE_COLUMNS = [

    "return_5d",

    "return_21d",

    "volatility_21d",

    "volatility_63d",

    "daily_range",

    "volume_zscore_20"
]

X = market_df[
    FEATURE_COLUMNS
].copy()

# =========================================================
# HANDLE NAN / INF
# =========================================================

X.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

X.dropna(inplace=True)

market_df = market_df.loc[
    X.index
].copy()

market_df.reset_index(
    drop=True,
    inplace=True
)

X.reset_index(
    drop=True,
    inplace=True
)

logger.info(
    f"Clean feature matrix shape -> "
    f"{X.shape}"
)

# =========================================================
# FEATURE SCALING
# =========================================================

logger.info(
    "Scaling features"
)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)

# =========================================================
# GAUSSIAN MIXTURE MODEL
# =========================================================

logger.info(
    "Training Gaussian Mixture Model"
)

model = GaussianMixture(

    n_components=N_REGIMES,

    covariance_type="full",

    max_iter=MAX_ITERATIONS,

    random_state=RANDOM_STATE
)

model.fit(
    X_scaled
)

logger.success(
    "Gaussian Mixture training complete"
)

# =========================================================
# REGIME PREDICTIONS
# =========================================================

logger.info(
    "Generating regime predictions"
)

regimes = model.predict(
    X_scaled
)

regime_probabilities = (
    model.predict_proba(
        X_scaled
    )
)

market_df["regime"] = regimes

# =========================================================
# REGIME PROBABILITIES
# =========================================================

for i in range(N_REGIMES):

    market_df[
        f"regime_probability_{i}"
    ] = regime_probabilities[:, i]

# =========================================================
# REGIME STATISTICS
# =========================================================

logger.info(
    "Computing regime statistics"
)

regime_summary = (

    market_df

    .groupby("regime")

    .agg({

        "return_5d": "mean",

        "return_21d": "mean",

        "volatility_21d": "mean",

        "volatility_63d": "mean",

        "daily_range": "mean",

        "volume_zscore_20": "mean"

    })
)

logger.info("====================================")
logger.info("REGIME SUMMARY")
logger.info("====================================")

logger.info(
    f"\n{regime_summary}"
)

logger.info("====================================")

# =========================================================
# SAVE OUTPUTS
# =========================================================

logger.info(
    "Saving regime outputs"
)

market_df.to_parquet(

    OUTPUT_DIR
    / "market_regimes.parquet",

    index=False
)

regime_summary.to_csv(

    OUTPUT_DIR
    / "regime_summary.csv"
)

logger.success(
    "Saved regime outputs"
)

# =========================================================
# FINAL LOGS
# =========================================================

logger.info("====================================")
logger.info("REGIME MODEL COMPLETE")
logger.info("====================================")