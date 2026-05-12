from pathlib import Path

import pandas as pd
import numpy as np

from scipy.stats import spearmanr

from loguru import logger

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data/panel/ranked"
)

MODEL_OUTPUT_DIR = (
    PROJECT_ROOT
    / "backtests/optimized"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports/diagnostics"
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
    LOG_DIR / "information_coefficient.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# =========================================================
# LOAD RANKED DATASET
# =========================================================

logger.info(
    "Loading ranked validation dataset"
)

df = pd.read_parquet(
    DATA_DIR
    / "validation_ranked.parquet"
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
    by=["timestamp"],
    inplace=True
)

# =========================================================
# PREDICTION PROXY
# =========================================================

logger.info(
    "Creating prediction proxy"
)

# Use rank percentile as prediction strength
df["prediction_score"] = (
    0.5 * df["momentum_252d_zscore"]
    +
    0.3 * df["momentum_126d_zscore"]
    -
    0.2 * df["volatility_63d_zscore"]
)

# =========================================================
# TARGET VARIABLE
# =========================================================

logger.info(
    "Preparing realized returns"
)

TARGET_COLUMN = "future_return_21d"

# Remove invalid rows
df = df.dropna(

    subset=[
        "prediction_score",
        TARGET_COLUMN
    ]
)

logger.info(
    f"Post-cleaning shape -> "
    f"{df.shape}"
)

# =========================================================
# DAILY IC COMPUTATION
# =========================================================

logger.info(
    "Computing daily IC values"
)

daily_ic_results = []

timestamps = sorted(
    df["timestamp"].unique()
)

for timestamp in timestamps:

    subset = df[
        df["timestamp"] == timestamp
    ]

    # Minimum sample threshold
    if len(subset) < 20:
        continue

    # Spearman rank correlation
    ic, p_value = spearmanr(

        subset["prediction_score"],

        subset[TARGET_COLUMN]
    )

    daily_ic_results.append({

        "timestamp": timestamp,

        "information_coefficient": ic,

        "p_value": p_value,

        "cross_section_size": len(subset)
    })

# =========================================================
# RESULTS DATAFRAME
# =========================================================

ic_df = pd.DataFrame(
    daily_ic_results
)

logger.info(
    f"IC observations -> "
    f"{len(ic_df)}"
)

# =========================================================
# REMOVE NAN IC VALUES
# =========================================================

ic_df.dropna(
    inplace=True
)

ic_df.reset_index(
    drop=True,
    inplace=True
)

# =========================================================
# ROLLING IC
# =========================================================

logger.info(
    "Computing rolling IC"
)

ROLLING_WINDOW = 21

ic_df["rolling_ic"] = (

    ic_df[
        "information_coefficient"
    ]

    .rolling(
        window=ROLLING_WINDOW
    )

    .mean()
)

# =========================================================
# SUMMARY STATISTICS
# =========================================================

mean_ic = (

    ic_df[
        "information_coefficient"
    ].mean()
)

std_ic = (

    ic_df[
        "information_coefficient"
    ].std()
)

information_ratio = (

    mean_ic / std_ic

    if std_ic > 0

    else np.nan
)

positive_ic_rate = (

    ic_df[
        "information_coefficient"
    ] > 0

).mean()

mean_p_value = (
    ic_df["p_value"].mean()
)

# =========================================================
# IC STABILITY
# =========================================================

ic_autocorrelation = (

    ic_df[
        "information_coefficient"
    ]

    .autocorr()
)

# =========================================================
# EXTREME IC ANALYSIS
# =========================================================

top_ic = (

    ic_df[
        "information_coefficient"
    ].max()
)

bottom_ic = (

    ic_df[
        "information_coefficient"
    ].min()
)

# =========================================================
# FINAL LOGGING
# =========================================================

logger.info("====================================")
logger.info("INFORMATION COEFFICIENT SUMMARY")
logger.info("====================================")

logger.info(
    f"Mean IC: {mean_ic:.6f}"
)

logger.info(
    f"IC Std Dev: {std_ic:.6f}"
)

logger.info(
    f"IC Information Ratio: "
    f"{information_ratio:.6f}"
)

logger.info(
    f"Positive IC Rate: "
    f"{positive_ic_rate:.4f}"
)

logger.info(
    f"Mean P-Value: "
    f"{mean_p_value:.6f}"
)

logger.info(
    f"IC Autocorrelation: "
    f"{ic_autocorrelation:.6f}"
)

logger.info(
    f"Best IC: {top_ic:.6f}"
)

logger.info(
    f"Worst IC: {bottom_ic:.6f}"
)

logger.info("====================================")

# =========================================================
# SAVE OUTPUTS
# =========================================================

logger.info(
    "Saving IC diagnostics"
)

ic_df.to_parquet(

    OUTPUT_DIR
    / "daily_information_coefficient.parquet",

    index=False
)

summary_df = pd.DataFrame({

    "metric": [

        "mean_ic",

        "std_ic",

        "information_ratio",

        "positive_ic_rate",

        "mean_p_value",

        "ic_autocorrelation",

        "best_ic",

        "worst_ic"
    ],

    "value": [

        mean_ic,

        std_ic,

        information_ratio,

        positive_ic_rate,

        mean_p_value,

        ic_autocorrelation,

        top_ic,

        bottom_ic
    ]
})

summary_df.to_csv(

    OUTPUT_DIR
    / "information_coefficient_summary.csv",

    index=False
)

logger.success(
    "Saved IC diagnostics"
)

# =========================================================
# FINAL LOGS
# =========================================================

logger.info("====================================")
logger.info("INFORMATION COEFFICIENT COMPLETE")
logger.info("====================================")