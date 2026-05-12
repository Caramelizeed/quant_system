from pathlib import Path

import pandas as pd
import numpy as np

from loguru import logger

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IC_REPORT_DIR = (
    PROJECT_ROOT
    / "reports/diagnostics"
)

FEATURE_STORE_DIR = (
    PROJECT_ROOT
    / "data/feature_store/meta_features"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports/diagnostics"
)

LOG_DIR = (
    PROJECT_ROOT
    / "logs"
)

# Create directories
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FEATURE_STORE_DIR.mkdir(
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
    LOG_DIR / "stability_filter.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# =========================================================
# LOAD IC RESULTS
# =========================================================

logger.info(
    "Loading IC feature scores"
)

ic_df = pd.read_csv(
    IC_REPORT_DIR
    / "feature_ic_scores.csv"
)

logger.info(
    f"Loaded IC scores -> "
    f"{ic_df.shape}"
)

# =========================================================
# BASIC CLEANING
# =========================================================

logger.info(
    "Cleaning IC dataframe"
)

ic_df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

ic_df.dropna(
    inplace=True
)

logger.info(
    f"Post-cleaning shape -> "
    f"{ic_df.shape}"
)

# =========================================================
# STABILITY METRICS
# =========================================================

logger.info(
    "Computing stability metrics"
)

# Stability score:
# higher mean IC
# lower IC variance

ic_df["stability_score"] = (

    ic_df["abs_mean_ic"]

    /

    (
        ic_df["std_ic"]
        + 1e-8
    )
)

# Penalize inconsistent directionality
ic_df["directional_consistency"] = (

    (
        ic_df["positive_ic_rate"]
        - 0.5
    ).abs()

    * 2
)

# Combined robustness metric
ic_df["robustness_score"] = (

    0.7
    * ic_df["stability_score"]

    +

    0.3
    * ic_df["directional_consistency"]
)

# =========================================================
# SORT FEATURES
# =========================================================

logger.info(
    "Ranking stable features"
)

ic_df.sort_values(

    by="robustness_score",

    ascending=False,

    inplace=True
)

ic_df.reset_index(

    drop=True,

    inplace=True
)

# =========================================================
# STABILITY FILTERING
# =========================================================

ROBUSTNESS_THRESHOLD = 0.12

stable_features_df = ic_df[

    ic_df[
        "robustness_score"
    ] >= ROBUSTNESS_THRESHOLD
]

unstable_features_df = ic_df[

    ic_df[
        "robustness_score"
    ] < ROBUSTNESS_THRESHOLD
]

stable_features = (
    stable_features_df[
        "feature"
    ].tolist()
)

unstable_features = (
    unstable_features_df[
        "feature"
    ].tolist()
)

# =========================================================
# FINAL REPORTING
# =========================================================

logger.info("====================================")
logger.info("STABILITY FILTER SUMMARY")
logger.info("====================================")

logger.info(
    f"Total Features: "
    f"{len(ic_df)}"
)

logger.info(
    f"Stable Features: "
    f"{len(stable_features)}"
)

logger.info(
    f"Removed Features: "
    f"{len(unstable_features)}"
)

logger.info(
    f"Robustness Threshold: "
    f"{ROBUSTNESS_THRESHOLD}"
)

logger.info("====================================")

logger.info(
    "\nTop Stable Features:\n"
    f"{stable_features_df.head(20)}"
)

# =========================================================
# SAVE OUTPUTS
# =========================================================

logger.info(
    "Saving stability outputs"
)

stable_features_df.to_csv(

    OUTPUT_DIR
    / "stable_features.csv",

    index=False
)

unstable_features_df.to_csv(

    OUTPUT_DIR
    / "unstable_features.csv",

    index=False
)

# Save stable feature list
with open(

    FEATURE_STORE_DIR
    / "stable_feature_list.txt",

    "w"
) as f:

    for feature in stable_features:

        f.write(
            f"{feature}\n"
        )

logger.success(
    "Saved stability outputs"
)

# =========================================================
# FINAL LOGGING
# =========================================================

logger.info("====================================")
logger.info("STABILITY FILTER COMPLETE")
logger.info("====================================")