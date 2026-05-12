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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports/diagnostics"
)

FEATURE_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/feature_store/meta_features"
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

FEATURE_OUTPUT_DIR.mkdir(
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
    LOG_DIR / "ic_filter.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# =========================================================
# LOAD DATASET
# =========================================================

logger.info(
    "Loading ranked training dataset"
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
# TARGET VARIABLE
# =========================================================

TARGET_COLUMN = "future_return_21d"

# =========================================================
# FEATURE CANDIDATES
# =========================================================

EXCLUDED_COLUMNS = [

    "timestamp",

    "future_return_21d",

    "future_return_rank",

    "cross_sectional_rank_label",

    "triple_barrier_label",

    "canonical_symbol",

    "provider_symbol",

    "sector",

    "country",

    "category",

    "source",

    "asset_type"
]

feature_columns = []

for column in df.columns:

    if column in EXCLUDED_COLUMNS:
        continue

    if pd.api.types.is_numeric_dtype(
        df[column]
    ):

        feature_columns.append(column)

logger.info(
    f"Feature candidates -> "
    f"{len(feature_columns)}"
)

# =========================================================
# IC COMPUTATION
# =========================================================

logger.info(
    "Computing feature IC scores"
)

results = []

timestamps = sorted(
    df["timestamp"].unique()
)

for feature in feature_columns:

    daily_ic_values = []

    for timestamp in timestamps:

        subset = df[
            df["timestamp"] == timestamp
        ]

        subset = subset.dropna(

            subset=[
                feature,
                TARGET_COLUMN
            ]
        )

        # Minimum cross-sectional size
        if len(subset) < 20:
            continue

        try:

            ic, _ = spearmanr(

                subset[feature],

                subset[TARGET_COLUMN]
            )

            if not np.isnan(ic):
                daily_ic_values.append(ic)

        except Exception:
            continue

    # Skip invalid features
    if len(daily_ic_values) == 0:
        continue

    mean_ic = np.mean(
        daily_ic_values
    )

    std_ic = np.std(
        daily_ic_values
    )

    abs_mean_ic = abs(
        mean_ic
    )

    information_ratio = (

        mean_ic / std_ic

        if std_ic > 0

        else np.nan
    )

    positive_ic_rate = (

        np.array(daily_ic_values) > 0

    ).mean()

    results.append({

        "feature": feature,

        "mean_ic": mean_ic,

        "abs_mean_ic": abs_mean_ic,

        "std_ic": std_ic,

        "information_ratio": information_ratio,

        "positive_ic_rate": positive_ic_rate,

        "observations": len(
            daily_ic_values
        )
    })

# =========================================================
# RESULTS DATAFRAME
# =========================================================

results_df = pd.DataFrame(
    results
)

results_df.sort_values(

    by="abs_mean_ic",

    ascending=False,

    inplace=True
)

results_df.reset_index(

    drop=True,

    inplace=True
)

logger.info(
    f"Computed IC for "
    f"{len(results_df)} features"
)

# =========================================================
# FEATURE FILTERING
# =========================================================

IC_THRESHOLD = 0.01

selected_features_df = results_df[

    results_df[
        "abs_mean_ic"
    ] >= IC_THRESHOLD
]

selected_features = (
    selected_features_df[
        "feature"
    ].tolist()
)

removed_features = (
    results_df[
        results_df[
            "abs_mean_ic"
        ] < IC_THRESHOLD
    ]["feature"]

    .tolist()
)

# =========================================================
# FINAL REPORTING
# =========================================================

logger.info("====================================")
logger.info("IC FEATURE FILTER SUMMARY")
logger.info("====================================")

logger.info(
    f"Total Features: "
    f"{len(results_df)}"
)

logger.info(
    f"Selected Features: "
    f"{len(selected_features)}"
)

logger.info(
    f"Removed Features: "
    f"{len(removed_features)}"
)

logger.info(
    f"IC Threshold: "
    f"{IC_THRESHOLD}"
)

logger.info("====================================")

logger.info(
    "\nTop Features:\n"
    f"{results_df.head(20)}"
)

# =========================================================
# SAVE OUTPUTS
# =========================================================

logger.info(
    "Saving IC filter outputs"
)

results_df.to_csv(

    OUTPUT_DIR
    / "feature_ic_scores.csv",

    index=False
)

selected_features_df.to_csv(

    OUTPUT_DIR
    / "selected_features.csv",

    index=False
)

pd.DataFrame({

    "feature": removed_features

}).to_csv(

    OUTPUT_DIR
    / "removed_features.csv",

    index=False
)

# Save selected feature list
with open(

    FEATURE_OUTPUT_DIR
    / "selected_feature_list.txt",

    "w"
) as f:

    for feature in selected_features:

        f.write(
            f"{feature}\n"
        )

logger.success(
    "Saved IC filter outputs"
)

# =========================================================
# FINAL LOGS
# =========================================================

logger.info("====================================")
logger.info("IC FILTER COMPLETE")
logger.info("====================================")