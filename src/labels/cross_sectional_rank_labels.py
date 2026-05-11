from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    PROJECT_ROOT / "data/panel/enhanced"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "data/panel/ranked"
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
    LOG_DIR / "cross_sectional_rank_labels.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# =========================================================
# CONFIGURATION
# =========================================================

FORWARD_HORIZON = 21

TOP_PERCENTILE = 0.80

BOTTOM_PERCENTILE = 0.20

# =========================================================
# LABEL FUNCTION
# =========================================================

def create_rank_labels(
    input_file: str,
    output_file: str
):

    logger.info(
        f"Loading {input_file}"
    )

    df = pd.read_parquet(
        INPUT_DIR / input_file
    )

    logger.info(
        f"Initial shape -> {df.shape}"
    )

    # =====================================================
    # TIMESTAMP STANDARDIZATION
    # =====================================================

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    # =====================================================
    # SORTING
    # =====================================================

    df.sort_values(

        by=["canonical_symbol", "timestamp"],

        inplace=True
    )

    df.reset_index(
        drop=True,
        inplace=True
    )

    # =====================================================
    # FUTURE RETURNS
    # =====================================================

    logger.info(
        "Creating future returns"
    )

    df["future_close"] = (

        df.groupby("canonical_symbol")["close"]

        .shift(-FORWARD_HORIZON)

    )

    df["future_return_21d"] = (

        (
            df["future_close"]

            / df["close"]
        )

        - 1.0

    )

    # =====================================================
    # DROP NAN FUTURE RETURNS
    # =====================================================

    before_rows = len(df)

    df.dropna(
        subset=["future_return_21d"],
        inplace=True
    )

    after_rows = len(df)

    logger.info(
        f"Removed {before_rows - after_rows} rows"
    )

    # =====================================================
    # CROSS-SECTIONAL RANKS
    # =====================================================

    logger.info(
        "Creating cross-sectional future return ranks"
    )

    df["future_return_rank"] = (

        df.groupby("timestamp")["future_return_21d"]

        .rank(
            method="average",
            pct=True
        )

    )

    # =====================================================
    # LABEL GENERATION
    # =====================================================

    logger.info(
        "Generating rank labels"
    )

    conditions = [

        (
            df["future_return_rank"]
            >= TOP_PERCENTILE
        ),

        (
            df["future_return_rank"]
            <= BOTTOM_PERCENTILE
        )

    ]

    choices = [
        1,
        -1
    ]

    df["cross_sectional_rank_label"] = np.select(

        conditions,

        choices,

        default=0

    )

    # =====================================================
    # LABEL DISTRIBUTION
    # =====================================================

    distribution = (

        df["cross_sectional_rank_label"]

        .value_counts(normalize=True)

    )

    logger.info(
        f"\nLabel Distribution:\n"
        f"{distribution}"
    )

    # =====================================================
    # CLEANUP
    # =====================================================

    df.drop(
        columns=["future_close"],
        inplace=True
    )

    # =====================================================
    # SAVE DATASET
    # =====================================================

    output_path = (
        OUTPUT_DIR / output_file
    )

    df.to_parquet(
        output_path,
        index=False
    )

    logger.success(
        f"Saved ranked dataset -> "
        f"{output_path}"
    )

    logger.info(
        f"Final shape -> {df.shape}"
    )

# =========================================================
# BUILD DATASETS
# =========================================================

create_rank_labels(

    input_file="train_panel_enhanced.parquet",

    output_file="train_ranked.parquet"
)

create_rank_labels(

    input_file="validation_panel_enhanced.parquet",

    output_file="validation_ranked.parquet"
)

create_rank_labels(

    input_file="test_panel_enhanced.parquet",

    output_file="test_ranked.parquet"
)

# =========================================================
# FINAL SUMMARY
# =========================================================

logger.info("====================================")
logger.info("CROSS SECTIONAL RANK LABELING COMPLETE")
logger.info("====================================")