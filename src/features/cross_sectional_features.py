from pathlib import Path
import pandas as pd
from loguru import logger

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PANEL_DIR = (
    PROJECT_ROOT / "data/panel"
)

ENHANCED_PANEL_DIR = (
    PROJECT_ROOT / "data/panel/enhanced"
)

LOG_DIR = (
    PROJECT_ROOT / "logs"
)

# Create directories
ENHANCED_PANEL_DIR.mkdir(
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
    LOG_DIR / "cross_sectional_features.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# =========================================================
# FEATURE LIST
# =========================================================

RANK_FEATURES = [

    "return_21d",
    "return_63d",

    "momentum_21d",
    "momentum_63d",
    "momentum_126d",
    "momentum_252d",

    "volatility_21d",
    "volatility_63d",

    "relative_volume_20",

    "dist_sma_20",
    "dist_sma_50",
    "dist_sma_200",

    "daily_range"
]

# =========================================================
# CROSS SECTIONAL FUNCTION
# =========================================================

def build_cross_sectional_features(
    input_file: str,
    output_file: str
):

    # =====================================================
    # LOAD PANEL
    # =====================================================

    logger.info(
        f"Loading {input_file}"
    )

    df = pd.read_parquet(
        PANEL_DIR / input_file
    )

    logger.info(
        f"Loaded shape -> {df.shape}"
    )

    # =====================================================
    # TIMESTAMP STANDARDIZATION
    # =====================================================

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    # =====================================================
    # SORT DATA
    # =====================================================

    df.sort_values(
        by=["timestamp", "canonical_symbol"],
        inplace=True
    )

    df.reset_index(
        drop=True,
        inplace=True
    )

    # =====================================================
    # CROSS-SECTIONAL RANKS
    # =====================================================

    logger.info(
        "Generating percentile rank features"
    )

    for feature in RANK_FEATURES:

        rank_column = (
            f"{feature}_rank"
        )

        logger.info(
            f"Creating {rank_column}"
        )

        # ---------------------------------------------
        # PERCENTILE RANK PER TIMESTAMP
        # ---------------------------------------------

        df[rank_column] = (

            df.groupby("timestamp")[feature]

            .rank(
                method="average",
                pct=True
            )

        )

    # =====================================================
    # Z-SCORE FEATURES
    # =====================================================

    logger.info(
        "Generating z-score features"
    )

    for feature in RANK_FEATURES:

        zscore_column = (
            f"{feature}_zscore"
        )

        logger.info(
            f"Creating {zscore_column}"
        )

        grouped = df.groupby(
            "timestamp"
        )[feature]

        cross_mean = grouped.transform(
            "mean"
        )

        cross_std = grouped.transform(
            "std"
        )

        df[zscore_column] = (

            (df[feature] - cross_mean)

            / cross_std

        )

    # =====================================================
    # RELATIVE MOMENTUM SPREAD
    # =====================================================

    logger.info(
        "Creating momentum spread features"
    )

    df["momentum_spread_21_63"] = (

        df["momentum_21d"]

        - df["momentum_63d"]

    )

    df["momentum_spread_63_252"] = (

        df["momentum_63d"]

        - df["momentum_252d"]

    )

    # =====================================================
    # VOLATILITY ADJUSTED MOMENTUM
    # =====================================================

    logger.info(
        "Creating volatility adjusted momentum"
    )

    df["vol_adjusted_momentum"] = (

        df["momentum_63d"]

        / df["volatility_21d"]

    )

    # =====================================================
    # CLEAN INFINITE VALUES
    # =====================================================

    logger.info(
        "Cleaning infinite values"
    )

    df.replace(
        [float("inf"), float("-inf")],
        pd.NA,
        inplace=True
    )

    # =====================================================
    # DROP NAN ROWS
    # =====================================================

    before_rows = len(df)

    df.dropna(inplace=True)

    after_rows = len(df)

    removed_rows = (
        before_rows - after_rows
    )

    logger.info(
        f"Removed {removed_rows} rows"
    )

    # =====================================================
    # SAVE ENHANCED PANEL
    # =====================================================

    output_path = (
        ENHANCED_PANEL_DIR / output_file
    )

    df.to_parquet(
        output_path,
        index=False
    )

    logger.success(
        f"Saved enhanced panel -> "
        f"{output_path}"
    )

    logger.info(
        f"Final shape -> {df.shape}"
    )

# =========================================================
# BUILD ENHANCED PANELS
# =========================================================

build_cross_sectional_features(

    input_file="train_panel.parquet",

    output_file="train_panel_enhanced.parquet"
)

build_cross_sectional_features(

    input_file="validation_panel.parquet",

    output_file="validation_panel_enhanced.parquet"
)

build_cross_sectional_features(

    input_file="test_panel.parquet",

    output_file="test_panel_enhanced.parquet"
)

# =========================================================
# FINAL SUMMARY
# =========================================================

logger.info("====================================")
logger.info("CROSS SECTIONAL FEATURE BUILD COMPLETE")
logger.info("====================================")