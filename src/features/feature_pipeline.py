from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEAN_DATA_DIR = (
    PROJECT_ROOT / "data/interim/cleaned"
)

FEATURE_STORE_DIR = (
    PROJECT_ROOT / "data/feature_store"
)

LOG_DIR = PROJECT_ROOT / "logs"

# Create directories
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
    LOG_DIR / "feature_pipeline.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# =========================================================
# FEATURE ENGINEERING FUNCTION
# =========================================================

def generate_features(asset_dir: Path):

    parquet_path = asset_dir / "daily.parquet"

    asset_name = asset_dir.name

    if not parquet_path.exists():

        logger.warning(
            f"{asset_name}: parquet missing"
        )

        return False

    try:

        # =================================================
        # LOAD CLEAN DATA
        # =================================================

        df = pd.read_parquet(parquet_path)

        logger.info(
            f"Generating features for {asset_name}"
        )

        # =================================================
        # SORT CHRONOLOGICALLY
        # =================================================

        df.sort_values(
            by="timestamp",
            inplace=True
        )

        df.reset_index(
            drop=True,
            inplace=True
        )

        # =================================================
        # PRICE SERIES
        # =================================================

        close = df["adj_close"]

        volume = df["volume"]

        # =================================================
        # RETURNS
        # =================================================

        # 1-day return
        df["return_1d"] = close.pct_change(1)

        # 5-day return
        df["return_5d"] = close.pct_change(5)

        # 21-day return
        df["return_21d"] = close.pct_change(21)

        # 63-day return
        df["return_63d"] = close.pct_change(63)

        # =================================================
        # LOG RETURNS
        # =================================================

        df["log_return_1d"] = np.log(
            close / close.shift(1)
        )

        # =================================================
        # VOLATILITY FEATURES
        # =================================================

        # 21-day annualized realized volatility
        df["volatility_21d"] = (

            df["log_return_1d"]
            .rolling(21)
            .std()

            * np.sqrt(252)

        )

        # 63-day annualized volatility
        df["volatility_63d"] = (

            df["log_return_1d"]
            .rolling(63)
            .std()

            * np.sqrt(252)

        )

        # =================================================
        # MOMENTUM FEATURES
        # =================================================

        df["momentum_21d"] = (
            close / close.shift(21)
        ) - 1

        df["momentum_63d"] = (
            close / close.shift(63)
        ) - 1

        df["momentum_126d"] = (
            close / close.shift(126)
        ) - 1

        df["momentum_252d"] = (
            close / close.shift(252)
        ) - 1

        # =================================================
        # MOVING AVERAGES
        # =================================================

        df["sma_20"] = (
            close.rolling(20).mean()
        )

        df["sma_50"] = (
            close.rolling(50).mean()
        )

        df["sma_200"] = (
            close.rolling(200).mean()
        )

        # =================================================
        # DISTANCE FROM MOVING AVERAGES
        # =================================================

        df["dist_sma_20"] = (
            (close - df["sma_20"])
            / df["sma_20"]
        )

        df["dist_sma_50"] = (
            (close - df["sma_50"])
            / df["sma_50"]
        )

        df["dist_sma_200"] = (
            (close - df["sma_200"])
            / df["sma_200"]
        )

        # =================================================
        # VOLUME FEATURES
        # =================================================

        # Rolling average volume
        df["avg_volume_20"] = (
            volume.rolling(20).mean()
        )

        # Relative volume
        df["relative_volume_20"] = (
            volume / df["avg_volume_20"]
        )

        # Volume z-score
        volume_mean = (
            volume.rolling(20).mean()
        )

        volume_std = (
            volume.rolling(20).std()
        )

        df["volume_zscore_20"] = (
            (volume - volume_mean)
            / volume_std
        )

        # =================================================
        # TREND FEATURES
        # =================================================

        # SMA crossover spread
        df["sma_spread_20_50"] = (
            (df["sma_20"] - df["sma_50"])
            / df["sma_50"]
        )

        df["sma_spread_50_200"] = (
            (df["sma_50"] - df["sma_200"])
            / df["sma_200"]
        )

        # =================================================
        # RANGE / INTRADAY FEATURES
        # =================================================

        df["daily_range"] = (
            (df["high"] - df["low"])
            / df["close"]
        )

        df["open_close_gap"] = (
            (df["open"] - df["close"].shift(1))
            / df["close"].shift(1)
        )

        # =================================================
        # DROP EARLY NAN WARMUP ROWS
        # =================================================

        df.dropna(inplace=True)

        df.reset_index(
            drop=True,
            inplace=True
        )

        # =================================================
        # OUTPUT DIRECTORY
        # =================================================

        output_dir = (
            FEATURE_STORE_DIR / asset_name
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            output_dir / "features.parquet"
        )

        # =================================================
        # SAVE FEATURES
        # =================================================

        df.to_parquet(
            output_path,
            index=False
        )

        logger.success(
            f"{asset_name}: generated "
            f"{len(df.columns)} features | "
            f"{len(df)} rows"
        )

        return True

    except Exception as e:

        logger.exception(
            f"{asset_name}: feature generation failed -> "
            f"{str(e)}"
        )

        return False

# =========================================================
# MAIN EXECUTION
# =========================================================

asset_dirs = [

    path for path in CLEAN_DATA_DIR.iterdir()
    if path.is_dir()

]

logger.info(
    f"Found {len(asset_dirs)} assets"
)

successful = 0
failed = 0

for asset_dir in asset_dirs:

    success = generate_features(asset_dir)

    if success:

        successful += 1

    else:

        failed += 1

# =========================================================
# FINAL SUMMARY
# =========================================================

logger.info("====================================")
logger.info("FEATURE GENERATION COMPLETE")
logger.info(f"Successful: {successful}")
logger.info(f"Failed: {failed}")
logger.info("====================================")


