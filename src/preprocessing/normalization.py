from pathlib import Path
import pandas as pd
from loguru import logger

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data/raw/yfinance"

CLEAN_DATA_DIR = (
    PROJECT_ROOT / "data/interim/cleaned"
)

LOG_DIR = PROJECT_ROOT / "logs"

# Create directories
CLEAN_DATA_DIR.mkdir(
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
    LOG_DIR / "normalization.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# =========================================================
# FINAL COLUMN ORDER
# =========================================================

FINAL_COLUMN_ORDER = [

    "timestamp",

    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",

    "canonical_symbol",
    "provider_symbol",

    "asset_type",
    "country",
    "sector",
    "category",

    "source"
]

# =========================================================
# NORMALIZATION FUNCTION
# =========================================================

def normalize_asset(asset_dir: Path):

    parquet_path = asset_dir / "daily.parquet"

    asset_name = asset_dir.name

    if not parquet_path.exists():

        logger.warning(
            f"{asset_name}: parquet missing"
        )

        return False

    try:

        # =================================================
        # LOAD DATA
        # =================================================

        df = pd.read_parquet(parquet_path)

        logger.info(
            f"Normalizing {asset_name}"
        )

        # =================================================
        # TIMESTAMP STANDARDIZATION
        # =================================================

        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        )

        # =================================================
        # REMOVE DUPLICATES
        # =================================================

        before = len(df)

        df.drop_duplicates(
            subset=["timestamp"],
            inplace=True
        )

        after = len(df)

        duplicates_removed = before - after

        if duplicates_removed > 0:

            logger.warning(
                f"{asset_name}: removed "
                f"{duplicates_removed} duplicates"
            )

        # =================================================
        # SORT CHRONOLOGICALLY
        # =================================================

        df.sort_values(
            by="timestamp",
            inplace=True
        )

        # =================================================
        # RESET INDEX
        # =================================================

        df.reset_index(
            drop=True,
            inplace=True
        )

        # =================================================
        # STANDARDIZE NUMERIC DTYPES
        # =================================================

        float_columns = [
            "open",
            "high",
            "low",
            "close",
            "adj_close"
        ]

        for col in float_columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).astype("float64")

        df["volume"] = pd.to_numeric(
            df["volume"],
            errors="coerce"
        ).fillna(0).astype("int64")

        # =================================================
        # REMOVE INVALID PRICE ROWS
        # =================================================

        df = df[

            (df["open"] > 0)
            & (df["high"] > 0)
            & (df["low"] > 0)
            & (df["close"] > 0)
            & (df["adj_close"] > 0)

        ]

        # =================================================
        # REMOVE INVALID HIGH/LOW ROWS
        # =================================================

        df = df[
            df["high"] >= df["low"]
        ]

        # =================================================
        # ENFORCE COLUMN ORDER
        # =================================================

        existing_columns = [
            col for col in FINAL_COLUMN_ORDER
            if col in df.columns
        ]

        df = df[existing_columns]

        # =================================================
        # OUTPUT DIRECTORY
        # =================================================

        output_dir = (
            CLEAN_DATA_DIR / asset_name
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            output_dir / "daily.parquet"
        )

        # =================================================
        # SAVE CLEAN DATA
        # =================================================

        df.to_parquet(
            output_path,
            index=False
        )

        logger.success(
            f"{asset_name}: normalized "
            f"{len(df)} rows"
        )

        return True

    except Exception as e:

        logger.exception(
            f"{asset_name}: normalization failed -> "
            f"{str(e)}"
        )

        return False

# =========================================================
# MAIN EXECUTION
# =========================================================

asset_dirs = [

    path for path in RAW_DATA_DIR.iterdir()
    if path.is_dir()

]

logger.info(
    f"Found {len(asset_dirs)} assets"
)

successful = 0
failed = 0

for asset_dir in asset_dirs:

    success = normalize_asset(asset_dir)

    if success:

        successful += 1

    else:

        failed += 1

# =========================================================
# FINAL SUMMARY
# =========================================================

logger.info("====================================")
logger.info("NORMALIZATION COMPLETE")
logger.info(f"Successful: {successful}")
logger.info(f"Failed: {failed}")
logger.info("====================================")