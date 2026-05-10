from pathlib import Path
import pandas as pd
from loguru import logger

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data/raw/yfinance"

REPORT_DIR = PROJECT_ROOT / "reports/validation"

LOG_DIR = PROJECT_ROOT / "logs"

# Create directories
REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# LOGGER
# =========================================================

logger.add(
    LOG_DIR / "schema_validation.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# =========================================================
# REQUIRED COLUMNS
# =========================================================

REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume"
]

# =========================================================
# VALIDATION RESULTS
# =========================================================

validation_results = []

# =========================================================
# VALIDATION FUNCTION
# =========================================================

def validate_asset(asset_dir: Path):

    parquet_path = asset_dir / "daily.parquet"

    asset_name = asset_dir.name

    if not parquet_path.exists():

        logger.warning(
            f"{asset_name}: parquet file missing"
        )

        return

    try:

        # =================================================
        # LOAD DATA
        # =================================================

        df = pd.read_parquet(parquet_path)

        logger.info(
            f"Validating {asset_name}"
        )

        # =================================================
        # BASIC METRICS
        # =================================================

        row_count = len(df)

        # =================================================
        # REQUIRED COLUMN CHECK
        # =================================================

        missing_columns = [
            col for col in REQUIRED_COLUMNS
            if col not in df.columns
        ]

        # =================================================
        # DUPLICATE CHECK
        # =================================================

        duplicate_count = df[
            "timestamp"
        ].duplicated().sum()

        # =================================================
        # NULL CHECKS
        # =================================================

        total_nulls = df[
            REQUIRED_COLUMNS
        ].isnull().sum().sum()

        # =================================================
        # NEGATIVE PRICE CHECK
        # =================================================

        negative_prices = (
            (df["open"] < 0).sum()
            + (df["high"] < 0).sum()
            + (df["low"] < 0).sum()
            + (df["close"] < 0).sum()
            + (df["adj_close"] < 0).sum()
        )

        # =================================================
        # NEGATIVE VOLUME CHECK
        # =================================================

        negative_volume = (
            df["volume"] < 0
        ).sum()

        # =================================================
        # HIGH-LOW CONSISTENCY
        # =================================================

        invalid_high_low = (
            (
                df["high"] < df["low"]
            ).sum()
        )

        # =================================================
        # CHRONOLOGICAL ORDER CHECK
        # =================================================

        timestamps = pd.to_datetime(
            df["timestamp"]
        )

        chronological = (
            timestamps.is_monotonic_increasing
        )

        # =================================================
        # SAVE RESULTS
        # =================================================

        validation_results.append({

            "asset": asset_name,

            "rows": row_count,

            "missing_columns":
                ",".join(missing_columns)
                if missing_columns else "None",

            "duplicate_timestamps":
                duplicate_count,

            "null_values":
                int(total_nulls),

            "negative_prices":
                int(negative_prices),

            "negative_volume":
                int(negative_volume),

            "invalid_high_low":
                int(invalid_high_low),

            "chronological":
                chronological
        })

        logger.success(
            f"{asset_name}: validation complete"
        )

    except Exception as e:

        logger.exception(
            f"{asset_name}: validation failed -> {str(e)}"
        )

# =========================================================
# MAIN VALIDATION LOOP
# =========================================================

asset_dirs = [
    path for path in RAW_DATA_DIR.iterdir()
    if path.is_dir()
]

logger.info(
    f"Found {len(asset_dirs)} assets"
)

for asset_dir in asset_dirs:

    validate_asset(asset_dir)

# =========================================================
# CREATE VALIDATION REPORT
# =========================================================

results_df = pd.DataFrame(validation_results)

output_path = (
    REPORT_DIR / "validation_summary.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

logger.info(
    f"Validation report saved -> {output_path}"
)

# =========================================================
# FINAL SUMMARY
# =========================================================

logger.info("====================================")
logger.info("VALIDATION COMPLETE")
logger.info(f"Assets Validated: {len(results_df)}")
logger.info("====================================")