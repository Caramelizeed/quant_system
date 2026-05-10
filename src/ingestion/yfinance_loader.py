from pathlib import Path
import pandas as pd
import yfinance as yf
import yaml
from loguru import logger

# =========================================================
# PATH CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

UNIVERSE_FILE = PROJECT_ROOT / "config/universe/all_tickers.yaml"
RAW_DATA_DIR = PROJECT_ROOT / "data/raw/yfinance"
LOG_DIR = PROJECT_ROOT / "logs"

# Create required directories
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# LOGGER CONFIGURATION
# =========================================================

logger.add(
    LOG_DIR / "yfinance_loader.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# =========================================================
# LOAD UNIVERSE
# =========================================================

with open(UNIVERSE_FILE, "r") as file:
    universe = yaml.safe_load(file)

logger.info(f"Loaded {len(universe)} assets from universe")

# =========================================================
# REQUIRED COLUMNS
# =========================================================

REQUIRED_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume"
]

# =========================================================
# DOWNLOAD FUNCTION
# =========================================================

def download_ticker(canonical_symbol: str, metadata: dict):

    # -----------------------------------------------------
    # GET PROVIDER SYMBOL
    # -----------------------------------------------------

    provider_symbol = metadata.get("yahoo_ticker")

    if provider_symbol is None:

        logger.error(
            f"{canonical_symbol}: missing yahoo_ticker"
        )

        return

    logger.info(
        f"Downloading {canonical_symbol} ({provider_symbol})"
    )

    try:

        # -------------------------------------------------
        # DOWNLOAD DATA
        # -------------------------------------------------

        df = yf.download(
            tickers=provider_symbol,
            start="2010-01-01",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True
        )

        # -------------------------------------------------
        # VALIDATE EMPTY DATAFRAME
        # -------------------------------------------------

        if df.empty:

            logger.warning(
                f"{canonical_symbol}: empty dataframe"
            )

            return

        # -------------------------------------------------
        # NORMALIZE COLUMN NAMES
        # -------------------------------------------------

        df.columns = [
            col.lower().replace(" ", "_")
            for col in df.columns
        ]

        # -------------------------------------------------
        # RESET INDEX
        # -------------------------------------------------

        df.reset_index(inplace=True)

        # -------------------------------------------------
        # STANDARDIZE TIMESTAMP COLUMN
        # -------------------------------------------------

        if "date" in df.columns:
            df.rename(
                columns={"date": "timestamp"},
                inplace=True
            )

        # -------------------------------------------------
        # VALIDATE REQUIRED COLUMNS
        # -------------------------------------------------

        missing_columns = [
            col for col in REQUIRED_COLUMNS
            if col not in df.columns
        ]

        if missing_columns:

            logger.error(
                f"{canonical_symbol}: missing columns -> "
                f"{missing_columns}"
            )

            return

        # -------------------------------------------------
        # REMOVE DUPLICATES
        # -------------------------------------------------

        before = len(df)

        df.drop_duplicates(
            subset=["timestamp"],
            inplace=True
        )

        after = len(df)

        duplicate_count = before - after

        if duplicate_count > 0:

            logger.warning(
                f"{canonical_symbol}: removed "
                f"{duplicate_count} duplicate rows"
            )

        # -------------------------------------------------
        # SORT BY TIME
        # -------------------------------------------------

        df.sort_values(
            by="timestamp",
            inplace=True
        )

        # -------------------------------------------------
        # ADD METADATA
        # -------------------------------------------------

        df["canonical_symbol"] = canonical_symbol
        df["provider_symbol"] = provider_symbol
        df["asset_type"] = metadata.get("asset_type")
        df["country"] = metadata.get("country")
        df["sector"] = metadata.get("sector")
        df["source"] = "yfinance"

        # -------------------------------------------------
        # OUTPUT DIRECTORY
        # -------------------------------------------------

        output_dir = RAW_DATA_DIR / canonical_symbol

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = output_dir / "daily.parquet"

        # -------------------------------------------------
        # SAVE PARQUET
        # -------------------------------------------------

        df.to_parquet(
            output_path,
            index=False
        )

        logger.success(
            f"{canonical_symbol}: saved "
            f"{len(df)} rows"
        )

    except Exception as e:

        logger.exception(
            f"{canonical_symbol}: FAILED -> {str(e)}"
        )

# =========================================================
# MAIN EXECUTION LOOP
# =========================================================

successful_downloads = 0
failed_downloads = 0

for canonical_symbol, metadata in universe.items():

    try:

        download_ticker(
            canonical_symbol,
            metadata
        )

        successful_downloads += 1

    except Exception:

        failed_downloads += 1

# =========================================================
# FINAL SUMMARY
# =========================================================

logger.info("======================================")
logger.info("DOWNLOAD PIPELINE COMPLETE")
logger.info(f"Successful: {successful_downloads}")
logger.info(f"Failed: {failed_downloads}")
logger.info("======================================")