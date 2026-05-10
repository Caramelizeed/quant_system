from pathlib import Path
import pandas as pd
from loguru import logger

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

LABEL_DIR = (
    PROJECT_ROOT / "data/labels/triple_barrier"
)

PROCESSED_DIR = (
    PROJECT_ROOT / "data/processed"
)

TRAIN_DIR = (
    PROCESSED_DIR / "train"
)

VALIDATION_DIR = (
    PROCESSED_DIR / "validation"
)

TEST_DIR = (
    PROCESSED_DIR / "test"
)

LOG_DIR = (
    PROJECT_ROOT / "logs"
)

# Create directories
TRAIN_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VALIDATION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TEST_DIR.mkdir(
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
    LOG_DIR / "train_test_split.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# =========================================================
# SPLIT YEARS
# =========================================================

TRAIN_END_YEAR = 2018

VALIDATION_END_YEAR = 2021

# =========================================================
# SPLIT FUNCTION
# =========================================================

def split_dataset(asset_dir: Path):

    label_path = (
        asset_dir / "labels.parquet"
    )

    asset_name = asset_dir.name

    if not label_path.exists():

        logger.warning(
            f"{asset_name}: labels missing"
        )

        return False

    try:

        # =================================================
        # LOAD DATA
        # =================================================

        df = pd.read_parquet(label_path)

        logger.info(
            f"Splitting {asset_name}"
        )

        # =================================================
        # TIMESTAMP CONVERSION
        # =================================================

        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
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
        # CREATE YEAR COLUMN
        # =================================================

        df["year"] = (
            df["timestamp"].dt.year
        )

        # =================================================
        # TRAIN SPLIT
        # =================================================

        train_df = df[
            df["year"] <= TRAIN_END_YEAR
        ].copy()

        # =================================================
        # VALIDATION SPLIT
        # =================================================

        validation_df = df[

            (df["year"] > TRAIN_END_YEAR)

            &

            (
                df["year"]
                <= VALIDATION_END_YEAR
            )

        ].copy()

        # =================================================
        # TEST SPLIT
        # =================================================

        test_df = df[
            df["year"] > VALIDATION_END_YEAR
        ].copy()

        # =================================================
        # REMOVE YEAR COLUMN
        # =================================================

        train_df.drop(
            columns=["year"],
            inplace=True
        )

        validation_df.drop(
            columns=["year"],
            inplace=True
        )

        test_df.drop(
            columns=["year"],
            inplace=True
        )

        # =================================================
        # VALIDATE SPLITS
        # =================================================

        if len(train_df) == 0:

            logger.warning(
                f"{asset_name}: empty train set"
            )

            return False

        if len(validation_df) == 0:

            logger.warning(
                f"{asset_name}: empty validation set"
            )

            return False

        if len(test_df) == 0:

            logger.warning(
                f"{asset_name}: empty test set"
            )

            return False

        # =================================================
        # OUTPUT DIRECTORIES
        # =================================================

        train_output_dir = (
            TRAIN_DIR / asset_name
        )

        validation_output_dir = (
            VALIDATION_DIR / asset_name
        )

        test_output_dir = (
            TEST_DIR / asset_name
        )

        train_output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        validation_output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        test_output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # =================================================
        # SAVE DATASETS
        # =================================================

        train_df.to_parquet(
            train_output_dir / "train.parquet",
            index=False
        )

        validation_df.to_parquet(
            validation_output_dir / "validation.parquet",
            index=False
        )

        test_df.to_parquet(
            test_output_dir / "test.parquet",
            index=False
        )

        # =================================================
        # LOG SPLIT SUMMARY
        # =================================================

        logger.success(

            f"{asset_name} | "

            f"train={len(train_df)} | "

            f"validation={len(validation_df)} | "

            f"test={len(test_df)}"

        )

        return True

    except Exception as e:

        logger.exception(
            f"{asset_name}: split failed -> "
            f"{str(e)}"
        )

        return False

# =========================================================
# MAIN EXECUTION
# =========================================================

asset_dirs = [

    path for path in LABEL_DIR.iterdir()
    if path.is_dir()

]

logger.info(
    f"Found {len(asset_dirs)} assets"
)

successful = 0
failed = 0

for asset_dir in asset_dirs:

    success = split_dataset(asset_dir)

    if success:

        successful += 1

    else:

        failed += 1

# =========================================================
# FINAL SUMMARY
# =========================================================

logger.info("====================================")
logger.info("TRAIN TEST SPLITTING COMPLETE")
logger.info(f"Successful: {successful}")
logger.info(f"Failed: {failed}")
logger.info("====================================")