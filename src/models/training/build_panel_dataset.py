from pathlib import Path
import pandas as pd
from loguru import logger

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_DIR = (
    PROJECT_ROOT / "data/processed/train"
)

VALIDATION_DIR = (
    PROJECT_ROOT / "data/processed/validation"
)

TEST_DIR = (
    PROJECT_ROOT / "data/processed/test"
)

PANEL_DIR = (
    PROJECT_ROOT / "data/panel"
)

LOG_DIR = (
    PROJECT_ROOT / "logs"
)

# Create directories
PANEL_DIR.mkdir(
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
    LOG_DIR / "panel_dataset.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# =========================================================
# LOAD SPLIT FUNCTION
# =========================================================

def load_split_dataset(
    split_dir: Path,
    split_name: str,
    file_name: str
):

    all_datasets = []

    asset_dirs = [

        path for path in split_dir.iterdir()
        if path.is_dir()

    ]

    logger.info(
        f"{split_name}: found "
        f"{len(asset_dirs)} assets"
    )

    for asset_dir in asset_dirs:

        asset_name = asset_dir.name

        parquet_path = (
            asset_dir / file_name
        )

        if not parquet_path.exists():

            logger.warning(
                f"{split_name} | "
                f"{asset_name}: missing file"
            )

            continue

        try:

            # =============================================
            # LOAD DATASET
            # =============================================

            df = pd.read_parquet(
                parquet_path
            )

            # =============================================
            # VALIDATION
            # =============================================

            if len(df) == 0:

                logger.warning(
                    f"{split_name} | "
                    f"{asset_name}: empty dataset"
                )

                continue

            # =============================================
            # TIMESTAMP STANDARDIZATION
            # =============================================

            df["timestamp"] = pd.to_datetime(
                df["timestamp"]
            )

            # =============================================
            # SORTING
            # =============================================

            df.sort_values(
                by="timestamp",
                inplace=True
            )

            df.reset_index(
                drop=True,
                inplace=True
            )

            # =============================================
            # APPEND
            # =============================================

            all_datasets.append(df)

            logger.success(
                f"{split_name} | "
                f"{asset_name}: loaded "
                f"{len(df)} rows"
            )

        except Exception as e:

            logger.exception(
                f"{split_name} | "
                f"{asset_name}: failed -> "
                f"{str(e)}"
            )

    # =====================================================
    # CONCATENATE PANEL
    # =====================================================

    if len(all_datasets) == 0:

        logger.error(
            f"{split_name}: no datasets loaded"
        )

        return None

    panel_df = pd.concat(
        all_datasets,
        axis=0,
        ignore_index=True
    )

    # =====================================================
    # FINAL SORT
    # =====================================================

    panel_df.sort_values(
        by=["timestamp", "canonical_symbol"],
        inplace=True
    )

    panel_df.reset_index(
        drop=True,
        inplace=True
    )

    logger.info(
        f"{split_name}: final panel shape -> "
        f"{panel_df.shape}"
    )

    return panel_df

# =========================================================
# BUILD TRAIN PANEL
# =========================================================

train_panel = load_split_dataset(

    split_dir=TRAIN_DIR,

    split_name="TRAIN",

    file_name="train.parquet"
)

# =========================================================
# BUILD VALIDATION PANEL
# =========================================================

validation_panel = load_split_dataset(

    split_dir=VALIDATION_DIR,

    split_name="VALIDATION",

    file_name="validation.parquet"
)

# =========================================================
# BUILD TEST PANEL
# =========================================================

test_panel = load_split_dataset(

    split_dir=TEST_DIR,

    split_name="TEST",

    file_name="test.parquet"
)

# =========================================================
# SAVE PANELS
# =========================================================

if train_panel is not None:

    train_panel.to_parquet(

        PANEL_DIR / "train_panel.parquet",

        index=False
    )

    logger.success(
        "Saved train panel"
    )

if validation_panel is not None:

    validation_panel.to_parquet(

        PANEL_DIR / "validation_panel.parquet",

        index=False
    )

    logger.success(
        "Saved validation panel"
    )

if test_panel is not None:

    test_panel.to_parquet(

        PANEL_DIR / "test_panel.parquet",

        index=False
    )

    logger.success(
        "Saved test panel"
    )

# =========================================================
# FINAL SUMMARY
# =========================================================

logger.info("====================================")
logger.info("PANEL DATASET BUILD COMPLETE")
logger.info("====================================")