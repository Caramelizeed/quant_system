from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_STORE_DIR = (
    PROJECT_ROOT / "data/feature_store"
)

LABEL_OUTPUT_DIR = (
    PROJECT_ROOT / "data/labels/triple_barrier"
)

LOG_DIR = PROJECT_ROOT / "logs"

# Create directories
LABEL_OUTPUT_DIR.mkdir(
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
    LOG_DIR / "triple_barrier.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# =========================================================
# LABELING PARAMETERS
# =========================================================

HOLDING_PERIOD = 21

TAKE_PROFIT_MULTIPLIER = 1.0

STOP_LOSS_MULTIPLIER = 1.0

# =========================================================
# TRIPLE BARRIER FUNCTION
# =========================================================

def generate_labels(asset_dir: Path):

    feature_path = (
        asset_dir / "features.parquet"
    )

    asset_name = asset_dir.name

    if not feature_path.exists():

        logger.warning(
            f"{asset_name}: features missing"
        )

        return False

    try:

        # =================================================
        # LOAD FEATURES
        # =================================================

        df = pd.read_parquet(feature_path)

        logger.info(
            f"Generating labels for {asset_name}"
        )

        # =================================================
        # REQUIRED COLUMNS
        # =================================================

        required_columns = [
            "timestamp",
            "adj_close",
            "volatility_21d"
        ]

        missing_columns = [

            col for col in required_columns
            if col not in df.columns

        ]

        if missing_columns:

            logger.error(
                f"{asset_name}: missing columns -> "
                f"{missing_columns}"
            )

            return False

        # =================================================
        # INITIALIZE LABEL COLUMN
        # =================================================

        df["triple_barrier_label"] = np.nan

        # =================================================
        # MAIN LABELING LOOP
        # =================================================

        for i in range(len(df) - HOLDING_PERIOD):

            current_price = (
                df.loc[i, "adj_close"]
            )

            annualized_volatility = (
                df.loc[i, "volatility_21d"]
            )

            current_volatility = (
                annualized_volatility
                * np.sqrt(HOLDING_PERIOD / 252)
)

            # ---------------------------------------------
            # SKIP INVALID VOLATILITY
            # ---------------------------------------------

            if pd.isna(current_volatility):

                continue

            # ---------------------------------------------
            # BARRIERS
            # ---------------------------------------------

            upper_barrier = (

                current_price *

                (
                    1
                    + TAKE_PROFIT_MULTIPLIER
                    * current_volatility
                )

            )

            lower_barrier = (

                current_price *

                (
                    1
                    - STOP_LOSS_MULTIPLIER
                    * current_volatility
                )

            )

            # ---------------------------------------------
            # FUTURE WINDOW
            # ---------------------------------------------

            future_prices = df.loc[
                i + 1 : i + HOLDING_PERIOD,
                "adj_close"
            ]

            label = 0

            # ---------------------------------------------
            # CHECK BARRIER HITS
            # ---------------------------------------------

            for future_price in future_prices:

                if future_price >= upper_barrier:

                    label = 1
                    break

                elif future_price <= lower_barrier:

                    label = -1
                    break

            # ---------------------------------------------
            # ASSIGN LABEL
            # ---------------------------------------------

            df.loc[
                i,
                "triple_barrier_label"
            ] = label

        # =================================================
        # REMOVE NAN LABELS
        # =================================================

        df.dropna(
            subset=["triple_barrier_label"],
            inplace=True
        )

        # =================================================
        # CONVERT LABEL TYPE
        # =================================================

        df["triple_barrier_label"] = (
            df["triple_barrier_label"]
            .astype(int)
        )

        # =================================================
        # LABEL DISTRIBUTION
        # =================================================

        label_counts = (
            df["triple_barrier_label"]
            .value_counts()
            .to_dict()
        )

        logger.info(
            f"{asset_name}: label distribution -> "
            f"{label_counts}"
        )

        # =================================================
        # OUTPUT DIRECTORY
        # =================================================

        output_dir = (
            LABEL_OUTPUT_DIR / asset_name
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            output_dir / "labels.parquet"
        )

        # =================================================
        # SAVE LABELS
        # =================================================

        df.to_parquet(
            output_path,
            index=False
        )

        logger.success(
            f"{asset_name}: generated "
            f"{len(df)} labeled rows"
        )

        return True

    except Exception as e:

        logger.exception(
            f"{asset_name}: labeling failed -> "
            f"{str(e)}"
        )

        return False

# =========================================================
# MAIN EXECUTION
# =========================================================

asset_dirs = [

    path for path in FEATURE_STORE_DIR.iterdir()
    if path.is_dir()

]

logger.info(
    f"Found {len(asset_dirs)} assets"
)

successful = 0
failed = 0

for asset_dir in asset_dirs:

    success = generate_labels(asset_dir)

    if success:

        successful += 1

    else:

        failed += 1

# =========================================================
# FINAL SUMMARY
# =========================================================

logger.info("====================================")
logger.info("TRIPLE BARRIER LABELING COMPLETE")
logger.info(f"Successful: {successful}")
logger.info(f"Failed: {failed}")
logger.info("====================================")