from pathlib import Path

import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    log_loss
)

from loguru import logger

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = (
    PROJECT_ROOT / "data/panel/ranked"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "backtests/walk_forward"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT / "checkpoints/walk_forward"
)

LOG_DIR = (
    PROJECT_ROOT / "logs"
)

# Create directories
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHECKPOINT_DIR.mkdir(
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
    LOG_DIR / "walk_forward.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# =========================================================
# CONFIGURATION
# =========================================================

INITIAL_TRAIN_YEARS = 5

VALIDATION_YEARS = 1

MIN_CLASS_COUNT = 25

# =========================================================
# LOAD DATA
# =========================================================

logger.info(
    "Loading ranked panel dataset"
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

df.reset_index(
    drop=True,
    inplace=True
)

df["year"] = (
    df["timestamp"]
    .dt.year
)

# =========================================================
# FEATURE COLUMNS
# =========================================================

EXCLUDED_COLUMNS = [

    "timestamp",

    "canonical_symbol",
    "provider_symbol",

    "asset_type",
    "country",
    "sector",
    "category",

    "source",

    "future_close",
    "future_return_21d",
    "future_return_rank",

    "triple_barrier_label",

    "cross_sectional_rank_label",

    "year"
]

FEATURE_COLUMNS = [

    col for col in df.columns

    if col not in EXCLUDED_COLUMNS
]

logger.info(
    f"Using {len(FEATURE_COLUMNS)} features"
)

# =========================================================
# TARGET
# =========================================================

TARGET_COLUMN = (
    "cross_sectional_rank_label"
)

# Convert labels:
# -1 → 0
#  0 → 1
#  1 → 2

label_mapping = {

    -1: 0,
     0: 1,
     1: 2
}

df[TARGET_COLUMN] = (

    df[TARGET_COLUMN]

    .map(label_mapping)
)

# =========================================================
# WALK FORWARD YEARS
# =========================================================

years = sorted(
    df["year"].unique()
)

logger.info(
    f"Available years -> {years}"
)

# =========================================================
# WALK FORWARD LOOP
# =========================================================

results = []

for i in range(

    INITIAL_TRAIN_YEARS,

    len(years) - VALIDATION_YEARS + 1
):

    # =====================================================
    # TRAIN / VALIDATION SPLITS
    # =====================================================

    train_years = years[:i]

    validation_years = years[
        i:i + VALIDATION_YEARS
    ]

    logger.info("====================================")

    logger.info(
        f"Train Years -> {train_years}"
    )

    logger.info(
        f"Validation Years -> "
        f"{validation_years}"
    )

    # =====================================================
    # SPLIT DATA
    # =====================================================

    train_df = df[
        df["year"].isin(train_years)
    ].copy()

    validation_df = df[
        df["year"].isin(validation_years)
    ].copy()

    # =====================================================
    # CLASS FILTER
    # =====================================================

    class_counts = (

        train_df[TARGET_COLUMN]

        .value_counts()
    )

    logger.info(
        f"Class Counts -> "
        f"{class_counts.to_dict()}"
    )

    if (

        class_counts.min()

        < MIN_CLASS_COUNT
    ):

        logger.warning(
            "Skipping due to "
            "insufficient class samples"
        )

        continue

    # =====================================================
    # FEATURES / TARGETS
    # =====================================================

    X_train = train_df[
        FEATURE_COLUMNS
    ]

    y_train = train_df[
        TARGET_COLUMN
    ]

    X_validation = validation_df[
        FEATURE_COLUMNS
    ]

    y_validation = validation_df[
        TARGET_COLUMN
    ]

    # =====================================================
    # MODEL
    # =====================================================

    model = XGBClassifier(

        objective="multi:softprob",

        num_class=3,

        n_estimators=300,

        max_depth=6,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=42,

        eval_metric="mlogloss",

        tree_method="hist",

        verbosity=0
    )

    # =====================================================
    # TRAIN MODEL
    # =====================================================

    logger.info(
        "Training model"
    )

    model.fit(
        X_train,
        y_train
    )

    # =====================================================
    # PREDICTIONS
    # =====================================================

    predictions = model.predict(
        X_validation
    )

    prediction_probabilities = (
        model.predict_proba(
            X_validation
        )
    )

    # =====================================================
    # METRICS
    # =====================================================

    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    loss = log_loss(
        y_validation,
        prediction_probabilities
    )

    logger.info(
        f"Validation Accuracy -> "
        f"{accuracy:.4f}"
    )

    logger.info(
        f"Validation Log Loss -> "
        f"{loss:.4f}"
    )

    # =====================================================
    # SAVE MODEL
    # =====================================================

    model_path = (

        CHECKPOINT_DIR

        / f"xgboost_"
          f"{validation_years[0]}.json"
    )

    model.save_model(
        model_path
    )

    logger.success(
        f"Saved model -> {model_path}"
    )

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    results.append({

        "train_start": min(train_years),

        "train_end": max(train_years),

        "validation_year": validation_years[0],

        "accuracy": accuracy,

        "log_loss": loss,

        "train_rows": len(train_df),

        "validation_rows": len(validation_df)
    })

# =========================================================
# RESULTS DATAFRAME
# =========================================================

results_df = pd.DataFrame(
    results
)

# =========================================================
# SUMMARY METRICS
# =========================================================

logger.info("====================================")
logger.info("WALK FORWARD SUMMARY")
logger.info("====================================")

logger.info(
    f"Mean Accuracy -> "
    f"{results_df['accuracy'].mean():.4f}"
)

logger.info(
    f"Mean Log Loss -> "
    f"{results_df['log_loss'].mean():.4f}"
)

logger.info("====================================")

# =========================================================
# SAVE RESULTS
# =========================================================

results_df.to_csv(

    OUTPUT_DIR
    / "walk_forward_results.csv",

    index=False
)

logger.success(
    "Saved walk-forward results"
)

logger.info("====================================")
logger.info("WALK FORWARD VALIDATION COMPLETE")
logger.info("====================================")