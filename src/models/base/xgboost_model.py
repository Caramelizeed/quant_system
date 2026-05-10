from pathlib import Path

import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss
)

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

MODEL_OUTPUT_DIR = (
    PROJECT_ROOT / "checkpoints"
)

LOG_DIR = (
    PROJECT_ROOT / "logs"
)

# Create directories
MODEL_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =========================================================
# LOGGER
# =========================================================

logger.add(
    LOG_DIR / "xgboost_model.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# =========================================================
# ASSET TO TRAIN
# =========================================================

ASSET = "AAPL"

# =========================================================
# LOAD DATA
# =========================================================

train_path = (
    TRAIN_DIR
    / ASSET
    / "train.parquet"
)

validation_path = (
    VALIDATION_DIR
    / ASSET
    / "validation.parquet"
)

logger.info(
    f"Loading datasets for {ASSET}"
)

train_df = pd.read_parquet(train_path)

validation_df = pd.read_parquet(validation_path)

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

    "triple_barrier_label"
]

FEATURE_COLUMNS = [

    col for col in train_df.columns
    if col not in EXCLUDED_COLUMNS

]

logger.info(
    f"Using {len(FEATURE_COLUMNS)} features"
)

# =========================================================
# PREPARE DATA
# =========================================================

X_train = train_df[
    FEATURE_COLUMNS
]

y_train = train_df[
    "triple_barrier_label"
]

X_validation = validation_df[
    FEATURE_COLUMNS
]

y_validation = validation_df[
    "triple_barrier_label"
]

# =========================================================
# LABEL MAPPING
# =========================================================

label_mapping = {
    -1: 0,
     0: 1,
     1: 2
}

y_train = y_train.map(
    label_mapping
)

y_validation = y_validation.map(
    label_mapping
)

# =========================================================
# MODEL
# =========================================================

model = XGBClassifier(

    objective="multi:softprob",

    num_class=3,

    n_estimators=200,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42,

    eval_metric="mlogloss"
)

# =========================================================
# TRAIN MODEL
# =========================================================

logger.info(
    "Training XGBoost model"
)

model.fit(
    X_train,
    y_train
)

logger.success(
    "Model training complete"
)

# =========================================================
# VALIDATION PREDICTIONS
# =========================================================

validation_predictions = model.predict(
    X_validation
)

validation_probabilities = model.predict_proba(
    X_validation
)

# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(
    y_validation,
    validation_predictions
)

loss = log_loss(
    y_validation,
    validation_probabilities
)

logger.info(
    f"Validation Accuracy: {accuracy:.4f}"
)

logger.info(
    f"Validation Log Loss: {loss:.4f}"
)

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

report = classification_report(
    y_validation,
    validation_predictions
)

logger.info(
    "\nClassification Report:\n"
    f"{report}"
)

# =========================================================
# CONFUSION MATRIX
# =========================================================

matrix = confusion_matrix(
    y_validation,
    validation_predictions
)

logger.info(
    "\nConfusion Matrix:\n"
    f"{matrix}"
)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance_df = pd.DataFrame({

    "feature": FEATURE_COLUMNS,

    "importance": model.feature_importances_

})

importance_df.sort_values(
    by="importance",
    ascending=False,
    inplace=True
)

logger.info(
    "\nTop 15 Features:\n"
    f"{importance_df.head(15)}"
)

# =========================================================
# SAVE FEATURE IMPORTANCE
# =========================================================

importance_path = (
    MODEL_OUTPUT_DIR
    / f"{ASSET}_feature_importance.csv"
)

importance_df.to_csv(
    importance_path,
    index=False
)

logger.success(
    f"Saved feature importance -> "
    f"{importance_path}"
)

# =========================================================
# FINAL SUMMARY
# =========================================================

logger.info("====================================")
logger.info("XGBOOST TRAINING COMPLETE")
logger.info(f"Asset: {ASSET}")
logger.info(f"Accuracy: {accuracy:.4f}")
logger.info(f"Log Loss: {loss:.4f}")
logger.info("====================================")