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

PANEL_DIR = (
    PROJECT_ROOT / "data/panel"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT / "checkpoints"
)

LOG_DIR = (
    PROJECT_ROOT / "logs"
)

# Create directories
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
    LOG_DIR / "global_xgboost.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# =========================================================
# LOAD PANEL DATASETS
# =========================================================

logger.info(
    "Loading panel datasets"
)

train_df = pd.read_parquet(
    PANEL_DIR / "train_panel.parquet"
)

validation_df = pd.read_parquet(
    PANEL_DIR / "validation_panel.parquet"
)

# =========================================================
# FEATURE SELECTION
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
# LABEL ENCODING
# =========================================================

label_mapping = {
    -1: 0,
     0: 1,
     1: 2
}

inverse_mapping = {
    0: -1,
    1: 0,
    2: 1
}

y_train = y_train.map(
    label_mapping
)

y_validation = y_validation.map(
    label_mapping
)

# =========================================================
# CLASS WEIGHTS
# =========================================================

class_counts = y_train.value_counts()

total_samples = len(y_train)

class_weights = {

    cls: total_samples / count

    for cls, count in class_counts.items()

}

sample_weights = y_train.map(
    class_weights
)

logger.info(
    f"Class weights -> {class_weights}"
)

# =========================================================
# MODEL CONFIGURATION
# =========================================================

model = XGBClassifier(

    objective="multi:softprob",

    num_class=3,

    n_estimators=300,

    max_depth=8,

    learning_rate=0.03,

    subsample=0.8,

    colsample_bytree=0.8,

    min_child_weight=5,

    gamma=1.0,

    reg_alpha=0.5,

    reg_lambda=1.0,

    random_state=42,

    eval_metric="mlogloss",

    tree_method="hist"
)

# =========================================================
# TRAIN MODEL
# =========================================================

logger.info(
    "Training global XGBoost model"
)

model.fit(

    X_train,

    y_train,

    sample_weight=sample_weights

)

logger.success(
    "Global model training complete"
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
    "\nTop 25 Features:\n"
    f"{importance_df.head(25)}"
)

# =========================================================
# SAVE FEATURE IMPORTANCE
# =========================================================

importance_df.to_csv(

    CHECKPOINT_DIR
    / "global_feature_importance.csv",

    index=False
)

logger.success(
    "Saved feature importance"
)

# =========================================================
# SAVE MODEL
# =========================================================

model.save_model(

    CHECKPOINT_DIR
    / "global_xgboost.json"
)

logger.success(
    "Saved global XGBoost model"
)

# =========================================================
# FINAL SUMMARY
# =========================================================

logger.info("====================================")
logger.info("GLOBAL XGBOOST TRAINING COMPLETE")
logger.info(f"Accuracy: {accuracy:.4f}")
logger.info(f"Log Loss: {loss:.4f}")
logger.info("====================================")