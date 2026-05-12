from pathlib import Path

import json

import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    log_loss,
    classification_report,
    confusion_matrix
)

from sklearn.utils.class_weight import (
    compute_class_weight
)

from loguru import logger

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = (
    PROJECT_ROOT
    / "data/panel/ranked"
)

FEATURE_STORE_DIR = (
    PROJECT_ROOT
    / "data/feature_store/meta_features"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints/trainer"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports/diagnostics"
)

LOG_DIR = (
    PROJECT_ROOT
    / "logs"
)

# Create directories
CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
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
    LOG_DIR / "trainer.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# =========================================================
# LOAD DATASETS
# =========================================================

logger.info(
    "Loading ranked datasets"
)

train_df = pd.read_parquet(
    DATA_DIR
    / "train_ranked.parquet"
)

validation_df = pd.read_parquet(
    DATA_DIR
    / "validation_ranked.parquet"
)

logger.info(
    f"Train shape -> {train_df.shape}"
)

logger.info(
    f"Validation shape -> "
    f"{validation_df.shape}"
)

# =========================================================
# LOAD FEATURE LISTS
# =========================================================

logger.info(
    "Loading representative features"
)

with open(

    FEATURE_STORE_DIR
    / "representative_feature_list.txt",

    "r"
) as f:

    representative_features = [

        line.strip()

        for line in f.readlines()
    ]

logger.info(
    "Loading stable features"
)

with open(

    FEATURE_STORE_DIR
    / "stable_feature_list.txt",

    "r"
) as f:

    stable_features = [

        line.strip()

        for line in f.readlines()
    ]

# =========================================================
# FINAL FEATURE UNIVERSE
# =========================================================

final_features = sorted(

    list(

        set(representative_features)

        &

        set(stable_features)
    )
)

logger.info(
    f"Final feature universe -> "
    f"{len(final_features)}"
)

logger.info(
    f"Features -> {final_features}"
)

# =========================================================
# TARGET VARIABLE
# =========================================================

TARGET_COLUMN = (
    "cross_sectional_rank_label"
)

# Convert labels:
# -1 -> 0
#  0 -> 1
#  1 -> 2

train_df[TARGET_COLUMN] = (
    train_df[TARGET_COLUMN]
    + 1
)

validation_df[TARGET_COLUMN] = (
    validation_df[TARGET_COLUMN]
    + 1
)

# =========================================================
# FEATURE MATRICES
# =========================================================

logger.info(
    "Building feature matrices"
)

X_train = train_df[
    final_features
].copy()

X_valid = validation_df[
    final_features
].copy()

y_train = train_df[
    TARGET_COLUMN
].copy()

y_valid = validation_df[
    TARGET_COLUMN
].copy()

# =========================================================
# CLEAN NAN / INF
# =========================================================

logger.info(
    "Cleaning matrices"
)

X_train.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

X_valid.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

X_train.fillna(
    0,
    inplace=True
)

X_valid.fillna(
    0,
    inplace=True
)

# =========================================================
# CLASS WEIGHTS
# =========================================================

logger.info(
    "Computing class weights"
)

classes = np.unique(
    y_train
)

weights = compute_class_weight(

    class_weight="balanced",

    classes=classes,

    y=y_train
)

class_weight_map = dict(
    zip(classes, weights)
)

logger.info(
    f"Class weights -> "
    f"{class_weight_map}"
)

sample_weights = y_train.map(
    class_weight_map
)

# =========================================================
# MODEL CONFIGURATION
# =========================================================

logger.info(
    "Initializing XGBoost model"
)

model = XGBClassifier(

    n_estimators=300,

    max_depth=6,

    learning_rate=0.03,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="multi:softprob",

    num_class=3,

    eval_metric="mlogloss",

    tree_method="hist",

    random_state=42,

    n_jobs=-1
)

# =========================================================
# TRAIN MODEL
# =========================================================

logger.info(
    "Training model"
)

model.fit(

    X_train,

    y_train,

    sample_weight=sample_weights
)

logger.success(
    "Training complete"
)

# =========================================================
# VALIDATION PREDICTIONS
# =========================================================

logger.info(
    "Generating validation predictions"
)

predictions = model.predict(
    X_valid
)

prediction_probabilities = (
    model.predict_proba(
        X_valid
    )
)

# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(

    y_valid,

    predictions
)

loss = log_loss(

    y_valid,

    prediction_probabilities
)

report = classification_report(

    y_valid,

    predictions,

    output_dict=True
)

conf_matrix = confusion_matrix(

    y_valid,

    predictions
)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance_df = pd.DataFrame({

    "feature": final_features,

    "importance": model.feature_importances_
})

importance_df.sort_values(

    by="importance",

    ascending=False,

    inplace=True
)

importance_df.reset_index(

    drop=True,

    inplace=True
)

# =========================================================
# FINAL REPORTING
# =========================================================

logger.info("====================================")
logger.info("TRAINER RESULTS")
logger.info("====================================")

logger.info(
    f"Accuracy: {accuracy:.4f}"
)

logger.info(
    f"Log Loss: {loss:.4f}"
)

logger.info(
    "\nTop Features:\n"
    f"{importance_df.head(20)}"
)

logger.info(
    "\nConfusion Matrix:\n"
    f"{conf_matrix}"
)

logger.info("====================================")

# =========================================================
# SAVE MODEL
# =========================================================

logger.info(
    "Saving model artifacts"
)

model.save_model(

    CHECKPOINT_DIR
    / "xgboost_trainer.json"
)

importance_df.to_csv(

    REPORT_DIR
    / "trainer_feature_importance.csv",

    index=False
)

# Save metrics
metrics = {

    "accuracy": float(
        accuracy
    ),

    "log_loss": float(
        loss
    ),

    "features": final_features,

    "num_features": len(
        final_features
    )
}

with open(

    REPORT_DIR
    / "trainer_metrics.json",

    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

logger.success(
    "Saved trainer outputs"
)

# =========================================================
# FINAL LOGS
# =========================================================

logger.info("====================================")
logger.info("TRAINER COMPLETE")
logger.info("====================================")