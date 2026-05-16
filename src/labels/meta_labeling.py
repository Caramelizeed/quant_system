from pathlib import Path

import json

import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    log_loss,
    confusion_matrix,
    classification_report
)

from loguru import logger

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data/panel/ranked"
)

FEATURE_STORE_DIR = (
    PROJECT_ROOT
    / "data/feature_store/meta_features"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports/diagnostics"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints/meta_labeling"
)

LOG_DIR = (
    PROJECT_ROOT
    / "logs"
)

REPORT_DIR.mkdir(
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
# LOGGER
# =========================================================

logger.add(
    LOG_DIR / "meta_labeling.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# =========================================================
# LOAD DATA
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
    f"Train shape -> "
    f"{train_df.shape}"
)

logger.info(
    f"Validation shape -> "
    f"{validation_df.shape}"
)

# =========================================================
# LOAD FEATURES
# =========================================================

logger.info(
    "Loading SHAP-selected features"
)

with open(

    FEATURE_STORE_DIR
    / "shap_selected_feature_list.txt",

    "r"
) as f:

    base_features = [

        line.strip()

        for line in f.readlines()
    ]

logger.info(
    f"Base features -> "
    f"{base_features}"
)

# =========================================================
# PRIMARY LABEL
# =========================================================

TARGET_COLUMN = (
    "cross_sectional_rank_label"
)

train_df[TARGET_COLUMN] = (
    train_df[TARGET_COLUMN]
    + 1
)

validation_df[TARGET_COLUMN] = (
    validation_df[TARGET_COLUMN]
    + 1
)

# =========================================================
# BUILD PRIMARY MODEL FEATURES
# =========================================================

logger.info(
    "Building primary feature matrices"
)

X_train_primary = train_df[
    base_features
].copy()

X_valid_primary = validation_df[
    base_features
].copy()

y_train_primary = train_df[
    TARGET_COLUMN
].copy()

y_valid_primary = validation_df[
    TARGET_COLUMN
].copy()

# =========================================================
# CLEAN MATRICES
# =========================================================

logger.info(
    "Cleaning feature matrices"
)

X_train_primary.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

X_valid_primary.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

X_train_primary.fillna(
    0,
    inplace=True
)

X_valid_primary.fillna(
    0,
    inplace=True
)

# =========================================================
# TRAIN PRIMARY MODEL
# =========================================================

logger.info(
    "Training primary directional model"
)

primary_model = XGBClassifier(

    n_estimators=200,

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

primary_model.fit(

    X_train_primary,

    y_train_primary
)

logger.success(
    "Primary model trained"
)

# =========================================================
# PRIMARY PREDICTIONS
# =========================================================

logger.info(
    "Generating directional probabilities"
)

train_probs = primary_model.predict_proba(
    X_train_primary
)

valid_probs = primary_model.predict_proba(
    X_valid_primary
)

train_pred = np.argmax(
    train_probs,
    axis=1
)

valid_pred = np.argmax(
    valid_probs,
    axis=1
)

# =========================================================
# CONFIDENCE FEATURES
# =========================================================

logger.info(
    "Building confidence features"
)

train_df[
    "prediction_confidence"
] = np.max(
    train_probs,
    axis=1
)

validation_df[
    "prediction_confidence"
] = np.max(
    valid_probs,
    axis=1
)

train_df[
    "prediction_entropy"
] = -np.sum(

    train_probs
    * np.log(
        train_probs + 1e-12
    ),

    axis=1
)

validation_df[
    "prediction_entropy"
] = -np.sum(

    valid_probs
    * np.log(
        valid_probs + 1e-12
    ),

    axis=1
)

train_df[
    "predicted_direction"
] = train_pred

validation_df[
    "predicted_direction"
] = valid_pred

# =========================================================
# META LABEL TARGET
# =========================================================

logger.info(
    "Creating meta labels"
)

# Trade success definition:
# profitable directional prediction

train_df[
    "meta_label"
] = (

    (
        (train_df["predicted_direction"] == 2)
        &
        (train_df["future_return_21d"] > 0)
    )

    |

    (
        (train_df["predicted_direction"] == 0)
        &
        (train_df["future_return_21d"] < 0)
    )

).astype(int)

validation_df[
    "meta_label"
] = (

    (
        (validation_df["predicted_direction"] == 2)
        &
        (validation_df["future_return_21d"] > 0)
    )

    |

    (
        (validation_df["predicted_direction"] == 0)
        &
        (validation_df["future_return_21d"] < 0)
    )

).astype(int)

logger.info(
    f"Train meta positives -> "
    f"{train_df['meta_label'].mean():.4f}"
)

logger.info(
    f"Validation meta positives -> "
    f"{validation_df['meta_label'].mean():.4f}"
)

# =========================================================
# META FEATURES
# =========================================================

meta_features = [

    "prediction_confidence",

    "prediction_entropy",

    "volatility_21d",

    "volatility_63d",

    "daily_range",

    "momentum_63d",

    "momentum_252d",

    "dist_sma_200",

    "relative_volume_20"
]

logger.info(
    f"Meta features -> "
    f"{meta_features}"
)

X_train_meta = train_df[
    meta_features
].copy()

X_valid_meta = validation_df[
    meta_features
].copy()

y_train_meta = train_df[
    "meta_label"
]

y_valid_meta = validation_df[
    "meta_label"
]

# =========================================================
# CLEAN META MATRICES
# =========================================================

X_train_meta.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

X_valid_meta.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

X_train_meta.fillna(
    0,
    inplace=True
)

X_valid_meta.fillna(
    0,
    inplace=True
)

# =========================================================
# TRAIN META MODEL
# =========================================================

logger.info(
    "Training meta model"
)

meta_model = XGBClassifier(

    n_estimators=150,

    max_depth=4,

    learning_rate=0.03,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="binary:logistic",

    eval_metric="logloss",

    tree_method="hist",

    random_state=42,

    n_jobs=-1
)

meta_model.fit(

    X_train_meta,

    y_train_meta
)

logger.success(
    "Meta model trained"
)

# =========================================================
# META PREDICTIONS
# =========================================================

logger.info(
    "Generating meta predictions"
)

meta_probs = meta_model.predict_proba(
    X_valid_meta
)[:, 1]

meta_predictions = (
    meta_probs > 0.5
).astype(int)

# =========================================================
# META METRICS
# =========================================================

meta_accuracy = accuracy_score(

    y_valid_meta,

    meta_predictions
)

meta_loss = log_loss(

    y_valid_meta,

    meta_probs
)

conf_matrix = confusion_matrix(

    y_valid_meta,

    meta_predictions
)

report = classification_report(

    y_valid_meta,

    meta_predictions,

    output_dict=True
)

# =========================================================
# TRADE FILTERING
# =========================================================

logger.info(
    "Evaluating trade filtering"
)

validation_df[
    "meta_probability"
] = meta_probs

validation_df[
    "take_trade"
] = meta_predictions

all_trades = validation_df[
    "future_return_21d"
]

filtered_trades = validation_df.loc[

    validation_df[
        "take_trade"
    ] == 1,

    "future_return_21d"
]

trade_improvement = (

    filtered_trades.mean()

    -

    all_trades.mean()
)

logger.info(
    f"Average Return All Trades -> "
    f"{all_trades.mean():.6f}"
)

logger.info(
    f"Average Return Filtered Trades -> "
    f"{filtered_trades.mean():.6f}"
)

logger.info(
    f"Trade Improvement -> "
    f"{trade_improvement:.6f}"
)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance_df = pd.DataFrame({

    "feature": meta_features,

    "importance": (
        meta_model.feature_importances_
    )
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
logger.info("META LABELING RESULTS")
logger.info("====================================")

logger.info(
    f"Meta Accuracy: "
    f"{meta_accuracy:.4f}"
)

logger.info(
    f"Meta Log Loss: "
    f"{meta_loss:.4f}"
)

logger.info(
    f"Filtered Trade Count: "
    f"{meta_predictions.sum()}"
)

logger.info(
    f"Trade Participation Rate: "
    f"{meta_predictions.mean():.4f}"
)

logger.info(
    "\nTop Meta Features:\n"
    f"{importance_df}"
)

logger.info(
    "\nConfusion Matrix:\n"
    f"{conf_matrix}"
)

logger.info("====================================")

# =========================================================
# SAVE MODELS
# =========================================================

logger.info(
    "Saving meta models"
)

primary_model.save_model(

    CHECKPOINT_DIR
    / "primary_directional_model.json"
)

meta_model.save_model(

    CHECKPOINT_DIR
    / "meta_model.json"
)

# =========================================================
# SAVE REPORTS
# =========================================================

importance_df.to_csv(

    REPORT_DIR
    / "meta_feature_importance.csv",

    index=False
)

validation_df[
    [
        "timestamp",
        "future_return_21d",
        "prediction_confidence",
        "prediction_entropy",
        "meta_probability",
        "take_trade"
    ]
].to_csv(

    REPORT_DIR
    / "meta_trade_predictions.csv",

    index=False
)

summary = {

    "meta_accuracy": float(
        meta_accuracy
    ),

    "meta_log_loss": float(
        meta_loss
    ),

    "trade_participation_rate": float(
        meta_predictions.mean()
    ),

    "trade_improvement": float(
        trade_improvement
    )
}

with open(

    REPORT_DIR
    / "meta_labeling_summary.json",

    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )

logger.success(
    "Saved meta labeling outputs"
)

# =========================================================
# FINAL LOGGING
# =========================================================

logger.info("====================================")
logger.info("META LABELING COMPLETE")
logger.info("====================================")