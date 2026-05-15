from pathlib import Path

import json

import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import RidgeClassifier

from sklearn.metrics import (
    accuracy_score,
    log_loss,
    classification_report,
    confusion_matrix
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
    / "checkpoints/ensemble"
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
# LOGGER
# =========================================================

logger.add(
    LOG_DIR / "ensemble_engine.log",
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
# LOAD FEATURE UNIVERSE
# =========================================================

logger.info(
    "Loading SHAP-selected features"
)

with open(

    FEATURE_STORE_DIR
    / "shap_selected_feature_list.txt",

    "r"
) as f:

    final_features = [

        line.strip()

        for line in f.readlines()
    ]

logger.info(
    f"Final features -> "
    f"{len(final_features)}"
)

logger.info(
    f"Features -> "
    f"{final_features}"
)

# =========================================================
# TARGET PROCESSING
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
# CLEAN MATRICES
# =========================================================

logger.info(
    "Cleaning feature matrices"
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
# MODEL DEFINITIONS
# =========================================================

logger.info(
    "Initializing ensemble models"
)

xgb_model = XGBClassifier(

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

rf_model = RandomForestClassifier(

    n_estimators=200,

    max_depth=8,

    random_state=42,

    n_jobs=-1
)

et_model = ExtraTreesClassifier(

    n_estimators=200,

    max_depth=8,

    random_state=42,

    n_jobs=-1
)

ridge_model = RidgeClassifier(

    alpha=1.0
)

# =========================================================
# TRAIN MODELS
# =========================================================

logger.info(
    "Training XGBoost"
)

xgb_model.fit(
    X_train,
    y_train
)

logger.info(
    "Training Random Forest"
)

rf_model.fit(
    X_train,
    y_train
)

logger.info(
    "Training Extra Trees"
)

et_model.fit(
    X_train,
    y_train
)

logger.info(
    "Training Ridge"
)

ridge_model.fit(
    X_train,
    y_train
)

logger.success(
    "All models trained"
)

# =========================================================
# MODEL PREDICTIONS
# =========================================================

logger.info(
    "Generating prediction probabilities"
)

xgb_probs = xgb_model.predict_proba(
    X_valid
)

rf_probs = rf_model.predict_proba(
    X_valid
)

et_probs = et_model.predict_proba(
    X_valid
)

# Ridge has no predict_proba
ridge_pred = ridge_model.predict(
    X_valid
)

ridge_probs = np.zeros(
    (len(ridge_pred), 3)
)

for i in range(3):

    ridge_probs[:, i] = (
        ridge_pred == i
    ).astype(float)

# =========================================================
# ENSEMBLE BLENDING
# =========================================================

logger.info(
    "Combining ensemble probabilities"
)

# Equal-weight ensemble
ensemble_probs = (

    xgb_probs

    + rf_probs

    + et_probs

    + ridge_probs

) / 4

ensemble_predictions = np.argmax(

    ensemble_probs,

    axis=1
)

# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(

    y_valid,

    ensemble_predictions
)

loss = log_loss(

    y_valid,

    ensemble_probs
)

report = classification_report(

    y_valid,

    ensemble_predictions,

    output_dict=True
)

conf_matrix = confusion_matrix(

    y_valid,

    ensemble_predictions
)

# =========================================================
# MODEL COMPARISON
# =========================================================

logger.info(
    "Evaluating individual models"
)

model_scores = []

individual_models = {

    "xgboost": (
        xgb_model,
        xgb_probs
    ),

    "random_forest": (
        rf_model,
        rf_probs
    ),

    "extra_trees": (
        et_model,
        et_probs
    )
}

for model_name, (
    model,
    probs
) in individual_models.items():

    preds = np.argmax(
        probs,
        axis=1
    )

    acc = accuracy_score(
        y_valid,
        preds
    )

    ll = log_loss(
        y_valid,
        probs
    )

    model_scores.append({

        "model": model_name,

        "accuracy": acc,

        "log_loss": ll
    })

# Add ensemble
model_scores.append({

    "model": "ensemble",

    "accuracy": accuracy,

    "log_loss": loss
})

scores_df = pd.DataFrame(
    model_scores
)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance_df = pd.DataFrame({

    "feature": final_features,

    "xgb_importance": (
        xgb_model.feature_importances_
    ),

    "rf_importance": (
        rf_model.feature_importances_
    ),

    "et_importance": (
        et_model.feature_importances_
    )
})

importance_df[
    "mean_importance"
] = (

    importance_df[
        [
            "xgb_importance",
            "rf_importance",
            "et_importance"
        ]
    ].mean(axis=1)
)

importance_df.sort_values(

    by="mean_importance",

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
logger.info("ENSEMBLE RESULTS")
logger.info("====================================")

logger.info(
    f"Ensemble Accuracy: "
    f"{accuracy:.4f}"
)

logger.info(
    f"Ensemble Log Loss: "
    f"{loss:.4f}"
)

logger.info(
    "\nModel Comparison:\n"
    f"{scores_df}"
)

logger.info(
    "\nTop Features:\n"
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
    "Saving ensemble artifacts"
)

xgb_model.save_model(

    CHECKPOINT_DIR
    / "ensemble_xgboost.json"
)

import joblib

joblib.dump(

    rf_model,

    CHECKPOINT_DIR
    / "random_forest.pkl"
)

joblib.dump(

    et_model,

    CHECKPOINT_DIR
    / "extra_trees.pkl"
)

joblib.dump(

    ridge_model,

    CHECKPOINT_DIR
    / "ridge_model.pkl"
)

# =========================================================
# SAVE REPORTS
# =========================================================

scores_df.to_csv(

    REPORT_DIR
    / "ensemble_model_scores.csv",

    index=False
)

importance_df.to_csv(

    REPORT_DIR
    / "ensemble_feature_importance.csv",

    index=False
)

metrics = {

    "ensemble_accuracy": float(
        accuracy
    ),

    "ensemble_log_loss": float(
        loss
    ),

    "num_features": len(
        final_features
    ),

    "features": final_features
}

with open(

    REPORT_DIR
    / "ensemble_metrics.json",

    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

logger.success(
    "Saved ensemble outputs"
)

# =========================================================
# FINAL LOGGING
# =========================================================

logger.info("====================================")
logger.info("ENSEMBLE ENGINE COMPLETE")
logger.info("====================================")