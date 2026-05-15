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
    confusion_matrix
)

from scipy.special import softmax

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
    LOG_DIR / "model_weighting.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# =========================================================
# LOAD DATA
# =========================================================

logger.info(
    "Loading datasets"
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

    final_features = [

        line.strip()

        for line in f.readlines()
    ]

logger.info(
    f"Features -> "
    f"{final_features}"
)

# =========================================================
# TARGET
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
# MATRICES
# =========================================================

logger.info(
    "Building matrices"
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
# CLEANING
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
# MODEL DEFINITIONS
# =========================================================

logger.info(
    "Initializing models"
)

models = {

    "xgboost": XGBClassifier(

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
    ),

    "random_forest": RandomForestClassifier(

        n_estimators=200,

        max_depth=8,

        random_state=42,

        n_jobs=-1
    ),

    "extra_trees": ExtraTreesClassifier(

        n_estimators=200,

        max_depth=8,

        random_state=42,

        n_jobs=-1
    ),

    "ridge": RidgeClassifier(
        alpha=1.0
    )
}

# =========================================================
# TRAIN MODELS
# =========================================================

logger.info(
    "Training models"
)

for name, model in models.items():

    logger.info(
        f"Training {name}"
    )

    model.fit(
        X_train,
        y_train
    )

logger.success(
    "All models trained"
)

# =========================================================
# GENERATE PREDICTIONS
# =========================================================

logger.info(
    "Generating probabilities"
)

model_probabilities = {}

model_metrics = []

for name, model in models.items():

    logger.info(
        f"Inference -> {name}"
    )

    if hasattr(
        model,
        "predict_proba"
    ):

        probs = model.predict_proba(
            X_valid
        )

    else:

        pred = model.predict(
            X_valid
        )

        probs = np.zeros(
            (len(pred), 3)
        )

        for i in range(3):

            probs[:, i] = (
                pred == i
            ).astype(float)

    model_probabilities[
        name
    ] = probs

    predictions = np.argmax(
        probs,
        axis=1
    )

    acc = accuracy_score(
        y_valid,
        predictions
    )

    ll = log_loss(
        y_valid,
        probs
    )

    sharpe_proxy = (
        acc / ll
    )

    model_metrics.append({

        "model": name,

        "accuracy": acc,

        "log_loss": ll,

        "sharpe_proxy": sharpe_proxy
    })

metrics_df = pd.DataFrame(
    model_metrics
)

# =========================================================
# COMPUTE MODEL WEIGHTS
# =========================================================

logger.info(
    "Computing adaptive weights"
)

# Use Sharpe-like score
scores = metrics_df[
    "sharpe_proxy"
].values

weights = softmax(
    scores
)

metrics_df[
    "ensemble_weight"
] = weights

logger.info(
    "\nModel Weights:\n"
    f"{metrics_df}"
)

# =========================================================
# WEIGHTED ENSEMBLE
# =========================================================

logger.info(
    "Building weighted ensemble"
)

ensemble_probs = np.zeros(
    (len(X_valid), 3)
)

for i, (
    model_name,
    probs
) in enumerate(

    model_probabilities.items()
):

    weight = weights[i]

    ensemble_probs += (
        probs * weight
    )

ensemble_predictions = np.argmax(

    ensemble_probs,

    axis=1
)

# =========================================================
# FINAL METRICS
# =========================================================

ensemble_accuracy = accuracy_score(

    y_valid,

    ensemble_predictions
)

ensemble_loss = log_loss(

    y_valid,

    ensemble_probs
)

conf_matrix = confusion_matrix(

    y_valid,

    ensemble_predictions
)

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance_frames = []

tree_models = {

    k: v

    for k, v in models.items()

    if hasattr(
        v,
        "feature_importances_"
    )
}

for name, model in tree_models.items():

    tmp = pd.DataFrame({

        "feature": final_features,

        f"{name}_importance": (
            model.feature_importances_
        )
    })

    importance_frames.append(
        tmp
    )

importance_df = importance_frames[0]

for frame in importance_frames[1:]:

    importance_df = importance_df.merge(

        frame,

        on="feature"
    )

importance_cols = [

    c

    for c in importance_df.columns

    if "importance" in c
]

importance_df[
    "mean_importance"
] = (

    importance_df[
        importance_cols
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
# REPORTING
# =========================================================

logger.info("====================================")
logger.info("WEIGHTED ENSEMBLE RESULTS")
logger.info("====================================")

logger.info(
    f"Accuracy: "
    f"{ensemble_accuracy:.4f}"
)

logger.info(
    f"Log Loss: "
    f"{ensemble_loss:.4f}"
)

logger.info(
    "\nWeighted Models:\n"
    f"{metrics_df}"
)

logger.info(
    "\nFeature Importance:\n"
    f"{importance_df}"
)

logger.info(
    "\nConfusion Matrix:\n"
    f"{conf_matrix}"
)

logger.info("====================================")

# =========================================================
# SAVE OUTPUTS
# =========================================================

logger.info(
    "Saving weighted ensemble outputs"
)

metrics_df.to_csv(

    REPORT_DIR
    / "weighted_model_metrics.csv",

    index=False
)

importance_df.to_csv(

    REPORT_DIR
    / "weighted_ensemble_importance.csv",

    index=False
)

summary = {

    "ensemble_accuracy": float(
        ensemble_accuracy
    ),

    "ensemble_log_loss": float(
        ensemble_loss
    ),

    "features": final_features,

    "weights": {

        row["model"]: float(
            row["ensemble_weight"]
        )

        for _, row in metrics_df.iterrows()
    }
}

with open(

    REPORT_DIR
    / "weighted_ensemble_summary.json",

    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )

logger.success(
    "Saved weighted ensemble outputs"
)

# =========================================================
# FINAL LOGGING
# =========================================================

logger.info("====================================")
logger.info("MODEL WEIGHTING COMPLETE")
logger.info("====================================")