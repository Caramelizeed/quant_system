from pathlib import Path

import numpy as np
import pandas as pd
import shap

from xgboost import XGBClassifier

from loguru import logger

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data/panel/ranked"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints/trainer"
)

FEATURE_STORE_DIR = (
    PROJECT_ROOT
    / "data/feature_store/meta_features"
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
    LOG_DIR / "shap_selection.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# =========================================================
# LOAD DATA
# =========================================================

logger.info(
    "Loading ranked training dataset"
)

df = pd.read_parquet(
    DATA_DIR
    / "train_ranked.parquet"
)

logger.info(
    f"Loaded shape -> "
    f"{df.shape}"
)

# =========================================================
# LOAD FEATURE UNIVERSE
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

# Match trainer.py feature universe
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
    f"Features -> "
    f"{final_features}"
)

# =========================================================
# TARGET PROCESSING
# =========================================================

TARGET_COLUMN = (
    "cross_sectional_rank_label"
)

df[TARGET_COLUMN] = (
    df[TARGET_COLUMN]
    + 1
)

# =========================================================
# FEATURE MATRIX
# =========================================================

logger.info(
    "Building feature matrix"
)

X = df[
    final_features
].copy()

y = df[
    TARGET_COLUMN
].copy()

# =========================================================
# CLEAN MATRIX
# =========================================================

logger.info(
    "Cleaning feature matrix"
)

X.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

X.fillna(
    0,
    inplace=True
)

logger.info(
    f"Clean matrix shape -> "
    f"{X.shape}"
)

# =========================================================
# SAMPLE DATA
# =========================================================

SAMPLE_SIZE = 100000

if len(X) > SAMPLE_SIZE:

    logger.info(
        f"Sampling {SAMPLE_SIZE} rows"
    )

    sample_idx = np.random.choice(

        len(X),

        SAMPLE_SIZE,

        replace=False
    )

    X_sample = X.iloc[
        sample_idx
    ]

    y_sample = y.iloc[
        sample_idx
    ]

else:

    X_sample = X
    y_sample = y

# =========================================================
# LOAD TRAINED MODEL
# =========================================================

logger.info(
    "Loading trainer model"
)

model = XGBClassifier()

model.load_model(

    CHECKPOINT_DIR
    / "xgboost_trainer.json"
)

logger.success(
    "Model loaded"
)

# =========================================================
# COMPUTE SHAP VALUES
# =========================================================

logger.info(
    "Initializing SHAP explainer"
)

explainer = shap.TreeExplainer(
    model
)

logger.info(
    "Computing SHAP values"
)

shap_values = explainer.shap_values(
    X_sample
)

logger.success(
    "SHAP computation complete"
)

# =========================================================
# MULTICLASS SHAP PROCESSING
# =========================================================

logger.info(
    "Processing SHAP outputs"
)

# SHAP output shape:
# (samples, features, classes)

shap_array = np.array(
    shap_values
)

logger.info(
    f"Raw SHAP shape -> "
    f"{shap_array.shape}"
)

# Handle multiclass tensors
if shap_array.ndim == 3:

    # Mean across samples and classes
    mean_shap = np.mean(

        np.abs(shap_array),

        axis=(0, 2)
    )

elif shap_array.ndim == 2:

    mean_shap = np.mean(

        np.abs(shap_array),

        axis=0
    )

else:

    raise ValueError(
        f"Unexpected SHAP dimensions: "
        f"{shap_array.shape}"
    )
# =========================================================
# AGGREGATE IMPORTANCE
# =========================================================

logger.info(
    "Aggregating SHAP importance"
)


importance_df = pd.DataFrame({

    "feature": final_features,

    "mean_abs_shap": mean_shap
})

importance_df.sort_values(

    by="mean_abs_shap",

    ascending=False,

    inplace=True
)

importance_df.reset_index(

    drop=True,

    inplace=True
)

# =========================================================
# NORMALIZED IMPORTANCE
# =========================================================

importance_df[
    "normalized_importance"
] = (

    importance_df[
        "mean_abs_shap"
    ]

    /

    importance_df[
        "mean_abs_shap"
    ].sum()
)

# =========================================================
# SHAP FILTER
# =========================================================

SHAP_THRESHOLD = 0.05

selected_df = importance_df[

    importance_df[
        "normalized_importance"
    ] >= SHAP_THRESHOLD
]

removed_df = importance_df[

    importance_df[
        "normalized_importance"
    ] < SHAP_THRESHOLD
]

selected_features = (
    selected_df[
        "feature"
    ].tolist()
)

# =========================================================
# FINAL REPORTING
# =========================================================

logger.info("====================================")
logger.info("SHAP FEATURE SELECTION SUMMARY")
logger.info("====================================")

logger.info(
    f"Input Features: "
    f"{len(final_features)}"
)

logger.info(
    f"Selected Features: "
    f"{len(selected_features)}"
)

logger.info(
    f"Removed Features: "
    f"{len(removed_df)}"
)

logger.info(
    f"SHAP Threshold: "
    f"{SHAP_THRESHOLD}"
)

logger.info("====================================")

logger.info(
    "\nSHAP Feature Ranking:\n"
    f"{importance_df}"
)

# =========================================================
# SAVE OUTPUTS
# =========================================================

logger.info(
    "Saving SHAP outputs"
)

importance_df.to_csv(

    REPORT_DIR
    / "shap_feature_importance.csv",

    index=False
)

selected_df.to_csv(

    REPORT_DIR
    / "shap_selected_features.csv",

    index=False
)

removed_df.to_csv(

    REPORT_DIR
    / "shap_removed_features.csv",

    index=False
)

with open(

    FEATURE_STORE_DIR
    / "shap_selected_feature_list.txt",

    "w"
) as f:

    for feature in selected_features:

        f.write(
            f"{feature}\n"
        )

logger.success(
    "Saved SHAP outputs"
)

# =========================================================
# FINAL LOGGING
# =========================================================

logger.info("====================================")
logger.info("SHAP FEATURE SELECTION COMPLETE")
logger.info("====================================")