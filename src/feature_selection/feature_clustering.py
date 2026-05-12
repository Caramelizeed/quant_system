from pathlib import Path

import pandas as pd
import numpy as np

from scipy.cluster.hierarchy import (
    linkage,
    fcluster
)

from scipy.spatial.distance import squareform

from loguru import logger

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data/panel/ranked"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports/diagnostics"
)

FEATURE_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/feature_store/meta_features"
)

LOG_DIR = (
    PROJECT_ROOT / "logs"
)

# Create directories
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FEATURE_OUTPUT_DIR.mkdir(
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
    LOG_DIR / "feature_clustering.log",
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
    f"Loaded shape -> {df.shape}"
)

# =========================================================
# FEATURE LIST
# =========================================================

logger.info(
    "Loading selected features"
)

selected_features_path = (
    FEATURE_OUTPUT_DIR
    / "selected_feature_list.txt"
)

with open(
    selected_features_path,
    "r"
) as f:

    selected_features = [

        line.strip()

        for line in f.readlines()
    ]

logger.info(
    f"Selected features -> "
    f"{len(selected_features)}"
)

# =========================================================
# FEATURE MATRIX
# =========================================================

feature_df = df[
    selected_features
].copy()

# =========================================================
# REMOVE NAN / INF
# =========================================================

logger.info(
    "Cleaning feature matrix"
)

feature_df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

feature_df.dropna(
    inplace=True
)

logger.info(
    f"Clean feature matrix shape -> "
    f"{feature_df.shape}"
)

# =========================================================
# CORRELATION MATRIX
# =========================================================

logger.info(
    "Computing correlation matrix"
)

correlation_matrix = (
    feature_df.corr(
        method="spearman"
    )
)

# =========================================================
# DISTANCE MATRIX
# =========================================================

logger.info(
    "Computing distance matrix"
)

distance_matrix = (

    1

    - correlation_matrix.abs()
)

# =========================================================
# HIERARCHICAL CLUSTERING
# =========================================================

logger.info(
    "Running hierarchical clustering"
)

condensed_distance = squareform(
    distance_matrix.values,
    checks=False
)

linkage_matrix = linkage(

    condensed_distance,

    method="average"
)

# =========================================================
# ASSIGN CLUSTERS
# =========================================================

CLUSTER_THRESHOLD = 0.35

cluster_labels = fcluster(

    linkage_matrix,

    t=CLUSTER_THRESHOLD,

    criterion="distance"
)

# =========================================================
# CLUSTER RESULTS
# =========================================================

cluster_df = pd.DataFrame({

    "feature": selected_features,

    "cluster": cluster_labels
})

cluster_df.sort_values(

    by=["cluster", "feature"],

    inplace=True
)

cluster_df.reset_index(

    drop=True,

    inplace=True
)

# =========================================================
# CLUSTER SUMMARY
# =========================================================

logger.info(
    "Building cluster summary"
)

cluster_summary = []

unique_clusters = sorted(
    cluster_df["cluster"].unique()
)

for cluster_id in unique_clusters:

    subset = cluster_df[
        cluster_df["cluster"] == cluster_id
    ]

    features = subset[
        "feature"
    ].tolist()

    cluster_summary.append({

        "cluster": cluster_id,

        "feature_count": len(features),

        "features": ", ".join(features)
    })

summary_df = pd.DataFrame(
    cluster_summary
)

# =========================================================
# REPRESENTATIVE FEATURES
# =========================================================

logger.info(
    "Selecting representative features"
)

representative_features = []

for cluster_id in unique_clusters:

    subset = cluster_df[
        cluster_df["cluster"] == cluster_id
    ]

    representative_feature = (
        subset.iloc[0]["feature"]
    )

    representative_features.append(
        representative_feature
    )

representative_df = pd.DataFrame({

    "feature": representative_features
})

# =========================================================
# FINAL LOGGING
# =========================================================

logger.info("====================================")
logger.info("FEATURE CLUSTERING SUMMARY")
logger.info("====================================")

logger.info(
    f"Input Features: "
    f"{len(selected_features)}"
)

logger.info(
    f"Clusters Found: "
    f"{len(unique_clusters)}"
)

logger.info(
    f"Representative Features: "
    f"{len(representative_features)}"
)

logger.info("====================================")

logger.info(
    "\nCluster Summary:\n"
    f"{summary_df.head(20)}"
)

# =========================================================
# SAVE OUTPUTS
# =========================================================

logger.info(
    "Saving clustering outputs"
)

cluster_df.to_csv(

    OUTPUT_DIR
    / "feature_clusters.csv",

    index=False
)

summary_df.to_csv(

    OUTPUT_DIR
    / "feature_cluster_summary.csv",

    index=False
)

representative_df.to_csv(

    OUTPUT_DIR
    / "representative_features.csv",

    index=False
)

# Save representative feature list
with open(

    FEATURE_OUTPUT_DIR
    / "representative_feature_list.txt",

    "w"
) as f:

    for feature in representative_features:

        f.write(
            f"{feature}\n"
        )

logger.success(
    "Saved clustering outputs"
)

# =========================================================
# FINAL LOGS
# =========================================================

logger.info("====================================")
logger.info("FEATURE CLUSTERING COMPLETE")
logger.info("====================================")