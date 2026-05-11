from pathlib import Path

import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

from loguru import logger

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT / "data/panel/ranked"
)

MODEL_DIR = (
    PROJECT_ROOT / "checkpoints"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "backtests"
)

LOG_DIR = (
    PROJECT_ROOT / "logs"
)

# Create directories
OUTPUT_DIR.mkdir(
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
    LOG_DIR / "cross_sectional_backtest.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# =========================================================
# LOAD DATA
# =========================================================

logger.info(
    "Loading validation ranked dataset"
)

df = pd.read_parquet(
    DATA_DIR / "validation_ranked.parquet"
)

logger.info(
    f"Validation shape -> {df.shape}"
)

# =========================================================
# LOAD MODEL
# =========================================================

logger.info(
    "Loading trained XGBoost model"
)

model = XGBClassifier()

model.load_model(
    MODEL_DIR / "global_xgboost.json"
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

    "cross_sectional_rank_label"
]

FEATURE_COLUMNS = [

    col for col in df.columns
    if col not in EXCLUDED_COLUMNS

]

logger.info(
    f"Using {len(FEATURE_COLUMNS)} features"
)

# =========================================================
# PREPARE FEATURES
# =========================================================

X = df[
    FEATURE_COLUMNS
]

# =========================================================
# MODEL PREDICTIONS
# =========================================================

logger.info(
    "Generating prediction probabilities"
)

prediction_probabilities = model.predict_proba(
    X
)

# =========================================================
# TARGET CLASS PROBABILITY
# =========================================================

# Class mapping:
# 0 -> bearish
# 1 -> neutral
# 2 -> bullish

df["bullish_probability"] = (
    prediction_probabilities[:, 2]
)

df["bearish_probability"] = (
    prediction_probabilities[:, 0]
)

# =========================================================
# LONG SHORT SCORE
# =========================================================

df["alpha_score"] = (

    df["bullish_probability"]

    - df["bearish_probability"]

)

# =========================================================
# PORTFOLIO CONFIGURATION
# =========================================================

TOP_PERCENTILE = 0.10

BOTTOM_PERCENTILE = 0.10

# =========================================================
# BACKTEST LOOP
# =========================================================

logger.info(
    "Running cross-sectional backtest"
)

portfolio_returns = []

unique_timestamps = sorted(
    df["timestamp"].unique()
)[::21]

for timestamp in unique_timestamps:

    daily_df = df[
        df["timestamp"] == timestamp
    ].copy()

    # =====================================================
    # RANK ASSETS
    # =====================================================

    daily_df.sort_values(

        by="alpha_score",

        ascending=False,

        inplace=True
    )

    # =====================================================
    # PORTFOLIO SIZES
    # =====================================================

    n_assets = len(daily_df)

    long_size = max(
        1,
        int(n_assets * TOP_PERCENTILE)
    )

    short_size = max(
        1,
        int(n_assets * BOTTOM_PERCENTILE)
    )

    # =====================================================
    # LONG / SHORT PORTFOLIOS
    # =====================================================

    long_portfolio = daily_df.head(
        long_size
    )

    short_portfolio = daily_df.tail(
        short_size
    )

    # =====================================================
    # FUTURE RETURNS
    # =====================================================

    long_return = (

        long_portfolio[
            "future_return_21d"
        ]

        .mean()

    )

    short_return = (

        short_portfolio[
            "future_return_21d"
        ]

        .mean()

    )

    # =====================================================
    # LONG SHORT RETURN
    # =====================================================

    portfolio_return = (

        long_return

        - short_return

    )

    portfolio_returns.append({

        "timestamp": timestamp,

        "long_return": long_return,

        "short_return": short_return,

        "portfolio_return": portfolio_return,

        "n_assets": n_assets
    })

# =========================================================
# RESULTS DATAFRAME
# =========================================================

results_df = pd.DataFrame(
    portfolio_returns
)

# =========================================================
# PERFORMANCE METRICS
# =========================================================

mean_return = results_df[
    "portfolio_return"
].mean()

std_return = results_df[
    "portfolio_return"
].std()

sharpe_ratio = (

    mean_return

    / std_return

) * np.sqrt(252)

cumulative_return = (

    1 + results_df["portfolio_return"]

).cumprod()

total_return = (
    cumulative_return.iloc[-1] - 1
)

# =========================================================
# MAX DRAWDOWN
# =========================================================

rolling_max = cumulative_return.cummax()

drawdown = (

    cumulative_return

    / rolling_max

) - 1

max_drawdown = drawdown.min()

# =========================================================
# WIN RATE
# =========================================================

win_rate = (

    results_df["portfolio_return"] > 0

).mean()

# =========================================================
# LOG RESULTS
# =========================================================

logger.info("====================================")
logger.info("BACKTEST RESULTS")
logger.info("====================================")

logger.info(
    f"Mean Daily Return: {mean_return:.6f}"
)

logger.info(
    f"Sharpe Ratio: {sharpe_ratio:.4f}"
)

logger.info(
    f"Total Return: {total_return:.4f}"
)

logger.info(
    f"Max Drawdown: {max_drawdown:.4f}"
)

logger.info(
    f"Win Rate: {win_rate:.4f}"
)

logger.info("====================================")

# =========================================================
# SAVE RESULTS
# =========================================================

results_path = (
    OUTPUT_DIR
    / "cross_sectional_backtest.parquet"
)

results_df.to_parquet(
    results_path,
    index=False
)

logger.success(
    f"Saved backtest results -> "
    f"{results_path}"
)