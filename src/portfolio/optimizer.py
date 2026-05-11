from pathlib import Path

import pandas as pd
import numpy as np

from xgboost import XGBClassifier

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
    PROJECT_ROOT / "backtests/optimized"
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
    LOG_DIR / "optimizer.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# =========================================================
# CONFIGURATION
# =========================================================

TOP_PERCENTILE = 0.10

BOTTOM_PERCENTILE = 0.10

MAX_SECTOR_WEIGHT = 0.25

REBALANCE_FREQUENCY = 21

MAX_POSITION_WEIGHT = 0.10

# =========================================================
# LOAD DATA
# =========================================================

logger.info(
    "Loading ranked validation dataset"
)

df = pd.read_parquet(
    DATA_DIR
    / "validation_ranked.parquet"
)

logger.info(
    f"Loaded shape -> {df.shape}"
)

# =========================================================
# LOAD MODEL
# =========================================================

logger.info(
    "Loading trained XGBoost model"
)

model = XGBClassifier()

model.load_model(
    MODEL_DIR
    / "global_xgboost.json"
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
# PREDICTIONS
# =========================================================

logger.info(
    "Generating prediction probabilities"
)

X = df[
    FEATURE_COLUMNS
]

prediction_probabilities = model.predict_proba(
    X
)

df["bullish_probability"] = (
    prediction_probabilities[:, 2]
)

df["bearish_probability"] = (
    prediction_probabilities[:, 0]
)

df["alpha_score"] = (

    df["bullish_probability"]

    - df["bearish_probability"]

)

# =========================================================
# REBALANCE DATES
# =========================================================

unique_timestamps = sorted(
    df["timestamp"].unique()
)[::REBALANCE_FREQUENCY]

# =========================================================
# BACKTEST LOOP
# =========================================================

logger.info(
    "Running optimized portfolio backtest"
)

portfolio_returns = []

for timestamp in unique_timestamps:

    daily_df = df[
        df["timestamp"] == timestamp
    ].copy()

    # =====================================================
    # SORT ALPHA
    # =====================================================

    daily_df.sort_values(

        by="alpha_score",

        ascending=False,

        inplace=True
    )

    n_assets = len(daily_df)

    target_long_size = max(
        1,
        int(n_assets * TOP_PERCENTILE)
    )

    target_short_size = max(
        1,
        int(n_assets * BOTTOM_PERCENTILE)
    )

    # =====================================================
    # SECTOR LIMITS
    # =====================================================

    max_assets_per_sector_long = max(
        1,
        int(target_long_size * MAX_SECTOR_WEIGHT)
    )

    max_assets_per_sector_short = max(
        1,
        int(target_short_size * MAX_SECTOR_WEIGHT)
    )

    # =====================================================
    # LONG PORTFOLIO
    # =====================================================

    long_portfolio = []

    long_sector_counts = {}

    for _, row in daily_df.iterrows():

        sector = row["sector"]

        count = long_sector_counts.get(
            sector,
            0
        )

        if count < max_assets_per_sector_long:

            long_portfolio.append(row)

            long_sector_counts[sector] = (
                count + 1
            )

        if len(long_portfolio) >= target_long_size:
            break

    # =====================================================
    # SHORT PORTFOLIO
    # =====================================================

    short_portfolio = []

    short_sector_counts = {}

    for _, row in daily_df.iloc[::-1].iterrows():

        sector = row["sector"]

        count = short_sector_counts.get(
            sector,
            0
        )

        if count < max_assets_per_sector_short:

            short_portfolio.append(row)

            short_sector_counts[sector] = (
                count + 1
            )

        if len(short_portfolio) >= target_short_size:
            break

    # =====================================================
    # DATAFRAMES
    # =====================================================

    long_portfolio = pd.DataFrame(
        long_portfolio
    )

    short_portfolio = pd.DataFrame(
        short_portfolio
    )

    # =====================================================
    # INVERSE VOLATILITY WEIGHTS
    # =====================================================

    # Use realized volatility proxy
    long_portfolio["inv_vol"] = (

        1.0

        / long_portfolio[
            "volatility_21d"
        ].clip(lower=1e-6)

    )

    short_portfolio["inv_vol"] = (

        1.0

        / short_portfolio[
            "volatility_21d"
        ].clip(lower=1e-6)

    )

    # =====================================================
    # NORMALIZE WEIGHTS
    # =====================================================

    long_portfolio["weight"] = (

        long_portfolio["inv_vol"]

        / long_portfolio["inv_vol"].sum()

    )

    short_portfolio["weight"] = (

        short_portfolio["inv_vol"]

        / short_portfolio["inv_vol"].sum()

    )

    # =====================================================
    # POSITION CAPS
    # =====================================================

    long_portfolio["weight"] = (

        long_portfolio["weight"]

        .clip(
            upper=MAX_POSITION_WEIGHT
        )

    )

    short_portfolio["weight"] = (

        short_portfolio["weight"]

        .clip(
            upper=MAX_POSITION_WEIGHT
        )

    )

    # Renormalize
    long_portfolio["weight"] = (

        long_portfolio["weight"]

        / long_portfolio["weight"].sum()

    )

    short_portfolio["weight"] = (

        short_portfolio["weight"]

        / short_portfolio["weight"].sum()

    )

    # =====================================================
    # PORTFOLIO RETURNS
    # =====================================================

    long_return = (

        long_portfolio["future_return_21d"]

        * long_portfolio["weight"]

    ).sum()

    short_return = (

        short_portfolio["future_return_21d"]

        * short_portfolio["weight"]

    ).sum()

    portfolio_return = (

        long_return

        - short_return
    )

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    portfolio_returns.append({

        "timestamp": timestamp,

        "long_return": long_return,

        "short_return": short_return,

        "portfolio_return": portfolio_return,

        "long_size": len(long_portfolio),

        "short_size": len(short_portfolio),

        "average_long_weight": (
            long_portfolio["weight"].mean()
        ),

        "average_short_weight": (
            short_portfolio["weight"].mean()
        )
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

mean_return = (
    results_df["portfolio_return"].mean()
)

volatility = (
    results_df["portfolio_return"].std()
)

annualized_return = (
    mean_return * 12
)

annualized_volatility = (
    volatility * np.sqrt(12)
)

sharpe_ratio = (

    annualized_return

    / annualized_volatility
)

# =========================================================
# CUMULATIVE RETURNS
# =========================================================

results_df["cumulative_return"] = (

    1 + results_df["portfolio_return"]

).cumprod()

# =========================================================
# DRAWDOWN
# =========================================================

rolling_max = (

    results_df["cumulative_return"]

    .cummax()
)

results_df["drawdown"] = (

    results_df["cumulative_return"]

    / rolling_max

) - 1

max_drawdown = (
    results_df["drawdown"].min()
)

# =========================================================
# WIN RATE
# =========================================================

win_rate = (

    results_df["portfolio_return"] > 0

).mean()

# =========================================================
# FINAL RESULTS
# =========================================================

logger.info("====================================")
logger.info("OPTIMIZED PORTFOLIO RESULTS")
logger.info("====================================")

logger.info(
    f"Annualized Return: "
    f"{annualized_return:.4f}"
)

logger.info(
    f"Annualized Volatility: "
    f"{annualized_volatility:.4f}"
)

logger.info(
    f"Sharpe Ratio: "
    f"{sharpe_ratio:.4f}"
)

logger.info(
    f"Max Drawdown: "
    f"{max_drawdown:.4f}"
)

logger.info(
    f"Win Rate: "
    f"{win_rate:.4f}"
)

logger.info("====================================")

# =========================================================
# SAVE RESULTS
# =========================================================

results_df.to_parquet(

    OUTPUT_DIR
    / "optimized_backtest.parquet",

    index=False
)

summary_df = pd.DataFrame({

    "metric": [

        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate"

    ],

    "value": [

        annualized_return,
        annualized_volatility,
        sharpe_ratio,
        max_drawdown,
        win_rate
    ]
})

summary_df.to_csv(

    OUTPUT_DIR
    / "optimizer_summary.csv",

    index=False
)

logger.success(
    "Saved optimized portfolio outputs"
)

logger.info("====================================")
logger.info("PORTFOLIO OPTIMIZATION COMPLETE")
logger.info("====================================")