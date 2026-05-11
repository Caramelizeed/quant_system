from pathlib import Path

import pandas as pd
import numpy as np

from loguru import logger

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BACKTEST_DIR = (
    PROJECT_ROOT
    / "backtests/optimized"
)

REGIME_DIR = (
    PROJECT_ROOT
    / "data/feature_store/regimes"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports/regime_analysis"
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
    LOG_DIR / "regime_performance.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# =========================================================
# LOAD DATA
# =========================================================

logger.info(
    "Loading optimized portfolio backtest"
)

backtest_df = pd.read_parquet(
    BACKTEST_DIR
    / "optimized_backtest.parquet"
)

logger.info(
    f"Backtest shape -> "
    f"{backtest_df.shape}"
)

logger.info(
    "Loading market regimes"
)

regime_df = pd.read_parquet(
    REGIME_DIR
    / "market_regimes.parquet"
)

logger.info(
    f"Regime dataframe shape -> "
    f"{regime_df.shape}"
)

# =========================================================
# TIMESTAMP STANDARDIZATION
# =========================================================

backtest_df["timestamp"] = pd.to_datetime(
    backtest_df["timestamp"]
)

regime_df["timestamp"] = pd.to_datetime(
    regime_df["timestamp"]
)

# =========================================================
# SORT DATAFRAMES
# =========================================================

backtest_df.sort_values(
    by="timestamp",
    inplace=True
)

regime_df.sort_values(
    by="timestamp",
    inplace=True
)

# =========================================================
# MERGE ASOF
# =========================================================

logger.info(
    "Aligning backtest with nearest regime"
)

df = pd.merge_asof(

    backtest_df,

    regime_df[[
        "timestamp",
        "regime"
    ]],

    on="timestamp",

    direction="backward"
)

logger.info(
    f"Merged shape -> {df.shape}"
)

# =========================================================
# REGIME COUNTS
# =========================================================

logger.info(
    "Regime counts:"
)

logger.info(
    f"\n{df['regime'].value_counts()}"
)

# =========================================================
# REGIME PERFORMANCE ANALYSIS
# =========================================================

logger.info(
    "Computing regime performance"
)

results = []

for regime in sorted(
    df["regime"].unique()
):

    regime_subset = df[
        df["regime"] == regime
    ].copy()

    # =====================================================
    # BASIC METRICS
    # =====================================================

    mean_return = (
        regime_subset[
            "portfolio_return"
        ].mean()
    )

    volatility = (
        regime_subset[
            "portfolio_return"
        ].std()
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

        if annualized_volatility > 0

        else np.nan
    )

    # =====================================================
    # WIN RATE
    # =====================================================

    win_rate = (

        regime_subset[
            "portfolio_return"
        ] > 0

    ).mean()

    # =====================================================
    # CUMULATIVE RETURNS
    # =====================================================

    cumulative_returns = (

        1

        + regime_subset[
            "portfolio_return"
        ]

    ).cumprod()

    # =====================================================
    # DRAWDOWN
    # =====================================================

    rolling_max = (
        cumulative_returns.cummax()
    )

    drawdown = (

        cumulative_returns

        / rolling_max

    ) - 1

    max_drawdown = (
        drawdown.min()
    )

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    results.append({

        "regime": regime,

        "observations": len(
            regime_subset
        ),

        "annualized_return": (
            annualized_return
        ),

        "annualized_volatility": (
            annualized_volatility
        ),

        "sharpe_ratio": (
            sharpe_ratio
        ),

        "max_drawdown": (
            max_drawdown
        ),

        "win_rate": (
            win_rate
        )
    })

# =========================================================
# RESULTS DATAFRAME
# =========================================================

results_df = pd.DataFrame(
    results
)

# =========================================================
# SORT RESULTS
# =========================================================

results_df.sort_values(

    by="sharpe_ratio",

    ascending=False,

    inplace=True
)

results_df.reset_index(
    drop=True,
    inplace=True
)

# =========================================================
# FINAL LOGGING
# =========================================================

logger.info("====================================")
logger.info("REGIME PERFORMANCE SUMMARY")
logger.info("====================================")

logger.info(
    f"\n{results_df}"
)

logger.info("====================================")

# =========================================================
# SAVE OUTPUTS
# =========================================================

logger.info(
    "Saving regime analysis"
)

results_df.to_csv(

    OUTPUT_DIR
    / "regime_performance.csv",

    index=False
)

logger.success(
    "Saved regime performance report"
)

# =========================================================
# FINAL LOGS
# =========================================================

logger.info("====================================")
logger.info("REGIME PERFORMANCE COMPLETE")
logger.info("====================================")