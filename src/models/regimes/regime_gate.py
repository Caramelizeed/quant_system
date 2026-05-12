from pathlib import Path

import pandas as pd
import numpy as np

from loguru import logger

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

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
    / "backtests/regime_gated"
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
    LOG_DIR / "regime_gate.log",
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
# ALIGN REGIMES
# =========================================================

logger.info(
    "Aligning regimes to rebalance dates"
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
# REGIME LEVERAGE MAP
# =========================================================

logger.info(
    "Applying regime leverage mapping"
)

REGIME_LEVERAGE = {

    0: 1.00,

    1: 0.50,

    2: 0.25
}

df["regime_leverage"] = (

    df["regime"]

    .map(REGIME_LEVERAGE)
)

# =========================================================
# GATED RETURNS
# =========================================================

logger.info(
    "Computing gated returns"
)

df["gated_return"] = (

    df["portfolio_return"]

    * df["regime_leverage"]
)

# =========================================================
# CUMULATIVE RETURNS
# =========================================================

df["cumulative_gated_return"] = (

    1 + df["gated_return"]

).cumprod()

df["cumulative_original_return"] = (

    1 + df["portfolio_return"]

).cumprod()

# =========================================================
# PERFORMANCE METRICS
# =========================================================

mean_return = (
    df["gated_return"].mean()
)

volatility = (
    df["gated_return"].std()
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

# =========================================================
# DRAWDOWN
# =========================================================

rolling_max = (

    df["cumulative_gated_return"]

    .cummax()
)

df["drawdown"] = (

    df["cumulative_gated_return"]

    / rolling_max

) - 1

max_drawdown = (
    df["drawdown"].min()
)

# =========================================================
# WIN RATE
# =========================================================

win_rate = (

    df["gated_return"] > 0

).mean()

# =========================================================
# LEVERAGE STATISTICS
# =========================================================

average_leverage = (
    df["regime_leverage"].mean()
)

max_leverage = (
    df["regime_leverage"].max()
)

min_leverage = (
    df["regime_leverage"].min()
)

# =========================================================
# REGIME COUNTS
# =========================================================

regime_counts = (
    df["regime"]
    .value_counts()
)

logger.info(
    f"\nRegime Counts:\n{regime_counts}"
)

# =========================================================
# FINAL RESULTS
# =========================================================

logger.info("====================================")
logger.info("REGIME GATED PERFORMANCE")
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

logger.info(
    f"Average Leverage: "
    f"{average_leverage:.4f}"
)

logger.info(
    f"Max Leverage: "
    f"{max_leverage:.4f}"
)

logger.info(
    f"Min Leverage: "
    f"{min_leverage:.4f}"
)

logger.info("====================================")

# =========================================================
# SAVE OUTPUTS
# =========================================================

logger.info(
    "Saving regime gated outputs"
)

df.to_parquet(

    OUTPUT_DIR
    / "regime_gated_backtest.parquet",

    index=False
)

summary_df = pd.DataFrame({

    "metric": [

        "annualized_return",

        "annualized_volatility",

        "sharpe_ratio",

        "max_drawdown",

        "win_rate",

        "average_leverage",

        "max_leverage",

        "min_leverage"
    ],

    "value": [

        annualized_return,

        annualized_volatility,

        sharpe_ratio,

        max_drawdown,

        win_rate,

        average_leverage,

        max_leverage,

        min_leverage
    ]
})

summary_df.to_csv(

    OUTPUT_DIR
    / "regime_gated_summary.csv",

    index=False
)

logger.success(
    "Saved regime gated outputs"
)

# =========================================================
# FINAL LOGS
# =========================================================

logger.info("====================================")
logger.info("REGIME GATING COMPLETE")
logger.info("====================================")