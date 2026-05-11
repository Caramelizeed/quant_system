from pathlib import Path

import pandas as pd
import numpy as np

from loguru import logger

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BACKTEST_DIR = (
    PROJECT_ROOT / "backtests"
)

OUTPUT_DIR = (
    BACKTEST_DIR / "attribution"
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
    LOG_DIR / "performance_attribution.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# =========================================================
# LOAD BACKTEST RESULTS
# =========================================================

logger.info(
    "Loading backtest results"
)

df = pd.read_parquet(
    BACKTEST_DIR
    / "cross_sectional_backtest.parquet"
)

logger.info(
    f"Loaded shape -> {df.shape}"
)

# =========================================================
# TIMESTAMP STANDARDIZATION
# =========================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df.sort_values(
    by="timestamp",
    inplace=True
)

df.reset_index(
    drop=True,
    inplace=True
)

# =========================================================
# CUMULATIVE RETURNS
# =========================================================

logger.info(
    "Computing cumulative returns"
)

df["cumulative_return"] = (

    1 + df["portfolio_return"]

).cumprod()

# =========================================================
# ROLLING SHARPE
# =========================================================

logger.info(
    "Computing rolling Sharpe ratio"
)

ROLLING_WINDOW = 20

rolling_mean = (

    df["portfolio_return"]

    .rolling(ROLLING_WINDOW)

    .mean()
)

rolling_std = (

    df["portfolio_return"]

    .rolling(ROLLING_WINDOW)

    .std()
)

df["rolling_sharpe"] = (

    rolling_mean

    / rolling_std

) * np.sqrt(252)

# =========================================================
# DRAWDOWN ANALYSIS
# =========================================================

logger.info(
    "Computing drawdowns"
)

rolling_max = (

    df["cumulative_return"]

    .cummax()
)

df["drawdown"] = (

    df["cumulative_return"]

    / rolling_max

) - 1

max_drawdown = (
    df["drawdown"].min()
)

# =========================================================
# LONG VS SHORT ATTRIBUTION
# =========================================================

logger.info(
    "Computing long-short attribution"
)

df["long_short_spread"] = (

    df["long_return"]

    - df["short_return"]

)

average_long_return = (
    df["long_return"].mean()
)

average_short_return = (
    df["short_return"].mean()
)

average_spread = (
    df["long_short_spread"].mean()
)

# =========================================================
# WIN RATE
# =========================================================

logger.info(
    "Computing win rate"
)

win_rate = (

    df["portfolio_return"] > 0

).mean()

# =========================================================
# VOLATILITY
# =========================================================

logger.info(
    "Computing volatility statistics"
)

annualized_volatility = (

    df["portfolio_return"].std()

) * np.sqrt(252)

# =========================================================
# SUMMARY METRICS
# =========================================================

logger.info("====================================")
logger.info("PERFORMANCE ATTRIBUTION SUMMARY")
logger.info("====================================")

logger.info(
    f"Average Long Return: "
    f"{average_long_return:.6f}"
)

logger.info(
    f"Average Short Return: "
    f"{average_short_return:.6f}"
)

logger.info(
    f"Average Long-Short Spread: "
    f"{average_spread:.6f}"
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
    f"Annualized Volatility: "
    f"{annualized_volatility:.4f}"
)

logger.info("====================================")

# =========================================================
# SAVE OUTPUTS
# =========================================================

logger.info(
    "Saving attribution outputs"
)

df.to_parquet(

    OUTPUT_DIR
    / "performance_attribution.parquet",

    index=False
)

summary_df = pd.DataFrame({

    "metric": [

        "average_long_return",
        "average_short_return",
        "average_spread",
        "max_drawdown",
        "win_rate",
        "annualized_volatility"

    ],

    "value": [

        average_long_return,
        average_short_return,
        average_spread,
        max_drawdown,
        win_rate,
        annualized_volatility
    ]
})

summary_df.to_csv(

    OUTPUT_DIR
    / "summary_metrics.csv",

    index=False
)

logger.success(
    "Saved attribution reports"
)

logger.info("====================================")
logger.info("PERFORMANCE ATTRIBUTION COMPLETE")
logger.info("====================================")