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
    BACKTEST_DIR / "scaled"
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
    LOG_DIR / "volatility_scaling.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# =========================================================
# CONFIGURATION
# =========================================================

TARGET_ANNUAL_VOLATILITY = 0.15

ROLLING_WINDOW = 20

TRADING_DAYS = 252/21

MAX_LEVERAGE = 3.0

MIN_LEVERAGE = 0.25

# =========================================================
# LOAD BACKTEST RESULTS
# =========================================================

logger.info(
    "Loading backtest returns"
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
# ROLLING VOLATILITY
# =========================================================

logger.info(
    "Computing rolling volatility"
)

df["rolling_volatility"] = (

    df["portfolio_return"]

    .rolling(ROLLING_WINDOW)

    .std()

) * np.sqrt(TRADING_DAYS)

# =========================================================
# VOLATILITY SCALING
# =========================================================

logger.info(
    "Computing volatility scaling"
)

df["target_leverage"] = (

    TARGET_ANNUAL_VOLATILITY

    / df["rolling_volatility"]

)

# =========================================================
# LEVERAGE CONSTRAINTS
# =========================================================

df["target_leverage"] = (

    df["target_leverage"]

    .clip(
        lower=MIN_LEVERAGE,
        upper=MAX_LEVERAGE
    )

)

# =========================================================
# SHIFT LEVERAGE
# =========================================================

# Prevent forward-looking bias

df["applied_leverage"] = (

    df["target_leverage"]

    .shift(1)

)

# =========================================================
# REMOVE NAN ROWS
# =========================================================

before_rows = len(df)

df.dropna(inplace=True)

after_rows = len(df)

logger.info(
    f"Removed {before_rows - after_rows} rows"
)

# =========================================================
# SCALED RETURNS
# =========================================================

logger.info(
    "Computing scaled returns"
)

df["scaled_return"] = (

    df["portfolio_return"]

    * df["applied_leverage"]

)

# =========================================================
# CUMULATIVE RETURNS
# =========================================================

logger.info(
    "Computing cumulative returns"
)

df["cumulative_scaled_return"] = (

    1 + df["scaled_return"]

).cumprod()

# =========================================================
# PERFORMANCE METRICS
# =========================================================

mean_return = (
    df["scaled_return"].mean()
)

volatility = (
    df["scaled_return"].std()
)

annualized_return = (
    mean_return * TRADING_DAYS
)

annualized_volatility = (
    volatility * np.sqrt(TRADING_DAYS)
)

sharpe_ratio = (

    annualized_return

    / annualized_volatility

)

# =========================================================
# DRAWDOWN ANALYSIS
# =========================================================

rolling_max = (

    df["cumulative_scaled_return"]

    .cummax()
)

df["drawdown"] = (

    df["cumulative_scaled_return"]

    / rolling_max

) - 1

max_drawdown = (
    df["drawdown"].min()
)

# =========================================================
# WIN RATE
# =========================================================

win_rate = (

    df["scaled_return"] > 0

).mean()

# =========================================================
# LEVERAGE STATISTICS
# =========================================================

average_leverage = (
    df["applied_leverage"].mean()
)

max_leverage = (
    df["applied_leverage"].max()
)

min_leverage = (
    df["applied_leverage"].min()
)

# =========================================================
# FINAL METRICS
# =========================================================

logger.info("====================================")
logger.info("VOLATILITY SCALING RESULTS")
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
    "Saving scaled portfolio results"
)

df.to_parquet(

    OUTPUT_DIR
    / "scaled_portfolio.parquet",

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
    / "volatility_scaling_summary.csv",

    index=False
)

logger.success(
    "Saved volatility scaling outputs"
)

logger.info("====================================")
logger.info("VOLATILITY SCALING COMPLETE")
logger.info("====================================")