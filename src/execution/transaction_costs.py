from pathlib import Path

import pandas as pd
import numpy as np

from loguru import logger

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    PROJECT_ROOT / "backtests/optimized"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "backtests/cost_adjusted"
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
    LOG_DIR / "transaction_costs.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# =========================================================
# CONFIGURATION
# =========================================================

# Transaction cost assumptions
COMMISSION_BPS = 5

SLIPPAGE_BPS = 10

TOTAL_COST_BPS = (

    COMMISSION_BPS

    + SLIPPAGE_BPS
)

# Convert bps to decimal
TOTAL_COST = (
    TOTAL_COST_BPS / 10000
)

# Approximate turnover assumption
# (full rebalance every period)

TURNOVER = 1.0

# Rebalance periods per year
PERIODS_PER_YEAR = 12

# =========================================================
# LOAD OPTIMIZED BACKTEST
# =========================================================

logger.info(
    "Loading optimized portfolio backtest"
)

df = pd.read_parquet(
    INPUT_DIR
    / "optimized_backtest.parquet"
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
# TRANSACTION COSTS
# =========================================================

logger.info(
    "Applying transaction costs"
)

# Cost per rebalance
df["transaction_cost"] = (

    TURNOVER

    * TOTAL_COST
)

# =========================================================
# COST-ADJUSTED RETURNS
# =========================================================

df["cost_adjusted_return"] = (

    df["portfolio_return"]

    - df["transaction_cost"]
)

# =========================================================
# CUMULATIVE RETURNS
# =========================================================

logger.info(
    "Computing cumulative returns"
)

df["cumulative_gross_return"] = (

    1 + df["portfolio_return"]

).cumprod()

df["cumulative_net_return"] = (

    1 + df["cost_adjusted_return"]

).cumprod()

# =========================================================
# PERFORMANCE METRICS
# =========================================================

gross_mean = (
    df["portfolio_return"].mean()
)

net_mean = (
    df["cost_adjusted_return"].mean()
)

gross_volatility = (
    df["portfolio_return"].std()
)

net_volatility = (
    df["cost_adjusted_return"].std()
)

# =========================================================
# ANNUALIZED METRICS
# =========================================================

gross_annual_return = (
    gross_mean * PERIODS_PER_YEAR
)

net_annual_return = (
    net_mean * PERIODS_PER_YEAR
)

gross_annual_volatility = (

    gross_volatility

    * np.sqrt(PERIODS_PER_YEAR)
)

net_annual_volatility = (

    net_volatility

    * np.sqrt(PERIODS_PER_YEAR)
)

gross_sharpe = (

    gross_annual_return

    / gross_annual_volatility
)

net_sharpe = (

    net_annual_return

    / net_annual_volatility
)

# =========================================================
# DRAWDOWN
# =========================================================

rolling_max = (

    df["cumulative_net_return"]

    .cummax()
)

df["drawdown"] = (

    df["cumulative_net_return"]

    / rolling_max

) - 1

max_drawdown = (
    df["drawdown"].min()
)

# =========================================================
# WIN RATE
# =========================================================

gross_win_rate = (

    df["portfolio_return"] > 0

).mean()

net_win_rate = (

    df["cost_adjusted_return"] > 0

).mean()

# =========================================================
# COST IMPACT
# =========================================================

average_cost = (
    df["transaction_cost"].mean()
)

total_cost_paid = (
    df["transaction_cost"].sum()
)

# =========================================================
# FINAL RESULTS
# =========================================================

logger.info("====================================")
logger.info("TRANSACTION COST ANALYSIS")
logger.info("====================================")

logger.info(
    f"Gross Annual Return: "
    f"{gross_annual_return:.4f}"
)

logger.info(
    f"Net Annual Return: "
    f"{net_annual_return:.4f}"
)

logger.info(
    f"Gross Sharpe Ratio: "
    f"{gross_sharpe:.4f}"
)

logger.info(
    f"Net Sharpe Ratio: "
    f"{net_sharpe:.4f}"
)

logger.info(
    f"Max Drawdown: "
    f"{max_drawdown:.4f}"
)

logger.info(
    f"Gross Win Rate: "
    f"{gross_win_rate:.4f}"
)

logger.info(
    f"Net Win Rate: "
    f"{net_win_rate:.4f}"
)

logger.info(
    f"Average Transaction Cost: "
    f"{average_cost:.6f}"
)

logger.info(
    f"Total Cost Paid: "
    f"{total_cost_paid:.6f}"
)

logger.info("====================================")

# =========================================================
# SAVE RESULTS
# =========================================================

logger.info(
    "Saving cost-adjusted results"
)

df.to_parquet(

    OUTPUT_DIR
    / "cost_adjusted_backtest.parquet",

    index=False
)

summary_df = pd.DataFrame({

    "metric": [

        "gross_annual_return",
        "net_annual_return",

        "gross_sharpe",
        "net_sharpe",

        "max_drawdown",

        "gross_win_rate",
        "net_win_rate",

        "average_transaction_cost",
        "total_cost_paid"
    ],

    "value": [

        gross_annual_return,
        net_annual_return,

        gross_sharpe,
        net_sharpe,

        max_drawdown,

        gross_win_rate,
        net_win_rate,

        average_cost,
        total_cost_paid
    ]
})

summary_df.to_csv(

    OUTPUT_DIR
    / "transaction_cost_summary.csv",

    index=False
)

logger.success(
    "Saved transaction cost outputs"
)

logger.info("====================================")
logger.info("TRANSACTION COST PIPELINE COMPLETE")
logger.info("====================================")