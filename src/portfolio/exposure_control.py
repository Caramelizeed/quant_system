from pathlib import Path

import pandas as pd
import numpy as np

from loguru import logger

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    PROJECT_ROOT / "reports/regime_allocator"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "reports/exposure_control"
)

LOG_DIR = (
    PROJECT_ROOT / "logs"
)

OUTPUT_DIR.mkdir(
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
    LOG_DIR / "exposure_control.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# =========================================================
# CONFIGURATION
# =========================================================

MAX_GROSS_EXPOSURE = 1.00

MAX_SINGLE_POSITION = 0.10

MAX_POSITIONS = 20

VOLATILITY_SHUTDOWN_THRESHOLD = 0.40

DRAWDOWN_SHUTDOWN_THRESHOLD = -0.25

ROLLING_VOL_WINDOW = 20

# =========================================================
# LOAD REGIME ALLOCATOR OUTPUT
# =========================================================

logger.info(
    "Loading regime allocator equity curve"
)

df = pd.read_csv(
    INPUT_DIR
    / "regime_equity_curve.csv"
)

logger.info(
    f"Loaded shape -> "
    f"{df.shape}"
)

# =========================================================
# TIMESTAMP PROCESSING
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
# BASIC RETURNS
# =========================================================

returns = df[
    "portfolio_return"
].copy()

# =========================================================
# ROLLING VOLATILITY
# =========================================================

logger.info(
    "Computing rolling volatility"
)

df["rolling_volatility"] = (

    returns

    .rolling(
        ROLLING_VOL_WINDOW
    )

    .std()
)

# =========================================================
# EQUITY CURVE
# =========================================================

logger.info(
    "Computing equity curve"
)

df["equity_curve"] = (

    1 + returns

).cumprod()

# =========================================================
# DRAWDOWNS
# =========================================================

rolling_max = (

    df["equity_curve"]

    .cummax()
)

df["drawdown"] = (

    df["equity_curve"]

    / rolling_max

) - 1

# =========================================================
# RISK FLAGS
# =========================================================

logger.info(
    "Applying risk controls"
)

df["volatility_shutdown"] = (

    df["rolling_volatility"]

    > VOLATILITY_SHUTDOWN_THRESHOLD
)

df["drawdown_shutdown"] = (

    df["drawdown"]

    < DRAWDOWN_SHUTDOWN_THRESHOLD
)

df["trading_enabled"] = ~(
    df["volatility_shutdown"]
    |
    df["drawdown_shutdown"]
)

# =========================================================
# EXPOSURE SCALING
# =========================================================

logger.info(
    "Applying dynamic exposure scaling"
)

df["exposure_multiplier"] = 1.0

# reduce exposure in high vol

high_vol_mask = (

    df["rolling_volatility"]

    > 0.20
)

df.loc[
    high_vol_mask,
    "exposure_multiplier"
] = 0.50

# reduce further in severe vol

extreme_vol_mask = (

    df["rolling_volatility"]

    > 0.30
)

df.loc[
    extreme_vol_mask,
    "exposure_multiplier"
] = 0.25

# disable trading entirely

df.loc[
    ~df["trading_enabled"],
    "exposure_multiplier"
] = 0.0

# =========================================================
# APPLY EXPOSURE CONTROL
# =========================================================

logger.info(
    "Computing controlled returns"
)

df["controlled_return"] = (

    df["portfolio_return"]

    * df["exposure_multiplier"]
)

# =========================================================
# CONTROLLED EQUITY CURVE
# =========================================================

df["controlled_equity_curve"] = (

    1 + df["controlled_return"]

).cumprod()

# =========================================================
# FINAL METRICS
# =========================================================

logger.info(
    "Computing final metrics"
)

mean_return = (
    df["controlled_return"]
    .mean()
)

volatility = (
    df["controlled_return"]
    .std()
)

if volatility > 0:

    sharpe = (
        mean_return
        / volatility
    )

else:

    sharpe = 0

rolling_max = (

    df["controlled_equity_curve"]

    .cummax()
)

controlled_drawdown = (

    df["controlled_equity_curve"]

    / rolling_max

) - 1

max_drawdown = (
    controlled_drawdown.min()
)

positive_rate = (

    df["controlled_return"]

    > 0

).mean()

average_exposure = (

    df["exposure_multiplier"]

    .mean()
)

# =========================================================
# RESULTS
# =========================================================

logger.info("====================================")
logger.info("EXPOSURE CONTROL RESULTS")
logger.info("====================================")

logger.info(
    f"Mean Return -> "
    f"{mean_return:.6f}"
)

logger.info(
    f"Volatility -> "
    f"{volatility:.6f}"
)

logger.info(
    f"Sharpe -> "
    f"{sharpe:.6f}"
)

logger.info(
    f"Max Drawdown -> "
    f"{max_drawdown:.6f}"
)

logger.info(
    f"Positive Rate -> "
    f"{positive_rate:.6f}"
)

logger.info(
    f"Average Exposure -> "
    f"{average_exposure:.6f}"
)

logger.info("====================================")

# =========================================================
# SAVE OUTPUTS
# =========================================================

logger.info(
    "Saving exposure control outputs"
)

df.to_csv(

    OUTPUT_DIR
    / "exposure_controlled_returns.csv",

    index=False
)

summary_df = pd.DataFrame({

    "metric": [

        "mean_return",
        "volatility",
        "sharpe",
        "max_drawdown",
        "positive_rate",
        "average_exposure"

    ],

    "value": [

        mean_return,
        volatility,
        sharpe,
        max_drawdown,
        positive_rate,
        average_exposure
    ]
})

summary_df.to_csv(

    OUTPUT_DIR
    / "exposure_control_summary.csv",

    index=False
)

logger.success(
    "Saved exposure control outputs"
)

logger.info("====================================")
logger.info("EXPOSURE CONTROL COMPLETE")
logger.info("====================================")