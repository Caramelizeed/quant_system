from pathlib import Path

import json

import numpy as np
import pandas as pd

from loguru import logger

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = (
    PROJECT_ROOT
    / "reports/diagnostics"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports/portfolio"
)

LOG_DIR = (
    PROJECT_ROOT
    / "logs"
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
    LOG_DIR / "confidence_position_sizing.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# =========================================================
# LOAD META PREDICTIONS
# =========================================================

logger.info(
    "Loading meta trade predictions"
)

df = pd.read_csv(

    REPORT_DIR
    / "meta_trade_predictions.csv"
)

logger.info(
    f"Loaded shape -> "
    f"{df.shape}"
)

# =========================================================
# CLEAN DATA
# =========================================================

logger.info(
    "Cleaning portfolio inputs"
)

df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

df.dropna(
    inplace=True
)

df.reset_index(
    drop=True,
    inplace=True
)

# =========================================================
# FILTER EXECUTED TRADES
# =========================================================

logger.info(
    "Filtering approved trades"
)

df = df[
    df["take_trade"] == 1
].copy()

logger.info(
    f"Approved trades -> "
    f"{len(df)}"
)

# =========================================================
# CLIP EXTREME RETURNS
# =========================================================

logger.info(
    "Clipping extreme returns"
)

df[
    "future_return_21d"
] = (

    df["future_return_21d"]

    .clip(
        lower=-0.50,
        upper=0.50
    )
)

# =========================================================
# PORTFOLIO CONSTRUCTION PARAMETERS
# =========================================================

logger.info(
    "Initializing portfolio constraints"
)

MIN_POSITIONS = 10

MAX_POSITIONS = 25

MAX_POSITION_WEIGHT = 0.10

MIN_META_PROBABILITY = 0.55

# =========================================================
# FILTER HIGH-CONFIDENCE TRADES
# =========================================================

logger.info(
    "Filtering high-confidence trades"
)

df = df[

    df["meta_probability"]

    >= MIN_META_PROBABILITY

].copy()

logger.info(
    f"Remaining trades -> "
    f"{len(df)}"
)

# =========================================================
# BUILD TIMESTAMP PORTFOLIOS
# =========================================================

logger.info(
    "Building diversified portfolios"
)

portfolio_frames = []

unique_timestamps = sorted(
    df["timestamp"].unique()
)

for timestamp in unique_timestamps:

    tmp = df[
        df["timestamp"] == timestamp
    ].copy()

    # rank by confidence

    tmp.sort_values(

        by="meta_probability",

        ascending=False,

        inplace=True
    )

    # keep top positions

    tmp = tmp.head(
        MAX_POSITIONS
    )

    # skip weakly diversified periods

    if len(tmp) < MIN_POSITIONS:

        continue

    # =====================================================
    # CONFIDENCE EDGE
    # =====================================================

    tmp[
        "confidence_edge"
    ] = (

        tmp["meta_probability"]

        - 0.50
    )

    tmp[
        "confidence_edge"
    ] = (

        tmp["confidence_edge"]

        .clip(lower=0)
    )

    # =====================================================
    # VOLATILITY SCALING
    # =====================================================

    tmp[
        "volatility_proxy"
    ] = (

        tmp["prediction_entropy"]

        + 1e-6
    )

    tmp[
        "raw_weight"
    ] = (

        tmp["confidence_edge"]

        /

        tmp["volatility_proxy"]
    )

    # =====================================================
    # NORMALIZE WEIGHTS
    # =====================================================

    weight_sum = (

        tmp["raw_weight"]

        .abs()

        .sum()
    )

    tmp[
        "portfolio_weight"
    ] = (

        tmp["raw_weight"]

        /

        (
            weight_sum
            + 1e-12
        )
    )

    # =====================================================
    # POSITION CAPS
    # =====================================================

    tmp[
        "portfolio_weight"
    ] = (

        tmp["portfolio_weight"]

        .clip(

            lower=-MAX_POSITION_WEIGHT,

            upper=MAX_POSITION_WEIGHT
        )
    )

    # renormalize after clipping

    renorm = (

        tmp["portfolio_weight"]

        .abs()

        .sum()
    )

    tmp[
        "portfolio_weight"
    ] = (

        tmp["portfolio_weight"]

        /

        (
            renorm
            + 1e-12
        )
    )

    portfolio_frames.append(
        tmp
    )

# =========================================================
# FINAL PORTFOLIO DATAFRAME
# =========================================================

df = pd.concat(
    portfolio_frames,
    ignore_index=True
)

logger.info(
    f"Final portfolio rows -> "
    f"{len(df)}"
)

# =========================================================
# COMPUTE PORTFOLIO RETURNS
# =========================================================

logger.info(
    "Computing diversified portfolio returns"
)

df[
    "weighted_return"
] = (

    df["portfolio_weight"]

    * df["future_return_21d"]
)

timestamp_returns = (

    df.groupby("timestamp")[
        "weighted_return"
    ]

    .sum()
)

# =========================================================
# PORTFOLIO METRICS
# =========================================================

portfolio_return = (
    timestamp_returns.mean()
)

portfolio_volatility = (
    timestamp_returns.std()
)

if portfolio_volatility > 0:

    sharpe_ratio = (

        portfolio_return

        / portfolio_volatility
    )

else:

    sharpe_ratio = 0

# =========================================================
# DRAWDOWN
# =========================================================

equity_curve = (

    1 + timestamp_returns
).cumprod()

rolling_max = (
    equity_curve.cummax()
)

drawdowns = (

    equity_curve

    / rolling_max

    - 1
)

max_drawdown = (
    drawdowns.min()
)

# =========================================================
# DIAGNOSTICS
# =========================================================

positive_rate = (
    (timestamp_returns > 0)
    .mean()
)

best_period = (
    timestamp_returns.max()
)

worst_period = (
    timestamp_returns.min()
)

avg_positions = (

    df.groupby("timestamp")
    .size()
    .mean()
)

gross_leverage = (

    df.groupby("timestamp")[
        "portfolio_weight"
    ]

    .apply(
        lambda x: x.abs().sum()
    )

    .mean()
)

# =========================================================
# FINAL REPORTING
# =========================================================

logger.info("====================================")
logger.info("DIVERSIFIED POSITION SIZING RESULTS")
logger.info("====================================")

logger.info(
    f"Mean Portfolio Return: "
    f"{portfolio_return:.6f}"
)

logger.info(
    f"Portfolio Volatility: "
    f"{portfolio_volatility:.6f}"
)

logger.info(
    f"Portfolio Sharpe: "
    f"{sharpe_ratio:.6f}"
)

logger.info(
    f"Max Drawdown: "
    f"{max_drawdown:.6f}"
)

logger.info(
    f"Positive Period Rate: "
    f"{positive_rate:.4f}"
)

logger.info(
    f"Average Positions: "
    f"{avg_positions:.2f}"
)

logger.info(
    f"Gross Leverage: "
    f"{gross_leverage:.4f}"
)

logger.info(
    f"Best Period Return: "
    f"{best_period:.6f}"
)

logger.info(
    f"Worst Period Return: "
    f"{worst_period:.6f}"
)

logger.info("====================================")
# =========================================================
# SAVE OUTPUTS
# =========================================================

logger.info(
    "Saving portfolio outputs"
)

df.to_csv(

    OUTPUT_DIR
    / "confidence_weighted_portfolio.csv",

    index=False
)

timestamp_returns.to_csv(

    OUTPUT_DIR
    / "timestamp_portfolio_returns.csv"
)

summary = {

    "mean_portfolio_return": float(
        portfolio_return
    ),

    "portfolio_volatility": float(
        portfolio_volatility
    ),

    "portfolio_sharpe": float(
        sharpe_ratio
    ),

    "max_drawdown": float(
        max_drawdown
    ),

    "gross_leverage": float(
        gross_leverage
    ),

    "positive_period_rate": float(
        positive_rate
    ),

    "approved_trades": int(
        len(df)
    )
}

with open(

    OUTPUT_DIR
    / "confidence_portfolio_summary.json",

    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )

logger.success(
    "Saved confidence-weighted portfolio"
)

# =========================================================
# FINAL LOGGING
# =========================================================

logger.info("====================================")
logger.info("CONFIDENCE POSITION SIZING COMPLETE")
logger.info("====================================")