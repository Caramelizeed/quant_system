from pathlib import Path

import json

import numpy as np
import pandas as pd

from loguru import logger

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

REGIME_DIR = (
    PROJECT_ROOT / "data/feature_store/regimes"
)

BACKTEST_DIR = (
    PROJECT_ROOT / "backtests/engine"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "reports/regime_allocator"
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
    LOG_DIR / "regime_allocator.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# =========================================================
# LOAD DATA
# =========================================================

logger.info(
    "Loading trade log"
)

trade_df = pd.read_csv(
    BACKTEST_DIR
    / "trade_log.csv"
)

logger.info(
    f"Trade log shape -> "
    f"{trade_df.shape}"
)

logger.info(
    "Loading regime data"
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
# TIMESTAMPS
# =========================================================

trade_df["timestamp"] = pd.to_datetime(
    trade_df["timestamp"]
).astype("datetime64[ms]")

regime_df["timestamp"] = pd.to_datetime(
    regime_df["timestamp"]
).astype("datetime64[ms]")

# =========================================================
# REGIME MERGE
# =========================================================

logger.info(
    "Merging regime information"
)

trade_df = pd.merge_asof(

    trade_df.sort_values("timestamp"),

    regime_df[
        ["timestamp", "regime"]
    ].sort_values("timestamp"),

    on="timestamp",

    direction="backward"
)

logger.info(
    f"Merged shape -> "
    f"{trade_df.shape}"
)

# =========================================================
# REGIME ANALYSIS
# =========================================================

logger.info(
    "Computing regime returns"
)

regime_stats = (

    trade_df

    .groupby("regime")[
        "future_return_21d"
    ]

    .agg([
        "mean",
        "std",
        "count"
    ])

    .reset_index()
)

regime_stats["sharpe_proxy"] = (

    regime_stats["mean"]

    /

    regime_stats["std"]
)

logger.info(
    f"\n{regime_stats}"
)

# =========================================================
# REGIME LABELING
# =========================================================

logger.info(
    "Building regime mappings"
)

regime_stats = regime_stats.sort_values(
    by="mean"
)

worst_regime = (
    regime_stats.iloc[0]["regime"]
)

best_regime = (
    regime_stats.iloc[-1]["regime"]
)

neutral_regimes = [

    r for r in regime_stats["regime"]

    if r not in [
        worst_regime,
        best_regime
    ]
]

logger.info(
    f"Best regime -> "
    f"{best_regime}"
)

logger.info(
    f"Worst regime -> "
    f"{worst_regime}"
)

logger.info(
    f"Neutral regimes -> "
    f"{neutral_regimes}"
)

# =========================================================
# ALLOCATION RULES
# =========================================================

REGIME_ALLOCATIONS = {

    int(best_regime): {

        "gross_leverage": 1.25,

        "max_position_weight": 0.10,

        "max_positions": 25,

        "trade_enabled": True
    },

    int(worst_regime): {

        "gross_leverage": 0.25,

        "max_position_weight": 0.03,

        "max_positions": 5,

        "trade_enabled": False
    }
}

# neutral

for regime in neutral_regimes:

    REGIME_ALLOCATIONS[int(regime)] = {

        "gross_leverage": 0.75,

        "max_position_weight": 0.05,

        "max_positions": 15,

        "trade_enabled": True
    }

logger.info(
    f"\nAllocation Map:\n"
    f"{REGIME_ALLOCATIONS}"
)

# =========================================================
# APPLY ALLOCATIONS
# =========================================================

logger.info(
    "Applying regime allocations"
)

trade_df["gross_leverage"] = (

    trade_df["regime"]

    .map(
        lambda x:
        REGIME_ALLOCATIONS[int(x)]
        ["gross_leverage"]
    )
)

trade_df["max_position_weight"] = (

    trade_df["regime"]

    .map(
        lambda x:
        REGIME_ALLOCATIONS[int(x)]
        ["max_position_weight"]
    )
)

trade_df["max_positions"] = (

    trade_df["regime"]

    .map(
        lambda x:
        REGIME_ALLOCATIONS[int(x)]
        ["max_positions"]
    )
)

trade_df["trade_enabled"] = (

    trade_df["regime"]

    .map(
        lambda x:
        REGIME_ALLOCATIONS[int(x)]
        ["trade_enabled"]
    )
)

# =========================================================
# FILTER DISABLED TRADES
# =========================================================

logger.info(
    "Filtering disabled regimes"
)

trade_df = trade_df[

    trade_df["trade_enabled"]

].copy()

logger.info(
    f"Remaining rows -> "
    f"{trade_df.shape}"
)

# =========================================================
# DYNAMIC POSITION WEIGHTS
# =========================================================

logger.info(
    "Building regime-aware portfolios"
)

portfolio_returns = []

equity_curve = []

current_equity = 1.0

timestamps = sorted(
    trade_df["timestamp"].unique()
)

for timestamp in timestamps:

    tmp = trade_df[

        trade_df["timestamp"]
        == timestamp

    ].copy()

    if len(tmp) == 0:

        continue

    leverage = (
        tmp["gross_leverage"]
        .iloc[0]
    )

    max_positions = int(
        tmp["max_positions"]
        .iloc[0]
    )

    max_weight = (
        tmp["max_position_weight"]
        .iloc[0]
    )

    # select highest confidence

    tmp.sort_values(

        by="prediction_probability",

        ascending=False,

        inplace=True
    )

    tmp = tmp.head(
        max_positions
    )

    # equal weights

    tmp["weight"] = (

        leverage

        / len(tmp)
    )

    # cap weights

    tmp["weight"] = (

        tmp["weight"]

        .clip(
            upper=max_weight
        )
    )

    # renormalize

    tmp["weight"] = (

        tmp["weight"]

        /

        tmp["weight"].sum()

        * leverage
    )

    portfolio_return = (

        tmp["weight"]

        * tmp["future_return_21d"]
    ).sum()

    portfolio_returns.append(
        portfolio_return
    )

    current_equity *= (
        1 + portfolio_return
    )

    equity_curve.append({

        "timestamp": timestamp,

        "portfolio_return": portfolio_return,

        "equity": current_equity
    })

# =========================================================
# FINAL METRICS
# =========================================================

logger.info(
    "Computing performance metrics"
)

portfolio_returns = pd.Series(
    portfolio_returns
)

mean_return = (
    portfolio_returns.mean()
)

volatility = (
    portfolio_returns.std()
)

if volatility > 0:

    sharpe = (
        mean_return
        / volatility
    )

else:

    sharpe = 0

equity_series = (

    1 + portfolio_returns
).cumprod()

rolling_max = (
    equity_series.cummax()
)

drawdowns = (

    equity_series
    / rolling_max

    - 1
)

max_drawdown = (
    drawdowns.min()
)

# =========================================================
# RESULTS
# =========================================================

logger.info("====================================")
logger.info("REGIME ALLOCATOR RESULTS")
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

logger.info("====================================")

# =========================================================
# SAVE OUTPUTS
# =========================================================

equity_df = pd.DataFrame(
    equity_curve
)

equity_df.to_csv(

    OUTPUT_DIR
    / "regime_equity_curve.csv",

    index=False
)

allocation_df = pd.DataFrame(

    REGIME_ALLOCATIONS

).T

allocation_df.to_csv(

    OUTPUT_DIR
    / "regime_allocations.csv"
)

summary = {

    "mean_return": float(
        mean_return
    ),

    "volatility": float(
        volatility
    ),

    "sharpe": float(
        sharpe
    ),

    "max_drawdown": float(
        max_drawdown
    )
}

with open(

    OUTPUT_DIR
    / "regime_allocator_summary.json",

    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )

logger.success(
    "Saved regime allocator outputs"
)

logger.info("====================================")
logger.info("REGIME ALLOCATOR COMPLETE")
logger.info("====================================")