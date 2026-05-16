from pathlib import Path

import json

import numpy as np
import pandas as pd

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

OUTPUT_DIR = (
    PROJECT_ROOT / "backtests/engine"
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
    LOG_DIR / "engine.log",
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# =========================================================
# CONFIGURATION
# =========================================================

INITIAL_TRAIN_YEARS = 5

TOP_K = 20

MAX_POSITION_WEIGHT = 0.10

TARGET_COLUMN = (
    "cross_sectional_rank_label"
)

# =========================================================
# LOAD DATA
# =========================================================

logger.info(
    "Loading ranked dataset"
)

df = pd.read_parquet(
    DATA_DIR
    / "train_ranked.parquet"
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

df["year"] = (
    df["timestamp"]
    .dt.year
)

# =========================================================
# LABEL MAPPING
# =========================================================

label_mapping = {

    -1: 0,
     0: 1,
     1: 2
}

df[TARGET_COLUMN] = (

    df[TARGET_COLUMN]

    .map(label_mapping)
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

    "future_return_21d",
    "future_return_rank",

    "cross_sectional_rank_label",

    "triple_barrier_label",

    "year"
]

FEATURE_COLUMNS = [

    col for col in df.columns

    if col not in EXCLUDED_COLUMNS
]

logger.info(
    f"Features -> "
    f"{FEATURE_COLUMNS}"
)

# =========================================================
# WALK FORWARD YEARS
# =========================================================

years = sorted(
    df["year"].unique()
)

logger.info(
    f"Available years -> "
    f"{years}"
)

# =========================================================
# STORAGE
# =========================================================

portfolio_returns = []

equity_curve = []

current_equity = 1.0

all_predictions = []

# =========================================================
# WALK FORWARD ENGINE
# =========================================================

for i in range(

    INITIAL_TRAIN_YEARS,

    len(years)
):

    logger.info("====================================")

    validation_year = years[i]

    train_years = years[:i]

    logger.info(
        f"Train years -> "
        f"{train_years}"
    )

    logger.info(
        f"Validation year -> "
        f"{validation_year}"
    )

    # =====================================================
    # SPLITS
    # =====================================================

    train_df = df[
        df["year"].isin(train_years)
    ].copy()

    validation_df = df[
        df["year"] == validation_year
    ].copy()

    # =====================================================
    # MATRICES
    # =====================================================

    X_train = train_df[
        FEATURE_COLUMNS
    ]

    y_train = train_df[
        TARGET_COLUMN
    ]

    X_validation = validation_df[
        FEATURE_COLUMNS
    ]

    y_validation = validation_df[
        TARGET_COLUMN
    ]

    # =====================================================
    # MODEL
    # =====================================================

    model = XGBClassifier(

        objective="multi:softprob",

        num_class=3,

        n_estimators=300,

        max_depth=6,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        tree_method="hist",

        random_state=42,

        eval_metric="mlogloss",

        verbosity=0
    )

    # =====================================================
    # TRAIN
    # =====================================================

    logger.info(
        "Training model"
    )

    model.fit(
        X_train,
        y_train
    )

    # =====================================================
    # PREDICTIONS
    # =====================================================

    logger.info(
        "Generating predictions"
    )

    prediction_probabilities = (

        model.predict_proba(
            X_validation
        )
    )

    predictions = (
        np.argmax(
            prediction_probabilities,
            axis=1
        )
    )

    validation_df[
        "prediction"
    ] = predictions

    validation_df[
        "prediction_probability"
    ] = (

        prediction_probabilities.max(
            axis=1
        )
    )

    # =====================================================
    # LONG SIGNALS ONLY
    # =====================================================

    validation_df = validation_df[

        validation_df["prediction"] == 2

    ].copy()

    # =====================================================
    # PORTFOLIO CONSTRUCTION
    # =====================================================

    yearly_returns = []

    timestamps = sorted(
        validation_df["timestamp"].unique()
    )

    for timestamp in timestamps:

        tmp = validation_df[

            validation_df["timestamp"]
            == timestamp

        ].copy()

        if len(tmp) == 0:

            continue

        # rank by confidence

        tmp.sort_values(

            by="prediction_probability",

            ascending=False,

            inplace=True
        )

        # top-k selection

        tmp = tmp.head(TOP_K)

        # equal weight portfolio

        tmp[
            "weight"
        ] = (

            1.0 / len(tmp)
        )

        # cap weights

        tmp[
            "weight"
        ] = (

            tmp["weight"]

            .clip(
                upper=MAX_POSITION_WEIGHT
            )
        )

        # renormalize

        tmp[
            "weight"
        ] = (

            tmp["weight"]

            /

            tmp["weight"].sum()
        )

        # portfolio return

        portfolio_return = (

            tmp["weight"]

            * tmp["future_return_21d"]
        ).sum()

        yearly_returns.append(
            portfolio_return
        )

        # save trades

        tmp[
            "portfolio_return"
        ] = portfolio_return

        all_predictions.append(
            tmp
        )

    # =====================================================
    # EQUITY CURVE
    # =====================================================

    yearly_returns = pd.Series(
        yearly_returns
    )

    mean_return = (
        yearly_returns.mean()
    )

    yearly_volatility = (
        yearly_returns.std()
    )

    if yearly_volatility > 0:

        sharpe = (

            mean_return

            / yearly_volatility
        )

    else:

        sharpe = 0

    current_equity *= (
        1 + mean_return
    )

    portfolio_returns.extend(
        yearly_returns.tolist()
    )

    equity_curve.append({

        "year": validation_year,

        "mean_return": mean_return,

        "volatility": yearly_volatility,

        "sharpe": sharpe,

        "equity": current_equity
    })

    accuracy = accuracy_score(
        y_validation,
        model.predict(X_validation)
    )

    logger.info(
        f"Accuracy -> "
        f"{accuracy:.4f}"
    )

    logger.info(
        f"Portfolio Return -> "
        f"{mean_return:.4f}"
    )

    logger.info(
        f"Sharpe -> "
        f"{sharpe:.4f}"
    )

# =========================================================
# FINAL RESULTS
# =========================================================

portfolio_returns = pd.Series(
    portfolio_returns
)

equity_df = pd.DataFrame(
    equity_curve
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

# =========================================================
# DRAWDOWN
# =========================================================

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
# SUMMARY
# =========================================================

logger.info("====================================")
logger.info("BACKTEST ENGINE RESULTS")
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

equity_df.to_csv(

    OUTPUT_DIR
    / "equity_curve.csv",

    index=False
)

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)

predictions_df.to_csv(

    OUTPUT_DIR
    / "trade_log.csv",

    index=False
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
    / "performance_summary.json",

    "w"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )

logger.success(
    "Saved backtest engine outputs"
)

logger.info("====================================")
logger.info("BACKTEST ENGINE COMPLETE")
logger.info("====================================")