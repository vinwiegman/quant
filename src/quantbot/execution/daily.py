"""Train on known labels, predict the latest bar, and reconcile a paper account."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from quantbot.broker import Broker, Order, Position, target_weights_to_orders
from quantbot.features import build_dataset, make_features
from quantbot.models import ModelName, get_model_factory
from quantbot.persistence import ExecutionStore


@dataclass(frozen=True)
class DailyExecutionResult:
    timestamp: str
    signal_date: str
    symbol: str
    model: str
    probability: float
    current_invested: bool
    target_weight: float
    latest_price: float
    market_is_open: bool | None
    orders: tuple[dict[str, float | str], ...]
    submitted: bool
    account_equity: float
    cash: float
    positions: tuple[dict[str, float | str], ...]


def run_daily_execution(
    market: pd.DataFrame,
    broker: Broker,
    *,
    symbol: str = "SPY",
    model: ModelName = "logistic",
    entry_threshold: float = 0.55,
    exit_threshold: float = 0.45,
    invested_weight: float = 0.95,
    min_trade_value: float = 25.0,
    submit: bool = False,
    log_path: str | Path | None = "logs/executions.jsonl",
    database_path: str | Path | None = None,
) -> DailyExecutionResult:
    """Build and optionally submit one idempotent daily paper-trading decision."""
    if not 0.0 < invested_weight <= 1.0:
        raise ValueError("invested_weight must be in (0, 1]")
    if not 0.0 <= exit_threshold < entry_threshold <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= exit < entry <= 1")
    required = {"close", "volume"}
    if not required <= set(market.columns):
        raise ValueError("market data must include close and volume")

    close = market["close"].dropna()
    volume = market["volume"].reindex(close.index)
    training = build_dataset(close, volume=volume)
    X_train = training.drop(columns="target")
    y_train = training["target"].astype(int)
    latest = make_features(close, volume=volume).iloc[[-1]]
    latest = latest.loc[:, X_train.columns]
    if latest.isna().any().any():
        raise ValueError("latest bar does not have a complete feature row")

    estimator = get_model_factory(model)()
    estimator.fit(X_train, y_train)
    probability = float(estimator.predict_proba(latest)[0, 1])

    positions = broker.positions()
    unexpected = sorted(set(positions) - {symbol})
    if unexpected:
        raise ValueError(f"account contains positions outside strategy scope: {unexpected}")
    current_invested = positions.get(symbol, None) is not None
    if probability >= entry_threshold:
        target_weight = invested_weight
    elif probability <= exit_threshold:
        target_weight = 0.0
    else:
        target_weight = invested_weight if current_invested else 0.0

    latest_price = float(close.iloc[-1])
    cash = float(broker.cash())
    equity_method = getattr(broker, "account_equity", None)
    equity = (
        float(equity_method())
        if callable(equity_method)
        else cash
        + sum(
            position.market_value(latest_price)
            for held_symbol, position in positions.items()
            if held_symbol == symbol
        )
    )
    orders = target_weights_to_orders(
        {symbol: target_weight},
        {symbol: latest_price},
        equity,
        positions,
        min_trade_value=min_trade_value,
    )
    if submit:
        for order in orders:
            broker.submit(order, latest_price)

    clock_method = getattr(broker, "market_is_open", None)
    market_is_open = bool(clock_method()) if callable(clock_method) else None
    result = DailyExecutionResult(
        timestamp=datetime.now(UTC).isoformat(),
        signal_date=pd.Timestamp(close.index[-1]).date().isoformat(),
        symbol=symbol,
        model=model,
        probability=probability,
        current_invested=current_invested,
        target_weight=target_weight,
        latest_price=latest_price,
        market_is_open=market_is_open,
        orders=tuple(_order_record(order) for order in orders),
        submitted=submit and bool(orders),
        account_equity=equity,
        cash=cash,
        positions=tuple(_position_record(position) for position in positions.values()),
    )
    if log_path is not None:
        _append_log(result, Path(log_path))
    if database_path is not None:
        ExecutionStore(database_path).record(result)
    return result


def _order_record(order: Order) -> dict[str, float | str]:
    return {
        "symbol": order.symbol,
        "quantity": order.quantity,
        "side": "buy" if order.quantity > 0 else "sell",
    }


def _position_record(position: Position) -> dict[str, float | str]:
    return {
        "symbol": position.symbol,
        "quantity": position.quantity,
        "avg_price": position.avg_price,
    }


def _append_log(result: DailyExecutionResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")
