from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from quantbot.cli import main
from quantbot.execution import DailyExecutionResult
from quantbot.persistence import ExecutionStore
from quantbot.reporting import write_paper_report


def _result(**changes) -> DailyExecutionResult:
    base = DailyExecutionResult(
        timestamp="2026-08-10T14:35:00+00:00",
        signal_date="2026-08-07",
        symbol="SPY",
        model="logistic",
        probability=0.61,
        current_invested=False,
        target_weight=0.95,
        latest_price=635.0,
        market_is_open=True,
        orders=({"symbol": "SPY", "quantity": 149.6, "side": "buy"},),
        submitted=True,
        account_equity=100_000.0,
        cash=100_000.0,
        positions=(),
    )
    return replace(base, **changes)


def test_store_round_trips_normalized_decision_state(tmp_path):
    store = ExecutionStore(tmp_path / "paper.sqlite3")
    store.record(_result())

    decisions = store.decisions()
    equity = store.equity_history()

    assert len(decisions) == 1
    assert decisions[0]["orders"][0]["quantity"] == pytest.approx(149.6)
    assert decisions[0]["submitted"] is True
    assert equity.iloc[0]["equity"] == pytest.approx(100_000.0)


def test_same_daily_decision_is_idempotent_and_preserves_submission(tmp_path):
    store = ExecutionStore(tmp_path / "paper.sqlite3")
    store.record(_result())
    store.record(_result(timestamp="2026-08-10T15:00:00+00:00", submitted=False))

    decisions = store.decisions()

    assert len(decisions) == 1
    assert decisions[0]["timestamp"] == "2026-08-10T15:00:00+00:00"
    assert decisions[0]["submitted"] is True
    assert len(store.equity_history()) == 1


def test_health_detects_fresh_stale_and_empty_state(tmp_path):
    store = ExecutionStore(tmp_path / "paper.sqlite3")
    assert store.health()["status"] == "empty"
    store.record(_result())

    fresh = store.health(as_of=datetime(2026, 8, 10, 15, tzinfo=UTC))
    stale = store.health(as_of=datetime(2026, 8, 13, 15, tzinfo=UTC))

    assert fresh["status"] == "healthy"
    assert stale["status"] == "stale"


def test_report_exports_portable_history_and_monitoring_page(tmp_path):
    store = ExecutionStore(tmp_path / "paper.sqlite3")
    store.record(_result())
    store.record(
        _result(
            signal_date="2026-08-08",
            timestamp="2026-08-11T14:35:00+00:00",
            account_equity=101_000.0,
            cash=5_000.0,
            current_invested=True,
            orders=(),
            positions=({"symbol": "SPY", "quantity": 150.0, "avg_price": 635.0},),
        )
    )
    output = tmp_path / "report"

    summary = write_paper_report(
        store,
        output,
        as_of=datetime(2026, 8, 11, 16, tzinfo=UTC),
    )

    assert summary["tracked_return"] == pytest.approx(0.01)
    assert (output / "paper_report.html").exists()
    assert (output / "equity_history.csv").exists()
    backup = json.loads((output / "paper_history.json").read_text())
    assert len(backup["decisions"]) == 2
    assert backup["decisions"][-1]["positions"][0]["symbol"] == "SPY"
    assert "healthy" in (output / "paper_report.html").read_text()


def test_newer_database_schema_is_rejected(tmp_path):
    path = tmp_path / "future.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version = 999")

    with pytest.raises(RuntimeError, match="newer than supported"):
        ExecutionStore(path)


def test_report_cli_can_fail_on_unhealthy_state(tmp_path):
    database = tmp_path / "empty.sqlite3"
    output = tmp_path / "report"

    exit_code = main(
        [
            "paper-report",
            "--database",
            str(database),
            "--out",
            str(output),
            "--fail-on-unhealthy",
        ]
    )

    assert exit_code == 1
