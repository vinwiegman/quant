"""Versioned SQLite persistence for paper-trading decisions and snapshots."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from quantbot.execution import DailyExecutionResult

SCHEMA_VERSION = 1


class ExecutionStore:
    """Persist and query paper decisions using only Python's standard library."""

    def __init__(self, path: str | Path = "state/paper_trading.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _migrate(self) -> None:
        with self._connect() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"database schema {version} is newer than supported version {SCHEMA_VERSION}"
                )
            if version == 0:
                connection.executescript(
                    """
                    CREATE TABLE decisions (
                        decision_key TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        signal_date TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        model TEXT NOT NULL,
                        probability REAL NOT NULL,
                        current_invested INTEGER NOT NULL,
                        target_weight REAL NOT NULL,
                        latest_price REAL NOT NULL,
                        market_is_open INTEGER,
                        submitted INTEGER NOT NULL,
                        account_equity REAL NOT NULL,
                        cash REAL NOT NULL,
                        UNIQUE(signal_date, symbol, model)
                    );
                    CREATE TABLE proposed_orders (
                        decision_key TEXT NOT NULL,
                        sequence INTEGER NOT NULL,
                        symbol TEXT NOT NULL,
                        quantity REAL NOT NULL,
                        side TEXT NOT NULL,
                        PRIMARY KEY(decision_key, sequence),
                        FOREIGN KEY(decision_key) REFERENCES decisions(decision_key)
                            ON DELETE CASCADE
                    );
                    CREATE TABLE position_snapshots (
                        decision_key TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        quantity REAL NOT NULL,
                        avg_price REAL NOT NULL,
                        PRIMARY KEY(decision_key, symbol),
                        FOREIGN KEY(decision_key) REFERENCES decisions(decision_key)
                            ON DELETE CASCADE
                    );
                    CREATE TABLE equity_snapshots (
                        decision_key TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        signal_date TEXT NOT NULL,
                        equity REAL NOT NULL,
                        cash REAL NOT NULL,
                        FOREIGN KEY(decision_key) REFERENCES decisions(decision_key)
                            ON DELETE CASCADE
                    );
                    CREATE INDEX decisions_timestamp_idx ON decisions(timestamp);
                    CREATE INDEX equity_timestamp_idx ON equity_snapshots(timestamp);
                    PRAGMA user_version = 1;
                    """
                )

    def record(self, result: DailyExecutionResult) -> str:
        """Upsert one daily decision and its exact observed account state."""
        key = f"{result.signal_date}:{result.symbol}:{result.model}"
        market_open = None if result.market_is_open is None else int(result.market_is_open)
        values = (
            key,
            result.timestamp,
            result.signal_date,
            result.symbol,
            result.model,
            result.probability,
            int(result.current_invested),
            result.target_weight,
            result.latest_price,
            market_open,
            int(result.submitted),
            result.account_equity,
            result.cash,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(decision_key) DO UPDATE SET
                    timestamp=excluded.timestamp,
                    probability=excluded.probability,
                    current_invested=excluded.current_invested,
                    target_weight=excluded.target_weight,
                    latest_price=excluded.latest_price,
                    market_is_open=excluded.market_is_open,
                    submitted=MAX(decisions.submitted, excluded.submitted),
                    account_equity=excluded.account_equity,
                    cash=excluded.cash
                """,
                values,
            )
            connection.execute("DELETE FROM proposed_orders WHERE decision_key = ?", (key,))
            connection.executemany(
                "INSERT INTO proposed_orders VALUES (?, ?, ?, ?, ?)",
                [
                    (key, sequence, order["symbol"], order["quantity"], order["side"])
                    for sequence, order in enumerate(result.orders)
                ],
            )
            connection.execute("DELETE FROM position_snapshots WHERE decision_key = ?", (key,))
            connection.executemany(
                "INSERT INTO position_snapshots VALUES (?, ?, ?, ?)",
                [
                    (key, position["symbol"], position["quantity"], position["avg_price"])
                    for position in result.positions
                ],
            )
            connection.execute(
                """
                INSERT INTO equity_snapshots VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(decision_key) DO UPDATE SET
                    timestamp=excluded.timestamp,
                    equity=excluded.equity,
                    cash=excluded.cash
                """,
                (key, result.timestamp, result.signal_date, result.account_equity, result.cash),
            )
        return key

    def decisions(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return chronological decisions in the JSON-log-compatible shape."""
        query = "SELECT * FROM decisions ORDER BY timestamp"
        parameters: tuple[int, ...] = ()
        if limit is not None:
            if limit <= 0:
                return []
            query = "SELECT * FROM decisions ORDER BY timestamp DESC LIMIT ?"
            parameters = (limit,)
        with self._connect() as connection:
            rows = list(connection.execute(query, parameters))
            if limit is not None:
                rows.reverse()
            output = []
            for row in rows:
                orders = connection.execute(
                    "SELECT symbol, quantity, side FROM proposed_orders "
                    "WHERE decision_key = ? ORDER BY sequence",
                    (row["decision_key"],),
                ).fetchall()
                positions = connection.execute(
                    "SELECT symbol, quantity, avg_price FROM position_snapshots "
                    "WHERE decision_key = ? ORDER BY symbol",
                    (row["decision_key"],),
                ).fetchall()
                item = dict(row)
                item["current_invested"] = bool(item["current_invested"])
                item["submitted"] = bool(item["submitted"])
                item["market_is_open"] = (
                    None if item["market_is_open"] is None else bool(item["market_is_open"])
                )
                item["orders"] = [dict(order) for order in orders]
                item["positions"] = [dict(position) for position in positions]
                output.append(item)
            return output

    def equity_history(self) -> pd.DataFrame:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT timestamp, signal_date, equity, cash "
                "FROM equity_snapshots ORDER BY timestamp"
            ).fetchall()
        if not rows:
            return pd.DataFrame(columns=["timestamp", "signal_date", "equity", "cash"])
        return pd.DataFrame([dict(row) for row in rows])

    def positions(self, decision_key: str | None = None) -> pd.DataFrame:
        with self._connect() as connection:
            if decision_key is None:
                row = connection.execute(
                    "SELECT decision_key FROM decisions ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                if row is None:
                    return pd.DataFrame(columns=["symbol", "quantity", "avg_price"])
                decision_key = str(row["decision_key"])
            rows = connection.execute(
                "SELECT symbol, quantity, avg_price FROM position_snapshots "
                "WHERE decision_key = ? ORDER BY symbol",
                (decision_key,),
            ).fetchall()
        return pd.DataFrame([dict(row) for row in rows])

    def health(
        self,
        *,
        as_of: datetime | None = None,
        stale_after_hours: float = 48.0,
    ) -> dict[str, Any]:
        """Summarize whether durable monitoring has recent, coherent state."""
        if stale_after_hours <= 0:
            raise ValueError("stale_after_hours must be positive")
        as_of = as_of or datetime.now(UTC)
        if as_of.tzinfo is None:
            as_of = as_of.replace(tzinfo=UTC)
        decisions = self.decisions()
        history = self.equity_history()
        if not decisions:
            return {
                "status": "empty",
                "decision_count": 0,
                "submitted_count": 0,
                "last_decision": None,
                "age_hours": None,
                "duplicate_signal_dates": 0,
            }
        latest = datetime.fromisoformat(decisions[-1]["timestamp"])
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=UTC)
        age_hours = max(
            0.0, (as_of.astimezone(UTC) - latest.astimezone(UTC)).total_seconds() / 3600
        )
        keys = [(item["signal_date"], item["symbol"], item["model"]) for item in decisions]
        duplicates = len(keys) - len(set(keys))
        status = "healthy"
        if age_hours > stale_after_hours:
            status = "stale"
        if duplicates or len(history) != len(decisions):
            status = "inconsistent"
        return {
            "status": status,
            "decision_count": len(decisions),
            "submitted_count": sum(bool(item["submitted"]) for item in decisions),
            "last_decision": decisions[-1]["timestamp"],
            "age_hours": age_hours,
            "duplicate_signal_dates": duplicates,
        }

    def export_json(self, path: str | Path) -> None:
        """Write a portable backup without exposing credentials."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "decisions": self.decisions(),
            "equity": self.equity_history().to_dict(orient="records"),
        }
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
