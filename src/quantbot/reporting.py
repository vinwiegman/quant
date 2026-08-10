"""Offline monitoring report generated from durable paper-trading state."""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from quantbot.persistence import ExecutionStore


def write_paper_report(
    store: ExecutionStore,
    out_dir: str | Path = "results/paper",
    *,
    as_of: datetime | None = None,
    stale_after_hours: float = 48.0,
) -> dict[str, Any]:
    """Export database tables and a self-contained operational HTML report."""
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    decisions = store.decisions()
    equity = store.equity_history()
    health = store.health(as_of=as_of, stale_after_hours=stale_after_hours)
    positions = store.positions()

    if equity.empty:
        first_equity = latest_equity = total_return = max_drawdown = None
    else:
        values = equity["equity"].astype(float)
        first_equity = float(values.iloc[0])
        latest_equity = float(values.iloc[-1])
        total_return = latest_equity / first_equity - 1.0 if first_equity else None
        drawdown = values / values.cummax() - 1.0
        max_drawdown = float(drawdown.min())

    summary = {
        **health,
        "first_equity": first_equity,
        "latest_equity": latest_equity,
        "tracked_return": total_return,
        "maximum_drawdown": max_drawdown,
        "position_count": len(positions),
    }
    equity.to_csv(destination / "equity_history.csv", index=False)
    positions.to_csv(destination / "latest_positions.csv", index=False)
    store.export_json(destination / "paper_history.json")
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    (destination / "paper_report.html").write_text(
        _render_html(summary, decisions, equity), encoding="utf-8"
    )
    return summary


def _render_html(summary: dict[str, Any], decisions: list[dict], equity) -> str:
    status = str(summary["status"])
    status_class = "ok" if status == "healthy" else "warn"
    latest = _money(summary["latest_equity"])
    tracked = _percent(summary["tracked_return"])
    drawdown = _percent(summary["maximum_drawdown"])
    rows = []
    for decision in reversed(decisions[-20:]):
        action = "hold"
        if decision["orders"]:
            action = ", ".join(
                f"{order['side']} {abs(float(order['quantity'])):.3f} {order['symbol']}"
                for order in decision["orders"]
            )
        rows.append(
            "<tr>"
            f"<td>{html.escape(decision['signal_date'])}</td>"
            f"<td>{float(decision['probability']):.3f}</td>"
            f"<td>{float(decision['target_weight']):.2f}</td>"
            f"<td>{html.escape(action)}</td>"
            f"<td>{'yes' if decision['submitted'] else 'dry'}</td>"
            "</tr>"
        )
    table_rows = "".join(rows) or '<tr><td colspan="5">No decisions recorded.</td></tr>'
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>quantbot paper history</title><style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:1000px;margin:30px auto;padding:0 20px;color:#172033}}
h1{{margin-bottom:4px}} .muted{{color:#64748b}} .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}}
.card{{border:1px solid #dbe2ea;border-radius:12px;padding:16px}} .value{{font-size:24px;font-weight:700}}
.ok{{color:#15803d}} .warn{{color:#b45309}} table{{width:100%;border-collapse:collapse}}
th,td{{padding:9px;border-bottom:1px solid #e5e7eb;text-align:left}} th{{font-size:12px;text-transform:uppercase;color:#64748b}}
svg{{width:100%;height:240px}} @media(max-width:700px){{.grid{{grid-template-columns:1fr 1fr}}}}
</style></head><body><h1>quantbot durable paper history</h1>
<div class="muted">Generated entirely from the local SQLite audit database.</div>
<div class="grid">
<div class="card"><div class="muted">Health</div><div class="value {status_class}">{html.escape(status)}</div></div>
<div class="card"><div class="muted">Latest equity</div><div class="value">{latest}</div></div>
<div class="card"><div class="muted">Tracked return</div><div class="value">{tracked}</div></div>
<div class="card"><div class="muted">Max drawdown</div><div class="value">{drawdown}</div></div>
</div><div class="card"><h2>Equity snapshots</h2>{_equity_svg(equity)}</div>
<div class="card"><h2>Recent decisions</h2><table><thead><tr><th>Signal date</th><th>Probability</th><th>Target</th><th>Action</th><th>Mode</th></tr></thead>
<tbody>{table_rows}</tbody></table></div>
<p class="muted">Decisions: {summary["decision_count"]} &middot; Submitted: {summary["submitted_count"]} &middot; Last: {html.escape(str(summary["last_decision"]))}</p>
</body></html>"""


def _equity_svg(equity) -> str:
    if len(equity) < 2:
        return '<p class="muted">At least two daily snapshots are required for a curve.</p>'
    values = equity["equity"].astype(float).to_numpy()
    low, high = float(values.min()), float(values.max())
    span = high - low or 1.0
    points = []
    for index, value in enumerate(values):
        x = 12.0 + 696.0 * index / (len(values) - 1)
        y = 12.0 + 196.0 * (1.0 - (value - low) / span)
        points.append(f"{x:.1f},{y:.1f}")
    color = "#15803d" if values[-1] >= values[0] else "#dc2626"
    label = f"Tracked equity from {_money(values[0])} to {_money(values[-1])}"
    return (
        f'<svg viewBox="0 0 720 220" role="img" aria-label="{html.escape(label)}">'
        f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
        'stroke-width="3" stroke-linejoin="round"/></svg>'
    )


def _money(value: float | None) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"${value:,.2f}"


def _percent(value: float | None) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.2%}"
