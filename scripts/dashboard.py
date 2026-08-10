"""Generate a self-contained HTML dashboard for the quantbot paper account.

Reads live state from the Alpaca paper account (credentials from .env) and the
local execution log, then writes results/dashboard.html. Re-run any time to
refresh. Open the file in any browser -- no server needed.

    python scripts/dashboard.py
"""

from __future__ import annotations

import html
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from quantbot.persistence import ExecutionStore

PROJECT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT / "logs" / "executions.jsonl"
DATABASE_PATH = PROJECT / "state" / "paper_trading.sqlite3"
OUT_PATH = PROJECT / "results" / "dashboard.html"


def _client():
    from dotenv import load_dotenv

    load_dotenv(PROJECT / ".env")
    key, secret = os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("ALPACA_API_KEY / ALPACA_SECRET_KEY missing (check .env)")
    from alpaca.trading.client import TradingClient

    return TradingClient(key, secret, paper=True)


def _fetch(client) -> dict:
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    account = client.get_account()
    positions = client.get_all_positions()
    orders = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.ALL, limit=25))
    try:
        from alpaca.trading.requests import GetPortfolioHistoryRequest

        hist = client.get_portfolio_history(
            history_filter=GetPortfolioHistoryRequest(period="1M", timeframe="1D")
        )
        equity_series = [(t, e) for t, e in zip(hist.timestamp, hist.equity, strict=False) if e]
    except Exception:  # noqa: BLE001 - chart is optional, never block the dashboard
        equity_series = []
    try:
        market_open = bool(client.get_clock().is_open)
    except Exception:  # noqa: BLE001
        market_open = None
    return {
        "account": account,
        "positions": positions,
        "orders": orders,
        "equity_series": equity_series,
        "market_open": market_open,
    }


def _decisions() -> list[dict]:
    if DATABASE_PATH.exists():
        return ExecutionStore(DATABASE_PATH).decisions()
    if not LOG_PATH.exists():
        return []
    rows = []
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _money(value) -> str:
    return f"${float(value):,.2f}"


def _sparkline(series: list[tuple[int, float]], w: int = 720, h: int = 220) -> str:
    if len(series) < 2:
        return '<p class="muted">Not enough history yet for an equity curve.</p>'
    values = [e for _, e in series]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    pad = 12
    n = len(values)
    pts = []
    for i, v in enumerate(values):
        x = pad + (w - 2 * pad) * i / (n - 1)
        y = pad + (h - 2 * pad) * (1 - (v - lo) / span)
        pts.append(f"{x:.1f},{y:.1f}")
    line = " ".join(pts)
    area = f"{pad},{h - pad} {line} {w - pad},{h - pad}"
    up = values[-1] >= values[0]
    color = "#16a34a" if up else "#dc2626"
    return f"""<svg viewBox="0 0 {w} {h}" width="100%" preserveAspectRatio="none" role="img" aria-label="Equity curve">
  <polygon points="{area}" fill="{color}" opacity="0.10" />
  <polyline points="{line}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" />
</svg>
<div class="axis"><span>{_money(lo)}</span><span>{_money(hi)}</span></div>"""


def _positions_rows(positions) -> str:
    if not positions:
        return '<tr><td colspan="5" class="muted">No open positions (in cash).</td></tr>'
    out = []
    for p in positions:
        pl = float(p.unrealized_pl)
        cls = "pos" if pl >= 0 else "neg"
        out.append(
            f"<tr><td>{html.escape(p.symbol)}</td><td>{float(p.qty):,.4f}</td>"
            f"<td>{_money(p.avg_entry_price)}</td><td>{_money(p.market_value)}</td>"
            f'<td class="{cls}">{_money(pl)}</td></tr>'
        )
    return "".join(out)


def _orders_rows(orders) -> str:
    filled = [o for o in orders if str(o.status.value) == "filled"][:12]
    if not filled:
        return '<tr><td colspan="5" class="muted">No filled trades yet.</td></tr>'
    out = []
    for o in filled:
        side = o.side.value
        cls = "pos" if side == "buy" else "neg"
        when = o.filled_at.strftime("%Y-%m-%d %H:%M") if o.filled_at else "-"
        out.append(
            f"<tr><td>{when}</td><td class='{cls}'>{side.upper()}</td>"
            f"<td>{html.escape(o.symbol)}</td><td>{float(o.qty):,.4f}</td>"
            f"<td>{_money(o.filled_avg_price)}</td></tr>"
        )
    return "".join(out)


def _decision_rows(decisions: list[dict]) -> str:
    if not decisions:
        return '<tr><td colspan="5" class="muted">No decisions logged yet.</td></tr>'
    out = []
    for d in reversed(decisions[-15:]):
        when = d["timestamp"][:16].replace("T", " ")
        prob = d["probability"]
        bar = int(round(prob * 100))
        if d["orders"]:
            action = ", ".join(f"{o['side'].upper()} {abs(o['quantity']):.3f}" for o in d["orders"])
        else:
            action = "hold"
        submitted = "yes" if d["submitted"] else "dry"
        scls = "pos" if d["submitted"] else "muted"
        out.append(
            f"<tr><td>{when}</td>"
            f'<td><div class="probwrap"><div class="probbar" style="width:{bar}%"></div>'
            f"<span>{prob:.3f}</span></div></td>"
            f"<td>{d['target_weight']:.2f}</td><td>{html.escape(action)}</td>"
            f'<td class="{scls}">{submitted}</td></tr>'
        )
    return "".join(out)


def build_html(data: dict, decisions: list[dict]) -> str:
    a = data["account"]
    equity = float(a.equity)
    last_equity = float(a.last_equity) if getattr(a, "last_equity", None) else equity
    day_pl = equity - last_equity
    baseline_equity = float(data.get("baseline_equity", equity))
    total_pl = equity - baseline_equity
    day_cls = "pos" if day_pl >= 0 else "neg"
    tot_cls = "pos" if total_pl >= 0 else "neg"
    mo = data.get("market_open")
    market = "OPEN" if mo else ("CLOSED" if mo is False else "-")
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>quantbot dashboard</title>
<style>
:root {{
  --bg:#f6f7f9; --card:#ffffff; --ink:#0f172a; --muted:#64748b; --line:#e2e8f0;
  --pos:#16a34a; --neg:#dc2626; --accent:#2563eb;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#0b1120; --card:#151d2e; --ink:#e2e8f0; --muted:#94a3b8; --line:#243049; }}
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; padding:28px; }}
.wrap {{ max-width:960px; margin:0 auto; }}
h1 {{ font-size:20px; margin:0 0 2px; }}
.sub {{ color:var(--muted); font-size:13px; margin-bottom:22px; }}
.hero {{ background:var(--card); border:1px solid var(--line); border-radius:14px;
  padding:22px 24px; margin-bottom:16px; }}
.equity {{ font-size:40px; font-weight:700; letter-spacing:-1px; }}
.plrow {{ display:flex; gap:20px; margin-top:6px; font-size:14px; font-weight:600; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-bottom:16px; }}
.stat {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px; }}
.stat .k {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
.stat .v {{ font-size:20px; font-weight:650; margin-top:3px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:18px 20px; margin-bottom:16px; }}
.card h2 {{ font-size:14px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); margin:0 0 12px; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th,td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); }}
th {{ color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.03em; }}
tr:last-child td {{ border-bottom:none; }}
.pos {{ color:var(--pos); }} .neg {{ color:var(--neg); }} .muted {{ color:var(--muted); }}
.axis {{ display:flex; justify-content:space-between; color:var(--muted); font-size:12px; margin-top:2px; }}
.probwrap {{ position:relative; background:var(--line); border-radius:5px; height:20px; width:120px; overflow:hidden; }}
.probbar {{ position:absolute; inset:0 auto 0 0; background:var(--accent); opacity:.35; }}
.probwrap span {{ position:relative; padding-left:8px; font-variant-numeric:tabular-nums; line-height:20px; }}
.foot {{ color:var(--muted); font-size:12px; text-align:center; margin-top:24px; }}
.pill {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px; font-weight:600;
  background:color-mix(in srgb, var(--accent) 15%, transparent); color:var(--accent); }}
</style></head>
<body><div class="wrap">
  <h1>quantbot &mdash; SPY paper account</h1>
  <div class="sub">Live Alpaca paper trading. Generated {generated}.</div>

  <div class="hero">
    <div class="equity">{_money(equity)}</div>
    <div class="plrow">
      <span class="{day_cls}">Today {day_pl:+,.2f}</span>
      <span class="{tot_cls}">All-time {total_pl:+,.2f}</span>
      <span class="muted">since durable tracking began</span>
    </div>
  </div>

  <div class="grid">
    <div class="stat"><div class="k">Cash</div><div class="v">{_money(a.cash)}</div></div>
    <div class="stat"><div class="k">Buying power</div><div class="v">{_money(a.buying_power)}</div></div>
    <div class="stat"><div class="k">Account</div><div class="v">{html.escape(str(a.status.value))}</div></div>
    <div class="stat"><div class="k">Trading</div><div class="v">{market}</div></div>
  </div>

  <div class="card">
    <h2>Equity curve (last month)</h2>
    {_sparkline(data["equity_series"])}
  </div>

  <div class="card">
    <h2>Positions</h2>
    <table><thead><tr><th>Symbol</th><th>Qty</th><th>Avg entry</th><th>Market value</th><th>Unrealized P/L</th></tr></thead>
    <tbody>{_positions_rows(data["positions"])}</tbody></table>
  </div>

  <div class="card">
    <h2>Recent filled trades</h2>
    <table><thead><tr><th>Filled</th><th>Side</th><th>Symbol</th><th>Qty</th><th>Price</th></tr></thead>
    <tbody>{_orders_rows(data["orders"])}</tbody></table>
  </div>

  <div class="card">
    <h2>Daily decisions <span class="pill">signal &rarr; action</span></h2>
    <table><thead><tr><th>When (UTC)</th><th>Signal prob.</th><th>Target</th><th>Action</th><th>Submitted</th></tr></thead>
    <tbody>{_decision_rows(decisions)}</tbody></table>
  </div>

  <div class="foot">Refresh with <code>python scripts/dashboard.py</code> &middot; entry &ge; 0.55, exit &le; 0.45</div>
</div></body></html>"""


def main() -> None:
    client = _client()
    data = _fetch(client)
    decisions = _decisions()
    if DATABASE_PATH.exists():
        history = ExecutionStore(DATABASE_PATH).equity_history()
        if not history.empty:
            data["equity_series"] = list(
                zip(history["signal_date"], history["equity"], strict=True)
            )
            data["baseline_equity"] = float(history["equity"].iloc[0])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(build_html(data, decisions), encoding="utf-8")
    print(f"Dashboard written to {OUT_PATH}")


if __name__ == "__main__":
    main()
