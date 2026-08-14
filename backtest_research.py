# FILE_VERSION: FTSE350_RANK1_CLOSE_TO_NEXT_DAY_1R_VS_2R_2026_08_14
"""
Backtest the CURRENT screener's rank-1 candidate as an overnight CFD strategy.

Strategy under test
-------------------
For each of the last 30 completed signal sessions:

1. Rebuild the current screener's liquid long/short candidate list using ONLY
   daily data through that signal day's close.
2. Take rank 1 only.
3. Enter a £30 notional CFD at that signal day's closing price (proxy for
   entering shortly before the London close).
4. Hold overnight into the next trading session.
5. Replay TWO exit policies independently using the exact stop/risk generated
   by screener.build_intraday_candidate():

      Policy A: stop or 1R target
      Policy B: stop or 2R target

   If neither target nor stop is hit, exit at the next session's 16:25/last
   available 5-minute close before 16:30.

6. Also record mark-to-market return at 08:00, 08:05, 08:15 and 09:30 so the
   opening effect can be seen directly.

Important execution caveat
--------------------------
The historical selector uses the completed daily candle and enters at that
day's closing price. That is a close-of-day proxy for an intended 16:20-16:25
entry, not an exact reconstruction of what the rank would have been before the
closing auction. It is therefore useful research, but slightly optimistic if
the final minutes materially changed the ranking.

CFD caveat
----------
Underlying LSE prices are used as the price proxy. Broker-specific CFD spread,
slippage and overnight financing are not available from Yahoo. A configurable
round-trip cost is therefore deducted from each trade.

Output: docs/backtest-research.html
"""

from __future__ import annotations

import datetime as dt
import html as html_lib
import math
import os
import sys
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

import screener as scr

LONDON = ZoneInfo("Europe/London")
UTC = dt.timezone.utc

OUTPUT_PATH = Path(__file__).resolve().parent / "docs" / "backtest-research.html"

DAYS = 30
NOTIONAL_GBP = 30.0
ROUND_TRIP_COST_BPS = float(os.environ.get("ROUND_TRIP_COST_BPS") or 20.0)

INK = "#12161F"
PANEL = "#1B2129"
HAIRLINE = "#2C333D"
BRASS = "#C9A24B"
SALMON = "#E8A493"
BULL = "#4FAE73"
BEAR = "#D1594B"
PAPER = "#ECE7DA"
MUTED = "#8B92A0"


def gbx_to_gbp(value: float) -> float:
    return float(value) / 100.0


def normalise(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in frame.columns for c in required):
        return pd.DataFrame()
    frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
    return frame.sort_index()


def ticker_frame(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        try:
            frame = data[ticker].copy()
        except Exception:
            try:
                frame = data.xs(ticker, axis=1, level=1).copy()
            except Exception:
                return pd.DataFrame()
    else:
        frame = data.copy()
    return normalise(frame)


def london_intraday(frame: pd.DataFrame) -> pd.DataFrame:
    frame = normalise(frame)
    if frame.empty:
        return frame
    idx = pd.DatetimeIndex(frame.index)
    if idx.tz is None:
        idx = idx.tz_localize(UTC)
    frame.index = idx.tz_convert(LONDON)
    return frame.sort_index()


def select_rank1(
    constituents: dict[str, str],
    daily_data: pd.DataFrame,
    signal_date: dt.date,
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []

    for epic, name in constituents.items():
        ticker = scr.epic_to_yahoo(epic)
        frame = ticker_frame(daily_data, ticker)
        if frame.empty:
            continue

        hist = frame[pd.DatetimeIndex(frame.index).date <= signal_date]
        if len(hist) < 25:
            continue

        try:
            candidate = scr.build_intraday_candidate(epic, name, hist.tail(90))
        except Exception:
            continue

        if candidate:
            item = dict(candidate)
            item["yahoo_ticker"] = ticker
            candidates.append(item)

    if not candidates:
        return None

    candidates.sort(key=lambda row: float(row.get("score", 0) or 0), reverse=True)
    return candidates[0]


def first_price_at_or_after(day: pd.DataFrame, time_value: dt.time) -> float | None:
    part = day[day.index.time >= time_value]
    if part.empty:
        return None
    return float(part.iloc[0]["Open"])


def close_at_or_before(day: pd.DataFrame, time_value: dt.time) -> float | None:
    part = day[day.index.time <= time_value]
    if part.empty:
        return None
    return float(part.iloc[-1]["Close"])


def directional_return(direction: str, entry_gbx: float, price_gbx: float) -> float:
    if entry_gbx <= 0:
        return 0.0
    if direction == "Long":
        return (price_gbx / entry_gbx - 1.0) * 100.0
    return (entry_gbx / price_gbx - 1.0) * 100.0


def pnl_from_prices(direction: str, entry_gbx: float, exit_gbx: float) -> float:
    entry_gbp = gbx_to_gbp(entry_gbx)
    exit_gbp = gbx_to_gbp(exit_gbx)
    if entry_gbp <= 0:
        return 0.0
    units = NOTIONAL_GBP / entry_gbp
    if direction == "Long":
        return units * (exit_gbp - entry_gbp)
    return units * (entry_gbp - exit_gbp)


def simulate_policy(
    day: pd.DataFrame,
    direction: str,
    entry: float,
    stop: float,
    target: float,
) -> dict[str, Any]:
    """
    Trade is already open from previous close. Evaluate from the next session's
    first 5-minute bar. Same-bar stop+target ambiguity is resolved conservatively
    as STOP first.
    """
    for _, bar in day.iterrows():
        high = float(bar["High"])
        low = float(bar["Low"])

        if direction == "Long":
            stop_hit = low <= stop
            target_hit = high >= target
        else:
            stop_hit = high >= stop
            target_hit = low <= target

        if stop_hit and target_hit:
            return {"exit": stop, "outcome": "STOP (same-bar conservative)"}
        if stop_hit:
            return {"exit": stop, "outcome": "STOP"}
        if target_hit:
            return {"exit": target, "outcome": "TARGET"}

    return {"exit": float(day.iloc[-1]["Close"]), "outcome": "DAY CLOSE"}


def simulate_day(
    pick: dict[str, Any],
    signal_date: dt.date,
    trade_date: dt.date,
    intraday: pd.DataFrame,
) -> dict[str, Any]:
    direction = str(pick["direction"]).title()
    entry = float(pick["entry"])
    stop = float(pick["stop"])
    risk = float(pick["risk_per_share"])

    target_1r = entry + risk if direction == "Long" else entry - risk
    target_2r = entry + 2.0 * risk if direction == "Long" else entry - 2.0 * risk

    result: dict[str, Any] = {
        "signal_date": signal_date,
        "trade_date": trade_date,
        "epic": str(pick["epic"]),
        "name": str(pick["name"]),
        "direction": direction,
        "score": float(pick.get("score", 0) or 0),
        "entry": entry,
        "stop": stop,
        "risk": risk,
        "target_1r": target_1r,
        "target_2r": target_2r,
        "open_return_pct": None,
        "r0805_pct": None,
        "r0815_pct": None,
        "r0930_pct": None,
        "one_r_outcome": "NO DATA",
        "one_r_exit": None,
        "one_r_net": 0.0,
        "two_r_outcome": "NO DATA",
        "two_r_exit": None,
        "two_r_net": 0.0,
    }

    day = intraday[
        (intraday.index.date == trade_date)
        & (intraday.index.time >= dt.time(8, 0))
        & (intraday.index.time < dt.time(16, 30))
    ].copy()

    if day.empty:
        return result

    open_price = float(day.iloc[0]["Open"])
    p0805 = close_at_or_before(day[day.index.time < dt.time(8, 10)], dt.time(8, 5))
    p0815 = close_at_or_before(day[day.index.time < dt.time(8, 20)], dt.time(8, 15))
    p0930 = close_at_or_before(day[day.index.time < dt.time(9, 35)], dt.time(9, 30))

    result["open_return_pct"] = directional_return(direction, entry, open_price)
    result["r0805_pct"] = directional_return(direction, entry, p0805) if p0805 else None
    result["r0815_pct"] = directional_return(direction, entry, p0815) if p0815 else None
    result["r0930_pct"] = directional_return(direction, entry, p0930) if p0930 else None

    policy1 = simulate_policy(day, direction, entry, stop, target_1r)
    policy2 = simulate_policy(day, direction, entry, stop, target_2r)

    cost = NOTIONAL_GBP * ROUND_TRIP_COST_BPS / 10_000.0

    gross1 = pnl_from_prices(direction, entry, float(policy1["exit"]))
    gross2 = pnl_from_prices(direction, entry, float(policy2["exit"]))

    result.update({
        "one_r_outcome": policy1["outcome"],
        "one_r_exit": float(policy1["exit"]),
        "one_r_net": gross1 - cost,
        "two_r_outcome": policy2["outcome"],
        "two_r_exit": float(policy2["exit"]),
        "two_r_net": gross2 - cost,
    })
    return result


def summary(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    pnl_key = f"{prefix}_net"
    outcome_key = f"{prefix}_outcome"
    valid = [r for r in rows if r[outcome_key] != "NO DATA"]
    wins = [r for r in valid if r[pnl_key] > 0]
    losses = [r for r in valid if r[pnl_key] < 0]
    total = sum(r[pnl_key] for r in valid)
    gross_profit = sum(r[pnl_key] for r in wins)
    gross_loss = abs(sum(r[pnl_key] for r in losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for r in valid:
        equity += r[pnl_key]
        peak = max(peak, equity)
        drawdown = min(drawdown, equity - peak)

    return {
        "trades": len(valid),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(valid) * 100 if valid else 0.0,
        "pnl": total,
        "pf": pf,
        "dd": drawdown,
        "avg": total / len(valid) if valid else 0.0,
    }


def fmt_pct(value: float | None) -> str:
    return "—" if value is None else f"{value:+.2f}%"


def fmt_price(value: float | None) -> str:
    return "—" if value is None else f"£{gbx_to_gbp(value):.2f}"


def render(rows: list[dict[str, Any]], generated: str) -> str:
    one = summary(rows, "one_r")
    two = summary(rows, "two_r")

    open_values = [r["open_return_pct"] for r in rows if r["open_return_pct"] is not None]
    v0805 = [r["r0805_pct"] for r in rows if r["r0805_pct"] is not None]
    v0815 = [r["r0815_pct"] for r in rows if r["r0815_pct"] is not None]
    v0930 = [r["r0930_pct"] for r in rows if r["r0930_pct"] is not None]

    def mean(values):
        return float(np.mean(values)) if values else 0.0

    pf1 = "∞" if math.isinf(one["pf"]) else f"{one['pf']:.2f}"
    pf2 = "∞" if math.isinf(two["pf"]) else f"{two['pf']:.2f}"

    better = "1R" if one["pnl"] > two["pnl"] else ("2R" if two["pnl"] > one["pnl"] else "Tie")

    cumulative1 = 0.0
    cumulative2 = 0.0
    html_rows = []
    for r in reversed(rows):
        # cumulative shown chronologically requires precompute below
        pass

    c1 = c2 = 0.0
    for r in rows:
        c1 += r["one_r_net"]
        c2 += r["two_r_net"]
        r["cum_1r"] = c1
        r["cum_2r"] = c2

    for r in reversed(rows):
        cls1 = "positive" if r["one_r_net"] > 0 else ("negative" if r["one_r_net"] < 0 else "neutral")
        cls2 = "positive" if r["two_r_net"] > 0 else ("negative" if r["two_r_net"] < 0 else "neutral")
        html_rows.append(
            "<tr>"
            f"<td>{r['signal_date'].strftime('%d %b')}</td>"
            f"<td>{r['trade_date'].strftime('%d %b')}</td>"
            f"<td><strong>{html_lib.escape(r['epic'])}</strong><br><span class='muted'>{html_lib.escape(r['name'])}</span></td>"
            f"<td>{html_lib.escape(r['direction'])}</td>"
            f"<td>{fmt_price(r['entry'])}</td>"
            f"<td>{fmt_pct(r['open_return_pct'])}</td>"
            f"<td>{fmt_pct(r['r0805_pct'])}</td>"
            f"<td>{fmt_pct(r['r0815_pct'])}</td>"
            f"<td>{fmt_pct(r['r0930_pct'])}</td>"
            f"<td>{fmt_price(r['stop'])}</td>"
            f"<td>{fmt_price(r['target_1r'])}</td>"
            f"<td>{html_lib.escape(r['one_r_outcome'])}</td>"
            f"<td class='{cls1}'>£{r['one_r_net']:+.2f}</td>"
            f"<td>{fmt_price(r['target_2r'])}</td>"
            f"<td>{html_lib.escape(r['two_r_outcome'])}</td>"
            f"<td class='{cls2}'>£{r['two_r_net']:+.2f}</td>"
            f"<td class='{cls1}'>£{r['cum_1r']:+.2f}</td>"
            f"<td class='{cls2}'>£{r['cum_2r']:+.2f}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FTSE 350 · Overnight rank-1 CFD research</title>
<style>
:root{{--ink:{INK};--panel:{PANEL};--line:{HAIRLINE};--brass:{BRASS};--salmon:{SALMON};--green:{BULL};--red:{BEAR};--paper:{PAPER};--muted:{MUTED}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--ink);color:var(--paper);font-family:Arial,sans-serif}}
.wrap{{max-width:1300px;margin:auto;padding:28px 18px 60px}}
nav{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px}} nav a{{color:var(--paper);text-decoration:none;border:1px solid var(--line);padding:8px 12px;border-radius:999px;font-size:13px}} nav a.active{{color:var(--brass);border-color:var(--brass)}}
h1{{font-size:28px;margin:5px 0}} h2{{font-size:18px;margin:26px 0 10px}} .sub,.note,.muted{{color:var(--muted)}} .sub{{font-size:12px;line-height:1.55}}
.callout{{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:15px;margin:18px 0;line-height:1.6;font-size:13px}} .callout strong{{color:var(--brass)}}
.grid{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;margin:16px 0}}
.stat{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}}
.label{{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.04em}} .value{{font-size:19px;margin-top:5px}}
.compare{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:16px 0}}
.policy{{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:16px}} .policy h3{{margin:0 0 12px;color:var(--brass)}} .policy-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}}
.table-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:8px;margin-top:14px}}
table{{width:100%;border-collapse:collapse;min-width:1650px;background:var(--panel)}} th,td{{padding:8px 9px;border-bottom:1px solid var(--line);font-size:11px;text-align:right;white-space:nowrap}} th{{color:var(--muted);font-size:9px;text-transform:uppercase}} th:nth-child(1),td:nth-child(1),th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3),th:nth-child(12),td:nth-child(12),th:nth-child(15),td:nth-child(15){{text-align:left}}
.positive{{color:var(--green)}} .negative{{color:var(--red)}} .neutral{{color:var(--muted)}}
@media(max-width:800px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}} .compare{{grid-template-columns:1fr}}}}
</style>
</head>
<body><main class="wrap">
<nav><a href="index.html">Daily screener</a><a href="intraday.html">Intraday</a><a href="backtest.html">Backtest</a><a class="active" href="backtest-research.html">Research</a></nav>
<div class="sub">FTSE 350 ex trusts · rank-1 close-to-next-session CFD replay</div>
<h1>£30 overnight rank-1 strategy · 1R versus 2R</h1>
<div class="sub">Generated {generated}</div>

<div class="callout"><strong>Exact hypothesis:</strong> use the current screener unchanged, take its rank-1 LONG or SHORT candidate at the end of the signal day, enter a £30 CFD at the closing-price proxy, hold overnight, and manage the following session using the screener's own structural stop. This page independently tests taking profit at 1R and 2R.</div>

<div class="grid">
<div class="stat"><div class="label">Signal days</div><div class="value">{len(rows)}</div></div>
<div class="stat"><div class="label">Avg move at open</div><div class="value {'positive' if mean(open_values)>=0 else 'negative'}">{mean(open_values):+.2f}%</div></div>
<div class="stat"><div class="label">Avg by 08:05</div><div class="value {'positive' if mean(v0805)>=0 else 'negative'}">{mean(v0805):+.2f}%</div></div>
<div class="stat"><div class="label">Avg by 08:15</div><div class="value {'positive' if mean(v0815)>=0 else 'negative'}">{mean(v0815):+.2f}%</div></div>
<div class="stat"><div class="label">Avg by 09:30</div><div class="value {'positive' if mean(v0930)>=0 else 'negative'}">{mean(v0930):+.2f}%</div></div>
<div class="stat"><div class="label">Better exit in sample</div><div class="value">{better}</div></div>
</div>

<div class="compare">
<section class="policy"><h3>1R target</h3><div class="policy-grid">
<div><div class="label">Net P/L</div><div class="value {'positive' if one['pnl']>=0 else 'negative'}">£{one['pnl']:+.2f}</div></div>
<div><div class="label">Win rate</div><div class="value">{one['win_rate']:.1f}%</div></div>
<div><div class="label">Profit factor</div><div class="value">{pf1}</div></div>
<div><div class="label">Average/trade</div><div class="value">£{one['avg']:+.2f}</div></div>
<div><div class="label">Max drawdown</div><div class="value negative">£{one['dd']:.2f}</div></div>
<div><div class="label">Trades</div><div class="value">{one['trades']}</div></div>
</div></section>

<section class="policy"><h3>2R target</h3><div class="policy-grid">
<div><div class="label">Net P/L</div><div class="value {'positive' if two['pnl']>=0 else 'negative'}">£{two['pnl']:+.2f}</div></div>
<div><div class="label">Win rate</div><div class="value">{two['win_rate']:.1f}%</div></div>
<div><div class="label">Profit factor</div><div class="value">{pf2}</div></div>
<div><div class="label">Average/trade</div><div class="value">£{two['avg']:+.2f}</div></div>
<div><div class="label">Max drawdown</div><div class="value negative">£{two['dd']:.2f}</div></div>
<div><div class="label">Trades</div><div class="value">{two['trades']}</div></div>
</div></section>
</div>

<h2>Daily replay</h2>
<div class="table-wrap"><table><thead><tr>
<th>Signal</th><th>Next day</th><th>Rank-1 stock</th><th>Bias</th><th>Entry</th>
<th>Open</th><th>08:05</th><th>08:15</th><th>09:30</th><th>Stop</th>
<th>1R</th><th>1R outcome</th><th>1R £</th>
<th>2R</th><th>2R outcome</th><th>2R £</th><th>Cum 1R</th><th>Cum 2R</th>
</tr></thead><tbody>{''.join(html_rows)}</tbody></table></div>

<p class="note"><strong>Close-entry proxy:</strong> the selection uses the completed signal-day candle and its closing price. Your intended real execution would be shortly before 16:30, so the backtest cannot perfectly reproduce the final few minutes or closing auction. A materially good result should therefore be confirmed over a longer sample before relying on it.</p>
<p class="note"><strong>CFD costs:</strong> results deduct {ROUND_TRIP_COST_BPS:.0f} bps from each £30 trade as a configurable execution-cost proxy. Broker-specific overnight financing is not included unless you incorporate it into that cost assumption.</p>
<p class="note">When a single five-minute candle touches both a stop and target, this research assumes the stop was hit first. That is deliberately conservative.</p>
</main></body></html>"""


def main() -> int:
    print("Fetching FTSE 350 operating-company universe...")
    constituents = scr.fetch_ftse250_constituents()
    tickers = [scr.epic_to_yahoo(epic) for epic in constituents]

    print(f"Downloading daily history for {len(tickers)} tickers...")
    daily_data = yf.download(
        tickers,
        period="6mo",
        interval="1d",
        group_by="ticker",
        threads=True,
        progress=False,
        auto_adjust=False,
    )

    calendar = normalise(
        yf.download("^FTSE", period="4mo", interval="1d", progress=False, auto_adjust=False)
    )
    if calendar.empty:
        raise RuntimeError("Unable to obtain FTSE trading calendar.")

    now = dt.datetime.now(UTC).astimezone(LONDON)
    session_dates = [d.date() for d in pd.DatetimeIndex(calendar.index) if d.date() < now.date()]

    # Need one completed next session to evaluate each signal.
    if len(session_dates) < DAYS + 1:
        raise RuntimeError("Not enough completed trading sessions for 30-day research.")

    signal_dates = session_dates[-(DAYS + 1):-1]
    trade_dates = session_dates[-DAYS:]

    selections: list[tuple[dt.date, dt.date, dict[str, Any] | None]] = []
    for signal_date, trade_date in zip(signal_dates, trade_dates):
        print(f"Selecting rank 1 at {signal_date} close for {trade_date}...")
        pick = select_rank1(constituents, daily_data, signal_date)
        selections.append((signal_date, trade_date, pick))

    unique_tickers = sorted({
        pick["yahoo_ticker"]
        for _, _, pick in selections
        if pick and pick.get("yahoo_ticker")
    })

    print(f"Downloading 5-minute next-session data for {len(unique_tickers)} unique picks...")
    intraday_cache: dict[str, pd.DataFrame] = {}
    for ticker in unique_tickers:
        intraday_cache[ticker] = london_intraday(
            yf.download(
                ticker,
                period="60d",
                interval="5m",
                progress=False,
                auto_adjust=False,
                prepost=False,
            )
        )

    rows: list[dict[str, Any]] = []
    for signal_date, trade_date, pick in selections:
        if not pick:
            print(f"{signal_date}: no rank-1 candidate")
            continue
        ticker = pick["yahoo_ticker"]
        print(f"Replaying {signal_date} -> {trade_date}: {pick['epic']} {pick['direction']}")
        rows.append(simulate_day(pick, signal_date, trade_date, intraday_cache[ticker]))

    if not rows:
        raise RuntimeError("No historical rank-1 trades could be reconstructed.")

    generated = now.strftime("%a %d %b %Y · %H:%M %Z")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(render(rows, generated), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
