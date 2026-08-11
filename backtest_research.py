# FILE_VERSION: FTSE350_ORB_RANK1_30DAY_RESEARCH_2026_08_11
"""
Historical research page for the live FTSE 350 day-trading strategy.

What it tests
-------------
For each of the last 30 completed London trading sessions:

1. Use ONLY daily data available at the PRIOR close.
2. Rebuild the same liquid long/short watchlist used by screener.py via
   build_intraday_candidate().
3. Select rank 1 only.
4. On the next session, establish the 08:00-08:15 opening range.
5. From 08:15 to 09:30, replay the current technical decision rules:
      - correct side of previous close
      - correct side of VWAP
      - opening turnover >= live threshold
      - controlled gap
      - not excessively stretched from VWAP
      - relative opening volume >= live ready threshold
      - post-opening-range higher-low/lower-high structure
      - break of the 15-minute opening range
      - no entry beyond MAX_CHASE_R
6. If a trade confirms, invest exactly £30 notional (fractional shares allowed).
7. Exit the full position at the displayed 2R target or stop. If neither is hit,
   exit at the final completed bar before 09:30.
8. If both stop and target are touched in the same five-minute candle, assume
   the stop was hit first (conservative).

Historical news caveat
----------------------
The live page can show current Yahoo headlines. Yahoo's search feed is not a
reliable point-in-time archive for every historical session, so this research
does NOT award or penalise a historical news catalyst. All historical entries
are therefore technical B-grade equivalents. This avoids look-ahead bias.

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

# Keep these defaults aligned with intraday_daytrader.py.
MAX_GAP_PCT = 3.0
MIN_OPENING_TURNOVER_GBP = 75_000.0
MAX_VWAP_DISTANCE_PCT = 1.50
ENTRY_BUFFER_PCT = 0.04
ATR_STOP_MULTIPLIER = 0.50
MIN_STOP_PCT = 0.75
MIN_READY_VOLUME_RATIO = 0.75
MAX_CHASE_R = 0.30
TARGET_R = 2.0

# Research-only execution cost assumption. 20 bps = 0.20% of £30 = 6p per trade.
SPREAD_BPS = float(os.environ.get("SPREAD_BPS") or 20.0)
COMMISSION_PER_TRADE = float(os.environ.get("COMMISSION_PER_TRADE") or 0.0)

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


def london_intraday(frame: pd.DataFrame) -> pd.DataFrame:
    frame = normalise(frame)
    if frame.empty:
        return frame
    idx = pd.DatetimeIndex(frame.index)
    if idx.tz is None:
        idx = idx.tz_localize(UTC)
    frame.index = idx.tz_convert(LONDON)
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


def atr_as_of(frame: pd.DataFrame, signal_date: dt.date, period: int = 14) -> float | None:
    hist = frame[pd.DatetimeIndex(frame.index).date <= signal_date].copy()
    if len(hist) < period + 1:
        return None
    prev_close = hist["Close"].shift(1)
    tr = pd.concat(
        [
            hist["High"] - hist["Low"],
            (hist["High"] - prev_close).abs(),
            (hist["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.tail(period).mean()
    return float(atr) if math.isfinite(float(atr)) and float(atr) > 0 else None


def previous_close(frame: pd.DataFrame, signal_date: dt.date) -> float | None:
    hist = frame[pd.DatetimeIndex(frame.index).date <= signal_date]
    if hist.empty:
        return None
    return float(hist.iloc[-1]["Close"])


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

        dates = pd.DatetimeIndex(frame.index).date
        hist = frame[dates <= signal_date]
        if len(hist) < 25:
            continue

        # Restrict to information known at the prior close.
        hist = hist.tail(70)
        try:
            candidate = scr.build_intraday_candidate(epic, name, hist)
        except Exception:
            continue
        if candidate:
            candidate = dict(candidate)
            candidate["yahoo_ticker"] = ticker
            candidates.append(candidate)

    if not candidates:
        return None

    candidates.sort(key=lambda r: float(r.get("score", 0) or 0), reverse=True)
    return candidates[0]


def prior_comparable_volume(
    intraday: pd.DataFrame,
    trade_date: dt.date,
    end_time: dt.time,
) -> float | None:
    totals: list[float] = []
    for day in sorted(set(intraday.index.date)):
        if day >= trade_date:
            continue
        part = intraday[
            (intraday.index.date == day)
            & (intraday.index.time >= dt.time(8, 0))
            & (intraday.index.time < end_time)
        ]
        total = float(part["Volume"].fillna(0).sum())
        if total > 0:
            totals.append(total)
    # Use up to the most recent five comparable sessions, matching the limited
    # 5-minute history available from Yahoo without using future information.
    return float(np.mean(totals[-5:])) if totals else None


def structure_ok(opening: pd.DataFrame, direction: str) -> bool:
    if len(opening) < 5:
        return False
    post = opening.iloc[3:]
    if direction == "Long":
        lows = post["Low"].astype(float)
        return len(lows) >= 2 and lows.iloc[-1] >= lows.iloc[0]
    highs = post["High"].astype(float)
    return len(highs) >= 2 and highs.iloc[-1] <= highs.iloc[0]


def simulate_day(
    candidate: dict[str, Any],
    trade_date: dt.date,
    signal_date: dt.date,
    daily_frame: pd.DataFrame,
    intraday: pd.DataFrame,
) -> dict[str, Any]:
    epic = str(candidate["epic"])
    name = str(candidate["name"])
    direction = str(candidate["direction"]).title()

    result: dict[str, Any] = {
        "date": trade_date,
        "signal_date": signal_date,
        "epic": epic,
        "name": name,
        "direction": direction,
        "watch_score": float(candidate.get("score", 0) or 0),
        "status": "NO TRADE",
        "reason": "",
        "entry": None,
        "stop": None,
        "target": None,
        "exit": None,
        "gross_pnl": 0.0,
        "cost": 0.0,
        "net_pnl": 0.0,
        "return_pct": 0.0,
        "equity": None,
    }

    day = intraday[
        (intraday.index.date == trade_date)
        & (intraday.index.time >= dt.time(8, 0))
        & (intraday.index.time < dt.time(9, 30))
    ].copy()

    if len(day) < 5:
        result["status"] = "NO DATA"
        result["reason"] = "Fewer than five completed 5-minute bars."
        return result

    opening_range = day[
        (day.index.time >= dt.time(8, 0))
        & (day.index.time < dt.time(8, 15))
    ]
    if len(opening_range) < 3:
        result["status"] = "NO DATA"
        result["reason"] = "Incomplete 08:00-08:15 opening range."
        return result

    pc = previous_close(daily_frame, signal_date)
    atr = atr_as_of(daily_frame, signal_date)
    if pc is None or pc <= 0:
        result["status"] = "NO DATA"
        result["reason"] = "Previous close unavailable."
        return result

    first_open = float(day.iloc[0]["Open"])
    gap_pct = ((first_open - pc) / pc) * 100
    or_high = float(opening_range["High"].max())
    or_low = float(opening_range["Low"].min())

    if abs(gap_pct) > MAX_GAP_PCT:
        result["status"] = "NO TRADE"
        result["reason"] = f"Opening gap {gap_pct:+.2f}% exceeded limit."
        return result

    trigger = (
        or_high * (1 + ENTRY_BUFFER_PCT / 100)
        if direction == "Long"
        else or_low * (1 - ENTRY_BUFFER_PCT / 100)
    )

    structure_stop = or_low if direction == "Long" else or_high
    atr_distance = (atr or 0.0) * ATR_STOP_MULTIPLIER
    minimum_distance = trigger * MIN_STOP_PCT / 100

    if direction == "Long":
        stop_distance = max(trigger - structure_stop, atr_distance, minimum_distance)
        stop = trigger - stop_distance
        target = trigger + TARGET_R * stop_distance
        max_entry = trigger + MAX_CHASE_R * stop_distance
    else:
        stop_distance = max(structure_stop - trigger, atr_distance, minimum_distance)
        stop = trigger + stop_distance
        target = trigger - TARGET_R * stop_distance
        max_entry = trigger - MAX_CHASE_R * stop_distance

    result["stop"] = stop
    result["target"] = target

    entry_idx = None
    entry_price = None

    # Replay the live assessment at each completed five-minute bar.
    for i in range(4, len(day)):
        opening = day.iloc[: i + 1]
        current = float(opening.iloc[-1]["Close"])
        end_time = (opening.index[-1] + pd.Timedelta(minutes=5)).time()

        typical = (opening["High"] + opening["Low"] + opening["Close"]) / 3
        vols = opening["Volume"].fillna(0)
        vwap = float((typical * vols).sum() / vols.sum()) if vols.sum() > 0 else current

        volume = float(vols.sum())
        avg_volume = prior_comparable_volume(intraday, trade_date, end_time)
        volume_ratio = volume / avg_volume if avg_volume and avg_volume > 0 else None
        turnover_gbp = volume * gbx_to_gbp(current)
        vwap_distance_pct = abs((current - vwap) / vwap) * 100 if vwap else 0.0

        if direction == "Long":
            side_prev = current > pc
            side_vwap = current > vwap
            broken = current >= trigger
            beyond = current > max_entry
        else:
            side_prev = current < pc
            side_vwap = current < vwap
            broken = current <= trigger
            beyond = current < max_entry

        hard_fail = (
            not side_prev
            or not side_vwap
            or turnover_gbp < MIN_OPENING_TURNOVER_GBP
            or vwap_distance_pct > MAX_VWAP_DISTANCE_PCT
        )
        ready = (
            not hard_fail
            and volume_ratio is not None
            and volume_ratio >= MIN_READY_VOLUME_RATIO
            and structure_ok(opening, direction)
        )

        if beyond:
            result["status"] = "MISSED"
            result["reason"] = "Breakout moved beyond the permitted chase zone."
            return result

        if broken and ready:
            entry_idx = i
            # Live user would enter after the confirming completed candle, so
            # use that candle's close rather than assuming a fill at the trigger.
            entry_price = current
            break

    if entry_idx is None or entry_price is None:
        result["status"] = "NO TRADE"
        result["reason"] = "No confirmed opening-range breakout before 09:30."
        return result

    result["entry"] = entry_price

    # Evaluate subsequent 5-minute candles. Entry confirmation occurs on the
    # close of entry_idx, so start outcome testing on the NEXT candle.
    exit_price = float(day.iloc[-1]["Close"])
    outcome = "TIME EXIT"

    for j in range(entry_idx + 1, len(day)):
        bar = day.iloc[j]
        bar_high = float(bar["High"])
        bar_low = float(bar["Low"])

        if direction == "Long":
            stop_hit = bar_low <= stop
            target_hit = bar_high >= target
        else:
            stop_hit = bar_high >= stop
            target_hit = bar_low <= target

        if stop_hit and target_hit:
            exit_price = stop
            outcome = "STOP (same-bar conservative)"
            break
        if stop_hit:
            exit_price = stop
            outcome = "STOP"
            break
        if target_hit:
            exit_price = target
            outcome = "2R TARGET"
            break

    shares = NOTIONAL_GBP / gbx_to_gbp(entry_price)
    if direction == "Long":
        gross = shares * (gbx_to_gbp(exit_price) - gbx_to_gbp(entry_price))
    else:
        gross = shares * (gbx_to_gbp(entry_price) - gbx_to_gbp(exit_price))

    spread_cost = NOTIONAL_GBP * (SPREAD_BPS / 10_000.0)
    cost = spread_cost + COMMISSION_PER_TRADE
    net = gross - cost

    result.update(
        {
            "status": outcome,
            "reason": "Trade confirmed under the current technical strategy.",
            "exit": exit_price,
            "gross_pnl": gross,
            "cost": cost,
            "net_pnl": net,
            "return_pct": (net / NOTIONAL_GBP) * 100,
        }
    )
    return result


def max_drawdown(values: list[float]) -> float:
    peak = values[0] if values else 0.0
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        worst = min(worst, value - peak)
    return worst


def render(results: list[dict[str, Any]], generated: str) -> str:
    traded = [r for r in results if r["entry"] is not None]
    wins = [r for r in traded if r["net_pnl"] > 0]
    losses = [r for r in traded if r["net_pnl"] < 0]
    no_trades = [r for r in results if r["entry"] is None]

    total_net = sum(r["net_pnl"] for r in results)
    cumulative = 0.0
    curve = [0.0]
    for r in results:
        cumulative += r["net_pnl"]
        r["equity"] = cumulative
        curve.append(cumulative)

    win_rate = (len(wins) / len(traded) * 100) if traded else 0.0
    gross_profit = sum(r["net_pnl"] for r in wins)
    gross_loss = abs(sum(r["net_pnl"] for r in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    dd = max_drawdown(curve)

    rows = []
    for r in reversed(results):
        pnl_class = "positive" if r["net_pnl"] > 0 else ("negative" if r["net_pnl"] < 0 else "neutral")
        rows.append(
            "<tr>"
            f"<td>{r['date'].strftime('%d %b %Y')}</td>"
            f"<td><strong>{html_lib.escape(r['epic'])}</strong><br><span class='muted'>{html_lib.escape(r['name'])}</span></td>"
            f"<td>{html_lib.escape(r['direction'])}</td>"
            f"<td>{html_lib.escape(r['status'])}</td>"
            f"<td>{'£'+format(gbx_to_gbp(r['entry']), '.2f') if r['entry'] is not None else '—'}</td>"
            f"<td>{'£'+format(gbx_to_gbp(r['stop']), '.2f') if r['stop'] is not None else '—'}</td>"
            f"<td>{'£'+format(gbx_to_gbp(r['target']), '.2f') if r['target'] is not None else '—'}</td>"
            f"<td>{'£'+format(gbx_to_gbp(r['exit']), '.2f') if r['exit'] is not None else '—'}</td>"
            f"<td class='{pnl_class}'>{r['net_pnl']:+.2f}</td>"
            f"<td class='{pnl_class}'>{r['return_pct']:+.2f}%</td>"
            f"<td>{r['equity']:+.2f}</td>"
            "</tr>"
        )

    pf_text = "∞" if math.isinf(profit_factor) else f"{profit_factor:.2f}"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FTSE 350 · 30-day rank-1 strategy research</title>
<style>
:root{{--ink:{INK};--panel:{PANEL};--line:{HAIRLINE};--brass:{BRASS};--salmon:{SALMON};--green:{BULL};--red:{BEAR};--paper:{PAPER};--muted:{MUTED}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--ink);color:var(--paper);font-family:Arial,sans-serif}}
.wrap{{max-width:1120px;margin:auto;padding:28px 18px 60px}}
nav{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px}} nav a{{color:var(--paper);text-decoration:none;border:1px solid var(--line);padding:8px 12px;border-radius:999px;font-size:13px}} nav a.active{{color:var(--brass);border-color:var(--brass)}}
h1{{font-size:28px;margin:5px 0}} h2{{font-size:18px;margin-top:26px}} .sub,.note,.muted{{color:var(--muted)}} .sub{{font-size:12px;line-height:1.55}}
.strategy{{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:15px;margin:18px 0;line-height:1.6;font-size:13px}} .strategy strong{{color:var(--brass)}}
.grid{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;margin:16px 0}}
.stat{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}}
.label{{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.04em}} .value{{font-size:20px;margin-top:5px}}
.table-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:8px;margin-top:14px}}
table{{width:100%;border-collapse:collapse;min-width:1000px;background:var(--panel)}} th,td{{padding:9px 10px;border-bottom:1px solid var(--line);font-size:12px;text-align:right;white-space:nowrap}} th{{color:var(--muted);font-size:10px;text-transform:uppercase}} th:nth-child(1),td:nth-child(1),th:nth-child(2),td:nth-child(2),th:nth-child(4),td:nth-child(4){{text-align:left}}
.positive{{color:var(--green)}} .negative{{color:var(--red)}} .neutral{{color:var(--muted)}}
@media(max-width:800px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
</style>
</head>
<body><main class="wrap">
<nav><a href="index.html">Pre-market</a><a href="intraday.html">Intraday</a><a href="backtest.html">Backtest</a><a class="active" href="backtest-research.html">Research</a></nav>
<div class="sub">FTSE 350 ex trusts · current-strategy historical replay</div>
<h1>£30 rank-1 strategy · last 30 trading days</h1>
<div class="sub">Generated {generated}</div>

<div class="strategy"><strong>What this page tests:</strong> each day it recreates the rank-1 pre-market candidate using only the prior close's data, then waits for the current 15-minute opening-range breakout rules. If the trade confirms, £30 notional is entered at the confirming five-minute close. Full position exits at the displayed 2R target, stop, or the final bar before 09:30. Days with no qualifying breakout remain uninvested.</div>

<div class="grid">
<div class="stat"><div class="label">Trading days tested</div><div class="value">{len(results)}</div></div>
<div class="stat"><div class="label">Trades taken</div><div class="value">{len(traded)}</div></div>
<div class="stat"><div class="label">No-trade / missed days</div><div class="value">{len(no_trades)}</div></div>
<div class="stat"><div class="label">Win rate</div><div class="value">{win_rate:.1f}%</div></div>
<div class="stat"><div class="label">Net P/L</div><div class="value {'positive' if total_net >= 0 else 'negative'}">£{total_net:+.2f}</div></div>
<div class="stat"><div class="label">End value vs £30/day stake</div><div class="value">£{30*len(traded)+total_net:.2f}</div></div>
</div>

<div class="grid">
<div class="stat"><div class="label">Winning trades</div><div class="value">{len(wins)}</div></div>
<div class="stat"><div class="label">Losing trades</div><div class="value">{len(losses)}</div></div>
<div class="stat"><div class="label">Profit factor</div><div class="value">{pf_text}</div></div>
<div class="stat"><div class="label">Max drawdown</div><div class="value negative">£{dd:.2f}</div></div>
<div class="stat"><div class="label">Stake per trade</div><div class="value">£30</div></div>
<div class="stat"><div class="label">Cost assumption</div><div class="value">{SPREAD_BPS:.0f} bps</div></div>
</div>

<h2>Daily replay</h2>
<div class="table-wrap"><table><thead><tr>
<th>Date</th><th>Rank-1 stock</th><th>Bias</th><th>Outcome</th><th>Entry</th><th>Stop</th><th>2R</th><th>Exit</th><th>Net £</th><th>Return</th><th>Cumulative £</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table></div>

<p class="note"><strong>Historical-news limitation:</strong> this replay deliberately does not use archived news sentiment because Yahoo's current headline search is not a reliable point-in-time archive. That prevents look-ahead bias. Historical entries therefore represent the technical B-grade form of the live strategy; the live page may upgrade a setup to A when a genuinely supportive current catalyst is present.</p>
<p class="note">This is a mechanical research replay, not a prediction. Five-minute bars still cannot show the exact sequence of prices inside each candle; where a single candle touches both stop and target, the research assumes the stop occurred first.</p>
</main></body></html>"""


def main() -> int:
    print("Fetching current FTSE 350 operating-company universe...")
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

    # Use FTSE index dates as the session calendar.
    calendar = normalise(
        yf.download(
            "^FTSE",
            period="4mo",
            interval="1d",
            progress=False,
            auto_adjust=False,
        )
    )
    if calendar.empty:
        raise RuntimeError("Unable to obtain FTSE trading calendar.")

    now = dt.datetime.now(UTC).astimezone(LONDON)
    completed_dates = [
        d.date()
        for d in pd.DatetimeIndex(calendar.index)
        if d.date() < now.date()
    ]

    if len(completed_dates) < DAYS + 1:
        raise RuntimeError("Not enough completed trading sessions for 30-day research.")

    trade_dates = completed_dates[-DAYS:]
    calendar_dates = [d.date() for d in pd.DatetimeIndex(calendar.index)]

    selections: list[tuple[dt.date, dt.date, dict[str, Any]]] = []
    for trade_date in trade_dates:
        idx = calendar_dates.index(trade_date)
        if idx == 0:
            continue
        signal_date = calendar_dates[idx - 1]
        print(f"Selecting rank 1 as of {signal_date} for trade date {trade_date}...")
        pick = select_rank1(constituents, daily_data, signal_date)
        if pick:
            selections.append((trade_date, signal_date, pick))
        else:
            selections.append(
                (
                    trade_date,
                    signal_date,
                    {
                        "epic": "—",
                        "name": "No candidate",
                        "direction": "—",
                        "score": 0,
                        "yahoo_ticker": "",
                    },
                )
            )

    unique_tickers = sorted(
        {
            pick["yahoo_ticker"]
            for _, _, pick in selections
            if pick.get("yahoo_ticker")
        }
    )

    intraday_cache: dict[str, pd.DataFrame] = {}
    daily_cache: dict[str, pd.DataFrame] = {}

    print(f"Downloading 5-minute history for {len(unique_tickers)} unique rank-1 stocks...")
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
        daily_cache[ticker] = ticker_frame(daily_data, ticker)

    results: list[dict[str, Any]] = []
    for trade_date, signal_date, pick in selections:
        ticker = pick.get("yahoo_ticker")
        if not ticker:
            results.append(
                {
                    "date": trade_date,
                    "signal_date": signal_date,
                    "epic": "—",
                    "name": "No pre-market candidate",
                    "direction": "—",
                    "watch_score": 0,
                    "status": "NO TRADE",
                    "reason": "No rank-1 candidate.",
                    "entry": None,
                    "stop": None,
                    "target": None,
                    "exit": None,
                    "gross_pnl": 0.0,
                    "cost": 0.0,
                    "net_pnl": 0.0,
                    "return_pct": 0.0,
                    "equity": None,
                }
            )
            continue

        print(f"Replaying {trade_date}: {pick['epic']} {pick['direction']}...")
        results.append(
            simulate_day(
                pick,
                trade_date,
                signal_date,
                daily_cache[ticker],
                intraday_cache[ticker],
            )
        )

    generated = now.strftime("%a %d %b %Y · %H:%M %Z")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(render(results, generated), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
