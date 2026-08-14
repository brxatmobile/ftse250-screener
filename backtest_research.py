# FILE_VERSION: FTSE350_RANK1_OVERNIGHT_LONG_SHORT_SPLIT_2026_08_14
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
ATR_STOP_MULTIPLIER = float(os.environ.get("OVERNIGHT_ATR_MULTIPLIER") or 0.50)
MIN_STOP_PCT = float(os.environ.get("OVERNIGHT_MIN_STOP_PCT") or 0.75)

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
        return ((price_gbx - entry_gbx) / entry_gbx) * 100.0
    return ((entry_gbx - price_gbx) / entry_gbx) * 100.0


def pnl_from_prices(direction: str, entry_gbx: float, exit_gbx: float) -> float:
    entry_gbp = gbx_to_gbp(entry_gbx)
    exit_gbp = gbx_to_gbp(exit_gbx)
    if entry_gbp <= 0:
        return 0.0
    units = NOTIONAL_GBP / entry_gbp
    if direction == "Long":
        return units * (exit_gbp - entry_gbp)
    return units * (entry_gbp - exit_gbp)


def atr14_as_of(frame: pd.DataFrame, as_of_date: dt.date) -> float | None:
    hist = frame[pd.DatetimeIndex(frame.index).date <= as_of_date].copy()
    if len(hist) < 15:
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
    value = float(tr.tail(14).mean())
    return value if math.isfinite(value) and value > 0 else None


def simulate_policy(
    day: pd.DataFrame,
    direction: str,
    entry: float,
    stop: float,
    target: float,
) -> dict[str, Any]:
    """
    Trade is open from the previous close. If the market gaps through the stop,
    use the opening price rather than pretending the stop filled at its level.
    """
    if day.empty:
        return {"exit": None, "outcome": "NO DATA"}

    opening = float(day.iloc[0]["Open"])
    if direction == "Long" and opening <= stop:
        return {"exit": opening, "outcome": "GAP STOP"}
    if direction == "Short" and opening >= stop:
        return {"exit": opening, "outcome": "GAP STOP"}

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
    daily_frame: pd.DataFrame,
) -> dict[str, Any]:
    direction = str(pick["direction"]).title()
    entry = float(pick["entry"])

    atr = atr14_as_of(daily_frame, signal_date)
    risk = max((atr or 0.0) * ATR_STOP_MULTIPLIER, entry * MIN_STOP_PCT / 100.0)

    if direction == "Long":
        stop = entry - risk
        target_1r = entry + risk
        target_2r = entry + 2.0 * risk
    else:
        stop = entry + risk
        target_1r = entry - risk
        target_2r = entry - 2.0 * risk

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
        "atr14": atr,
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

    gross1 = pnl_from_prices(direction, entry, float(policy1["exit"])) if policy1["exit"] is not None else 0.0
    gross2 = pnl_from_prices(direction, entry, float(policy2["exit"])) if policy2["exit"] is not None else 0.0

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
    cost = NOTIONAL_GBP * ROUND_TRIP_COST_BPS / 10_000.0

    # Convert the fixed-time directional returns into actual £ P/L on a £30 CFD.
    for r in rows:
        for suffix, pct_key in (
            ("open", "open_return_pct"),
            ("0805", "r0805_pct"),
            ("0815", "r0815_pct"),
            ("0930", "r0930_pct"),
        ):
            pct = r.get(pct_key)
            r[f"{suffix}_net"] = (
                NOTIONAL_GBP * float(pct) / 100.0 - cost
                if pct is not None else 0.0
            )

    strategies = [
        ("Open", "open_net"),
        ("08:05", "0805_net"),
        ("08:15", "0815_net"),
        ("09:30", "0930_net"),
        ("1R", "one_r_net"),
        ("2R", "two_r_net"),
    ]

    def money_stats(group_rows: list[dict[str, Any]], key: str) -> dict[str, float]:
        vals = [float(r.get(key, 0.0) or 0.0) for r in group_rows]
        total = sum(vals)
        wins = [v for v in vals if v > 0]
        losses = [v for v in vals if v < 0]
        gp = sum(wins)
        gl = abs(sum(losses))
        pf = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)

        equity = 0.0
        peak = 0.0
        dd = 0.0
        for v in vals:
            equity += v
            peak = max(peak, equity)
            dd = min(dd, equity - peak)

        return {
            "trades": len(vals),
            "pnl": total,
            "avg": total / len(vals) if vals else 0.0,
            "win_rate": len(wins) / len(vals) * 100 if vals else 0.0,
            "pf": pf,
            "dd": dd,
        }

    longs = [r for r in rows if r["direction"] == "Long"]
    shorts = [r for r in rows if r["direction"] == "Short"]
    groups = [("Combined", rows), ("LONG", longs), ("SHORT", shorts)]

    all_stats = {
        group_name: {
            label: money_stats(group_rows, key)
            for label, key in strategies
        }
        for group_name, group_rows in groups
    }

    def cls(value: float) -> str:
        return "positive" if value >= 0 else "negative"

    def pf_text(value: float) -> str:
        return "∞" if math.isinf(value) else f"{value:.2f}"

    def group_section(group_name: str, group_rows: list[dict[str, Any]]) -> str:
        deployed = len(group_rows) * NOTIONAL_GBP
        if group_rows:
            best_label = max(
                strategies,
                key=lambda item: all_stats[group_name][item[0]]["pnl"],
            )[0]
            best_pnl = all_stats[group_name][best_label]["pnl"]
        else:
            best_label = "—"
            best_pnl = 0.0

        body = []
        for label, _ in strategies:
            s = all_stats[group_name][label]
            return_on_stake = (s["pnl"] / deployed * 100.0) if deployed else 0.0
            body.append(
                "<tr>"
                f"<td><strong>{label}</strong></td>"
                f"<td>{s['trades']}</td>"
                f"<td>£{deployed:,.2f}</td>"
                f"<td class='{cls(s['pnl'])}'><strong>£{s['pnl']:+.2f}</strong></td>"
                f"<td class='{cls(return_on_stake)}'>{return_on_stake:+.2f}%</td>"
                f"<td class='{cls(s['avg'])}'>£{s['avg']:+.2f}</td>"
                f"<td>{s['win_rate']:.1f}%</td>"
                f"<td>{pf_text(s['pf'])}</td>"
                f"<td class='negative'>£{s['dd']:.2f}</td>"
                "</tr>"
            )

        return f"""
        <section class="group-block">
          <div class="group-head">
            <div>
              <div class="eyebrow">{group_name}</div>
              <h2>{group_name} monetary performance</h2>
            </div>
            <div class="best">
              <span>Best exit</span>
              <strong>{best_label}</strong>
              <b class="{cls(best_pnl)}">£{best_pnl:+.2f}</b>
            </div>
          </div>
          <div class="money-strip">
            <div><span>Trades</span><strong>{len(group_rows)}</strong></div>
            <div><span>Stake per trade</span><strong>£{NOTIONAL_GBP:.2f}</strong></div>
            <div><span>Total stake deployed</span><strong>£{deployed:,.2f}</strong></div>
            <div><span>Best net gain/loss</span><strong class="{cls(best_pnl)}">£{best_pnl:+.2f}</strong></div>
          </div>
          <div class="table-wrap">
            <table class="summary-table">
              <thead><tr>
                <th>Exit</th><th>Trades</th><th>Stake deployed</th>
                <th>Net gain/loss</th><th>Return on stake</th>
                <th>Avg £/trade</th><th>Win rate</th>
                <th>Profit factor</th><th>Max drawdown</th>
              </tr></thead>
              <tbody>{''.join(body)}</tbody>
            </table>
          </div>
        </section>
        """

    combined_best = max(
        strategies,
        key=lambda item: all_stats["Combined"][item[0]]["pnl"]
    )[0]
    combined_best_pnl = all_stats["Combined"][combined_best]["pnl"]

    open_values = [r["open_return_pct"] for r in rows if r["open_return_pct"] is not None]
    avg_open = float(np.mean(open_values)) if open_values else 0.0
    median_open = float(np.median(open_values)) if open_values else 0.0

    html_rows = []
    for r in reversed(rows):
        html_rows.append(
            "<tr>"
            f"<td>{r['signal_date'].strftime('%d %b')}</td>"
            f"<td>{r['trade_date'].strftime('%d %b')}</td>"
            f"<td><strong>{html_lib.escape(r['epic'])}</strong><br><span class='muted'>{html_lib.escape(r['name'])}</span></td>"
            f"<td>{html_lib.escape(r['direction'])}</td>"
            f"<td>{fmt_price(r['entry'])}</td>"
            f"<td>{fmt_pct(r['open_return_pct'])}</td><td class='{cls(r['open_net'])}'>£{r['open_net']:+.2f}</td>"
            f"<td>{fmt_pct(r['r0805_pct'])}</td><td class='{cls(r['0805_net'])}'>£{r['0805_net']:+.2f}</td>"
            f"<td>{fmt_pct(r['r0815_pct'])}</td><td class='{cls(r['0815_net'])}'>£{r['0815_net']:+.2f}</td>"
            f"<td>{fmt_pct(r['r0930_pct'])}</td><td class='{cls(r['0930_net'])}'>£{r['0930_net']:+.2f}</td>"
            f"<td>{fmt_price(r['stop'])}</td>"
            f"<td>{fmt_price(r['target_1r'])}</td>"
            f"<td>{html_lib.escape(r['one_r_outcome'])}</td>"
            f"<td class='{cls(r['one_r_net'])}'>£{r['one_r_net']:+.2f}</td>"
            f"<td>{fmt_price(r['target_2r'])}</td>"
            f"<td>{html_lib.escape(r['two_r_outcome'])}</td>"
            f"<td class='{cls(r['two_r_net'])}'>£{r['two_r_net']:+.2f}</td>"
            "</tr>"
        )

    sections = "".join(group_section(name, group_rows) for name, group_rows in groups)

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FTSE 350 · Overnight LONG vs SHORT research</title>
<style>
:root{{--ink:{INK};--panel:{PANEL};--line:{HAIRLINE};--brass:{BRASS};--green:{BULL};--red:{BEAR};--paper:{PAPER};--muted:{MUTED}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--ink);color:var(--paper);font-family:Arial,sans-serif}}
.wrap{{max-width:1420px;margin:auto;padding:28px 18px 60px}}
nav{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px}}
nav a{{color:var(--paper);text-decoration:none;border:1px solid var(--line);padding:8px 12px;border-radius:999px;font-size:13px}}
nav a.active{{color:var(--brass);border-color:var(--brass)}}
h1{{font-size:29px;margin:5px 0}}h2{{font-size:20px;margin:0}}
.sub,.note,.muted{{color:var(--muted)}}.sub{{font-size:12px}}
.callout{{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:15px;margin:18px 0;font-size:13px;line-height:1.6}}
.hero{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;margin:16px 0 28px}}
.hero>div,.money-strip>div,.best{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}}
.hero span,.money-strip span,.best span{{display:block;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.04em}}
.hero strong,.money-strip strong{{display:block;font-size:19px;margin-top:5px}}
.group-block{{margin:30px 0 38px}}
.group-head{{display:flex;justify-content:space-between;gap:16px;align-items:flex-end;margin-bottom:10px}}
.eyebrow{{color:var(--brass);font-size:10px;text-transform:uppercase;letter-spacing:.12em;margin-bottom:5px}}
.best{{min-width:180px;text-align:right}}.best strong{{display:block;font-size:18px;margin:4px 0}}.best b{{font-size:17px}}
.money-strip{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:10px 0}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:8px;margin-top:12px}}
table{{width:100%;border-collapse:collapse;background:var(--panel)}}.summary-table{{min-width:1000px}}.daily-table{{min-width:1780px}}
th,td{{padding:9px;border-bottom:1px solid var(--line);font-size:11px;text-align:right;white-space:nowrap}}
th{{font-size:9px;color:var(--muted);text-transform:uppercase}}
.summary-table th:first-child,.summary-table td:first-child{{text-align:left}}
.daily-table th:nth-child(-n+4),.daily-table td:nth-child(-n+4),
.daily-table th:nth-child(16),.daily-table td:nth-child(16),
.daily-table th:nth-child(19),.daily-table td:nth-child(19){{text-align:left}}
.positive{{color:var(--green)}}.negative{{color:var(--red)}}
@media(max-width:900px){{
.hero{{grid-template-columns:repeat(2,1fr)}}.money-strip{{grid-template-columns:repeat(2,1fr)}}
.group-head{{align-items:flex-start;flex-direction:column}}.best{{width:100%;text-align:left}}
}}
</style></head><body><main class="wrap">
<nav><a href="index.html">Daily screener</a><a href="intraday.html">Intraday</a><a href="backtest.html">Backtest</a><a class="active" href="backtest-research.html">Research</a></nav>
<div class="sub">FTSE 350 · rank-1 overnight CFD replay</div>
<h1>£30 overnight strategy · LONG versus SHORT</h1>
<div class="sub">Generated {generated}</div>
<div class="callout"><strong>Money-first comparison:</strong> every signal represents one £30 CFD position. The report separates LONG and SHORT recommendations and shows the actual simulated <strong>£ gain or loss</strong> for selling at the open, 08:05, 08:15, 09:30, 1R or 2R. “Total stake deployed” is £30 × number of trades; it is not a portfolio balance.</div>
<div class="hero">
<div><span>Total trades</span><strong>{len(rows)}</strong></div>
<div><span>LONG trades</span><strong>{len(longs)}</strong></div>
<div><span>SHORT trades</span><strong>{len(shorts)}</strong></div>
<div><span>Average directional open</span><strong class="{cls(avg_open)}">{avg_open:+.2f}%</strong></div>
<div><span>Median directional open</span><strong class="{cls(median_open)}">{median_open:+.2f}%</strong></div>
<div><span>Best combined exit</span><strong>{combined_best}<br><span class="{cls(combined_best_pnl)}">£{combined_best_pnl:+.2f}</span></strong></div>
</div>
{sections}
<h2 style="margin-top:38px;">Individual £30 trades</h2>
<div class="table-wrap"><table class="daily-table"><thead><tr>
<th>Signal</th><th>Next day</th><th>Rank-1 stock</th><th>Bias</th><th>Entry</th>
<th>Open %</th><th>Open £</th><th>08:05 %</th><th>08:05 £</th>
<th>08:15 %</th><th>08:15 £</th><th>09:30 %</th><th>09:30 £</th>
<th>Stop</th><th>1R</th><th>1R outcome</th><th>1R £</th>
<th>2R</th><th>2R outcome</th><th>2R £</th>
</tr></thead><tbody>{''.join(html_rows)}</tbody></table></div>
<p class="note"><strong>Entry limitation:</strong> the completed daily candle recreates the ranking and its close is the entry proxy. A real 16:20–16:25 entry occurs before the official close, so this is not a perfect point-in-time reconstruction.</p>
<p class="note">Each strategy deducts {ROUND_TRIP_COST_BPS:.0f} bps from each £30 trade as the generic execution-cost proxy. Broker-specific overnight financing is not separately modelled.</p>
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
    daily_cache: dict[str, pd.DataFrame] = {}
    for ticker in unique_tickers:
        daily_cache[ticker] = ticker_frame(daily_data, ticker)
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
        rows.append(simulate_day(pick, signal_date, trade_date, intraday_cache[ticker], daily_cache[ticker]))

    if not rows:
        raise RuntimeError("No historical rank-1 trades could be reconstructed.")

    generated = now.strftime("%a %d %b %Y · %H:%M %Z")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(render(rows, generated), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
