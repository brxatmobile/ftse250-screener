# FILE_VERSION: FTSE350_0700_0805_WITH_INTRADAY_CANDLES_2026_08_08
from __future__ import annotations

import argparse
import datetime as dt
import html as html_lib
import json
import math
import os
import re
import sys
from pathlib import Path
from string import Template
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

LONDON = ZoneInfo("Europe/London")
UTC = dt.timezone.utc


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return float(default)
    return float(str(raw).strip())


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return int(default)
    return int(str(raw).strip())


CAPITAL = env_float("CAPITAL", 5000)
RISK_PCT = env_float("RISK_PCT", 1)
POOL_N = env_int("CANDIDATE_POOL_N", 80)
MAX_RECOMMENDATIONS = env_int("MAX_ACTIONABLE_TRADES", 5)
TARGET_R = env_float("TARGET_R", 2)
MAX_GAP_PCT = env_float("MAX_GAP_PCT", 3.0)
MIN_RANGE_PCT = env_float("MIN_OPENING_RANGE_PCT", 0.25)
MAX_RANGE_PCT = env_float("MAX_OPENING_RANGE_PCT", 4.5)
MIN_VOLUME_RATIO = env_float("MIN_OPENING_VOLUME_RATIO", 0.50)
MIN_OPENING_TURNOVER_GBP = env_float("MIN_OPENING_TURNOVER_GBP", 75000)
MAX_VWAP_DISTANCE_PCT = env_float("MAX_VWAP_DISTANCE_PCT", 1.50)
ENTRY_BUFFER_PCT = env_float("ENTRY_BUFFER_PCT", 0.04)
MAX_ENTRY_EXTENSION_R = env_float("MAX_ENTRY_EXTENSION_R", 0.35)
MIN_RECOMMENDATION_SCORE = env_float("MIN_ACTIONABLE_SCORE", 68)

ROOT = Path(__file__).resolve().parent
DAILY_INDEX_PATH = ROOT / "docs" / "index.html"
OUTPUT_PATH = ROOT / "docs" / "intraday.html"

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


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def now_london() -> dt.datetime:
    return dt.datetime.now(UTC).astimezone(LONDON)


def scan_stage(now: dt.datetime) -> tuple[str, dt.time, int]:
    """
    Opening breakout scan using the first completed London-market five-minute bar.

    The workflow is scheduled shortly after 08:05 so the 08:00-08:05 bar has
    time to appear in the market-data feed.
    """
    return "08:05 OPENING BREAKOUT SCAN", dt.time(8, 5), 1


def normalise(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(column not in frame.columns for column in required):
        return pd.DataFrame()
    frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
    idx = pd.DatetimeIndex(frame.index)
    if idx.tz is None:
        idx = idx.tz_localize(UTC)
    frame.index = idx.tz_convert(LONDON)
    return frame.sort_index()


def load_pool() -> list[dict[str, Any]]:
    if not DAILY_INDEX_PATH.exists():
        raise RuntimeError("docs/index.html is missing; run the daily screener first.")

    page = DAILY_INDEX_PATH.read_text(encoding="utf-8")
    match = re.search(
        r'<script[^>]*id=["\']daily-shortlist-data["\'][^>]*>(.*?)</script>',
        page,
        flags=re.I | re.S,
    )
    if not match:
        raise RuntimeError("The daily page contains no embedded intraday pool.")

    payload = json.loads(match.group(1).replace("<\\/", "</"))
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("The embedded intraday candidate pool is empty.")

    generated = dt.datetime.fromisoformat(
        str(payload["generated_at_utc"]).replace("Z", "+00:00")
    )
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=UTC)
    if generated.astimezone(LONDON).date() != now_london().date():
        raise RuntimeError("The candidate pool is not from today.")

    return candidates[:POOL_N]


def history(ticker: str) -> pd.DataFrame:
    return normalise(
        yf.download(
            ticker,
            period="5d",
            interval="5m",
            progress=False,
            auto_adjust=False,
            prepost=False,
        )
    )


def prior_close(frame: pd.DataFrame, today: dt.date) -> float | None:
    prior = frame[frame.index.date < today]
    return float(prior.iloc[-1]["Close"]) if not prior.empty else None


def prior_window_volume(
    frame: pd.DataFrame,
    today: dt.date,
    end_time: dt.time,
) -> float | None:
    totals: list[float] = []
    for date_value in sorted(set(frame.index.date)):
        if date_value >= today:
            continue
        day = frame[frame.index.date == date_value]
        window = day[
            (day.index.time >= dt.time(8, 0))
            & (day.index.time < end_time)
        ]
        total = float(window["Volume"].fillna(0).sum())
        if total > 0:
            totals.append(total)
    return float(np.mean(totals)) if totals else None


def assess(
    candidate: dict[str, Any],
    now: dt.datetime,
    stage_name: str,
    window_end: dt.time,
    expected_bars: int,
) -> dict[str, Any]:
    direction = str(candidate.get("direction", "")).title()
    ticker = str(candidate.get("yahoo_ticker") or f"{candidate['epic']}.L")
    base = {
        "epic": candidate.get("epic", ticker.replace(".L", "")),
        "name": candidate.get("name", ""),
        "ticker": ticker,
        "direction": direction,
        "daily_score": float(candidate.get("score", 0)),
        "daily_pattern": candidate.get("pattern", ""),
        "status": "REJECTED",
        "score": 0.0,
        "recommendation": "No trade.",
        "passed": [],
        "failed": [],
        "stage": stage_name,
    }

    if direction not in {"Long", "Short"}:
        base["failed"] = ["Daily candidate has no valid long or short direction."]
        return base

    frame = history(ticker)
    if frame.empty:
        base["status"] = "DATA ONLY"
        base["failed"] = ["No intraday market data was available."]
        return base

    today = now.date()
    today_frame = frame[frame.index.date == today]
    opening = today_frame[
        (today_frame.index.time >= dt.time(8, 0))
        & (today_frame.index.time < window_end)
    ]
    bars = len(opening)
    base["bars"] = bars

    pc = prior_close(frame, today)
    if opening.empty or pc is None or pc <= 0:
        base["status"] = "DATA ONLY"
        base["failed"] = ["Previous close or opening data was unavailable."]
        return base

    first_open = float(opening.iloc[0]["Open"])
    high = float(opening["High"].max())
    low = float(opening["Low"].min())
    close = float(opening.iloc[-1]["Close"])
    volume = float(opening["Volume"].fillna(0).sum())

    typical = (opening["High"] + opening["Low"] + opening["Close"]) / 3
    volume_series = opening["Volume"].fillna(0)
    vwap = (
        float((typical * volume_series).sum() / volume_series.sum())
        if volume_series.sum() > 0
        else close
    )

    gap_pct = ((first_open - pc) / pc) * 100
    range_pct = ((high - low) / pc) * 100
    move_pct = ((close - pc) / pc) * 100
    location_pct = ((close - low) / max(high - low, 1e-9)) * 100
    vwap_distance_pct = abs((close - vwap) / vwap) * 100 if vwap else 0.0

    average_volume = prior_window_volume(frame, today, window_end)
    volume_ratio = (
        volume / average_volume
        if average_volume is not None and average_volume > 0
        else None
    )
    turnover_gbp = volume * gbx_to_gbp(close)

    completed = today_frame[
        (today_frame.index.time >= dt.time(8, 0))
        & (today_frame.index.time < window_end)
    ]
    current = float(completed.iloc[-1]["Close"])
    current_time = completed.index[-1]

    if direction == "Long":
        trigger = high * (1 + ENTRY_BUFFER_PCT / 100)
        stop = low
        initial_risk = trigger - stop
        maximum_entry = trigger + MAX_ENTRY_EXTENSION_R * initial_risk
        risk = current - stop
        target = current + TARGET_R * risk
        correct_side_previous_close = close > pc
        correct_side_vwap = close > vwap
        favourable_location = location_pct >= 58
        triggered = current >= trigger
        beyond_entry = current > maximum_entry
    else:
        trigger = low * (1 - ENTRY_BUFFER_PCT / 100)
        stop = high
        initial_risk = stop - trigger
        maximum_entry = trigger - MAX_ENTRY_EXTENSION_R * initial_risk
        risk = stop - current
        target = current - TARGET_R * risk
        correct_side_previous_close = close < pc
        correct_side_vwap = close < vwap
        favourable_location = location_pct <= 42
        triggered = current <= trigger
        beyond_entry = current < maximum_entry

    passed: list[str] = []
    failed: list[str] = []
    score = 0.0

    def check(condition: bool, points: float, yes: str, no: str) -> None:
        nonlocal score
        if condition:
            score += points
            passed.append(yes)
        else:
            failed.append(no)

    check(
        bars >= expected_bars,
        10,
        f"{bars} completed five-minute bars were available.",
        f"Only {bars} of {expected_bars} required bars were available.",
    )
    check(
        correct_side_previous_close,
        14,
        f"Price moved in the intended {direction.lower()} direction from yesterday's close.",
        "Price did not move in the intended direction from yesterday's close.",
    )
    check(
        correct_side_vwap,
        16,
        f"Price was on the correct side of VWAP for a {direction.lower()}.",
        f"Price was on the wrong side of VWAP for a {direction.lower()}.",
    )
    check(
        abs(gap_pct) <= MAX_GAP_PCT,
        8,
        "The opening gap was controlled.",
        f"The opening gap of {gap_pct:+.2f}% was excessive.",
    )
    check(
        MIN_RANGE_PCT <= range_pct <= MAX_RANGE_PCT,
        10,
        "The early opening range was usable.",
        f"The opening range of {range_pct:.2f}% was unsuitable.",
    )
    check(
        favourable_location,
        12,
        "Price held near the favourable end of the opening range.",
        "Price did not hold near the favourable end of the opening range.",
    )
    check(
        volume_ratio is not None and volume_ratio >= MIN_VOLUME_RATIO,
        10,
        f"Volume was {volume_ratio:.2f}× the comparable opening window."
        if volume_ratio is not None
        else "",
        "Comparable opening volume was unavailable or too weak.",
    )
    check(
        turnover_gbp >= MIN_OPENING_TURNOVER_GBP,
        10,
        f"Opening turnover was about £{turnover_gbp:,.0f}.",
        f"Opening turnover of about £{turnover_gbp:,.0f} was too low.",
    )
    check(
        vwap_distance_pct <= MAX_VWAP_DISTANCE_PCT,
        5,
        "Price was not excessively stretched from VWAP.",
        f"Price was {vwap_distance_pct:.2f}% from VWAP and overextended.",
    )
    check(
        initial_risk > 0 and risk > 0,
        5,
        "The opening-range stop structure was valid.",
        "The stop structure was invalid.",
    )

    mandatory_failure = (
        bars < expected_bars
        or not correct_side_previous_close
        or not correct_side_vwap
        or abs(gap_pct) > MAX_GAP_PCT
        or range_pct < MIN_RANGE_PCT
        or range_pct > MAX_RANGE_PCT
        or turnover_gbp < MIN_OPENING_TURNOVER_GBP
        or initial_risk <= 0
        or risk <= 0
    )

    if bars < expected_bars:
        status = "DATA ONLY"
        recommendation = (
            f"No recommendation — only {bars} of {expected_bars} required bars "
            f"were available for the {stage_name.lower()}."
        )
    elif mandatory_failure:
        status = "REJECTED"
        recommendation = "No trade. A mandatory tradability check failed."
    elif beyond_entry:
        status = "DO NOT CHASE"
        recommendation = (
            f"The {direction.lower()} move has already travelled beyond the "
            f"maximum acceptable entry of £{gbx_to_gbp(maximum_entry):.2f}."
        )
    elif triggered and score >= MIN_RECOMMENDATION_SCORE:
        status = "ACTIONABLE"
        recommendation = (
            f"{direction} setup triggered. Current completed-bar price is about "
            f"£{gbx_to_gbp(current):.2f}; verify the live quote and spread."
        )
    elif score >= MIN_RECOMMENDATION_SCORE:
        status = "WAIT FOR BREAK"
        relation = "above" if direction == "Long" else "below"
        recommendation = (
            f"Valid conditional {direction.lower()} setup. Enter only on a clean "
            f"move {relation} £{gbx_to_gbp(trigger):.2f}; do not chase beyond "
            f"£{gbx_to_gbp(maximum_entry):.2f}."
        )
    else:
        status = "REJECTED"
        recommendation = "No trade. The setup was not strong enough overall."

    risk_budget = CAPITAL * RISK_PCT / 100
    risk_gbp = gbx_to_gbp(risk) if risk > 0 else 0
    current_gbp = gbx_to_gbp(current)
    shares = max(
        0,
        min(
            math.floor(risk_budget / risk_gbp) if risk_gbp > 0 else 0,
            math.floor(CAPITAL / current_gbp) if current_gbp > 0 else 0,
        ),
    )

    opening_candles = [
        {
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "time": idx.strftime("%H:%M"),
        }
        for idx, row in opening.iterrows()
    ]

    base.update(
        {
            "opening_candles": opening_candles,
            "status": status,
            "recommendation": recommendation,
            "score": min(score, 100),
            "passed": passed,
            "failed": failed,
            "previous_close": pc,
            "gap_pct": gap_pct,
            "move_pct": move_pct,
            "range_pct": range_pct,
            "close": close,
            "vwap": vwap,
            "location_pct": location_pct,
            "volume_ratio": volume_ratio,
            "turnover_gbp": turnover_gbp,
            "current": current,
            "current_time": current_time,
            "trigger": trigger,
            "maximum_entry": maximum_entry,
            "stop": stop,
            "target": target,
            "shares": shares,
            "position_value": shares * current_gbp,
            "planned_risk": shares * risk_gbp,
        }
    )
    return base


def money(value: Any) -> str:
    return f"£{gbx_to_gbp(float(value)):.2f}" if finite(value) else "—"


def candlestick_svg(candles: list[dict[str, Any]]) -> str:
    """Render a compact SVG candlestick chart from completed opening bars."""
    if not candles:
        return "<div class='mini-chart empty-chart'>No completed opening candles</div>"

    width = 360
    height = 120
    pad_x = 18
    pad_y = 12
    prices = [
        float(candle[key])
        for candle in candles
        for key in ("open", "high", "low", "close")
        if finite(candle.get(key))
    ]
    if not prices:
        return "<div class='mini-chart empty-chart'>No usable candle data</div>"

    low_price = min(prices)
    high_price = max(prices)
    span = max(high_price - low_price, 1e-9)

    def y(price: float) -> float:
        usable = height - 2 * pad_y
        return pad_y + ((high_price - price) / span) * usable

    count = len(candles)
    slot = (width - 2 * pad_x) / max(count, 1)
    body_width = max(5.0, min(18.0, slot * 0.52))
    parts = [
        f'<svg class="mini-candles" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="Opening candlestick chart">'
    ]

    for index, candle in enumerate(candles):
        o = float(candle["open"])
        h = float(candle["high"])
        l = float(candle["low"])
        c = float(candle["close"])
        x = pad_x + slot * (index + 0.5)
        bullish = c >= o
        colour = BULL if bullish else BEAR
        body_top = min(y(o), y(c))
        body_bottom = max(y(o), y(c))
        body_height = max(2.0, body_bottom - body_top)

        parts.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{y(h):.1f}" y2="{y(l):.1f}" '
            f'stroke="{colour}" stroke-width="1.4"/>'
        )
        parts.append(
            f'<rect x="{x-body_width/2:.1f}" y="{body_top:.1f}" '
            f'width="{body_width:.1f}" height="{body_height:.1f}" '
            f'fill="{colour}" rx="1"/>'
        )
        if count <= 8:
            parts.append(
                f'<text x="{x:.1f}" y="{height-2:.1f}" text-anchor="middle" '
                f'font-size="8" fill="{MUTED}">{html_lib.escape(str(candle["time"]))}</text>'
            )

    parts.append("</svg>")
    return "".join(parts)


def card(item: dict[str, Any], nap: bool = False) -> str:
    label = ("NAP — " if nap else "") + str(item["status"])
    direction_colour = BULL if item["direction"] == "Long" else BEAR
    status_colour = (
        direction_colour
        if item["status"] == "ACTIONABLE"
        else BRASS
    )
    passed = "".join(
        f"<li>{html_lib.escape(value)}</li>" for value in item.get("passed", [])
    ) or "<li>None</li>"
    failed = "".join(
        f"<li>{html_lib.escape(value)}</li>" for value in item.get("failed", [])
    ) or "<li>None</li>"
    volume_ratio = item.get("volume_ratio")
    volume_text = f"{volume_ratio:.2f}×" if finite(volume_ratio) else "—"

    return f"""
<section class="pick">
<div class="pick-head"><div>
<strong class="epic">{html_lib.escape(str(item['epic']))}</strong>
<span class="name">{html_lib.escape(str(item['name']))}</span>
<div class="pattern" style="color:{direction_colour}">{item['direction']} · {html_lib.escape(str(item.get('daily_pattern','')))}</div>
</div><div class="score"><span style="color:{status_colour}">{html_lib.escape(label)}</span><strong>{item['score']:.0f}/100</strong></div></div>
<p class="recommendation" data-original="{html_lib.escape(item['recommendation'], quote=True)}">{html_lib.escape(item['recommendation'])}</p>
<div class="chart-wrap"><div class="chart-title">Opening price action</div>{candlestick_svg(item.get("opening_candles", []))}</div>
<div class="metrics">
<div><span>Latest completed price</span><strong>{money(item.get('current'))} at {item.get('current_time').strftime('%H:%M') if item.get('current_time') else '—'}</strong></div>
<div><span>Entry trigger</span><strong>{money(item.get('trigger'))}</strong></div>
<div><span>Maximum entry</span><strong>{money(item.get('maximum_entry'))}</strong></div>
<div><span>Stop / 2R target</span><strong>{money(item.get('stop'))} / {money(item.get('target'))}</strong></div>
<div><span>Opening gap</span><strong>{item.get('gap_pct',0):+.2f}%</strong></div>
<div><span>Opening move / range</span><strong>{item.get('move_pct',0):+.2f}% / {item.get('range_pct',0):.2f}%</strong></div>
<div><span>Close / VWAP</span><strong>{money(item.get('close'))} / {money(item.get('vwap'))}</strong></div>
<div><span>Relative volume / turnover</span><strong>{volume_text} / £{item.get('turnover_gbp',0):,.0f}</strong></div>
<div><span>Indicative size</span><strong>{item.get('shares',0)} shares</strong></div>
<div><span>Position value / risk</span><strong>£{item.get('position_value',0):,.2f} / £{item.get('planned_risk',0):,.2f}</strong></div>
</div>
<details><summary>Assessment details</summary><div class="detail-grid">
<div><h3>Passed</h3><ul>{passed}</ul></div>
<div><h3>Failed or cautions</h3><ul>{failed}</ul></div>
</div></details></section>"""


def rejected_row(item: dict[str, Any]) -> str:
    reason = "; ".join(item.get("failed", [])[:2]) or item.get(
        "recommendation",
        "No trade.",
    )
    return (
        f"<li><strong>{html_lib.escape(str(item['epic']))}</strong> "
        f"({html_lib.escape(str(item['direction']))}) — "
        f"{html_lib.escape(str(item['status']))}: "
        f"{html_lib.escape(reason)}</li>"
    )


def build_html(
    results: list[dict[str, Any]],
    generated: dt.datetime,
    stage_name: str,
) -> str:
    recommended = sorted(
        [
            row
            for row in results
            if row["status"] in {"ACTIONABLE", "WAIT FOR BREAK"}
            and row["score"] >= MIN_RECOMMENDATION_SCORE
        ],
        key=lambda row: (
            1 if row["status"] == "ACTIONABLE" else 0,
            row["score"],
        ),
        reverse=True,
    )[:MAX_RECOMMENDATIONS]

    rejected = [row for row in results if row not in recommended]
    rejected.sort(key=lambda row: row["score"], reverse=True)

    actionable = [row for row in recommended if row["status"] == "ACTIONABLE"]
    waiting = [row for row in recommended if row["status"] == "WAIT FOR BREAK"]
    nap_epic = recommended[0]["epic"] if recommended else None

    if actionable and waiting:
        decision = (
            f"{len(actionable)} actionable and {len(waiting)} conditional setup"
            f"{'s' if len(waiting) != 1 else ''} from the {stage_name.lower()}."
        )
    elif actionable:
        decision = (
            f"{len(actionable)} actionable trade"
            f"{'s' if len(actionable) != 1 else ''} from the {stage_name.lower()}."
        )
    elif waiting:
        decision = (
            f"{len(waiting)} valid conditional setup"
            f"{'s' if len(waiting) != 1 else ''}; wait for the displayed trigger."
        )
    else:
        decision = (
            f"NO TRADE — the {stage_name.lower()} found no actionable or valid "
            "trigger-based long or short setup."
        )

    cards = "".join(
        card(row, nap=(row["epic"] == nap_epic))
        for row in recommended
    )
    rejected_html = "".join(rejected_row(row) for row in rejected) or "<li>None</li>"
    expiry = dt.datetime.combine(
        generated.date(),
        dt.time(9, 30),
        tzinfo=LONDON,
    ).isoformat()

    template = Template("""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>FTSE 350 08:05 breakout trades</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:$INK;color:$PAPER;font-family:Arial,sans-serif}
.wrap{max-width:920px;margin:auto;padding:22px 14px 56px}a{color:$BRASS}.header{display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;border-bottom:1px solid $HAIRLINE;padding-bottom:16px}
h1{margin:4px 0 0;font-size:27px}h3{font-size:13px;margin:0 0 5px}.kicker{color:$SALMON;font-size:12px;text-transform:uppercase;letter-spacing:.09em}
.time,.pattern,.name,.footer{color:$MUTED;font-size:12px}.decision,.pick,.rejected,.expired{background:$PANEL;border:1px solid $HAIRLINE;border-radius:9px;padding:15px;margin:14px 0}
.decision strong{color:$BRASS}.pick-head{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}.epic{color:$BRASS;font-size:19px}.name{margin-left:8px}
.score{display:flex;flex-direction:column;text-align:right;font-size:13px;font-weight:700}.chart-wrap{margin:12px 0;background:#12161F;border:1px solid #2C333D;border-radius:7px;padding:10px}.chart-title{font-size:11px;color:#8B92A0;margin-bottom:5px}.mini-candles{width:100%;height:120px;display:block}.empty-chart{color:#8B92A0;font-size:12px;padding:20px;text-align:center}.score strong{font-size:22px;margin-top:3px}.recommendation{font-size:14px;line-height:1.5}
.metrics{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:9px;margin-top:12px}.metrics div{background:$INK;border:1px solid $HAIRLINE;border-radius:6px;padding:9px}
.metrics span{display:block;color:$MUTED;font-size:11px;margin-bottom:4px}.metrics strong{font-size:12px;line-height:1.35}details{margin-top:12px}summary{cursor:pointer;color:$BRASS;font-size:13px}
.detail-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:10px}ul{padding-left:18px;color:$MUTED;font-size:12px;line-height:1.5}
.expired{display:none;border-color:$SALMON}.expired h2{color:$SALMON;margin-top:0}.footer{line-height:1.6;border-top:1px solid $HAIRLINE;padding-top:14px;margin-top:20px}
@media(max-width:760px){.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.detail-grid{grid-template-columns:1fr}h1{font-size:22px}}
</style></head><body><main class="wrap">
<div class="header"><div><div class="kicker">FTSE 350 ex trusts · 08:05 long and short breakout strategies</div><h1>$STAGE</h1></div><div class="time">$DATE<br>$TIME</div></div>
<div class="decision"><strong>$DECISION</strong> <a href="index.html">Daily watchlist</a> · <a href="backtest.html">Backtest</a></div>
<div id="expired-content" class="expired"><h2>Past the 09:30 cutoff</h2><p>The opening setups below are retained for review only. Do not use their levels now.</p></div>
<div id="live-content">$CARDS
<details class="rejected"><summary>Other candidates assessed but not recommended</summary><ul>$REJECTED</ul></details></div>
<p class="footer">The opening scan uses the first completed 08:00–08:05 five-minute bar. Longs require strength above the previous close and VWAP; shorts require weakness below both. ACTIONABLE means the opening-range trigger has broken without being chased. WAIT FOR BREAK is conditional and must not be entered early.</p>
</main><script>
(function(){const expiry=new Date('$EXPIRY');const expired=document.getElementById('expired-content');
function enforce(){if(new Date()>=expiry){expired.style.display='block';document.querySelectorAll('.recommendation').forEach(function(el){const original=el.dataset.original||el.textContent;el.dataset.original=original;el.textContent='PAST 09:30 — Do not follow this recommendation now. Original assessment: '+original;});}}
enforce();setInterval(enforce,30000);})();
</script></body></html>""")

    return template.substitute(
        INK=INK,
        PAPER=PAPER,
        BRASS=BRASS,
        HAIRLINE=HAIRLINE,
        SALMON=SALMON,
        MUTED=MUTED,
        PANEL=PANEL,
        STAGE=html_lib.escape(stage_name),
        DATE=generated.strftime("%a %d %b %Y"),
        TIME=generated.strftime("%H:%M %Z"),
        DECISION=html_lib.escape(decision),
        CARDS=cards
        or (
            "<section class='pick'><strong>NO TRADE</strong>"
            "<p>No candidate produced a valid early long or short setup.</p></section>"
        ),
        REJECTED=rejected_html,
        EXPIRY=expiry,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["auto", "analyse", "expire"],
        default="auto",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = now_london()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    mode = (
        "expire"
        if now.time() >= dt.time(9, 30)
        else "analyse"
    ) if args.mode == "auto" else args.mode

    if mode == "expire":
        print("Retaining the existing page; browser-side logic marks it expired.")
        return 0

    stage_name, window_end, expected_bars = scan_stage(now)
    candidates = load_pool()

    print(
        f"{stage_name}: assessing {len(candidates)} FTSE 350 long and short "
        f"candidates using bars through {window_end.strftime('%H:%M')}."
    )

    results = [
        assess(candidate, now, stage_name, window_end, expected_bars)
        for candidate in candidates
    ]

    OUTPUT_PATH.write_text(
        build_html(results, now, stage_name),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_PATH}")

    for row in sorted(results, key=lambda value: value["score"], reverse=True):
        print(
            f"{row['epic']:<7} {row['direction']:<5} "
            f"{row['status']:<15} {row['score']:>5.1f} "
            f"{row['recommendation']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
