# FILE_VERSION: FTSE350_VOLATILITY_AWARE_RISK_2026_08_10
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
NEWS_COUNT = env_int("NEWS_COUNT", 3)
ATR_STOP_MULTIPLIER = env_float("ATR_STOP_MULTIPLIER", 0.50)
MIN_STOP_PCT = env_float("MIN_STOP_PCT", 0.75)
OPENING_RANGE_MINUTES = env_int("OPENING_RANGE_MINUTES", 15)
MIN_READY_VOLUME_RATIO = env_float("MIN_READY_VOLUME_RATIO", 0.75)
MIN_A_GRADE_VOLUME_RATIO = env_float("MIN_A_GRADE_VOLUME_RATIO", 1.10)
MAX_CHASE_R = env_float("MAX_CHASE_R", 0.30)

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
    market_open = now.replace(hour=8, minute=0, second=0, microsecond=0)
    completed = now.replace(second=0, microsecond=0) - dt.timedelta(minutes=now.minute % 5)
    if completed <= market_open:
        completed = market_open + dt.timedelta(minutes=5)
    cap = now.replace(hour=9, minute=30, second=0, microsecond=0)
    completed = min(completed, cap)
    expected_bars = max(1, int((completed - market_open).total_seconds() // 300))
    if expected_bars < 3:
        stage = f"OPENING OBSERVATION — {expected_bars} completed 5-minute bar{'s' if expected_bars != 1 else ''}"
    else:
        stage = f"15-MIN OPENING-RANGE STRATEGY — {expected_bars} completed 5-minute bars"
    return stage, completed.time(), expected_bars


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


def recent_news(candidate: dict[str, Any]) -> list[dict[str, str]]:
    query = str(candidate.get("name") or candidate.get("epic") or "").strip()
    if not query:
        return []
    try:
        items = yf.Search(
            query,
            max_results=1,
            news_count=max(NEWS_COUNT * 2, 6),
            enable_fuzzy_query=False,
            timeout=10,
            raise_errors=False,
        ).news or []
    except Exception:
        return []

    output = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        content = raw.get("content") if isinstance(raw.get("content"), dict) else raw
        title = str(content.get("title") or raw.get("title") or "").strip()
        if not title:
            continue
        provider = content.get("provider") or raw.get("publisher") or ""
        if isinstance(provider, dict):
            publisher = str(provider.get("displayName") or provider.get("name") or "")
        else:
            publisher = str(provider)
        url_obj = content.get("clickThroughUrl") or content.get("canonicalUrl")
        if isinstance(url_obj, dict):
            url = str(url_obj.get("url") or "")
        else:
            url = str(content.get("link") or raw.get("link") or raw.get("url") or "")
        output.append({"title": title, "publisher": publisher, "url": url})
        if len(output) >= NEWS_COUNT:
            break
    return output


def news_html(item: dict[str, Any]) -> str:
    items = item.get("news") or []
    if not items:
        return (
            '<div class="news-box"><div class="news-heading">News context</div>'
            '<div class="news-empty">No recent company-specific headline found.</div></div>'
        )
    rows = []
    for news in items:
        title = html_lib.escape(str(news.get("title", "")))
        publisher = html_lib.escape(str(news.get("publisher", "")))
        url = str(news.get("url", "")).strip()
        if url.startswith(("http://", "https://")):
            title_html = (
                f'<a href="{html_lib.escape(url, quote=True)}" target="_blank" '
                f'rel="noopener noreferrer">{title}</a>'
            )
        else:
            title_html = title
        rows.append(f'<li>{title_html}<div class="news-meta">{publisher}</div></li>')
    return (
        '<div class="news-box"><div class="news-heading">Recent news</div>'
        f'<ul class="news-list">{"".join(rows)}</ul></div>'
    )




def finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

def daily_atr_from_history(ticker: str, period: int = 14) -> float | None:
    """Latest 14-day ATR in the same GBX units as London price data."""
    try:
        daily = normalise(
            yf.download(
                ticker,
                period="3mo",
                interval="1d",
                progress=False,
                auto_adjust=False,
                prepost=False,
            )
        )
    except Exception:
        return None

    if daily.empty or len(daily) < period + 1:
        return None

    prev_close = daily["Close"].shift(1)
    true_range = pd.concat(
        [
            daily["High"] - daily["Low"],
            (daily["High"] - prev_close).abs(),
            (daily["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.rolling(period).mean().iloc[-1]
    return float(atr) if finite_number(atr) and float(atr) > 0 else None


def classify_news(news_items: list[dict[str, str]]) -> tuple[str, int]:
    if not news_items:
        return "No fresh catalyst found", 0

    text = " ".join(item.get("title", "").lower() for item in news_items)
    positive_terms = (
        "upgrade", "raises guidance", "beats", "record", "contract win",
        "acquisition", "takeover", "bid", "profit rises", "strong trading",
        "dividend increase", "buyback",
    )
    negative_terms = (
        "downgrade", "profit warning", "cuts guidance", "misses", "investigation",
        "regulator", "loss widens", "dividend cut", "placing", "rights issue",
    )
    pos = sum(term in text for term in positive_terms)
    neg = sum(term in text for term in negative_terms)
    if pos > neg and pos > 0:
        return "Supportive catalyst", 1
    if neg > pos and neg > 0:
        return "Adverse catalyst", -1
    return "Recent news, no clear directional catalyst", 0


def morning_structure(opening: pd.DataFrame, direction: str) -> tuple[bool, str]:
    if len(opening) < 5:
        return False, "Not enough post-opening-range bars yet to confirm structure"
    post = opening.iloc[3:]
    if direction == "Long":
        lows = post["Low"].astype(float)
        ok = lows.iloc[-1] >= lows.iloc[0]
        return ok, "Higher-low structure confirmed" if ok else "Higher lows not confirmed"
    highs = post["High"].astype(float)
    ok = highs.iloc[-1] <= highs.iloc[0]
    return ok, "Lower-high structure confirmed" if ok else "Lower highs not confirmed"


def assess(
    candidate: dict[str, Any],
    now: dt.datetime,
    stage_name: str,
    window_end: dt.time,
    expected_bars: int,
) -> dict[str, Any]:
    direction = str(candidate.get("direction", "")).title()
    ticker = str(candidate.get("yahoo_ticker") or f"{candidate['epic']}.L")
    daily_atr = daily_atr_from_history(ticker)

    base = {
        "epic": candidate.get("epic", ticker.replace(".L", "")),
        "name": candidate.get("name", ""),
        "ticker": ticker,
        "direction": direction,
        "daily_score": float(candidate.get("score", 0) or 0),
        "daily_pattern": candidate.get("pattern", ""),
        "status": "NO TRADE",
        "grade": "NO TRADE",
        "recommendation": "No trade.",
        "why": [],
        "failed": [],
        "stage": stage_name,
    }

    if direction not in {"Long", "Short"}:
        base["failed"] = ["No valid long/short daily bias."]
        return base

    frame = history(ticker)
    if frame.empty:
        base["status"] = "DATA ONLY"
        base["grade"] = "DATA ONLY"
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
        base["grade"] = "DATA ONLY"
        base["failed"] = ["Previous close or opening data was unavailable."]
        return base

    opening_range = today_frame[
        (today_frame.index.time >= dt.time(8, 0))
        & (today_frame.index.time < dt.time(8, OPENING_RANGE_MINUTES))
    ]

    news_items = recent_news(candidate)

    if len(opening_range) < 3:
        current = float(opening.iloc[-1]["Close"])
        base.update({
            "status": "WATCH",
            "grade": "WATCH",
            "recommendation": (
                "Do not trade yet. Wait for three completed five-minute candles "
                "to establish the 08:00–08:15 opening range."
            ),
            "current": current,
            "current_time": opening.index[-1],
            "opening_candles": [
                {"open": float(r["Open"]), "high": float(r["High"]),
                 "low": float(r["Low"]), "close": float(r["Close"]),
                 "time": idx.strftime("%H:%M")}
                for idx, r in opening.iterrows()
            ],
            "news": news_items,
        })
        return base

    first_open = float(opening.iloc[0]["Open"])
    close = float(opening.iloc[-1]["Close"])
    current = close
    current_time = opening.index[-1]
    high = float(opening["High"].max())
    low = float(opening["Low"].min())
    volume = float(opening["Volume"].fillna(0).sum())

    typical = (opening["High"] + opening["Low"] + opening["Close"]) / 3
    volume_series = opening["Volume"].fillna(0)
    vwap = float((typical * volume_series).sum() / volume_series.sum()) if volume_series.sum() > 0 else close

    gap_pct = ((first_open - pc) / pc) * 100
    move_pct = ((close - pc) / pc) * 100
    range_pct = ((high - low) / pc) * 100
    vwap_distance_pct = abs((close - vwap) / vwap) * 100 if vwap else 0.0

    average_volume = prior_window_volume(frame, today, window_end)
    volume_ratio = volume / average_volume if average_volume and average_volume > 0 else None
    turnover_gbp = volume * gbx_to_gbp(close)

    or_high = float(opening_range["High"].max())
    or_low = float(opening_range["Low"].min())

    if direction == "Long":
        trigger = or_high * (1 + ENTRY_BUFFER_PCT / 100)
        side_prev = current > pc
        side_vwap = current > vwap
        broken = current >= trigger
        structure_stop = or_low
    else:
        trigger = or_low * (1 - ENTRY_BUFFER_PCT / 100)
        side_prev = current < pc
        side_vwap = current < vwap
        broken = current <= trigger
        structure_stop = or_high

    atr_distance = (daily_atr or 0.0) * ATR_STOP_MULTIPLIER
    min_distance = trigger * MIN_STOP_PCT / 100

    if direction == "Long":
        stop_distance = max(trigger - structure_stop, atr_distance, min_distance)
        stop = trigger - stop_distance
        target_1r = trigger + stop_distance
        target_2r = trigger + TARGET_R * stop_distance
        max_entry = trigger + MAX_CHASE_R * stop_distance
        beyond = current > max_entry
    else:
        stop_distance = max(structure_stop - trigger, atr_distance, min_distance)
        stop = trigger + stop_distance
        target_1r = trigger - stop_distance
        target_2r = trigger - TARGET_R * stop_distance
        max_entry = trigger - MAX_CHASE_R * stop_distance
        beyond = current < max_entry

    structure_ok, structure_text = morning_structure(opening, direction)
    news_label, catalyst_bias = classify_news(news_items)

    why, failed = [], []

    def flag(cond, yes, no):
        if cond:
            why.append(yes)
            return True
        failed.append(no)
        return False

    trend_ok = flag(side_prev, "Price is moving in the daily-bias direction versus yesterday's close.",
                    "Price is not moving in the daily-bias direction versus yesterday's close.")
    vwap_ok = flag(side_vwap, "Price is on the correct side of VWAP.",
                   "Price is on the wrong side of VWAP.")
    vol_ok = flag(volume_ratio is not None and volume_ratio >= MIN_READY_VOLUME_RATIO,
                  f"Relative opening volume is {volume_ratio:.2f}×." if volume_ratio is not None else "Relative volume confirmed.",
                  "Opening volume is too weak or unavailable.")
    liquidity_ok = flag(turnover_gbp >= MIN_OPENING_TURNOVER_GBP,
                        f"Opening turnover is about £{turnover_gbp:,.0f}.",
                        f"Opening turnover of about £{turnover_gbp:,.0f} is too low.")
    structure_pass = flag(structure_ok, structure_text + ".", structure_text + ".")
    gap_ok = flag(abs(gap_pct) <= MAX_GAP_PCT, "Opening gap is controlled.",
                  f"Opening gap of {gap_pct:+.2f}% is too large.")
    stretch_ok = flag(vwap_distance_pct <= MAX_VWAP_DISTANCE_PCT,
                      "Price is not excessively stretched from VWAP.",
                      f"Price is {vwap_distance_pct:.2f}% from VWAP and looks extended.")

    catalyst_supports = (
        (catalyst_bias == 1 and direction == "Long")
        or (catalyst_bias == -1 and direction == "Short")
    )
    catalyst_conflicts = (
        (catalyst_bias == -1 and direction == "Long")
        or (catalyst_bias == 1 and direction == "Short")
    )
    if catalyst_supports:
        why.append("Recent headline catalyst supports the technical direction.")
    elif catalyst_conflicts:
        failed.append("Recent headline catalyst conflicts with the technical direction.")
    else:
        why.append(news_label + ".")

    hard_fail = not (trend_ok and vwap_ok and liquidity_ok and gap_ok and stretch_ok) or catalyst_conflicts
    ready = not hard_fail and vol_ok and structure_pass

    if hard_fail:
        status, grade = "FAILED SETUP", "NO TRADE"
        recommendation = "No trade — a core setup condition has failed."
    elif beyond:
        status, grade = "DON'T CHASE", "MISSED"
        recommendation = (
            f"Breakout occurred but price is beyond the acceptable entry of "
            f"£{gbx_to_gbp(max_entry):.2f}. Stand aside."
        )
    elif broken and ready:
        strong_volume = volume_ratio is not None and volume_ratio >= MIN_A_GRADE_VOLUME_RATIO
        grade = "A" if strong_volume and catalyst_supports else "B"
        status = f"ENTER {direction.upper()}"
        recommendation = (
            f"{grade}-grade breakout is live. Entry from £{gbx_to_gbp(trigger):.2f}; "
            f"do not chase beyond £{gbx_to_gbp(max_entry):.2f}."
        )
    elif ready:
        strong_volume = volume_ratio is not None and volume_ratio >= MIN_A_GRADE_VOLUME_RATIO
        grade = "A" if strong_volume and catalyst_supports else "B"
        status = "READY — WAIT FOR BREAK"
        relation = "above" if direction == "Long" else "below"
        recommendation = (
            f"{grade}-grade setup. Wait for a clean break {relation} "
            f"£{gbx_to_gbp(trigger):.2f}. Do not enter before the break."
        )
    else:
        status, grade = "WATCH", "WATCH"
        recommendation = (
            "Potential setup, but confirmation is incomplete. Wait for volume/structure "
            "to improve before considering the opening-range break."
        )

    risk_budget = CAPITAL * RISK_PCT / 100
    risk_gbp = gbx_to_gbp(stop_distance) if stop_distance > 0 else 0
    current_gbp = gbx_to_gbp(current)
    shares = max(0, min(
        math.floor(risk_budget / risk_gbp) if risk_gbp > 0 else 0,
        math.floor(CAPITAL / current_gbp) if current_gbp > 0 else 0,
    ))

    opening_candles = [
        {"open": float(r["Open"]), "high": float(r["High"]),
         "low": float(r["Low"]), "close": float(r["Close"]),
         "time": idx.strftime("%H:%M")}
        for idx, r in opening.iterrows()
    ]

    base.update({
        "opening_candles": opening_candles,
        "news": news_items,
        "news_label": news_label,
        "status": status,
        "grade": grade,
        "recommendation": recommendation,
        "why": why,
        "failed": failed,
        "previous_close": pc,
        "gap_pct": gap_pct,
        "move_pct": move_pct,
        "range_pct": range_pct,
        "close": close,
        "vwap": vwap,
        "volume_ratio": volume_ratio,
        "turnover_gbp": turnover_gbp,
        "current": current,
        "current_time": current_time,
        "opening_range_high": or_high,
        "opening_range_low": or_low,
        "trigger": trigger,
        "maximum_entry": max_entry,
        "stop": stop,
        "target_1r": target_1r,
        "target": target_2r,
        "daily_atr": daily_atr,
        "stop_distance": stop_distance,
        "shares": shares,
        "position_value": shares * current_gbp,
        "planned_risk": shares * risk_gbp,
    })
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
    direction_colour = BULL if item.get("direction") == "Long" else BEAR
    badge = f"{'NAP — ' if nap else ''}{item.get('status','')}"
    why = "".join(f"<li>{html_lib.escape(v)}</li>" for v in item.get("why", [])) or "<li>None</li>"
    failed = "".join(f"<li>{html_lib.escape(v)}</li>" for v in item.get("failed", [])) or "<li>None</li>"
    volume_ratio = item.get("volume_ratio")
    volume_text = f"{volume_ratio:.2f}×" if finite(volume_ratio) else "—"

    return f"""
<section class="pick">
<div class="pick-head"><div>
<strong class="epic">{html_lib.escape(str(item.get('epic','')))}</strong>
<span class="name">{html_lib.escape(str(item.get('name','')))}</span>
<div class="pattern" style="color:{direction_colour}">{item.get('direction','')} · 15-minute opening-range breakout</div>
</div><div class="score"><span>{html_lib.escape(badge)}</span><strong>{html_lib.escape(str(item.get('grade','')))}</strong></div></div>

<div class="action-box"><div class="action-title">WHAT TO DO NOW</div>
<div class="action-text">{html_lib.escape(str(item.get('recommendation','')))}</div></div>

<div class="levels">
<div><span>Opening range</span><strong>{money(item.get('opening_range_low'))} – {money(item.get('opening_range_high'))}</strong></div>
<div><span>Entry trigger</span><strong>{money(item.get('trigger'))}</strong></div>
<div><span>Do not chase beyond</span><strong>{money(item.get('maximum_entry'))}</strong></div>
<div><span>Stop</span><strong>{money(item.get('stop'))}</strong></div>
<div><span>Target 1 (1R)</span><strong>{money(item.get('target_1r'))}</strong></div>
<div><span>Target 2 (2R)</span><strong>{money(item.get('target'))}</strong></div>
</div>

<div class="chart-wrap"><div class="chart-title">Morning price action — completed 5-minute candles</div>{candlestick_svg(item.get("opening_candles", []))}</div>
{news_html(item)}

<div class="metrics">
<div><span>Latest price</span><strong>{money(item.get('current'))} at {item.get('current_time').strftime('%H:%M') if item.get('current_time') else '—'}</strong></div>
<div><span>Close / VWAP</span><strong>{money(item.get('close'))} / {money(item.get('vwap'))}</strong></div>
<div><span>Relative volume</span><strong>{volume_text}</strong></div>
<div><span>Opening turnover</span><strong>£{item.get('turnover_gbp',0):,.0f}</strong></div>
<div><span>Gap / move</span><strong>{item.get('gap_pct',0):+.2f}% / {item.get('move_pct',0):+.2f}%</strong></div>
<div><span>Indicative size</span><strong>{item.get('shares',0)} shares</strong></div>
</div>

<details><summary>Why this setup is / isn’t ready</summary><div class="detail-grid">
<div><h3>Supporting evidence</h3><ul>{why}</ul></div>
<div><h3>Missing / failed conditions</h3><ul>{failed}</ul></div>
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
    priority = {
        "ENTER LONG": 5,
        "ENTER SHORT": 5,
        "READY — WAIT FOR BREAK": 4,
        "WATCH": 2,
        "DON'T CHASE": 1,
        "FAILED SETUP": 0,
        "NO TRADE": 0,
        "DATA ONLY": 0,
    }
    grade_rank = {"A": 3, "B": 2, "WATCH": 1, "MISSED": 0, "NO TRADE": 0, "DATA ONLY": 0}
    recommended = sorted(
        [row for row in results if priority.get(row.get("status",""), 0) >= 2],
        key=lambda row: (
            priority.get(row.get("status",""), 0),
            grade_rank.get(row.get("grade",""), 0),
        ),
        reverse=True,
    )[:MAX_RECOMMENDATIONS]

    rejected = [row for row in results if row not in recommended]
    rejected.sort(key=lambda row: row["score"], reverse=True)

    actionable = [row for row in recommended if str(row.get("status","")).startswith("ENTER ")]
    waiting = [row for row in recommended if row.get("status") == "READY — WAIT FOR BREAK"]
    nap_candidates = [row for row in actionable if row.get("grade") == "A"]
    nap_epic = nap_candidates[0]["epic"] if nap_candidates else None

    if actionable:
        decision = (
            f"{len(actionable)} live opening-range breakout"
            f"{'s' if len(actionable) != 1 else ''}. Follow the entry/stop rules shown below."
        )
    elif waiting:
        decision = (
            f"{len(waiting)} setup{'s' if len(waiting) != 1 else ''} ready, "
            "but none has broken the 15-minute opening range yet."
        )
    elif recommended:
        decision = "No trade yet. The candidates below remain on watch, but confirmation is incomplete."
    else:
        decision = "NO TRADE — no candidate currently meets the strategy conditions."

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
<meta name="viewport" content="width=device-width,initial-scale=1"><title>FTSE 350 15-minute opening-range strategy</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:$INK;color:$PAPER;font-family:Arial,sans-serif}
.wrap{max-width:920px;margin:auto;padding:22px 14px 56px}a{color:$BRASS}.header{display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;border-bottom:1px solid $HAIRLINE;padding-bottom:16px}
h1{margin:4px 0 0;font-size:27px}h3{font-size:13px;margin:0 0 5px}.kicker{color:$SALMON;font-size:12px;text-transform:uppercase;letter-spacing:.09em}
.time,.pattern,.name,.footer{color:$MUTED;font-size:12px}.decision,.pick,.rejected,.expired{background:$PANEL;border:1px solid $HAIRLINE;border-radius:9px;padding:15px;margin:14px 0}
.decision strong{color:$BRASS}.action-box{background:#12161F;border:1px solid $BRASS;border-radius:7px;padding:12px;margin:12px 0}.action-title{font-size:10px;color:$BRASS;letter-spacing:.08em;text-transform:uppercase;margin-bottom:5px}.action-text{font-size:14px;line-height:1.5}.levels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px;margin:12px 0}.levels div{background:#12161F;border:1px solid $HAIRLINE;border-radius:6px;padding:9px}.levels span{display:block;color:$MUTED;font-size:10px;margin-bottom:4px}.levels strong{font-size:13px}.pick-head{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}.epic{color:$BRASS;font-size:19px}.name{margin-left:8px}
.score{display:flex;flex-direction:column;text-align:right;font-size:13px;font-weight:700}.chart-wrap{margin:12px 0;background:#12161F;border:1px solid #2C333D;border-radius:7px;padding:10px}.chart-title{font-size:11px;color:#8B92A0;margin-bottom:5px}.mini-candles{width:100%;height:150px;display:block}.empty-chart{color:#8B92A0;font-size:12px;padding:20px;text-align:center}.news-box{margin:12px 0;background:#12161F;border:1px solid #2C333D;border-radius:7px;padding:10px 12px}.news-heading{font-size:11px;color:#C9A24B;text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}.news-list{margin:0;padding-left:18px}.news-list li{margin:5px 0;color:#ECE7DA;font-size:12px;line-height:1.35}.news-list a{color:#ECE7DA;text-decoration:none}.news-meta,.news-empty{font-size:10px;color:#8B92A0;margin-top:2px}.score strong{font-size:22px;margin-top:3px}.recommendation{font-size:14px;line-height:1.5}
.metrics{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:9px;margin-top:12px}.metrics div{background:$INK;border:1px solid $HAIRLINE;border-radius:6px;padding:9px}
.metrics span{display:block;color:$MUTED;font-size:11px;margin-bottom:4px}.metrics strong{font-size:12px;line-height:1.35}details{margin-top:12px}summary{cursor:pointer;color:$BRASS;font-size:13px}
.detail-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:10px}ul{padding-left:18px;color:$MUTED;font-size:12px;line-height:1.5}
.expired{display:none;border-color:$SALMON}.expired h2{color:$SALMON;margin-top:0}.footer{line-height:1.6;border-top:1px solid $HAIRLINE;padding-top:14px;margin-top:20px}
@media(max-width:760px){.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.detail-grid{grid-template-columns:1fr}h1{font-size:22px}}
</style></head><body><main class="wrap">
<div class="header"><div><div class="kicker">FTSE 350 ex trusts · 15-minute opening-range strategy</div><h1>$STAGE</h1></div><div class="time">$DATE<br>$TIME</div></div>
<div class="decision"><strong>$DECISION</strong> <a href="index.html">Daily watchlist</a> · <a href="backtest.html">Backtest</a></div>
<div id="expired-content" class="expired"><h2>Past the 09:30 cutoff</h2><p>The opening setups below are retained for review only. Do not use their levels now.</p></div>
<div id="live-content">$CARDS
<details class="rejected"><summary>Other candidates assessed but not recommended</summary><ul>$REJECTED</ul></details></div>
<p class="footer">Strategy: no trade decision before 08:15. The first three completed five-minute candles define the 08:00–08:15 opening range. READY means conditions align but the range has not broken. ENTER LONG/SHORT appears only after a confirmed break. DON’T CHASE means price has travelled too far beyond the planned entry. Stops use the widest of opening-range structure, 0.50× daily ATR and a 0.75% minimum distance.</p>
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
