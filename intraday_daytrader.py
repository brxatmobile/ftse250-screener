"""
FTSE opening-hour day-trade assessment using pattern-first daily selection.

This is an add-on to the existing screener.py. It does not alter the daily
screener or backtest. The daily screener supplies the shortlist; this script
assesses the completed 08:00-09:00 London opening hour using 5-minute bars.

Normal scheduled behaviour (Europe/London):
  09:00-09:20  Build docs/intraday.html
  After 10:00   Keep the assessment visible but mark it as no longer actionable

Manual examples:
  python intraday_daytrader.py --mode analyse --force
  python intraday_daytrader.py --mode expire --force

Environment variables:
  CAPITAL                    default 5000
  RISK_PCT                   default 1
  TOP_N                      default 5
  MAX_GAP_PCT                default 2.0
  TARGET_R                   default 2.0
  MIN_INTRADAY_SCORE         default 65
  MIN_OPENING_VOLUME_RATIO   default 0.60
  MAX_VWAP_DISTANCE_PCT      default 1.25
  ENTRY_BUFFER_PCT           default 0.05
"""

# FILE_VERSION: INTRADAY_PATTERN_VOLUME_TREND_2026_08_02

from __future__ import annotations

import argparse
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

CAPITAL = float(os.environ.get("CAPITAL", "5000"))
RISK_PCT = float(os.environ.get("RISK_PCT", "1"))
TOP_N = int(os.environ.get("TOP_N", "5"))
MAX_GAP_PCT = float(os.environ.get("MAX_GAP_PCT", "2.0"))
TARGET_R = float(os.environ.get("TARGET_R", "2.0"))
MIN_INTRADAY_SCORE = float(os.environ.get("MIN_INTRADAY_SCORE", "65"))
MIN_OPENING_VOLUME_RATIO = float(os.environ.get("MIN_OPENING_VOLUME_RATIO", "0.60"))
MAX_VWAP_DISTANCE_PCT = float(os.environ.get("MAX_VWAP_DISTANCE_PCT", "1.25"))
ENTRY_BUFFER_PCT = float(os.environ.get("ENTRY_BUFFER_PCT", "0.05"))

OUTPUT_PATH = Path(__file__).resolve().parent / "docs" / "intraday.html"

INK = "#12161F"
PANEL = "#1B2129"
HAIRLINE = "#2C333D"
BRASS = "#C9A24B"
SALMON = "#E8A493"
BULL = "#4FAE73"
BEAR = "#D1594B"
PAPER = "#ECE7DA"
MUTED = "#8B92A0"


def finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def london_now() -> dt.datetime:
    return dt.datetime.now(UTC).astimezone(LONDON)


def normalise_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if any(column not in frame.columns for column in needed):
        return pd.DataFrame()
    frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
    if frame.empty:
        return frame
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        index = index.tz_localize(UTC)
    frame.index = index.tz_convert(LONDON)
    return frame.sort_index()


def enforce_window(now: dt.datetime, mode: str, force: bool) -> None:
    if force:
        return
    if now.weekday() >= 5:
        print(f"Skipping: {now:%A} is not a trading weekday.")
        raise SystemExit(0)
    valid = (
        mode == "analyse" and now.hour == 9 and 0 <= now.minute <= 20
    ) or (
        mode == "expire" and now.hour == 10 and 0 <= now.minute <= 20
    )
    if not valid:
        print(f"Skipping {mode} at {now:%Y-%m-%d %H:%M %Z}: outside its London window.")
        raise SystemExit(0)


def auto_mode(now: dt.datetime) -> str:
    if now.hour == 10:
        return "expire"
    return "analyse"


def get_daily_candidates() -> list[dict[str, Any]]:
    print("Getting the daily shortlist through the existing screener.py...")
    constituents = scr.fetch_ftse250_constituents()
    epics = list(constituents)
    tickers = [scr.epic_to_yahoo(epic) for epic in epics]
    ticker_to_epic = dict(zip(tickers, epics))

    data = yf.download(
        tickers,
        period="3mo",
        interval="1d",
        group_by="ticker",
        threads=True,
        progress=False,
        auto_adjust=False,
    )

    ranked: list[dict[str, Any]] = []
    for ticker in tickers:
        try:
            frame = data[ticker] if isinstance(data.columns, pd.MultiIndex) else data
        except Exception:
            continue
        if frame is None or frame.empty:
            continue
        frame = frame.dropna(subset=["Open", "High", "Low", "Close"])
        if frame.empty:
            continue
        epic = ticker_to_epic[ticker]
        try:
            result = scr.analyze(epic, constituents[epic], frame)
        except Exception as exc:
            print(f"Skipping {epic}: daily analysis failed: {exc}")
            continue
        if result:
            result["yahoo_ticker"] = ticker
            ranked.append(result)

    ranked.sort(key=lambda row: float(row.get("score", 0)), reverse=True)
    for daily_rank, row in enumerate(ranked, start=1):
        row["daily_rank"] = daily_rank
    return ranked[:TOP_N]


def get_intraday_history(ticker: str) -> pd.DataFrame:
    raw = yf.download(
        ticker,
        period="5d",
        interval="5m",
        progress=False,
        auto_adjust=False,
        prepost=False,
    )
    return normalise_frame(raw)


def previous_close_from_history(frame: pd.DataFrame, today: dt.date) -> float | None:
    earlier = frame[frame.index.date < today]
    if earlier.empty:
        return None
    close = earlier.iloc[-1]["Close"]
    return float(close) if finite_number(close) else None


def average_prior_opening_volume(frame: pd.DataFrame, today: dt.date) -> float | None:
    totals: list[float] = []
    for trading_date in sorted(set(frame.index.date)):
        if trading_date >= today:
            continue
        day = frame[frame.index.date == trading_date]
        opening = day[(day.index.time >= dt.time(8, 0)) & (day.index.time < dt.time(9, 0))]
        if not opening.empty:
            volume = float(opening["Volume"].fillna(0).sum())
            if volume > 0:
                totals.append(volume)
    return float(np.mean(totals)) if totals else None


def first_hour_pattern(opening: pd.DataFrame) -> tuple[str, float]:
    first_open = float(opening.iloc[0]["Open"])
    last_close = float(opening.iloc[-1]["Close"])
    high = float(opening["High"].max())
    low = float(opening["Low"].min())
    span = max(high - low, 1e-9)
    body = abs(last_close - first_open)
    upper = high - max(first_open, last_close)
    lower = min(first_open, last_close) - low

    if last_close > first_open and body / span >= 0.60:
        return "Strong bullish opening candle", 10.0
    if last_close < first_open and body / span >= 0.60:
        return "Strong bearish opening candle", 10.0
    if lower / span >= 0.45 and body / span <= 0.40:
        return "Bullish rejection from the opening low", 8.0
    if upper / span >= 0.45 and body / span <= 0.40:
        return "Bearish rejection from the opening high", 8.0
    return "Indecisive opening-hour structure", 2.0


def assess_candidate(candidate: dict[str, Any], now: dt.datetime) -> dict[str, Any]:
    ticker = candidate["yahoo_ticker"]
    frame = get_intraday_history(ticker)
    direction = str(candidate.get("direction", "Long"))
    long_side = direction.lower() == "long"

    result: dict[str, Any] = {
        "epic": candidate.get("epic", ticker.replace(".L", "")),
        "name": candidate.get("name", ""),
        "ticker": ticker,
        "direction": direction,
        "daily_pattern": candidate.get("pattern", ""),
        "daily_pattern_base": float(candidate.get("pattern_base", 0) or 0),
        "daily_strong_pattern": bool(candidate.get("strong_pattern", False)),
        "daily_trend_aligned": bool(candidate.get("trend_aligned", False)),
        "daily_volume_ratio": float(candidate.get("vol_ratio", 1.0) or 1.0),
        "daily_score": float(candidate.get("score", 0)),
        "daily_rank": int(candidate.get("daily_rank", 0) or 0),
        "status": "NO TRADE",
        "recommendation": "No usable opening-hour data",
        "intraday_score": 0.0,
        "checks": [],
    }
    if frame.empty:
        return result

    today = now.date()
    today_frame = frame[frame.index.date == today]
    opening = today_frame[(today_frame.index.time >= dt.time(8, 0)) & (today_frame.index.time < dt.time(9, 0))]
    if len(opening) < 10:
        result["recommendation"] = f"Only {len(opening)} of 12 expected opening-hour bars were available"
        return result

    previous_close = previous_close_from_history(frame, today)
    if previous_close is None or previous_close <= 0:
        result["recommendation"] = "Previous close unavailable"
        return result

    open_price = float(opening.iloc[0]["Open"])
    high = float(opening["High"].max())
    low = float(opening["Low"].min())
    close = float(opening.iloc[-1]["Close"])
    volume = float(opening["Volume"].fillna(0).sum())
    typical = (opening["High"] + opening["Low"] + opening["Close"]) / 3
    volume_series = opening["Volume"].fillna(0)
    vwap = float((typical * volume_series).sum() / volume_series.sum()) if volume_series.sum() > 0 else close
    gap_pct = ((open_price - previous_close) / previous_close) * 100
    move_pct = ((close - previous_close) / previous_close) * 100
    range_pct = ((high - low) / previous_close) * 100
    close_location = ((close - low) / max(high - low, 1e-9)) * 100
    vwap_distance_pct = abs((close - vwap) / vwap) * 100 if vwap else 0

    prior_avg_volume = average_prior_opening_volume(frame, today)
    volume_ratio = volume / prior_avg_volume if prior_avg_volume and prior_avg_volume > 0 else None
    pattern_text, pattern_points = first_hour_pattern(opening)

    score = 0.0
    checks: list[str] = []
    blockers: list[str] = []

    # Carry the research-backed daily evidence into the opening-hour decision.
    daily_pattern_base = float(result.get("daily_pattern_base", 0) or 0)
    daily_strong_pattern = bool(result.get("daily_strong_pattern", False)) or daily_pattern_base >= 7.0
    daily_volume_ratio = float(result.get("daily_volume_ratio", 1.0) or 1.0)
    daily_trend_aligned = bool(result.get("daily_trend_aligned", False))

    if daily_strong_pattern:
        score += 10
        checks.append(f"Strong daily pattern ({daily_pattern_base:.1f} base)")
    else:
        blockers.append(f"Daily pattern base {daily_pattern_base:.1f} is below the researched 7.0 threshold")

    if daily_volume_ratio >= 2.0:
        score += 5
        checks.append(f"Daily volume was very strong at {daily_volume_ratio:.2f}× average")
    elif daily_volume_ratio >= 1.5:
        score += 3
        checks.append(f"Daily volume was elevated at {daily_volume_ratio:.2f}× average")

    if daily_trend_aligned:
        score += 3
        checks.append("Daily direction is aligned with SMA20")

    aligned_previous_close = close > previous_close if long_side else close < previous_close
    if aligned_previous_close:
        score += 15
        checks.append("Direction agrees with the previous close")
    else:
        blockers.append("Opening hour contradicts the daily direction")

    aligned_vwap = close > vwap if long_side else close < vwap
    if aligned_vwap:
        score += 15
        checks.append("Price is on the correct side of VWAP")
    else:
        blockers.append("Price is on the wrong side of VWAP")

    if abs(gap_pct) <= MAX_GAP_PCT:
        score += 10
        checks.append("Opening gap is within the permitted range")
    else:
        blockers.append(f"Opening gap {gap_pct:+.2f}% is excessive")

    favourable_location = close_location >= 70 if long_side else close_location <= 30
    if favourable_location:
        score += 15
        checks.append("09:00 close is near the favourable end of the opening range")
    elif 40 <= close_location <= 60:
        blockers.append("09:00 close is trapped near the middle of the opening range")

    if volume_ratio is None:
        score += 5
        checks.append("Relative opening volume unavailable; neutral weighting used")
    elif volume_ratio >= 1.0:
        score += 15
        checks.append(f"Opening volume is {volume_ratio:.2f}× its recent first-hour average")
    elif volume_ratio >= MIN_OPENING_VOLUME_RATIO:
        score += 8
        checks.append(f"Opening volume is acceptable at {volume_ratio:.2f}× average")
    else:
        blockers.append(f"Opening volume is weak at {volume_ratio:.2f}× average")

    if vwap_distance_pct <= MAX_VWAP_DISTANCE_PCT:
        score += 10
        checks.append("Price is not excessively extended from VWAP")
    else:
        blockers.append(f"Price is {vwap_distance_pct:.2f}% from VWAP and may be overextended")

    pattern_agrees = (
        long_side and ("bullish" in pattern_text.lower())
    ) or (
        (not long_side) and ("bearish" in pattern_text.lower())
    )
    if pattern_agrees:
        score += pattern_points
        checks.append(pattern_text)
    elif "indecisive" in pattern_text.lower():
        score += pattern_points
        checks.append(pattern_text)
    else:
        blockers.append(pattern_text + " opposes the daily direction")

    after_nine = today_frame[today_frame.index.time >= dt.time(9, 0)]
    reference_entry = float(after_nine.iloc[0]["Open"]) if not after_nine.empty else close
    daily_stop = float(candidate.get("stop", low if long_side else high))

    if long_side:
        trigger = high * (1 + ENTRY_BUFFER_PCT / 100)
        stop = max(low, daily_stop) if daily_stop < reference_entry else low
        risk = trigger - stop
        target = trigger + TARGET_R * risk
        invalidated = reference_entry <= stop
    else:
        trigger = low * (1 - ENTRY_BUFFER_PCT / 100)
        stop = min(high, daily_stop) if daily_stop > reference_entry else high
        risk = stop - trigger
        target = trigger - TARGET_R * risk
        invalidated = reference_entry >= stop

    if invalidated or risk <= 0 or not all(finite_number(v) for v in [trigger, stop, target]):
        blockers.append("The calculated entry/stop structure is invalid")

    risk_budget = CAPITAL * RISK_PCT / 100
    shares = 0
    position_value = 0.0
    planned_risk = 0.0
    if risk > 0:
        trigger_gbp = scr.gbx_to_gbp(trigger)
        risk_gbp = scr.gbx_to_gbp(risk)
        if trigger_gbp > 0 and risk_gbp > 0:
            shares = max(0, min(math.floor(risk_budget / risk_gbp), math.floor(CAPITAL / trigger_gbp)))
            position_value = shares * trigger_gbp
            planned_risk = shares * risk_gbp
    if shares <= 0:
        blockers.append("Capital/risk settings do not support one whole share")

    hard_blockers = [
        text for text in blockers
        if text.startswith("Opening hour contradicts")
        or text.startswith("Price is on the wrong side")
        or text.startswith("Opening gap")
        or text.startswith("The calculated")
        or text.startswith("Daily pattern base")
    ]

    if not hard_blockers and score >= 80:
        status = "STRONG SETUP"
        nap_prefix = "NAP — " if long_side and result.get("daily_rank") == 1 else ""
        recommendation = f"{nap_prefix}Strong pattern-led setup. Consider only after a clean break and hold beyond the opening-range {'high' if long_side else 'low'}; do not chase the trigger."
    elif not hard_blockers and score >= MIN_INTRADAY_SCORE:
        status = "WATCH"
        recommendation = f"Pattern qualifies but confirmation is incomplete. Require a clean opening-range break, acceptable live spread and continued volume before considering entry."
    else:
        status = "NO TRADE"
        recommendation = "Opening-hour evidence is not strong enough for the proposed day-trade plan."

    result.update({
        "status": status,
        "recommendation": recommendation,
        "intraday_score": min(100.0, score),
        "checks": checks,
        "blockers": blockers,
        "previous_close_gbx": previous_close,
        "open_gbx": open_price,
        "gap_pct": gap_pct,
        "move_pct": move_pct,
        "range_pct": range_pct,
        "hour_high_gbx": high,
        "hour_low_gbx": low,
        "hour_close_gbx": close,
        "close_location_pct": close_location,
        "vwap_gbx": vwap,
        "vwap_distance_pct": vwap_distance_pct,
        "hour_volume": int(volume),
        "volume_ratio": volume_ratio,
        "opening_pattern": pattern_text,
        "entry_trigger_gbx": trigger,
        "stop_gbx": stop,
        "target_gbx": target,
        "shares": shares,
        "position_value_gbp": position_value,
        "planned_risk_gbp": planned_risk,
    })
    return result


def money_gbx(value: Any) -> str:
    return f"£{scr.gbx_to_gbp(float(value)):.2f}" if finite_number(value) else "—"


def render_result(item: dict[str, Any]) -> str:
    status = item["status"]
    colour = BULL if status == "STRONG SETUP" else (BRASS if status == "WATCH" else MUTED)
    checks = "".join(f"<li>{html_lib.escape(text)}</li>" for text in item.get("checks", []))
    blockers = "".join(f"<li>{html_lib.escape(text)}</li>" for text in item.get("blockers", []))
    volume_ratio = item.get("volume_ratio")
    volume_text = f"{volume_ratio:.2f}×" if finite_number(volume_ratio) else "—"

    return f"""
    <section class="pick">
      <div class="pick-head">
        <div>
          <strong class="epic">{html_lib.escape(str(item['epic']))}</strong>
          <span class="name">{html_lib.escape(str(item['name']))}</span>
          <div class="pattern">{html_lib.escape(str(item['direction']))} · {html_lib.escape(str(item['daily_pattern']))} · daily {item['daily_score']:.1f}/100</div>
        </div>
        <div class="score"><span style="color:{colour}">{html_lib.escape(status)}</span><strong>{item['intraday_score']:.0f}/100</strong></div>
      </div>
      <p class="recommendation">{html_lib.escape(item['recommendation'])}</p>
      <div class="metrics">
        <div><span>Previous close</span><strong>{money_gbx(item.get('previous_close_gbx'))}</strong></div>
        <div><span>Opening gap</span><strong>{item.get('gap_pct', 0):+.2f}%</strong></div>
        <div><span>09:00 move</span><strong>{item.get('move_pct', 0):+.2f}%</strong></div>
        <div><span>Opening range</span><strong>{item.get('range_pct', 0):.2f}%</strong></div>
        <div><span>09:00 close / VWAP</span><strong>{money_gbx(item.get('hour_close_gbx'))} / {money_gbx(item.get('vwap_gbx'))}</strong></div>
        <div><span>Close in range</span><strong>{item.get('close_location_pct', 0):.0f}%</strong></div>
        <div><span>Relative volume</span><strong>{volume_text}</strong></div>
        <div><span>Opening pattern</span><strong>{html_lib.escape(item.get('opening_pattern', '—'))}</strong></div>
        <div><span>Entry trigger</span><strong>{money_gbx(item.get('entry_trigger_gbx'))}</strong></div>
        <div><span>Stop / target</span><strong>{money_gbx(item.get('stop_gbx'))} / {money_gbx(item.get('target_gbx'))}</strong></div>
        <div><span>Indicative size</span><strong>{item.get('shares', 0)} shares</strong></div>
        <div><span>Value / planned risk</span><strong>£{item.get('position_value_gbp', 0):.2f} / £{item.get('planned_risk_gbp', 0):.2f}</strong></div>
      </div>
      <details><summary>Assessment details</summary>
        <div class="detail-grid"><div><h3>Positive checks</h3><ul>{checks or '<li>None</li>'}</ul></div><div><h3>Warnings</h3><ul>{blockers or '<li>None</li>'}</ul></div></div>
      </details>
    </section>"""


def build_live_html(results: list[dict[str, Any]], generated_at: dt.datetime) -> str:
    from string import Template

    ranked = sorted(results, key=lambda row: float(row.get("intraday_score", 0)), reverse=True)
    actionable = sum(row["status"] in {"STRONG SETUP", "WATCH"} for row in ranked)
    expiry_iso = dt.datetime.combine(generated_at.date(), dt.time(10, 0), tzinfo=LONDON).isoformat()
    cards = "".join(render_result(item) for item in ranked)
    template = Template("""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FTSE opening-hour day-trade review</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:$INK;color:$PAPER;font-family:Arial,sans-serif}.wrap{max-width:900px;margin:auto;padding:22px 14px 56px}a{color:$BRASS}.header{display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;border-bottom:1px solid $HAIRLINE;padding-bottom:16px}h1{margin:4px 0 0;font-size:27px}h3{font-size:13px;margin:0 0 5px}.kicker{color:$SALMON;font-size:12px;text-transform:uppercase;letter-spacing:.09em}.time,.pattern,.name,.footer{color:$MUTED;font-size:12px}.summary,.pick{background:$PANEL;border:1px solid $HAIRLINE;border-radius:9px;padding:15px;margin:14px 0}.summary strong{color:$BRASS}.expiry{color:$SALMON;font-weight:700}.pick-head{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}.epic{color:$BRASS;font-size:19px}.name{margin-left:8px}.score{display:flex;flex-direction:column;text-align:right;font-size:13px;font-weight:700}.score strong{font-size:22px;margin-top:3px}.recommendation{font-size:14px;line-height:1.5}.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;margin-top:12px}.metrics div{background:$INK;border:1px solid $HAIRLINE;border-radius:6px;padding:9px}.metrics span{display:block;color:$MUTED;font-size:11px;margin-bottom:4px}.metrics strong{font-size:12px;line-height:1.35}details{margin-top:12px}summary{cursor:pointer;color:$BRASS;font-size:13px}.detail-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:10px}ul{padding-left:18px;color:$MUTED;font-size:12px;line-height:1.5}.expired{display:none;background:$PANEL;border:1px solid $SALMON;border-radius:9px;padding:16px;margin:14px 0;color:$PAPER}.expired h2{margin:0 0 8px;color:$SALMON}.expired p{margin:0;line-height:1.5}.footer{line-height:1.6;border-top:1px solid $HAIRLINE;padding-top:14px;margin-top:20px}@media(max-width:680px){.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.detail-grid{grid-template-columns:1fr}h1{font-size:22px}}
</style></head><body><main class="wrap">
<div class="header"><div><div class="kicker">FTSE · opening-hour decision support</div><h1>09:00 day-trade review</h1></div><div class="time">$DATE<br>$TIME</div></div>
<div id="expired-content" class="expired"><h2>Past the 10:00 cutoff</h2><p>The assessment below is retained for review, but it should not be followed after 10:00 London time. Opening-hour triggers, stops and targets may no longer be valid.</p></div>
<div id="live-content"><div class="summary"><strong>$ACTIONABLE of $TOTAL candidates remain worth reviewing.</strong> These are conditional setups, not market orders. <span class="expiry">Only follow this assessment before 10:00 London time.</span> <a href="index.html">Daily watchlist</a> · <a href="backtest.html">Backtest</a></div>$CARDS</div>
<p class="footer">Method: candidates must first have a strong daily candlestick pattern (base at least 7), with daily volume and SMA20 alignment used as secondary evidence. They are then reassessed using completed 08:00–09:00 five-minute bars, previous close, opening gap, VWAP, opening-range position, relative first-hour volume and opening-hour candle structure. Entry levels are conditional opening-range triggers. Check live broker prices, spreads and news before taking any action. This is decision support, not financial advice.</p>
</main><script>
(function(){const expiry=new Date('$EXPIRY');const expired=document.getElementById('expired-content');function enforce(){if(new Date()>=expiry){expired.style.display='block';document.querySelectorAll('.recommendation').forEach(function(el){if(!el.dataset.original){el.dataset.original=el.textContent;}el.textContent='PAST 10:00 — Do not follow this recommendation now. Original assessment: '+el.dataset.original;});}}enforce();setInterval(enforce,30000);})();
</script></body></html>""")
    return template.substitute(
        INK=INK, PAPER=PAPER, BRASS=BRASS, HAIRLINE=HAIRLINE, SALMON=SALMON,
        MUTED=MUTED, PANEL=PANEL, DATE=generated_at.strftime("%a %d %b %Y"),
        TIME=generated_at.strftime("%H:%M %Z"), ACTIONABLE=actionable,
        TOTAL=len(ranked), CARDS=cards, EXPIRY=expiry_iso,
    )


def build_expired_html(now: dt.datetime) -> str:
    from string import Template

    template = Template("""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FTSE intraday recommendations expired</title><style>:root{color-scheme:dark}body{margin:0;background:$INK;color:$PAPER;font-family:Arial,sans-serif}main{max-width:720px;margin:auto;padding:44px 18px}section{background:$PANEL;border:1px solid $HAIRLINE;border-radius:9px;padding:24px}h1{margin-top:0}p{line-height:1.6;color:$MUTED}a{color:$BRASS}.time{font-size:12px;color:$SALMON}</style></head><body><main><section><div class="time">$STAMP</div><h1>Today’s recommendations have expired</h1><p>The 10:00 London cutoff has passed. The opening-hour setups have been removed because they are no longer intended for execution.</p><p>The page will be rebuilt after tomorrow’s completed opening hour.</p><p><a href="index.html">Daily watchlist</a> · <a href="backtest.html">Backtest</a></p></section></main></body></html>""")
    return template.substitute(
        INK=INK, PAPER=PAPER, PANEL=PANEL, HAIRLINE=HAIRLINE, MUTED=MUTED,
        BRASS=BRASS, SALMON=SALMON, STAMP=now.strftime("%a %d %b %Y · %H:%M %Z"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["auto", "analyse", "expire"], default="auto")
    parser.add_argument("--force", action="store_true", help="Ignore London time-window checks")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = london_now()
    mode = auto_mode(now) if args.mode == "auto" else args.mode
    enforce_window(now, mode, args.force)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if mode == "expire":
        # The live page now retains its cards and changes its warning automatically
        # in the browser after 10:00 London time. Do not overwrite it.
        print("Past 10:00: retaining the existing intraday report with its expiry warning.")
        return 0

    candidates = get_daily_candidates()
    results = [assess_candidate(candidate, now) for candidate in candidates]
    OUTPUT_PATH.write_text(build_live_html(results, now), encoding="utf-8")
    print(f"Day-trade report written to {OUTPUT_PATH}")
    for item in sorted(results, key=lambda row: row.get("intraday_score", 0), reverse=True):
        print(f"{item['epic']:<7} {item['status']:<12} {item['intraday_score']:>5.1f}  {item['recommendation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
