"""
FTSE 250 daily candlestick screener.

- Scrapes the current FTSE 250 constituent list from the London Stock Exchange
  site (so the universe stays current without hardcoding tickers).
- Pulls ~2 months of daily OHLCV for every constituent via Yahoo Finance.
- Ranks stocks pattern-first, then volume and SMA20 trend alignment; RSI is displayed but not scored.
- Picks the top 5 as a next-session watchlist and writes a self-contained HTML report to docs/index.html.

Run manually with:  python screener.py
Configure capital / risk % via CAPITAL and RISK_PCT below, or environment
variables CAPITAL and RISK_PCT (used by the GitHub Actions workflow).

Yahoo Finance returns London Stock Exchange equity prices in pence. Pattern
analysis remains in pence, but monetary display and indicative sizing are
converted to pounds. The completed daily candle is used only to build a
next-session watchlist; it is not treated as an executable entry price.
"""

# FILE_VERSION: PATTERN_VOLUME_TREND_LIVE_2026_08_02
import os
import re
import sys
import math
import datetime as dt
import html as html_lib
from html.parser import HTMLParser

import requests
import pandas as pd
import numpy as np
import yfinance as yf

CAPITAL = float(os.environ.get("CAPITAL", "5000"))
RISK_PCT = float(os.environ.get("RISK_PCT", "1"))
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "docs", "index.html")

# Yahoo Finance normally reports London-listed equity prices in GBX (pence).
PENCE_PER_POUND = 100.0

INK = "#12161F"
PANEL = "#1B2129"
HAIRLINE = "#2C333D"
BRASS = "#C9A24B"
SALMON = "#E8A493"
BULL = "#4FAE73"
BEAR = "#D1594B"
PAPER = "#ECE7DA"
MUTED = "#8B92A0"


def gbx_to_gbp(value):
    """Convert a Yahoo Finance London-market price from pence to pounds."""
    return float(value) / PENCE_PER_POUND


def calculate_position_size(capital, risk_amount, entry_gbx, risk_per_share_gbx):
    """
    Return a whole-share position size.

    The size is constrained by both:
      1. maximum cash risk at the stop; and
      2. available capital/notional value.

    Prices supplied by Yahoo for .L equities are in pence, so both entry and
    per-share risk are converted to pounds before sizing.
    """
    entry_gbp = gbx_to_gbp(entry_gbx)
    risk_per_share_gbp = gbx_to_gbp(risk_per_share_gbx)

    if entry_gbp <= 0 or risk_per_share_gbp <= 0:
        return 0

    shares_by_risk = math.floor(risk_amount / risk_per_share_gbp)
    shares_by_capital = math.floor(capital / entry_gbp)
    return max(0, min(shares_by_risk, shares_by_capital))


def _clean_company_name(value):
    """Remove HTML markup/entities and tidy a company/security name."""
    value = re.sub(r"<[^>]+>", " ", str(value))
    value = html_lib.unescape(value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" \t\r\n-–|")


LSE_URL = "https://www.lse.co.uk/indices/ftse-250/constituents.html"
LSE_EXPECTED_MIN = 200
LSE_EXPECTED_MAX = 260


def _make_lse_session():
    """Create a browser-like session for the static constituent page."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/150.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-GB,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    })
    return session


def _extract_first_table_after_marker(page_html, marker):
    """Return the first complete HTML table appearing after marker text."""
    marker_match = re.search(
        re.escape(marker).replace(r"\ ", r"\s+"),
        page_html,
        flags=re.IGNORECASE,
    )
    if not marker_match:
        raise RuntimeError(
            f"Could not find the FTSE 250 constituent marker {marker!r}."
        )

    table_start = page_html.find("<table", marker_match.end())
    if table_start < 0:
        raise RuntimeError("No constituent table found after the FTSE 250 marker.")

    # Handle nested markup safely enough for ordinary HTML tables by counting
    # table open/close tags from the first table after the marker.
    token_re = re.compile(r"</?table\b[^>]*>", re.IGNORECASE)
    depth = 0
    for token in token_re.finditer(page_html, table_start):
        if token.group(0).lower().startswith("</table"):
            depth -= 1
            if depth == 0:
                return page_html[table_start:token.end()]
        else:
            depth += 1

    raise RuntimeError("The FTSE 250 constituent table was not properly closed.")


def _parse_ftse250_constituent_table(page_html):
    """
    Parse only the table immediately following the statement that its shares
    make up the FTSE 250. This deliberately ignores all sidebar, chat, news,
    riser/faller and other share links elsewhere on the page.
    """
    marker = "The following shares make up the FTSE 250"
    table_html = _extract_first_table_after_marker(page_html, marker)

    constituents = {}
    row_re = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
    anchor_re = re.compile(
        r"<a\b[^>]*href=[\"'][^\"']*(?:shareprice=([A-Z0-9.]+)|/share-prices/[^\"']+)[^\"']*[\"'][^>]*>(.*?)</a>",
        re.IGNORECASE | re.DOTALL,
    )

    for row_match in row_re.finditer(table_html):
        row_html = row_match.group(1)
        row_text = _clean_company_name(row_html)

        epic = None
        name = None

        # Preferred path: read the ticker from the shareprice query parameter
        # and the company name from that same anchor.
        for anchor_match in anchor_re.finditer(row_html):
            query_epic = anchor_match.group(1)
            label = _clean_company_name(anchor_match.group(2))
            visible = re.search(r"^(.*?)\s*\(([A-Z0-9.]{1,10})\.?\)\s*$", label)

            if query_epic:
                epic = query_epic.upper().strip().rstrip(".")
                name = label
                if visible and visible.group(2).upper().rstrip(".") == epic:
                    name = visible.group(1).strip()
                break

            if visible:
                epic = visible.group(2).upper().strip().rstrip(".")
                name = visible.group(1).strip()
                break

        # Fallback: the displayed Share cell always ends with "(EPIC)".
        if not epic:
            visible = re.search(
                r"(.+?)\s*\(([A-Z0-9.]{1,10})\.?\)(?=\s|$)",
                row_text,
                flags=re.IGNORECASE,
            )
            if visible:
                name = visible.group(1).strip()
                epic = visible.group(2).upper().strip().rstrip(".")

        if not epic or not name:
            continue
        if not re.fullmatch(r"[A-Z0-9.]{1,10}", epic):
            continue
        if epic.lower() in {"share", "price", "change"}:
            continue

        constituents.setdefault(epic, name)

    count = len(constituents)
    if not LSE_EXPECTED_MIN <= count <= LSE_EXPECTED_MAX:
        sample = ", ".join(list(constituents)[:10]) or "none"
        raise RuntimeError(
            "The isolated FTSE 250 constituent table produced an unexpected "
            f"count of {count}; expected {LSE_EXPECTED_MIN}-{LSE_EXPECTED_MAX}. "
            f"Sample codes: {sample}. Refusing to publish an unverified universe."
        )

    return constituents


def fetch_ftse250_constituents():
    """
    Fetch the FTSE 250 universe automatically from the static lse.co.uk page.

    Only the first table immediately following the exact sentence
    'The following shares make up the FTSE 250' is parsed. No links outside
    that table are eligible, which prevents AIM/sidebar shares from entering.
    """
    session = _make_lse_session()
    try:
        response = session.get(LSE_URL, timeout=(20, 60), allow_redirects=True)
        print(
            f"  Constituents response: HTTP {response.status_code}; "
            f"{len(response.content):,} bytes; final URL: {response.url}"
        )
        response.raise_for_status()
        constituents = _parse_ftse250_constituent_table(response.text)
        print(
            f"  Parsed {len(constituents)} FTSE 250 constituents from the "
            "isolated constituent table."
        )
        return constituents
    finally:
        session.close()

def epic_to_yahoo(epic):
    return epic.rstrip(".") + ".L"


def rsi(closes, period=14):
    closes = np.asarray(closes, dtype=float)
    if len(closes) < period + 1:
        return None
    diffs = np.diff(closes[-(period + 1):])
    gains = diffs[diffs > 0].sum()
    losses = -diffs[diffs < 0].sum()
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return 100 - 100 / (1 + rs)


def sma(vals, period):
    if len(vals) < period:
        return None
    return float(np.mean(vals[-period:]))


def detect_pattern(candles):
    """candles: list of dicts with open/high/low/close, oldest -> newest."""
    if len(candles) < 3:
        return None
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]

    def body(c):
        return abs(c["close"] - c["open"])

    def rng(c):
        return (c["high"] - c["low"]) or 0.0001

    def is_bull(c):
        return c["close"] > c["open"]

    def is_bear(c):
        return c["close"] < c["open"]

    def lower_wick(c):
        return min(c["open"], c["close"]) - c["low"]

    def upper_wick(c):
        return c["high"] - max(c["open"], c["close"])

    if (is_bear(c1) and body(c1) > rng(c1) * 0.4 and body(c2) < body(c1) * 0.4
            and is_bull(c3) and body(c3) > rng(c3) * 0.4
            and c3["close"] > (c1["open"] + c1["close"]) / 2):
        return {"name": "Morning star", "dir": "bull", "base": 9}
    if (is_bull(c1) and body(c1) > rng(c1) * 0.4 and body(c2) < body(c1) * 0.4
            and is_bear(c3) and body(c3) > rng(c3) * 0.4
            and c3["close"] < (c1["open"] + c1["close"]) / 2):
        return {"name": "Evening star", "dir": "bear", "base": 9}
    if is_bear(c2) and is_bull(c3) and c3["open"] <= c2["close"] and c3["close"] >= c2["open"]:
        return {"name": "Bullish engulfing", "dir": "bull", "base": 8}
    if is_bull(c2) and is_bear(c3) and c3["open"] >= c2["close"] and c3["close"] <= c2["open"]:
        return {"name": "Bearish engulfing", "dir": "bear", "base": 8}
    if body(c3) > 0 and lower_wick(c3) >= body(c3) * 2 and upper_wick(c3) <= body(c3) * 0.35:
        return {
            "name": "Hammer" if is_bull(c3) else "Hanging man",
            "dir": "bull" if is_bull(c3) else "bear",
            "base": 6.5,
        }
    if body(c3) > 0 and upper_wick(c3) >= body(c3) * 2 and lower_wick(c3) <= body(c3) * 0.35:
        return {
            "name": "Shooting star" if is_bear(c3) else "Inverted hammer",
            "dir": "bear" if is_bear(c3) else "bull",
            "base": 6,
        }
    if body(c3) >= rng(c3) * 0.85:
        return {
            "name": "Bullish marubozu" if is_bull(c3) else "Bearish marubozu",
            "dir": "bull" if is_bull(c3) else "bear",
            "base": 7,
        }
    if body(c3) <= rng(c3) * 0.08:
        return {"name": "Doji", "dir": "bull" if is_bull(c3) else "bear", "base": 3}
    return None


STRATEGY_TEXT = {
    "Morning star": "Three-candle reversal confirmed. Enter near the open, trail the stop up as price holds above the pattern low.",
    "Evening star": "Three-candle top confirmed. Short near the open, cover on any close back above the pattern high.",
    "Bullish engulfing": "Momentum reversal off a down-move. Buy on strength through the prior candle's high, exit if it fails back below entry.",
    "Bearish engulfing": "Momentum reversal off an up-move. Short the break of the prior low, cover if price reclaims the entry level.",
    "Hammer": "Rejection of lower prices. Buy on a move back above the hammer's body, invalidated on a close below the wick low.",
    "Hanging man": "Rejection at highs after an advance. Treat as a warning to take profit or short on confirmation.",
    "Shooting star": "Rejection at highs. Short on confirmation below the star's low, cover if price closes back above the high.",
    "Inverted hammer": "Early reversal signal. Buy only on next-candle confirmation above the high; skip if unconfirmed.",
    "Bullish marubozu": "Strong one-sided conviction. Buy the continuation, trail stop under the candle's own low.",
    "Bearish marubozu": "Strong one-sided selling. Short the continuation, trail stop above the candle's own high.",
    "Doji": "Indecision candle. Only trade in the direction of the next move that clears today's high or low; size down.",
}


def analyze(epic, name, df):
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if len(df) < 16:
        return None
    candles = [
        {
            "date": idx,
            "open": r.Open,
            "high": r.High,
            "low": r.Low,
            "close": r.Close,
            "volume": r.Volume,
        }
        for idx, r in df.iterrows()
    ]
    pattern = detect_pattern(candles)
    if not pattern:
        return None

    closes = [c["close"] for c in candles]
    vols = [c["volume"] for c in candles]
    r = rsi(closes, 14)
    s20 = sma(closes, min(20, len(closes) - 1))
    avg_vol = sma(vols[:-1], min(19, len(vols) - 2)) if len(vols) > 2 else None
    last_vol = vols[-1]
    last = candles[-1]
    bull = pattern["dir"] == "bull"

    # Research-backed live ranking:
    #   1) pattern quality is the primary factor;
    #   2) volume is the strongest secondary ranking input;
    #   3) SMA20 alignment receives a modest bonus;
    #   4) RSI is retained for display only and does not alter the score.
    pattern_base = float(pattern["base"])
    strong_pattern = pattern_base >= 7.0

    trend_aligned = bool(
        s20 is not None
        and ((bull and last["close"] > s20) or ((not bull) and last["close"] < s20))
    )

    vol_ratio = (last_vol / avg_vol) if avg_vol else 1.0

    raw_score = pattern_base
    if strong_pattern:
        raw_score += 3.0

    if vol_ratio >= 2.0:
        raw_score += 3.0
    elif vol_ratio >= 1.5:
        raw_score += 1.0

    if trend_aligned:
        raw_score += 1.0

    # Maximum possible raw score is 16 (9 + 3 + 3 + 1).
    # Convert to the existing 0-10 display scale without changing ranking.
    score = max(0.0, min(10.0, raw_score * 10.0 / 16.0))

    # These values remain in pence because all Yahoo .L OHLC data is in pence.
    pattern_low = min(candles[-1]["low"], candles[-2]["low"])
    pattern_high = max(candles[-1]["high"], candles[-2]["high"])
    entry = last["close"]
    stop = pattern_low * 0.997 if bull else pattern_high * 1.003
    risk_per_share = abs(entry - stop)
    target = entry + risk_per_share * 2 if bull else entry - risk_per_share * 2

    return {
        "epic": epic,
        "name": name,
        "score": score,
        "pattern": pattern["name"],
        "pattern_base": pattern_base,
        "strong_pattern": strong_pattern,
        "direction": "Long" if bull else "Short",
        "rsi": r,
        "sma20": s20,
        "trend_aligned": trend_aligned,
        "vol_ratio": vol_ratio,
        "entry": entry,
        "stop": stop,
        "target": target,
        "risk_per_share": risk_per_share,
        "strategy": STRATEGY_TEXT.get(
            pattern["name"], "Trade only on confirmed follow-through."
        ),
        "candles": candles[-10:],
    }


def render_html(results, capital, risk_pct, generated_at):
    risk_amount = capital * risk_pct / 100

    def candle_svg(candles):
        w, h, pad = 220, 60, 4
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        top, bot = max(highs), min(lows)
        span = (top - bot) or 1
        cw = (w - pad * 2) / len(candles)

        def y(v):
            return h - pad - ((v - bot) / span) * (h - pad * 2)

        parts = [
            f'<svg width="100%" viewBox="0 0 {w} {h}" '
            f'preserveAspectRatio="xMidYMid meet">'
        ]
        for i, c in enumerate(candles):
            cx = pad + i * cw + cw / 2
            up = c["close"] >= c["open"]
            color = BULL if up else BEAR
            body_top = y(max(c["open"], c["close"]))
            body_bot = y(min(c["open"], c["close"]))
            parts.append(
                f'<line x1="{cx:.1f}" x2="{cx:.1f}" '
                f'y1="{y(c["high"]):.1f}" y2="{y(c["low"]):.1f}" '
                f'stroke="{color}" stroke-width="1"/>'
                f'<rect x="{cx - cw * 0.32:.1f}" y="{body_top:.1f}" '
                f'width="{cw * 0.64:.1f}" '
                f'height="{max(1.4, body_bot - body_top):.1f}" fill="{color}"/>'
            )
        parts.append("</svg>")
        return "".join(parts)

    rows = []
    for i, r in enumerate(results):
        shares = calculate_position_size(
            capital=capital,
            risk_amount=risk_amount,
            entry_gbx=r["entry"],
            risk_per_share_gbx=r["risk_per_share"],
        )

        entry_gbp = gbx_to_gbp(r["entry"])
        stop_gbp = gbx_to_gbp(r["stop"])
        target_gbp = gbx_to_gbp(r["target"])
        risk_per_share_gbp = gbx_to_gbp(r["risk_per_share"])
        position_value = shares * entry_gbp
        planned_risk = shares * risk_per_share_gbp
        dir_color = BULL if r["direction"] == "Long" else BEAR

        rows.append(f"""
        <div style="background:{PANEL};border:1px solid {HAIRLINE};border-radius:8px;margin-bottom:12px;overflow:hidden;">
          <div style="display:flex;padding:16px 18px;gap:16px;align-items:center;flex-wrap:wrap;">
            <div style="font-family:'Fraunces',serif;font-size:22px;color:{BRASS};width:26px;text-align:center;">{i+1}</div>
            <div style="width:130px;min-width:110px;">{candle_svg(r['candles'])}</div>
            <div style="flex:1;min-width:160px;">
              <div><span style="font-family:'IBM Plex Mono',monospace;font-weight:500;font-size:15px;">{r['epic']}</span>
              <span style="font-size:12px;color:{MUTED};"> {r['name']}</span></div>
              <div style="font-size:12px;color:{dir_color};margin-top:4px;">{r['direction']} &middot; {r['pattern']}</div>
            </div>
            <div style="text-align:right;">
              <div style="font-family:'Fraunces',serif;font-size:24px;font-weight:600;color:{BRASS};">{r['score']:.1f}</div>
              <div style="font-size:10px;color:{MUTED};text-transform:uppercase;">/ 10</div>
            </div>
          </div>
          <div style="border-top:1px solid {HAIRLINE};padding:14px 18px;">
            <p style="font-size:13.5px;line-height:1.6;color:{PAPER};margin:0 0 12px;">{r['strategy']}</p>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:12px;">
              <div><div style="font-size:11px;color:{MUTED};">Reference close (not entry)</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:15px;">£{entry_gbp:.2f}</div></div>
              <div><div style="font-size:11px;color:{MUTED};">Provisional stop</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:15px;color:{BEAR};">£{stop_gbp:.2f}</div></div>
              <div><div style="font-size:11px;color:{MUTED};">Reference 2R target</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:15px;color:{BULL};">£{target_gbp:.2f}</div></div>
            </div>
            <div style="background:{INK};border:1px solid {HAIRLINE};border-radius:6px;padding:12px 14px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;">
              <div><div style="font-size:11px;color:{MUTED};">Indicative maximum size</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:14px;">{shares} shares &middot; ~£{position_value:.2f}</div>
                <div style="font-size:10px;color:{MUTED};margin-top:3px;">At reference close; recalculate from actual entry. Risk ~£{planned_risk:.2f}</div></div>
              <div style="text-align:right;"><div style="font-size:11px;color:{MUTED};">RSI(14) / vs SMA20 / volume</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:14px;">
                  {f"{r['rsi']:.0f}" if r['rsi'] is not None else "—"} /
                  {"above" if (r['sma20'] is not None and r['entry'] > r['sma20']) else ("below" if r['sma20'] is not None else "—")} /
                  {r['vol_ratio']:.1f}x
                </div></div>
            </div>
          </div>
        </div>""")

    if not results:
        rows_html = (
            f'<div style="color:{MUTED};font-size:14px;padding:20px;'
            f'text-align:center;">No qualifying candlestick setups found today.</div>'
        )
    else:
        rows_html = "".join(rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FTSE 250 &middot; Next-session watchlist</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
body {{ background:{INK}; color:{PAPER}; font-family:'Inter',sans-serif; margin:0; padding:0; }}
.wrap {{ max-width:760px; margin:0 auto; padding:32px 20px 60px; box-sizing:border-box; }}
.site-nav {{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:18px; }}
.site-nav a {{ text-decoration:none; border:1px solid {HAIRLINE}; border-radius:999px; padding:8px 12px; color:{PAPER}; font-size:13px; }}
.site-nav a.active {{ border-color:{BRASS}; color:{BRASS}; }}
</style>
</head>
<body>
<div class="wrap">
  <nav class="site-nav"><a class="active" href="index.html">Daily screener</a><a href="backtest.html">Backtest</a><a href="intraday.html">Intraday</a></nav>
  <div style="display:flex;justify-content:space-between;align-items:baseline;border-bottom:1px solid {HAIRLINE};padding-bottom:18px;margin-bottom:24px;flex-wrap:wrap;gap:8px;">
    <div>
      <div style="font-size:11px;letter-spacing:.12em;color:{SALMON};text-transform:uppercase;margin-bottom:4px;">FTSE 250 &middot; Daily-candle watchlist</div>
      <h1 style="font-family:'Fraunces',serif;font-size:28px;font-weight:600;margin:0;">Next-session watchlist</h1>
    </div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:{MUTED};text-align:right;">
      <div>{generated_at}</div>
      <div style="margin-top:6px;"><a href="backtest.html" style="color:{BRASS};text-decoration:none;">View backtest analysis</a></div>
    </div>
  </div>
  <div style="font-size:12px;color:{MUTED};margin-bottom:14px;">
    Capital £{capital:,.2f} &middot; maximum risk £{risk_amount:.2f} per trade ({risk_pct:g}%) &middot; screened from completed daily candles.
  </div>
  <div style="background:{PANEL};border:1px solid {HAIRLINE};border-radius:8px;padding:14px 16px;margin-bottom:22px;font-size:12px;line-height:1.6;color:{PAPER};">
    <strong style="color:{SALMON};">Use as a watchlist, not an automatic order.</strong><br>
    Before entering, confirm the next-session price has not opened beyond the stop, check the opening gap and liquidity, and require intraday confirmation such as a 5- or 15-minute close in the signal direction. Recalculate position size and the 2R target from the actual entry price.
  </div>
  {rows_html}
  <p style="font-size:11.5px;color:{MUTED};line-height:1.6;margin-top:24px;border-top:1px solid {HAIRLINE};padding-top:16px;">
    Generated automatically from completed daily public-market data. London-listed Yahoo Finance prices are converted from pence to pounds for display.
    The reference close, provisional stop, target and size are planning aids only; actual entry, target and size must be recalculated from the next-session execution price.
    This is informational and educational only, not financial advice. Day trading carries a high risk of loss. Verify prices and liquidity independently.
  </p>
</div>
</body>
</html>"""


def main():
    print("Fetching FTSE 250 constituent list...")
    constituents = fetch_ftse250_constituents()
    print(f"Found {len(constituents)} constituents.")

    epics = list(constituents.keys())
    yahoo_tickers = [epic_to_yahoo(e) for e in epics]
    ticker_to_epic = dict(zip(yahoo_tickers, epics))

    print("Downloading price history (this can take a minute)...")
    data = yf.download(
        yahoo_tickers,
        period="2mo",
        interval="1d",
        group_by="ticker",
        threads=True,
        progress=False,
        auto_adjust=False,
    )

    results = []
    for yt in yahoo_tickers:
        try:
            df = data[yt] if isinstance(data.columns, pd.MultiIndex) else data
        except Exception:
            continue
        if df is None or df.empty:
            continue
        epic = ticker_to_epic[yt]
        name = constituents[epic]
        try:
            r = analyze(epic, name, df)
        except Exception as exc:
            print(f"  skipping {epic}: {exc}")
            continue
        if r:
            results.append(r)

    results.sort(key=lambda x: x["score"], reverse=True)
    top5 = results[:5]

    print(f"Scored {len(results)} qualifying setups; writing top {len(top5)}.")

    generated_at = dt.datetime.now(dt.timezone.utc).strftime(
        "%a %d %b %Y, %H:%M UTC"
    )
    report_html = render_html(top5, CAPITAL, RISK_PCT, generated_at)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        output_file.write(report_html)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
