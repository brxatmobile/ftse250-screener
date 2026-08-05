# FILE_VERSION: FTSE350_EARLY_LONG_SHORT_POOL_BLANK_ENV_SAFE_2026_08_05
"""
FTSE 350 ex-investment-trust daily candlestick screener.

- Scrapes the current FTSE 250 constituent list from the London Stock Exchange
  site (so the universe stays current without hardcoding tickers).
- Pulls ~2 months of daily OHLCV for every constituent via Yahoo Finance.
- Scores each stock on candlestick pattern + RSI(14) + trend (SMA20) + volume.
- Picks the top 5 and writes a self-contained HTML report to docs/index.html.

Run manually with:  python screener.py
Configure capital / risk % via CAPITAL and RISK_PCT below, or environment
variables CAPITAL and RISK_PCT (used by the GitHub Actions workflow).

Yahoo Finance returns London Stock Exchange equity prices in pence. Pattern
analysis remains in pence, but monetary display, position sizing and position
value calculations are converted to pounds.
"""

import os
import re
import sys
import math
import json
import datetime as dt
import html as html_lib
import random
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
import numpy as np
import yfinance as yf

CAPITAL = env_float("CAPITAL", 5000)
RISK_PCT = env_float("RISK_PCT", 1)
INTRADAY_POOL_SIZE = env_int("INTRADAY_POOL_SIZE", 80)
MIN_DAILY_TURNOVER_GBP = env_float("MIN_DAILY_TURNOVER_GBP", 2000000)
MIN_SHARE_PRICE_GBP = env_float("MIN_SHARE_PRICE_GBP", 1.00)
MIN_ATR_PCT = env_float("MIN_ATR_PCT", 1.0)
MAX_ATR_PCT = env_float("MAX_ATR_PCT", 6.0)
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "docs", "index.html")
LSE_URL = "https://www.lse.co.uk/indices/ftse-350/constituents.html"
LSE_HOME_URL = "https://www.lse.co.uk/"
LSE_INDICES_URL = "https://www.lse.co.uk/indices/"

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


def _make_lse_session():
    """Create a retrying session with normal browser request headers."""
    session = requests.Session()

    retry = Retry(
        total=4,
        connect=4,
        read=4,
        status=4,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/150.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8"
        ),
        "Accept-Language": "en-GB,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
    })
    return session


def _clean_company_name(value):
    """Remove HTML tags/entities and tidy whitespace from a company name."""
    value = re.sub(r"<[^>]+>", " ", value)
    value = html_lib.unescape(value)
    value = re.sub(r"\s+", " ", value).strip(" \t\r\n-–|")
    return value


def _parse_lse_constituents(page_html):
    """
    Parse ticker -> name pairs from LSE HTML.

    Several patterns are supported because LSE has used slightly different
    link capitalisation, attribute order and markup over time.
    """
    seen = {}

    anchor_pattern = re.compile(
        r"""<a\b[^>]*href=["'][^"']*
            (?:SharePrice\.html|share-prices/[^"'?]+)
            [^"']*[?&](?:amp;)?shareprice=([A-Z0-9.]{1,10})
            [^"']*["'][^>]*>
            (.*?)
            </a>""",
        re.IGNORECASE | re.DOTALL | re.VERBOSE,
    )

    for match in anchor_pattern.finditer(page_html):
        epic = html_lib.unescape(match.group(1)).upper().strip().rstrip(".")
        label = _clean_company_name(match.group(2))
        label = re.sub(
            rf"\s*\(\s*{re.escape(epic)}\.?\s*\)\s*$",
            "",
            label,
            flags=re.IGNORECASE,
        ).strip()

        if epic and label and epic not in seen:
            seen[epic] = label

    if len(seen) < 100:
        display_pattern = re.compile(
            r"""<a\b[^>]*href=["'][^"']*(?:SharePrice\.html|share-prices/)[^"']*["'][^>]*>
                (.*?)
                \(\s*([A-Z0-9.]{1,10})\.?\s*\)
                \s*</a>""",
            re.IGNORECASE | re.DOTALL | re.VERBOSE,
        )
        for match in display_pattern.finditer(page_html):
            name = _clean_company_name(match.group(1))
            epic = html_lib.unescape(match.group(2)).upper().strip().rstrip(".")
            if epic and name and epic not in seen:
                seen[epic] = name

    if len(seen) < 100:
        decoded = html_lib.unescape(page_html).replace("\\u0026", "&").replace("\\/", "/")
        query_pattern = re.compile(
            r"""shareprice=([A-Z0-9.]{1,10})
                (?:&|&amp;)share=[^"'<>\\\s]+
                [^>]{0,500}>
                \s*([^<>{}\[\]]{2,120}?)
                \s*\(\s*\1\.?\s*\)""",
            re.IGNORECASE | re.DOTALL | re.VERBOSE,
        )
        for match in query_pattern.finditer(decoded):
            epic = match.group(1).upper().strip().rstrip(".")
            name = _clean_company_name(match.group(2))
            if epic and name and epic not in seen:
                seen[epic] = name

    return seen


TRUST_NAME_TERMS = (
    "INVESTMENT TRUST", "INVESTMENT COMPANY", "CAPITAL GEARING",
    "GLOBAL SMALLER COMPANIES", "WORLDWIDE HEALTHCARE TRUST",
    "SCOTTISH MORTGAGE", "ALLIANCE WITAN", "TEMPLE BAR",
    "F&C INVESTMENT TRUST", "POLAR CAPITAL GLOBAL FINANCIALS TRUST",
    "INFRASTRUCTURE PLC", "RENEWABLES INFRASTRUCTURE",
)
FUND_NAME_TERMS = (
    " ETF", "FUND PLC", "INCOME FUND", "PROPERTY INCOME",
    "REIT PLC", "REAL ESTATE INVESTMENT TRUST",
)

def is_excluded_collective(name):
    """Exclude investment trusts, listed funds, ETFs and REIT-style vehicles."""
    upper = f" {str(name).upper()} "
    return (" TRUST " in upper or " ETF " in upper or " REIT " in upper or any(term in upper for term in TRUST_NAME_TERMS + FUND_NAME_TERMS))


def fetch_ftse250_constituents():
    """
    Scrape ticker -> name pairs from the LSE FTSE 250 constituents page.

    The request first visits the LSE home and indices pages so the session
    receives ordinary site cookies before requesting the constituent page.
    """
    session = _make_lse_session()

    try:
        for warmup_url, referer in (
            (LSE_HOME_URL, None),
            (LSE_INDICES_URL, LSE_HOME_URL),
        ):
            warmup_headers = {}
            if referer:
                warmup_headers["Referer"] = referer

            try:
                warmup = session.get(
                    warmup_url,
                    headers=warmup_headers,
                    timeout=(15, 30),
                    allow_redirects=True,
                )
                print(
                    f"  LSE warm-up: {warmup.status_code} "
                    f"{warmup.url} ({len(warmup.content):,} bytes)"
                )
            except requests.RequestException as exc:
                print(f"  LSE warm-up warning: {exc}")

            time.sleep(random.uniform(0.6, 1.2))

        response = session.get(
            LSE_URL,
            headers={"Referer": LSE_INDICES_URL},
            timeout=(20, 60),
            allow_redirects=True,
        )

        print(
            f"  LSE constituents response: HTTP {response.status_code}; "
            f"final URL: {response.url}; {len(response.content):,} bytes"
        )

        if response.status_code == 403:
            server = response.headers.get("server", "unknown")
            request_id = (
                response.headers.get("cf-ray")
                or response.headers.get("x-request-id")
                or "not supplied"
            )
            raise RuntimeError(
                "LSE returned HTTP 403 Forbidden even after browser-style "
                "headers, cookies and a normal navigation sequence. "
                f"Server={server}; request-id={request_id}. This normally means "
                "the LSE site is blocking the GitHub-hosted runner's IP address, "
                "rather than a parsing problem in screener.py."
            )

        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        content_encoding = response.headers.get("content-encoding", "").lower()

        if content_encoding not in ("", "identity", "gzip", "deflate"):
            raise RuntimeError(
                "LSE returned an unsupported compressed response: "
                f"content-encoding={content_encoding!r}. "
                "The request should only advertise gzip and deflate."
            )

        if not response.encoding:
            response.encoding = response.apparent_encoding or "utf-8"

        page_html = response.text

        if "html" not in content_type and not page_html.lstrip().startswith("<"):
            raise RuntimeError(
                "LSE returned an unexpected response instead of HTML: "
                f"content-type={content_type or 'not supplied'}; "
                f"content-encoding={content_encoding or 'identity'}."
            )
        constituents = _parse_lse_constituents(page_html)

        if len(constituents) < 100:
            page_title = re.search(
                r"<title[^>]*>(.*?)</title>",
                page_html,
                flags=re.IGNORECASE | re.DOTALL,
            )
            title = _clean_company_name(page_title.group(1)) if page_title else "unknown"
            preview = re.sub(r"\s+", " ", _clean_company_name(page_html[:1000]))[:300]

            raise RuntimeError(
                f"Only parsed {len(constituents)} constituents from the LSE page. "
                f"Page title: {title!r}. Response preview: {preview!r}. "
                "The LSE page layout may have changed or an access-check page "
                "may have been returned."
            )

        before_filter = len(constituents)
        constituents = {
            epic: name for epic, name in constituents.items()
            if not is_excluded_collective(name)
        }
        print(
            f"  Parsed {before_filter} FTSE 350 constituents; "
            f"retained {len(constituents)} operating companies after trust/fund exclusions."
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

    score = pattern["base"] * 0.55
    if r is not None:
        if bull and r < 35:
            score += 1.8
        elif bull and r < 55:
            score += 0.8
        elif bull and r > 72:
            score -= 1.6
        elif not bull and r > 65:
            score += 1.8
        elif not bull and r > 45:
            score += 0.8
        elif not bull and r < 28:
            score -= 1.6

    if s20 is not None:
        if bull and last["close"] > s20:
            score += 1.2
        if bull and last["close"] < s20:
            score -= 0.8
        if not bull and last["close"] < s20:
            score += 1.2
        if not bull and last["close"] > s20:
            score -= 0.8

    vol_ratio = (last_vol / avg_vol) if avg_vol else 1.0
    if vol_ratio > 1.5:
        score += 1.3
    elif vol_ratio < 0.7:
        score -= 0.6

    score = max(0.0, min(10.0, score))

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
        "direction": "Long" if bull else "Short",
        "rsi": r,
        "sma20": s20,
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


def build_intraday_candidate(epic, name, df):
    """
    Build a liquid FTSE 350 intraday candidate in either direction.

    The daily candlestick top five remains separate. This pool is designed for
    the 08:22 and 08:37 London opening scans and therefore prioritises:
    liquidity, useful volatility, and a clear recent directional bias.
    """
    frame = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"]).copy()
    if len(frame) < 25:
        return None

    highs = frame["High"].astype(float)
    lows = frame["Low"].astype(float)
    closes = frame["Close"].astype(float)
    volumes = frame["Volume"].fillna(0).astype(float)

    close = float(closes.iloc[-1])
    price_gbp = gbx_to_gbp(close)
    if price_gbp < MIN_SHARE_PRICE_GBP:
        return None

    previous_close = closes.shift(1)
    true_ranges = pd.concat(
        [
            highs - lows,
            (highs - previous_close).abs(),
            (lows - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = float(true_ranges.tail(14).mean())
    atr_pct = (atr14 / close) * 100 if close > 0 else 0.0

    avg_volume_20 = float(volumes.tail(20).mean())
    last_volume = float(volumes.iloc[-1])
    avg_turnover_gbp = avg_volume_20 * price_gbp
    volume_ratio = last_volume / avg_volume_20 if avg_volume_20 > 0 else 0.0

    sma20 = float(closes.tail(20).mean())
    momentum_5d = ((close / float(closes.iloc[-6])) - 1.0) * 100
    momentum_20d = ((close / float(closes.iloc[-21])) - 1.0) * 100
    trend_pct = ((close / sma20) - 1.0) * 100 if sma20 else 0.0

    if avg_turnover_gbp < MIN_DAILY_TURNOVER_GBP:
        return None
    if not (MIN_ATR_PCT <= atr_pct <= MAX_ATR_PCT):
        return None

    long_bias = close >= sma20 and momentum_5d >= 0.5
    short_bias = close <= sma20 and momentum_5d <= -0.5

    if not long_bias and not short_bias:
        return None

    direction = "Long" if long_bias else "Short"
    directional_trend = trend_pct if direction == "Long" else -trend_pct
    directional_momentum_5d = momentum_5d if direction == "Long" else -momentum_5d
    directional_momentum_20d = momentum_20d if direction == "Long" else -momentum_20d

    score = 0.0
    score += min(25.0, max(0.0, 10.0 + directional_trend * 3.0))
    score += min(22.0, max(0.0, 8.0 + directional_momentum_5d * 2.0))
    score += min(13.0, max(0.0, 5.0 + directional_momentum_20d * 0.7))
    score += min(15.0, max(0.0, volume_ratio * 7.5))
    score += min(15.0, max(0.0, atr_pct * 3.0))
    score += min(
        10.0,
        max(0.0, math.log10(max(avg_turnover_gbp, 1.0)) - 5.0) * 5.0,
    )

    if direction == "Long":
        structural_stop = float(lows.tail(3).min()) * 0.997
        risk = max(close - structural_stop, 0.0)
        target = close + 2.0 * risk
    else:
        structural_stop = float(highs.tail(3).max()) * 1.003
        risk = max(structural_stop - close, 0.0)
        target = close - 2.0 * risk

    return {
        "epic": epic,
        "name": name,
        "direction": direction,
        "pattern": f"Liquid {direction.lower()} trend/volatility candidate",
        "score": min(100.0, score),
        "entry": close,
        "stop": structural_stop,
        "target": target,
        "risk_per_share": risk,
        "daily_sma20": sma20,
        "daily_momentum_5d_pct": momentum_5d,
        "daily_momentum_20d_pct": momentum_20d,
        "daily_volume_ratio": volume_ratio,
        "avg_daily_turnover_gbp": avg_turnover_gbp,
        "atr14_pct": atr_pct,
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
              <div><div style="font-size:11px;color:{MUTED};">Entry (last close)</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:15px;">£{entry_gbp:.2f}</div></div>
              <div><div style="font-size:11px;color:{MUTED};">Sell / stop trigger</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:15px;color:{BEAR};">£{stop_gbp:.2f}</div></div>
              <div><div style="font-size:11px;color:{MUTED};">Take-profit target</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:15px;color:{BULL};">£{target_gbp:.2f}</div></div>
            </div>
            <div style="background:{INK};border:1px solid {HAIRLINE};border-radius:6px;padding:12px 14px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;">
              <div><div style="font-size:11px;color:{MUTED};">Suggested size</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:14px;">{shares} shares &middot; ~£{position_value:.2f}</div>
                <div style="font-size:10px;color:{MUTED};margin-top:3px;">Risk at stop ~£{planned_risk:.2f}</div></div>
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
<title>FTSE 350 ex trusts &middot; Today's top five</title>
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
  <nav class="site-nav"><a class="active" href="index.html">Daily screener</a><a href="intraday.html">Intraday</a><a href="backtest.html">Backtest</a><a href="backtest-research.html">Research</a></nav>
  <div style="display:flex;justify-content:space-between;align-items:baseline;border-bottom:1px solid {HAIRLINE};padding-bottom:18px;margin-bottom:24px;flex-wrap:wrap;gap:8px;">
    <div>
      <div style="font-size:11px;letter-spacing:.12em;color:{SALMON};text-transform:uppercase;margin-bottom:4px;">FTSE 350 ex trusts &middot; daily review</div>
      <h1 style="font-family:'Fraunces',serif;font-size:28px;font-weight:600;margin:0;">Today's top five</h1>
    </div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:{MUTED};text-align:right;">{generated_at}</div>
  </div>
  <div style="font-size:12px;color:{MUTED};margin-bottom:22px;">
    Capital £{capital:,.2f} &middot; maximum risk £{risk_amount:.2f} per trade ({risk_pct:g}%) &middot; screened automatically from FTSE 350 operating companies, excluding trusts and funds.
  </div>
  {rows_html}
  <p style="font-size:11.5px;color:{MUTED};line-height:1.6;margin-top:24px;border-top:1px solid {HAIRLINE};padding-top:16px;">
    Generated automatically from public market data. London-listed Yahoo Finance prices are converted from pence to pounds for display and sizing.
    This is informational and educational only, not financial advice. Day trading carries a high risk of loss.
    Verify prices independently before placing any trade.
  </p>
</div>
</body>
</html>"""



def _json_safe(value):
    """Convert numpy/pandas scalars into JSON-safe Python values."""
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def embed_daily_shortlist(report_html, intraday_pool, generated_at_iso):
    """Embed the broad liquid long-candidate pool used by the intraday review."""
    candidates = []
    for rank, row in enumerate(intraday_pool, start=1):
        item = {key: _json_safe(value) for key, value in row.items()}
        item["daily_rank"] = rank
        item["yahoo_ticker"] = epic_to_yahoo(str(item.get("epic", "")))
        candidates.append(item)

    payload = {
        "schema_version": 2,
        "generated_at_utc": generated_at_iso,
        "selection": "liquid_long_trend_volume_pool",
        "candidates": candidates,
    }
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    data = data.replace("</", "<\\/")
    block = f'<script id="daily-shortlist-data" type="application/json">{data}</script>'
    if "</body>" not in report_html:
        raise RuntimeError("Daily HTML has no </body> tag for shortlist data")
    return report_html.replace("</body>", block + "</body>", 1)

def main():
    print("Fetching FTSE 350 constituent list...")
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
    intraday_candidates = []
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
            intraday_candidate = build_intraday_candidate(epic, name, df)
            if intraday_candidate:
                intraday_candidates.append(intraday_candidate)
            r = analyze(epic, name, df)
        except Exception as exc:
            print(f"  skipping {epic}: {exc}")
            continue
        if r:
            results.append(r)

    results.sort(key=lambda x: x["score"], reverse=True)
    top5 = results[:5]
    intraday_candidates.sort(key=lambda x: x["score"], reverse=True)
    intraday_pool = intraday_candidates[:INTRADAY_POOL_SIZE]

    print(
        f"Scored {len(results)} candlestick setups; writing top {len(top5)}. "
        f"Saved {len(intraday_pool)} liquid long candidates for intraday review."
    )

    generated_dt = dt.datetime.now(dt.timezone.utc)
    generated_at = generated_dt.strftime("%a %d %b %Y, %H:%M UTC")
    report_html = render_html(top5, CAPITAL, RISK_PCT, generated_at)
    report_html = embed_daily_shortlist(
        report_html, intraday_pool, generated_dt.isoformat()
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        output_file.write(report_html)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
