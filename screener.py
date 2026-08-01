"""
FTSE 250 daily candlestick screener.

- Scrapes the current FTSE 250 constituent list from the London Stock Exchange
  site (so the universe stays current without hardcoding tickers).
- Pulls ~2 months of daily OHLCV for every constituent via Yahoo Finance.
- Scores each stock on candlestick pattern + RSI(14) + trend (SMA20) + volume.
- Picks the top 5 and writes a self-contained HTML report to docs/index.html.

Run manually with:  python screener.py
Configure capital / risk % via CAPITAL and RISK_PCT below, or environment
variables CAPITAL and RISK_PCT (used by the GitHub Actions workflow).
"""

import os
import re
import sys
import math
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

CAPITAL = float(os.environ.get("CAPITAL", "5000"))
RISK_PCT = float(os.environ.get("RISK_PCT", "1"))
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "docs", "index.html")
LSE_URL = "https://www.lse.co.uk/indices/ftse-250/constituents.html"
LSE_HOME_URL = "https://www.lse.co.uk/"
LSE_INDICES_URL = "https://www.lse.co.uk/indices/"

INK = "#12161F"
PANEL = "#1B2129"
HAIRLINE = "#2C333D"
BRASS = "#C9A24B"
SALMON = "#E8A493"
BULL = "#4FAE73"
BEAR = "#D1594B"
PAPER = "#ECE7DA"
MUTED = "#8B92A0"


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

    # Current/typical LSE share links. The link body usually ends "(EPIC)".
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

        # Remove the final ticker in brackets, e.g. "Greggs (GRG)".
        label = re.sub(
            rf"\s*\(\s*{re.escape(epic)}\.?\s*\)\s*$",
            "",
            label,
            flags=re.IGNORECASE,
        ).strip()

        if epic and label and epic not in seen:
            seen[epic] = label

    # Fallback for pages where the ticker is present only in the displayed text.
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

    # Final fallback for embedded JSON/escaped HTML containing shareprice=.
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


def fetch_ftse250_constituents():
    """
    Scrape ticker -> name pairs from the LSE FTSE 250 constituents page.

    The request first visits the LSE home and indices pages so the session
    receives ordinary site cookies before requesting the constituent page.
    """
    session = _make_lse_session()

    try:
        # Establish an ordinary navigation session and collect site cookies.
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
                # A warm-up failure is not fatal; the main request may still work.
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

        # requests transparently decodes gzip and deflate. We deliberately do
        # not advertise Brotli ("br"), because GitHub's Python environment may
        # not have a Brotli decoder installed. Advertising br without support
        # can produce an HTTP 200 response whose body appears as binary noise.
        if content_encoding not in ("", "identity", "gzip", "deflate"):
            raise RuntimeError(
                "LSE returned an unsupported compressed response: "
                f"content-encoding={content_encoding!r}. "
                "The request should only advertise gzip and deflate."
            )

        # Use the server-provided encoding where possible, otherwise default
        # to UTF-8. LSE pages are HTML and should decode to readable markup.
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

        print(f"  Parsed {len(constituents)} LSE constituents.")
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
        return {"name": "Hammer" if is_bull(c3) else "Hanging man",
                "dir": "bull" if is_bull(c3) else "bear", "base": 6.5}
    if body(c3) > 0 and upper_wick(c3) >= body(c3) * 2 and lower_wick(c3) <= body(c3) * 0.35:
        return {"name": "Shooting star" if is_bear(c3) else "Inverted hammer",
                "dir": "bear" if is_bear(c3) else "bull", "base": 6}
    if body(c3) >= rng(c3) * 0.85:
        return {"name": "Bullish marubozu" if is_bull(c3) else "Bearish marubozu",
                "dir": "bull" if is_bull(c3) else "bear", "base": 7}
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
        {"date": idx, "open": r.Open, "high": r.High, "low": r.Low, "close": r.Close, "volume": r.Volume}
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
        if bull and r < 35: score += 1.8
        elif bull and r < 55: score += 0.8
        elif bull and r > 72: score -= 1.6
        elif not bull and r > 65: score += 1.8
        elif not bull and r > 45: score += 0.8
        elif not bull and r < 28: score -= 1.6

    if s20 is not None:
        if bull and last["close"] > s20: score += 1.2
        if bull and last["close"] < s20: score -= 0.8
        if not bull and last["close"] < s20: score += 1.2
        if not bull and last["close"] > s20: score -= 0.8

    vol_ratio = (last_vol / avg_vol) if avg_vol else 1.0
    if vol_ratio > 1.5: score += 1.3
    elif vol_ratio < 0.7: score -= 0.6

    score = max(0.0, min(10.0, score))

    pattern_low = min(candles[-1]["low"], candles[-2]["low"])
    pattern_high = max(candles[-1]["high"], candles[-2]["high"])
    entry = last["close"]
    stop = pattern_low * 0.997 if bull else pattern_high * 1.003
    risk_per_share = abs(entry - stop)
    target = entry + risk_per_share * 2 if bull else entry - risk_per_share * 2

    return {
        "epic": epic, "name": name, "score": score, "pattern": pattern["name"],
        "direction": "Long" if bull else "Short", "rsi": r, "sma20": s20,
        "vol_ratio": vol_ratio, "entry": entry, "stop": stop, "target": target,
        "risk_per_share": risk_per_share,
        "strategy": STRATEGY_TEXT.get(pattern["name"], "Trade only on confirmed follow-through."),
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

        parts = [f'<svg width="100%" viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet">']
        for i, c in enumerate(candles):
            cx = pad + i * cw + cw / 2
            up = c["close"] >= c["open"]
            color = BULL if up else BEAR
            body_top = y(max(c["open"], c["close"]))
            body_bot = y(min(c["open"], c["close"]))
            parts.append(
                f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{y(c["high"]):.1f}" y2="{y(c["low"]):.1f}" '
                f'stroke="{color}" stroke-width="1"/>'
                f'<rect x="{cx - cw * 0.32:.1f}" y="{body_top:.1f}" width="{cw * 0.64:.1f}" '
                f'height="{max(1.4, body_bot - body_top):.1f}" fill="{color}"/>'
            )
        parts.append("</svg>")
        return "".join(parts)

    rows = []
    for i, r in enumerate(results):
        shares = math.floor(risk_amount / r["risk_per_share"]) if r["risk_per_share"] > 0 else 0
        position_value = shares * r["entry"]
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
                <div style="font-family:'IBM Plex Mono',monospace;font-size:15px;">£{r['entry']:.2f}</div></div>
              <div><div style="font-size:11px;color:{MUTED};">Sell / stop trigger</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:15px;color:{BEAR};">£{r['stop']:.2f}</div></div>
              <div><div style="font-size:11px;color:{MUTED};">Take-profit target</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:15px;color:{BULL};">£{r['target']:.2f}</div></div>
            </div>
            <div style="background:{INK};border:1px solid {HAIRLINE};border-radius:6px;padding:12px 14px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;">
              <div><div style="font-size:11px;color:{MUTED};">Suggested size</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:14px;">{shares} shares &middot; ~£{position_value:.2f}</div></div>
              <div style="text-align:right;"><div style="font-size:11px;color:{MUTED};">RSI(14) / vs SMA20 / volume</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:14px;">
                  {f"{r['rsi']:.0f}" if r['rsi'] is not None else "—"} /
                  {"above" if (r['sma20'] is not None and r['entry']>r['sma20']) else ("below" if r['sma20'] is not None else "—")} /
                  {r['vol_ratio']:.1f}x
                </div></div>
            </div>
          </div>
        </div>""")

    if not results:
        rows_html = f'<div style="color:{MUTED};font-size:14px;padding:20px;text-align:center;">No qualifying candlestick setups found today.</div>'
    else:
        rows_html = "".join(rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FTSE 250 &middot; Today's top five</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
body {{ background:{INK}; color:{PAPER}; font-family:'Inter',sans-serif; margin:0; padding:0; }}
.wrap {{ max-width:760px; margin:0 auto; padding:32px 20px 60px; box-sizing:border-box; }}
</style>
</head>
<body>
<div class="wrap">
  <div style="display:flex;justify-content:space-between;align-items:baseline;border-bottom:1px solid {HAIRLINE};padding-bottom:18px;margin-bottom:24px;flex-wrap:wrap;gap:8px;">
    <div>
      <div style="font-size:11px;letter-spacing:.12em;color:{SALMON};text-transform:uppercase;margin-bottom:4px;">FTSE 250 &middot; 09:30 review</div>
      <h1 style="font-family:'Fraunces',serif;font-size:28px;font-weight:600;margin:0;">Today's top five</h1>
    </div>
    <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;color:{MUTED};text-align:right;">{generated_at}</div>
  </div>
  <div style="font-size:12px;color:{MUTED};margin-bottom:22px;">
    Capital £{capital:,.0f} &middot; risking £{risk_amount:.2f} per trade ({risk_pct:g}%) &middot; screened automatically each morning from the full FTSE 250.
  </div>
  {rows_html}
  <p style="font-size:11.5px;color:{MUTED};line-height:1.6;margin-top:24px;border-top:1px solid {HAIRLINE};padding-top:16px;">
    Generated automatically from public market data. This is informational and educational only, not financial advice.
    Day trading carries a high risk of loss. Verify prices independently before placing any trade.
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
        yahoo_tickers, period="2mo", interval="1d",
        group_by="ticker", threads=True, progress=False, auto_adjust=False,
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
        except Exception as e:
            print(f"  skipping {epic}: {e}")
            continue
        if r:
            results.append(r)

    results.sort(key=lambda x: x["score"], reverse=True)
    top5 = results[:5]

    print(f"Scored {len(results)} qualifying setups; writing top {len(top5)}.")

    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%a %d %b %Y, %H:%M UTC")
    html = render_html(top5, CAPITAL, RISK_PCT, generated_at)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
