"""
FTSE 250 daily candlestick screener.

- Scrapes the current FTSE 250 constituent list from the London Stock Exchange
  site (so the universe stays current without hardcoding tickers).
- Pulls ~2 months of daily OHLCV for every constituent via Yahoo Finance.
- Scores each stock on candlestick pattern + RSI(14) + trend (SMA20) + volume.
- Picks the top 5 as a next-session watchlist and writes a self-contained HTML report to docs/index.html.

Run manually with:  python screener.py
Configure capital / risk % via CAPITAL and RISK_PCT below, or environment
variables CAPITAL and RISK_PCT (used by the GitHub Actions workflow).

Yahoo Finance returns London Stock Exchange equity prices in pence. Pattern
analysis remains in pence, but monetary display and indicative sizing are
converted to pounds. The completed daily candle is used only to build a
next-session watchlist; it is not treated as an executable entry price.
"""

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
CONSTITUENTS_URL = "https://en.wikipedia.org/wiki/FTSE_250_Index"

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


def _normalise_column_name(column):
    """Flatten and normalise a pandas HTML-table column heading."""
    if isinstance(column, tuple):
        text = " ".join(str(part) for part in column if str(part) != "nan")
    else:
        text = str(column)
    return re.sub(r"\s+", " ", text).strip().lower()


class _WikipediaTableParser(HTMLParser):
    """Collect HTML table text using only Python's standard library."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables = []
        self._table_depth = 0
        self._current_table = None
        self._current_row = None
        self._current_cell = None

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._current_table = []
        elif self._table_depth == 1 and tag == "tr":
            self._current_row = []
        elif self._table_depth == 1 and tag in {"th", "td"}:
            self._current_cell = []

    def handle_data(self, data):
        if self._table_depth == 1 and self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if self._table_depth == 1 and tag in {"th", "td"}:
            if self._current_row is not None and self._current_cell is not None:
                text = re.sub(r"\s+", " ", "".join(self._current_cell)).strip()
                self._current_row.append(text)
            self._current_cell = None
        elif self._table_depth == 1 and tag == "tr":
            if self._current_table is not None and self._current_row:
                self._current_table.append(self._current_row)
            self._current_row = None
        elif tag == "table":
            if self._table_depth == 1 and self._current_table:
                self.tables.append(self._current_table)
                self._current_table = None
            if self._table_depth > 0:
                self._table_depth -= 1


def _normalise_header(value):
    value = re.sub(r"\[[^]]*]", "", str(value))
    return re.sub(r"\s+", " ", value).strip().lower()


def _parse_ftse250_wikipedia(page_html):
    """Parse the FTSE 250 constituent table without optional dependencies.

    This intentionally avoids ``pandas.read_html`` because that requires an
    optional parser such as lxml. Only tables with company and ticker headers
    and roughly 250 constituent rows are considered. The function fails closed
    if the table cannot be identified uniquely.
    """
    parser = _WikipediaTableParser()
    try:
        parser.feed(page_html)
        parser.close()
    except Exception as exc:
        raise RuntimeError(f"Could not parse FTSE 250 constituent HTML: {exc}") from exc

    candidates = []
    for rows in parser.tables:
        if not rows:
            continue

        header_index = None
        company_index = None
        ticker_index = None

        # Wikipedia can use one or more heading rows. Inspect the first few.
        for row_index, row in enumerate(rows[:5]):
            headers = [_normalise_header(cell) for cell in row]
            company_index = next(
                (i for i, h in enumerate(headers) if h in {"company", "constituent", "name"}),
                None,
            )
            ticker_index = next(
                (i for i, h in enumerate(headers) if h in {"ticker", "ticker symbol", "epic"}),
                None,
            )
            if company_index is not None and ticker_index is not None:
                header_index = row_index
                break

        if header_index is None:
            continue

        constituents = {}
        for row in rows[header_index + 1:]:
            if max(company_index, ticker_index) >= len(row):
                continue

            raw_ticker = re.sub(r"\[[^]]*]", "", row[ticker_index])
            raw_name = re.sub(r"\[[^]]*]", "", row[company_index])
            ticker = html_lib.unescape(raw_ticker).strip().upper().rstrip(".")
            name = _clean_company_name(raw_name)

            if not re.fullmatch(r"[A-Z0-9.]{1,10}", ticker):
                continue
            if not name:
                continue
            constituents[ticker] = name

        if 240 <= len(constituents) <= 260:
            candidates.append(constituents)

    if len(candidates) != 1:
        counts = [len(candidate) for candidate in candidates]
        raise RuntimeError(
            "Could not uniquely identify the FTSE 250 constituents table. "
            f"Plausible table sizes: {counts}. Refusing to publish an unverified universe."
        )

    constituents = candidates[0]
    print(f"  Parsed {len(constituents)} verified FTSE 250 constituents.")
    return constituents


def fetch_ftse250_constituents():
    """Return a verified ticker-to-company mapping for the FTSE 250."""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; FTSE250Screener/1.0)",
        "Accept-Language": "en-GB,en;q=0.9",
    }
    response = requests.get(CONSTITUENTS_URL, headers=headers, timeout=(20, 60))
    response.raise_for_status()
    print(
        f"  Constituents response: HTTP {response.status_code}; "
        f"{len(response.content):,} bytes"
    )
    return _parse_ftse250_wikipedia(response.text)


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
</style>
</head>
<body>
<div class="wrap">
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
