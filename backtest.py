"""
Backtest the FTSE 250 screener against real historical data.

Reuses the exact same pattern-detection / scoring / stop-target logic as
screener.py, so the backtest reflects what the live tool would actually have
recommended on a given day.

For each signal day D (a trading day with enough prior history for RSI/SMA):
  - Run the scorer using only data up to and including D's close.
  - Take the top 5 by score -> the next-session watchlist.
  - Entry = the NEXT trading day's opening price, not the signal-day close.
  - The original setup stop is retained and the 2R target is recalculated from
    the actual next-session opening entry.
  - If the market opens beyond the stop, the trade is skipped as untradeable.
  - Outcome is evaluated on the NEXT trading day (D+1), a same-day / day-trade
    exit: if D+1's high reaches the target, count it a win at the target;
    if D+1's low reaches the stop, count it a loss at the stop; if neither,
    exit at D+1's close (flat P&L for that leftover distance).
  - IMPORTANT CAVEAT: daily bars don't tell us whether the high or the low
    came first intraday. If a day's range touches BOTH the stop and the
    target, this script conservatively assumes the STOP was hit first
    (worst case), and flags this in the output so you can see when it
    happened.

Usage:
  python backtest.py --days 75
  python backtest.py --start 2026-07-27 --end 2026-07-29
  Environment variables: CAPITAL, RISK_PCT, SPREAD_BPS,
  COMMISSION_PER_TRADE and BACKTEST_DAYS. If no date range or --days value is
  supplied, BACKTEST_DAYS is used and defaults to 75.
"""

# FILE_VERSION: BACKTEST_HTML_NAV_2026_08_02


import os
import sys
import argparse
import datetime as dt
import html as html_lib

import pandas as pd
import numpy as np
import yfinance as yf

import screener as scr  # reuse fetch_ftse250_constituents, analyze, detect_pattern, etc.

CAPITAL = float(os.environ.get("CAPITAL", "5000"))
RISK_PCT = float(os.environ.get("RISK_PCT", "1"))
SPREAD_BPS = float(os.environ.get("SPREAD_BPS", "20"))
COMMISSION_PER_TRADE = float(os.environ.get("COMMISSION_PER_TRADE", "0"))
BACKTEST_DAYS = int(os.environ.get("BACKTEST_DAYS", "75"))
BACKTEST_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "docs", "backtest.html")


def get_trading_days(index, start=None, end=None, last_n=None):
    days = list(index)
    if start:
        days = [d for d in days if d.date() >= start]
    if end:
        days = [d for d in days if d.date() <= end]
    if last_n:
        # leave room for at least one day AFTER the signal day to evaluate outcome
        days = days[-(last_n + 1):-1] if len(days) > last_n else days[:-1]
    return days


def analyze_as_of(epic, name, df_full, as_of_idx):
    """Run screener.analyze() using only data up to (inclusive of) as_of_idx."""
    window = df_full.iloc[max(0, as_of_idx - 40): as_of_idx + 1]
    return scr.analyze(epic, name, window)


def run_backtest(start=None, end=None, last_n=None, capital=CAPITAL, risk_pct=RISK_PCT):
    print("Fetching FTSE 250 constituent list...")
    constituents = scr.fetch_ftse250_constituents()
    epics = list(constituents.keys())
    yahoo_tickers = [scr.epic_to_yahoo(e) for e in epics]
    ticker_to_epic = dict(zip(yahoo_tickers, epics))

    print(f"Downloading price history for {len(yahoo_tickers)} tickers...")
    data = yf.download(
        yahoo_tickers, period="6mo", interval="1d",
        group_by="ticker", threads=True, progress=False, auto_adjust=False,
    )

    sample_df = None
    for yt in yahoo_tickers:
        try:
            d = data[yt] if isinstance(data.columns, pd.MultiIndex) else data
        except Exception:
            continue
        if d is not None and not d.empty:
            sample_df = d
            break
    if sample_df is None:
        raise RuntimeError("No price data returned at all — check network/tickers.")

    signal_days = get_trading_days(sample_df.index, start=start, end=end, last_n=last_n)
    if not signal_days:
        raise RuntimeError("No signal days selected — check --start/--end/--days.")

    print(f"Signal days: {[d.date().isoformat() for d in signal_days]}")

    risk_amount = capital * risk_pct / 100
    all_trades = []

    for signal_day in signal_days:
        day_results = []
        for yt in yahoo_tickers:
            try:
                df = data[yt] if isinstance(data.columns, pd.MultiIndex) else data
            except Exception:
                continue
            if df is None or df.empty:
                continue

            # Work only with genuine, complete trading rows for this ticker.
            # yfinance's combined calendar can contain rows where one ticker's
            # Open/High/Low/Close are NaN. Using iloc + 1 on those rows was the
            # source of NaN entries, exits and total P/L.
            clean_df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
            if clean_df.empty or signal_day not in clean_df.index:
                continue

            epic = ticker_to_epic[yt]
            name = constituents[epic]
            as_of_idx = clean_df.index.get_loc(signal_day)
            if not isinstance(as_of_idx, (int, np.integer)):
                continue
            try:
                r = analyze_as_of(epic, name, clean_df, int(as_of_idx))
            except Exception:
                continue
            if not r or int(as_of_idx) + 1 >= len(clean_df):
                continue
            r["_df"] = clean_df
            r["_as_of_idx"] = int(as_of_idx)
            day_results.append(r)

        day_results.sort(key=lambda x: x["score"], reverse=True)

        for r in day_results[:5]:
            df = r["_df"]
            next_row = df.iloc[r["_as_of_idx"] + 1]
            next_date = df.index[r["_as_of_idx"] + 1]
            bull = r["direction"] == "Long"

            signal_close = float(r["entry"])
            entry = float(next_row.Open)
            stop = float(r["stop"])
            next_high = float(next_row.High)
            next_low = float(next_row.Low)
            next_close = float(next_row.Close)

            if not all(np.isfinite(v) and v > 0 for v in
                       (signal_close, entry, stop, next_high, next_low, next_close)):
                print(f"  skipping {r['epic']} on {signal_day.date()}: non-finite OHLC value")
                continue

            gap_pct = ((entry / signal_close) - 1.0) * 100.0

            invalid_open = (bull and entry <= stop) or ((not bull) and entry >= stop)
            if invalid_open:
                all_trades.append({
                    "signal_date": signal_day.date().isoformat(),
                    "exit_date": next_date.date().isoformat(),
                    "epic": r["epic"], "name": r["name"], "score": r["score"],
                    "pattern": r["pattern"], "direction": r["direction"],
                    "signal_close": signal_close, "entry": entry, "stop": stop,
                    "target": None, "exit_price": None,
                    "outcome": "skipped — opened beyond stop",
                    "shares": 0, "gross_pnl": 0.0, "costs": 0.0, "pnl": 0.0,
                    "ambiguous": False, "gap_pct": gap_pct, "skipped": True,
                })
                continue

            risk_per_share = abs(entry - stop)
            target = entry + risk_per_share * 2 if bull else entry - risk_per_share * 2

            hit_stop = (next_low <= stop) if bull else (next_high >= stop)
            hit_target = (next_high >= target) if bull else (next_low <= target)
            ambiguous = hit_stop and hit_target

            if ambiguous:
                exit_price, outcome = stop, "stop (ambiguous — target also touched same day)"
            elif hit_stop:
                exit_price, outcome = stop, "stop"
            elif hit_target:
                exit_price, outcome = target, "target"
            else:
                exit_price, outcome = next_close, "closed at session close"

            entry_gbp = entry / 100.0
            risk_per_share_gbp = risk_per_share / 100.0
            risk_sized_shares = int(risk_amount / risk_per_share_gbp) if risk_per_share_gbp > 0 else 0
            affordable_shares = int(capital / entry_gbp) if entry_gbp > 0 else 0
            shares = max(0, min(risk_sized_shares, affordable_shares))

            pnl_per_share_pence = (exit_price - entry) if bull else (entry - exit_price)
            gross_pnl = (pnl_per_share_pence / 100.0) * shares
            spread_cost = ((entry_gbp + (exit_price / 100.0)) * shares) * (SPREAD_BPS / 20000.0)
            costs = spread_cost + (COMMISSION_PER_TRADE if shares > 0 else 0.0)
            pnl = gross_pnl - costs
            if not all(np.isfinite(v) for v in (gross_pnl, costs, pnl)):
                print(f"  skipping {r['epic']} on {signal_day.date()}: non-finite P/L")
                continue

            all_trades.append({
                "signal_date": signal_day.date().isoformat(),
                "exit_date": next_date.date().isoformat(),
                "epic": r["epic"], "name": r["name"], "score": r["score"],
                "pattern": r["pattern"], "direction": r["direction"],
                "signal_close": signal_close, "entry": entry, "stop": stop,
                "target": target, "exit_price": exit_price, "outcome": outcome,
                "shares": shares, "gross_pnl": gross_pnl, "costs": costs,
                "pnl": pnl, "ambiguous": ambiguous, "gap_pct": gap_pct,
                "skipped": False,
            })

    return all_trades


def print_report(trades, capital, risk_pct):
    print("\n" + "=" * 92)
    print(
        f"BACKTEST REPORT  (capital £{capital:,.2f}, risk {risk_pct:g}% per trade, "
        f"spread {SPREAD_BPS:g} bps, commission £{COMMISSION_PER_TRADE:.2f})"
    )
    print("=" * 92)

    by_day = {}
    for t in trades:
        by_day.setdefault(t["signal_date"], []).append(t)

    for day, day_trades in sorted(by_day.items()):
        print(f"\nSignal day: {day}  (entered next session, {day_trades[0]['exit_date']})")
        for t in day_trades:
            if t["skipped"]:
                print(
                    f"  {t['epic']:<6} {t['direction']:<5} {t['pattern']:<20} "
                    f"score {t['score']:.1f}  signal close £{t['signal_close']/100:.2f} "
                    f"-> open £{t['entry']/100:.2f} ({t['gap_pct']:+.2f}%)"
                )
                print(f"         SKIPPED: {t['outcome']}")
                continue

            flag = "  [!] ambiguous — both stop & target touched" if t["ambiguous"] else ""
            print(
                f"  {t['epic']:<6} {t['direction']:<5} {t['pattern']:<20} score {t['score']:.1f}  "
                f"open £{t['entry']/100:.2f} -> exit £{t['exit_price']/100:.2f} "
                f"({t['outcome']}){flag}"
            )
            print(
                f"         gap {t['gap_pct']:+.2f}% | {t['shares']} shares | "
                f"gross £{t['gross_pnl']:+.2f} | costs £{t['costs']:.2f} | "
                f"net £{t['pnl']:+.2f}"
            )

    executed = [t for t in trades if not t["skipped"] and t["shares"] > 0]
    skipped = sum(1 for t in trades if t["skipped"])
    zero_size = sum(1 for t in trades if not t["skipped"] and t["shares"] == 0)
    pnls = [float(t["pnl"]) for t in executed if np.isfinite(t["pnl"])]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    flats = len(pnls) - len(wins) - len(losses)
    total_pnl = sum(pnls)
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0

    equity = capital
    peak = capital
    max_drawdown = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    print("\n" + "-" * 92)
    print(
        f"Signals: {len(trades)} | Executed: {len(executed)} | Skipped: {skipped} | "
        f"Zero-size: {zero_size}"
    )
    print(f"Wins: {len(wins)} | Losses: {len(losses)} | Flat: {flats}")
    print(f"Win rate: {(len(wins)/len(executed)*100 if executed else 0):.1f}%")
    total_gross_pnl = sum(float(t["gross_pnl"]) for t in executed if np.isfinite(t["gross_pnl"]))
    total_costs = sum(float(t["costs"]) for t in executed if np.isfinite(t["costs"]))
    ending_capital = capital + total_pnl
    period_return_pct = (total_pnl / capital * 100.0) if capital else 0.0

    period_start = min((t["signal_date"] for t in trades), default="n/a")
    period_end = max((t["exit_date"] for t in trades), default="n/a")

    print(f"Average win: £{avg_win:+.2f} | Average loss: £{avg_loss:+.2f}")
    print(f"Profit factor: {profit_factor:.2f} | Maximum drawdown: £{max_drawdown:.2f}")
    print("-" * 92)
    print("OVERALL PERIOD P/L STATEMENT")
    print(f"Period analysed: {period_start} to {period_end}")
    print(f"Starting capital: £{capital:,.2f}")
    print(f"Gross trading P/L: £{total_gross_pnl:+,.2f}")
    print(f"Estimated trading costs: £{total_costs:,.2f}")
    print(f"Net trading P/L: £{total_pnl:+,.2f}")
    print(f"Ending capital: £{ending_capital:,.2f}")
    print(f"Return on starting capital: {period_return_pct:+.2f}%")
    print("-" * 92)
    print("\nDaily bars still cannot reveal whether the intraday high or low occurred first.")
    print("When both stop and target are touched, the test assumes the stop was hit first.")


def render_html_report(trades, capital, risk_pct, generated_at):
    """Create a responsive HTML backtest report suitable for GitHub Pages."""
    executed = [t for t in trades if not t["skipped"] and t["shares"] > 0]
    skipped = sum(1 for t in trades if t["skipped"])
    zero_size = sum(1 for t in trades if not t["skipped"] and t["shares"] == 0)
    pnls = [float(t["pnl"]) for t in executed if np.isfinite(t["pnl"])]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    flats = len(pnls) - len(wins) - len(losses)
    total_pnl = sum(pnls)
    total_gross_pnl = sum(float(t["gross_pnl"]) for t in executed if np.isfinite(t["gross_pnl"]))
    total_costs = sum(float(t["costs"]) for t in executed if np.isfinite(t["costs"]))
    ending_capital = capital + total_pnl
    period_return_pct = (total_pnl / capital * 100.0) if capital else 0.0
    period_start = min((t["signal_date"] for t in trades), default="n/a")
    period_end = max((t["exit_date"] for t in trades), default="n/a")
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    equity = capital
    peak = capital
    max_drawdown = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    cards = []
    for t in sorted(trades, key=lambda x: (x["signal_date"], x["epic"]), reverse=True):
        safe_name = html_lib.escape(str(t["name"]))
        safe_pattern = html_lib.escape(str(t["pattern"]))
        safe_outcome = html_lib.escape(str(t["outcome"]))
        if t["skipped"]:
            price_line = (
                f"Signal close £{t['signal_close']/100:.2f} → open £{t['entry']/100:.2f} "
                f"({t['gap_pct']:+.2f}%)"
            )
            result_line = f"Skipped: {safe_outcome}"
            result_class = "neutral"
        else:
            price_line = (
                f"Open £{t['entry']/100:.2f} → exit £{t['exit_price']/100:.2f}"
            )
            result_line = (
                f"{t['shares']} shares · gross £{t['gross_pnl']:+.2f} · "
                f"costs £{t['costs']:.2f} · net £{t['pnl']:+.2f}"
            )
            result_class = "positive" if t["pnl"] > 0 else "negative" if t["pnl"] < 0 else "neutral"

        cards.append(f"""
        <article class="trade-card">
          <div class="trade-head">
            <div><strong>{html_lib.escape(t['epic'])}</strong> <span>{safe_name}</span></div>
            <div class="score">{t['score']:.1f}</div>
          </div>
          <div class="meta">{t['signal_date']} → {t['exit_date']} · {html_lib.escape(t['direction'])} · {safe_pattern}</div>
          <div class="prices">{price_line}</div>
          <div class="outcome">{safe_outcome}</div>
          <div class="{result_class}">{result_line}</div>
        </article>""")

    cards_html = "".join(cards) or '<p class="empty">No trades were produced.</p>'
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FTSE 250 Backtest Analysis</title>
<style>
:root {{ --ink:#12161F; --panel:#1B2129; --line:#2C333D; --brass:#C9A24B; --paper:#ECE7DA; --muted:#8B92A0; --green:#4FAE73; --red:#D1594B; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--ink); color:var(--paper); font-family:Arial,sans-serif; }}
.wrap {{ max-width:900px; margin:auto; padding:22px 14px 50px; }}
a {{ color:var(--brass); }}
.site-nav {{ display:flex;gap:8px;flex-wrap:wrap;margin-bottom:18px; }}
.site-nav a {{ text-decoration:none;border:1px solid var(--line);border-radius:999px;padding:8px 12px;color:var(--paper);font-size:13px; }}
.site-nav a.active {{ border-color:var(--brass);color:var(--brass); }}
h1 {{ margin:6px 0 4px; font-size:28px; }}
.sub {{ color:var(--muted); font-size:13px; margin-bottom:18px; }}
.summary {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:16px 0; }}
.stat {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; }}
.label {{ color:var(--muted); font-size:11px; text-transform:uppercase; }}
.value {{ font-size:19px; margin-top:5px; }}
.statement {{ background:var(--panel); border:1px solid var(--brass); border-radius:9px; padding:16px; margin:18px 0; }}
.statement-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px 18px; }}
.trade-card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; margin:10px 0; }}
.trade-head {{ display:flex; justify-content:space-between; gap:12px; }}
.trade-head span,.meta,.outcome {{ color:var(--muted); }}
.score {{ color:var(--brass); font-weight:bold; }}
.meta,.prices,.outcome {{ font-size:13px; margin-top:7px; }}
.positive {{ color:var(--green); margin-top:8px; }}
.negative {{ color:var(--red); margin-top:8px; }}
.neutral {{ color:var(--muted); margin-top:8px; }}
.note {{ color:var(--muted); font-size:12px; line-height:1.5; margin-top:20px; }}
@media (max-width:650px) {{
  .wrap {{ padding:16px 10px 40px; }}
  h1 {{ font-size:23px; }}
  .summary {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
  .statement-grid {{ grid-template-columns:1fr; }}
  .trade-head {{ align-items:flex-start; }}
}}
</style>
</head>
<body><main class="wrap">
<nav class="site-nav"><a href="index.html">Daily screener</a><a class="active" href="backtest.html">Backtest</a><a href="intraday.html">Intraday</a></nav>
<h1>Backtest analysis</h1>
<div class="sub">Generated {generated_at} · {period_start} to {period_end}</div>
<section class="summary">
  <div class="stat"><div class="label">Executed</div><div class="value">{len(executed)}</div></div>
  <div class="stat"><div class="label">Win rate</div><div class="value">{(len(wins)/len(executed)*100 if executed else 0):.1f}%</div></div>
  <div class="stat"><div class="label">Profit factor</div><div class="value">{profit_factor:.2f}</div></div>
  <div class="stat"><div class="label">Max drawdown</div><div class="value">£{max_drawdown:.2f}</div></div>
</section>
<section class="statement">
<h2>Overall period P/L</h2>
<div class="statement-grid">
<div>Starting capital: <strong>£{capital:,.2f}</strong></div>
<div>Ending capital: <strong>£{ending_capital:,.2f}</strong></div>
<div>Gross P/L: <strong>£{total_gross_pnl:+,.2f}</strong></div>
<div>Estimated costs: <strong>£{total_costs:,.2f}</strong></div>
<div>Net P/L: <strong>£{total_pnl:+,.2f}</strong></div>
<div>Return: <strong>{period_return_pct:+.2f}%</strong></div>
<div>Wins / losses / flat: <strong>{len(wins)} / {len(losses)} / {flats}</strong></div>
<div>Skipped / zero-size: <strong>{skipped} / {zero_size}</strong></div>
</div>
</section>
<h2>Trade detail</h2>
{cards_html}
<p class="note">Daily bars cannot show whether the intraday high or low occurred first. If both stop and target were touched, the test assumes the stop was hit first. This report is informational only.</p>
</main></body></html>"""


def write_html_report(trades, capital, risk_pct):
    generated_at = dt.datetime.now(dt.timezone.utc).strftime("%a %d %b %Y, %H:%M UTC")
    report_html = render_html_report(trades, capital, risk_pct, generated_at)
    os.makedirs(os.path.dirname(BACKTEST_OUTPUT_PATH), exist_ok=True)
    with open(BACKTEST_OUTPUT_PATH, "w", encoding="utf-8") as output_file:
        output_file.write(report_html)
    print(f"Wrote {BACKTEST_OUTPUT_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=None, help="Use the last N trading days as signal days")
    ap.add_argument("--start", type=str, default=None, help="YYYY-MM-DD")
    ap.add_argument("--end", type=str, default=None, help="YYYY-MM-DD")
    args = ap.parse_args()

    start = dt.date.fromisoformat(args.start) if args.start else None
    end = dt.date.fromisoformat(args.end) if args.end else None
    last_n = args.days if not (start or end) else None
    if not (start or end or last_n):
        last_n = BACKTEST_DAYS

    trades = run_backtest(start=start, end=end, last_n=last_n, capital=CAPITAL, risk_pct=RISK_PCT)
    print_report(trades, CAPITAL, RISK_PCT)
    write_html_report(trades, CAPITAL, RISK_PCT)


if __name__ == "__main__":
    sys.exit(main())
