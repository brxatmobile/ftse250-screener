"""
Backtest the FTSE 250 screener against real historical data.

Reuses the exact same pattern-detection / scoring / stop-target logic as
screener.py, so the backtest reflects what the live tool would actually have
recommended on a given day.

For each signal day D (a trading day with enough prior history for RSI/SMA):
  - Run the scorer using only data up to and including D's close.
  - Take the top 5 by score -> that's what the 09:30 review would have shown.
  - "Entry" = D's close (proxy for next-day open).
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
  python backtest.py --days 3
  python backtest.py --start 2026-07-27 --end 2026-07-29
  (env vars CAPITAL / RISK_PCT work the same as screener.py)
"""

import os
import sys
import argparse
import datetime as dt

import pandas as pd
import numpy as np
import yfinance as yf

import screener as scr  # reuse fetch_ftse250_constituents, analyze, detect_pattern, etc.

CAPITAL = float(os.environ.get("CAPITAL", "5000"))
RISK_PCT = float(os.environ.get("RISK_PCT", "1"))


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
        yahoo_tickers, period="4mo", interval="1d",
        group_by="ticker", threads=True, progress=False, auto_adjust=False,
    )

    # Use the first ticker with data to establish the trading-day calendar
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
            if df is None or df.empty or signal_day not in df.index:
                continue
            epic = ticker_to_epic[yt]
            name = constituents[epic]
            as_of_idx = df.index.get_loc(signal_day)
            try:
                r = analyze_as_of(epic, name, df, as_of_idx)
            except Exception:
                continue
            if not r:
                continue
            # need a next trading day to evaluate the outcome
            if as_of_idx + 1 >= len(df):
                continue
            r["_df"] = df
            r["_as_of_idx"] = as_of_idx
            day_results.append(r)

        day_results.sort(key=lambda x: x["score"], reverse=True)
        top5 = day_results[:5]

        for r in top5:
            df = r["_df"]
            next_row = df.iloc[r["_as_of_idx"] + 1]
            next_date = df.index[r["_as_of_idx"] + 1]
            bull = r["direction"] == "Long"
            entry, stop, target = r["entry"], r["stop"], r["target"]
            hit_stop = (next_row.Low <= stop) if bull else (next_row.High >= stop)
            hit_target = (next_row.High >= target) if bull else (next_row.Low <= target)

            ambiguous = hit_stop and hit_target
            if hit_stop and hit_target:
                exit_price, outcome = stop, "stop (ambiguous — target also touched same day)"
            elif hit_stop:
                exit_price, outcome = stop, "stop"
            elif hit_target:
                exit_price, outcome = target, "target"
            else:
                exit_price, outcome = next_row.Close, "closed flat (neither hit)"

            # Yahoo Finance quotes London-listed .L shares in pence (GBp), not pounds.
            # Keep all stop/target comparisons in pence, but convert monetary values
            # to GBP for position sizing and P&L.
            entry_gbp = entry / 100.0
            risk_per_share_gbp = r["risk_per_share"] / 100.0

            risk_sized_shares = (
                int(risk_amount / risk_per_share_gbp)
                if risk_per_share_gbp > 0 else 0
            )
            affordable_shares = int(capital / entry_gbp) if entry_gbp > 0 else 0
            shares = min(risk_sized_shares, affordable_shares)

            pnl_per_share_pence = (exit_price - entry) if bull else (entry - exit_price)
            pnl = (pnl_per_share_pence / 100.0) * shares

            all_trades.append({
                "signal_date": signal_day.date().isoformat(),
                "exit_date": next_date.date().isoformat(),
                "epic": r["epic"], "name": r["name"], "score": r["score"],
                "pattern": r["pattern"], "direction": r["direction"],
                "entry": entry, "stop": stop, "target": target,
                "exit_price": exit_price, "outcome": outcome,
                "shares": shares, "pnl": pnl, "ambiguous": ambiguous,
            })

    return all_trades


def print_report(trades, capital, risk_pct):
    print("\n" + "=" * 78)
    print(f"BACKTEST REPORT  (capital £{capital:,.0f}, risk {risk_pct:g}% per trade)")
    print("=" * 78)
    total_pnl = 0.0
    by_day = {}
    for t in trades:
        by_day.setdefault(t["signal_date"], []).append(t)

    for day, day_trades in sorted(by_day.items()):
        print(f"\nSignal day: {day}  (traded next session, {day_trades[0]['exit_date']})")
        for t in day_trades:
            flag = "  [!] ambiguous — both stop & target touched same day" if t["ambiguous"] else ""
            print(f"  {t['epic']:<6} {t['direction']:<5} {t['pattern']:<20} score {t['score']:.1f}  "
                  f"entry £{t['entry'] / 100:.2f} -> exit £{t['exit_price'] / 100:.2f} "
                  f"({t['outcome']}){flag}")
            print(f"         {t['shares']} shares  ->  P&L £{t['pnl']:+.2f}")
            total_pnl += t["pnl"]

    n = len(trades)
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = sum(1 for t in trades if t["pnl"] < 0)
    flats = n - wins - losses
    print("\n" + "-" * 78)
    print(f"Total trades: {n}   Wins: {wins}   Losses: {losses}   Flat: {flats}")
    print(f"TOTAL P&L across all trades: £{total_pnl:+.2f}")
    print("-" * 78)
    print("\nNote: ambiguous days (stop and target both touched) are scored as a stop")
    print("(worst case) since daily bars can't show which was hit first intraday.")


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
        last_n = 3

    trades = run_backtest(start=start, end=end, last_n=last_n, capital=CAPITAL, risk_pct=RISK_PCT)
    print_report(trades, CAPITAL, RISK_PCT)


if __name__ == "__main__":
    sys.exit(main())
