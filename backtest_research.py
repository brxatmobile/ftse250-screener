"""
Research backtest for the FTSE 250 screener.

Purpose
-------
Compare the current live score with fixed alternative weightings for the factors
already measured by screener.py: candlestick pattern, RSI, SMA20 trend alignment
and volume ratio. The script uses the first 70% of dates as the research period
and reports performance separately on the final 30% holdout period.

It writes docs/backtest-research.html. It does not modify screener.py or the live scoring.
"""

# FILE_VERSION: BACKTEST_RESEARCH_LONG_ONLY_RANK1_OUTPUT_FIX_2026_08_02

import argparse
import datetime as dt
import html as html_lib
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf

import screener as scr


def env_float(name, default):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return float(default)
    return float(raw.strip())


def env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return int(default)
    return int(raw.strip())


CAPITAL = env_float("CAPITAL", 5000)
RISK_PCT = env_float("RISK_PCT", 1)
SPREAD_BPS = env_float("SPREAD_BPS", 20)
COMMISSION_PER_TRADE = env_float("COMMISSION_PER_TRADE", 0)
BACKTEST_DAYS = env_int("BACKTEST_DAYS", 75)
OUTPUT_PATH = os.path.abspath(os.path.join(os.getcwd(), "docs", "backtest-research.html"))

PATTERN_BASE = {
    "Morning star": 9.0,
    "Evening star": 9.0,
    "Bullish engulfing": 8.0,
    "Bearish engulfing": 8.0,
    "Hammer": 6.5,
    "Hanging man": 6.5,
    "Shooting star": 6.0,
    "Inverted hammer": 6.0,
    "Bullish marubozu": 7.0,
    "Bearish marubozu": 7.0,
    "Doji": 3.0,
}


def get_trading_days(index, start=None, end=None, last_n=None):
    days = list(index)
    if start:
        days = [d for d in days if d.date() >= start]
    if end:
        days = [d for d in days if d.date() <= end]
    if last_n:
        days = days[-(last_n + 1):-1] if len(days) > last_n else days[:-1]
    return days


def analyze_as_of(epic, name, df_full, as_of_idx):
    window = df_full.iloc[max(0, as_of_idx - 40): as_of_idx + 1]
    return scr.analyze(epic, name, window)


def factor_values(r):
    bull = r["direction"] == "Long"
    entry = float(r["entry"])
    sma20 = float(r["sma20"]) if r.get("sma20") is not None else None
    rsi = float(r["rsi"]) if r.get("rsi") is not None else None
    vol = float(r.get("vol_ratio") or 1.0)

    trend_aligned = bool(sma20 is not None and ((bull and entry > sma20) or ((not bull) and entry < sma20)))
    sma_distance_pct = ((entry / sma20) - 1.0) * 100.0 if sma20 and sma20 > 0 else 0.0
    directional_sma_distance = sma_distance_pct if bull else -sma_distance_pct

    rsi_supportive = False
    rsi_extreme = False
    if rsi is not None:
        rsi_supportive = (bull and rsi < 55) or ((not bull) and rsi > 45)
        rsi_extreme = (bull and rsi < 35) or ((not bull) and rsi > 65)

    return {
        "pattern_base": PATTERN_BASE.get(r["pattern"], 0.0),
        "rsi_supportive": rsi_supportive,
        "rsi_extreme": rsi_extreme,
        "trend_aligned": trend_aligned,
        "directional_sma_distance": directional_sma_distance,
        "high_volume": vol >= 1.5,
        "very_high_volume": vol >= 2.0,
        "vol_ratio": vol,
    }


def model_scores(r):
    f = factor_values(r)
    baseline = float(r["score"])

    # Fixed alternatives. They deliberately alter only ranking, not trade exits.
    return {
        "Current score": baseline,
        "Pattern weighted": baseline + 0.45 * f["pattern_base"],
        "RSI weighted": baseline + (2.2 if f["rsi_extreme"] else 1.0 if f["rsi_supportive"] else -0.8),
        "Trend weighted": baseline + (2.5 if f["trend_aligned"] else -1.5),
        "Volume weighted": baseline + (3.0 if f["very_high_volume"] else 2.0 if f["high_volume"] else -0.5),
        "Trend + volume": baseline + (1.8 if f["trend_aligned"] else -1.0) + (2.2 if f["very_high_volume"] else 1.4 if f["high_volume"] else -0.4),
    }


def evaluate_candidate(r, next_row, next_date, capital, risk_pct):
    bull = r["direction"] == "Long"
    signal_close = float(r["entry"])
    entry = float(next_row.Open)
    stop = float(r["stop"])
    next_high = float(next_row.High)
    next_low = float(next_row.Low)
    next_close = float(next_row.Close)

    values = (signal_close, entry, stop, next_high, next_low, next_close)
    if not all(np.isfinite(v) and v > 0 for v in values):
        return None

    gap_pct = ((entry / signal_close) - 1.0) * 100.0
    invalid_open = (bull and entry <= stop) or ((not bull) and entry >= stop)

    base = {
        "exit_date": next_date.date().isoformat(),
        "epic": r["epic"],
        "name": r["name"],
        "pattern": r["pattern"],
        "direction": r["direction"],
        "baseline_score": float(r["score"]),
        "rsi": float(r["rsi"]) if r.get("rsi") is not None else None,
        "sma20": float(r["sma20"]) if r.get("sma20") is not None else None,
        "vol_ratio": float(r.get("vol_ratio") or 1.0),
        "signal_close": signal_close,
        "entry": entry,
        "stop": stop,
        "gap_pct": gap_pct,
        **factor_values(r),
        "model_scores": model_scores(r),
    }

    if invalid_open:
        return {**base, "skipped": True, "outcome": "opened beyond stop", "target": None,
                "exit_price": None, "shares": 0, "gross_pnl": 0.0, "costs": 0.0,
                "pnl": 0.0, "r_multiple": 0.0, "ambiguous": False}

    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0 or not np.isfinite(risk_per_share):
        return None
    target = entry + 2 * risk_per_share if bull else entry - 2 * risk_per_share

    hit_stop = next_low <= stop if bull else next_high >= stop
    hit_target = next_high >= target if bull else next_low <= target
    ambiguous = hit_stop and hit_target

    if ambiguous or hit_stop:
        exit_price = stop
        outcome = "stop" if not ambiguous else "stop (both touched)"
    elif hit_target:
        exit_price = target
        outcome = "target"
    else:
        exit_price = next_close
        outcome = "session close"

    risk_amount = capital * risk_pct / 100.0
    entry_gbp = entry / 100.0
    risk_per_share_gbp = risk_per_share / 100.0
    risk_sized = int(risk_amount / risk_per_share_gbp) if risk_per_share_gbp > 0 else 0
    affordable = int(capital / entry_gbp) if entry_gbp > 0 else 0
    shares = max(0, min(risk_sized, affordable))

    pnl_per_share = (exit_price - entry) if bull else (entry - exit_price)
    gross_pnl = pnl_per_share / 100.0 * shares
    spread_cost = ((entry_gbp + exit_price / 100.0) * shares) * (SPREAD_BPS / 20000.0)
    costs = spread_cost + (COMMISSION_PER_TRADE if shares > 0 else 0.0)
    pnl = gross_pnl - costs
    r_multiple = pnl_per_share / risk_per_share

    if not all(np.isfinite(v) for v in (gross_pnl, costs, pnl, r_multiple)):
        return None

    return {**base, "skipped": False, "outcome": outcome, "target": target,
            "exit_price": exit_price, "shares": shares, "gross_pnl": gross_pnl,
            "costs": costs, "pnl": pnl, "r_multiple": r_multiple,
            "ambiguous": ambiguous}


def build_candidate_history(start=None, end=None, last_n=None):
    print("Fetching FTSE 250 constituent list...")
    constituents = scr.fetch_ftse250_constituents()
    epics = list(constituents.keys())
    yahoo_tickers = [scr.epic_to_yahoo(e) for e in epics]
    ticker_to_epic = dict(zip(yahoo_tickers, epics))

    print(f"Downloading daily history for {len(yahoo_tickers)} tickers...")
    data = yf.download(
        yahoo_tickers, period="6mo", interval="1d", group_by="ticker",
        threads=True, progress=False, auto_adjust=False,
    )

    cleaned = {}
    for yt in yahoo_tickers:
        try:
            df = data[yt] if isinstance(data.columns, pd.MultiIndex) else data
        except Exception:
            continue
        if df is None or df.empty:
            continue
        df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
        if not df.empty:
            cleaned[yt] = df

    if not cleaned:
        raise RuntimeError("No usable price history was returned.")

    sample_df = next(iter(cleaned.values()))
    signal_days = get_trading_days(sample_df.index, start=start, end=end, last_n=last_n)
    if not signal_days:
        raise RuntimeError("No signal days selected.")

    history = defaultdict(list)
    for signal_day in signal_days:
        print(f"Analysing {signal_day.date().isoformat()}...")
        for yt, df in cleaned.items():
            if signal_day not in df.index:
                continue
            loc = df.index.get_loc(signal_day)
            if not isinstance(loc, (int, np.integer)) or int(loc) + 1 >= len(df):
                continue
            try:
                r = analyze_as_of(ticker_to_epic[yt], constituents[ticker_to_epic[yt]], df, int(loc))
            except Exception:
                continue
            if not r:
                continue
            result = evaluate_candidate(r, df.iloc[int(loc) + 1], df.index[int(loc) + 1], CAPITAL, RISK_PCT)
            if result is not None:
                result["signal_date"] = signal_day.date().isoformat()
                history[result["signal_date"]].append(result)

    return dict(history)


def selected_trades(history, model_name, dates, direction=None, limit=5, skip_first=0):
    """Return ranked trades for each day.

    When direction is supplied, ranking is performed inside that direction. Therefore
    direction="Long", limit=1 means the highest-ranked long candidate of each day,
    regardless of whether a short candidate scored more highly overall.
    """
    trades = []
    for date in dates:
        candidates = history.get(date, [])
        if direction:
            candidates = [t for t in candidates if t["direction"] == direction]
        ranked = sorted(candidates, key=lambda x: x["model_scores"][model_name], reverse=True)
        trades.extend(ranked[skip_first: skip_first + limit])
    return trades


def summary(trades):
    executed = [t for t in trades if not t["skipped"] and t["shares"] > 0]
    pnls = [t["pnl"] for t in executed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss else (math.inf if gross_profit else 0.0)
    equity = CAPITAL
    peak = CAPITAL
    max_dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return {
        "signals": len(trades),
        "executed": len(executed),
        "wins": len(wins),
        "win_rate": len(wins) / len(executed) * 100 if executed else 0.0,
        "net_pnl": sum(pnls),
        "avg_r": float(np.mean([t["r_multiple"] for t in executed])) if executed else 0.0,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
    }


def bootstrap_difference(on_values, off_values, iterations=2000, seed=20260802):
    if len(on_values) < 5 or len(off_values) < 5:
        return None
    rng = np.random.default_rng(seed)
    diffs = np.empty(iterations)
    a = np.asarray(on_values, dtype=float)
    b = np.asarray(off_values, dtype=float)
    for i in range(iterations):
        diffs[i] = rng.choice(a, size=len(a), replace=True).mean() - rng.choice(b, size=len(b), replace=True).mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def factor_analysis(history, dates):
    candidates = [
        t for d in dates for t in history.get(d, [])
        if t["direction"] == "Long" and not t["skipped"] and t["shares"] > 0
    ]
    definitions = {
        "Strong pattern (base ≥ 7)": lambda t: t["pattern_base"] >= 7,
        "Supportive RSI": lambda t: t["rsi_supportive"],
        "Extreme RSI": lambda t: t["rsi_extreme"],
        "SMA20 trend alignment": lambda t: t["trend_aligned"],
        "Volume ≥ 1.5× average": lambda t: t["high_volume"],
        "Volume ≥ 2.0× average": lambda t: t["very_high_volume"],
    }
    rows = []
    for name, predicate in definitions.items():
        on = [t["r_multiple"] for t in candidates if predicate(t)]
        off = [t["r_multiple"] for t in candidates if not predicate(t)]
        if not on or not off:
            continue
        effect = float(np.mean(on) - np.mean(off))
        ci = bootstrap_difference(on, off)
        if ci and ci[0] > 0 and len(on) >= 100:
            confidence = "High"
        elif ci and ci[0] > 0 and len(on) >= 40:
            confidence = "Moderate"
        elif effect > 0:
            confidence = "Low"
        else:
            confidence = "No positive evidence"
        rows.append({
            "factor": name,
            "n_on": len(on),
            "n_off": len(off),
            "avg_r_on": float(np.mean(on)),
            "avg_r_off": float(np.mean(off)),
            "effect": effect,
            "ci": ci,
            "confidence": confidence,
        })
    rows.sort(key=lambda x: x["effect"], reverse=True)
    return rows


def fmt_pf(value):
    return "∞" if math.isinf(value) else f"{value:.2f}"


def render_html(history, dates, train_dates, test_dates, model_results, ranking_results, factors):
    baseline_test = model_results["Current score"]["test"]
    best_name = max(model_results, key=lambda n: model_results[n]["test"]["net_pnl"])
    best_test = model_results[best_name]["test"]
    primary = factors[0] if factors else None

    model_rows = []
    for name, result in model_results.items():
        full, train, test = result["full"], result["train"], result["test"]
        change = test["net_pnl"] - baseline_test["net_pnl"]
        cls = "positive" if change > 0 else "negative" if change < 0 else "neutral"
        model_rows.append(f"""
        <tr>
          <td><strong>{html_lib.escape(name)}</strong></td>
          <td>£{full['net_pnl']:+,.2f}</td><td>{full['win_rate']:.1f}%</td><td>{fmt_pf(full['profit_factor'])}</td><td>£{full['max_drawdown']:,.2f}</td>
          <td>£{train['net_pnl']:+,.2f}</td>
          <td>£{test['net_pnl']:+,.2f}</td><td>{test['win_rate']:.1f}%</td><td>{fmt_pf(test['profit_factor'])}</td>
          <td class="{cls}">£{change:+,.2f}</td>
        </tr>""")


    ranking_rows = []
    for name, groups in ranking_results.items():
        for label, stats in groups.items():
            row_class = "neutral" if "Short" in label else ""
            ranking_rows.append(f"""
            <tr class="{row_class}">
              <td><strong>{html_lib.escape(name)}</strong></td>
              <td>{html_lib.escape(label)}</td>
              <td>{stats['signals']}</td><td>{stats['executed']}</td><td>{stats['wins']}</td>
              <td>{stats['win_rate']:.1f}%</td><td>£{stats['net_pnl']:+,.2f}</td>
              <td>{stats['avg_r']:+.2f}R</td><td>{fmt_pf(stats['profit_factor'])}</td>
              <td>£{stats['max_drawdown']:,.2f}</td>
            </tr>""")

    factor_rows = []
    for f in factors:
        ci_text = "n/a" if not f["ci"] else f"{f['ci'][0]:+.2f}R to {f['ci'][1]:+.2f}R"
        factor_rows.append(f"""
        <tr><td>{html_lib.escape(f['factor'])}</td><td>{f['n_on']}</td>
        <td>{f['avg_r_on']:+.2f}R</td><td>{f['avg_r_off']:+.2f}R</td>
        <td>{f['effect']:+.2f}R</td><td>{ci_text}</td><td>{f['confidence']}</td></tr>""")

    if primary:
        primary_html = f"""
        <section class="callout">
          <div class="eyebrow">Strongest measured factor for long candidates in the holdout period</div>
          <h2>{html_lib.escape(primary['factor'])}</h2>
          <p>Average improvement: <strong>{primary['effect']:+.2f}R per candidate</strong>. Confidence: <strong>{primary['confidence']}</strong>.</p>
          <p class="muted">Factor present: {primary['n_on']} candidates at {primary['avg_r_on']:+.2f}R average; absent: {primary['n_off']} candidates at {primary['avg_r_off']:+.2f}R average.</p>
        </section>"""
    else:
        primary_html = '<section class="callout"><h2>No factor conclusion</h2></section>'

    generated = dt.datetime.now(dt.timezone.utc).strftime("%a %d %b %Y, %H:%M UTC")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FTSE 250 scoring research</title>
<style>
:root{{--ink:#12161f;--panel:#1b2129;--line:#303843;--paper:#ece7da;--muted:#9aa1ac;--gold:#c9a24b;--green:#5eb77c;--red:#dc6a5d}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--ink);color:var(--paper);font-family:Arial,sans-serif}} .wrap{{max-width:1120px;margin:auto;padding:22px 14px 50px}}
nav{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:18px}} nav a{{color:var(--paper);text-decoration:none;border:1px solid var(--line);border-radius:999px;padding:8px 12px;font-size:13px}} nav a.active{{color:var(--gold);border-color:var(--gold)}}
h1{{margin:4px 0}} .muted,.sub{{color:var(--muted)}} .sub{{margin-bottom:18px;font-size:13px}} .callout{{border:1px solid var(--gold);background:var(--panel);padding:16px;border-radius:10px;margin:18px 0}} .callout h2{{margin:5px 0 8px}} .eyebrow{{color:var(--gold);font-size:11px;text-transform:uppercase;letter-spacing:.08em}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:16px 0}} .stat{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}} .label{{color:var(--muted);font-size:11px;text-transform:uppercase}} .value{{font-size:19px;margin-top:5px}}
.table-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:8px;margin:12px 0 22px}} table{{width:100%;border-collapse:collapse;min-width:900px;background:var(--panel)}} th,td{{padding:10px;border-bottom:1px solid var(--line);font-size:13px;text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}} th{{color:var(--muted);font-size:11px;text-transform:uppercase}} .positive{{color:var(--green)}} .negative{{color:var(--red)}} .neutral{{color:var(--muted)}} .note{{font-size:12px;line-height:1.5;color:var(--muted)}}
@media(max-width:650px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}} h1{{font-size:24px}}}}
</style></head><body><main class="wrap">
<nav><a href="index.html">Daily screener</a><a href="backtest.html">Backtest</a><a class="active" href="backtest-research.html">Research</a><a href="intraday.html">Intraday</a></nav>
<h1>Scoring-model research backtest</h1><div class="sub">Generated {generated} · {dates[0]} to {dates[-1]} · first 70% research / final 30% holdout</div>
<div class="grid">
<div class="stat"><div class="label">Signal days</div><div class="value">{len(dates)}</div></div>
<div class="stat"><div class="label">Research days</div><div class="value">{len(train_dates)}</div></div>
<div class="stat"><div class="label">Holdout days</div><div class="value">{len(test_dates)}</div></div>
<div class="stat"><div class="label">Best holdout model</div><div class="value" style="font-size:15px">{html_lib.escape(best_name)}</div></div>
</div>
{primary_html}
<h2>Long-only model comparison: top five long candidates</h2>
<div class="table-wrap"><table><thead><tr><th>Model</th><th>Full P/L</th><th>Full win</th><th>Full PF</th><th>Full DD</th><th>Research P/L</th><th>Holdout P/L</th><th>Holdout win</th><th>Holdout PF</th><th>vs current</th></tr></thead><tbody>{''.join(model_rows)}</tbody></table></div>
<h2>Rank and direction breakdown — holdout period</h2>
<div class="table-wrap"><table><thead><tr><th>Model</th><th>Selection</th><th>Signals</th><th>Executed</th><th>Wins</th><th>Win rate</th><th>P/L</th><th>Average R</th><th>PF</th><th>Drawdown</th></tr></thead><tbody>{''.join(ranking_rows)}</tbody></table></div>
<h2>Long-only holdout factor evidence</h2>
<div class="table-wrap"><table><thead><tr><th>Factor</th><th>Present n</th><th>Present avg</th><th>Absent avg</th><th>Effect</th><th>95% bootstrap interval</th><th>Confidence</th></tr></thead><tbody>{''.join(factor_rows)}</tbody></table></div>
<p class="note">Headline model results use the five highest-ranked long candidates each day. “Rank 1 long” means the highest-scoring long candidate, even when a short candidate ranks above it overall. Short signals remain in a separate research row and do not affect long-only win rate, P/L or profit factor. Entry, stop, target, spread, commission and same-day exit rules remain identical. Daily bars cannot determine which occurred first when stop and target are both touched; those cases are scored as stops.</p>
</main></body></html>"""


def main():
    print("Running BACKTEST_RESEARCH_LONG_ONLY_RANK1_OUTPUT_FIX_2026_08_02")
    print(f"Working directory: {os.getcwd()}")
    print(f"Research output: {OUTPUT_PATH}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    args = parser.parse_args()

    start = dt.date.fromisoformat(args.start) if args.start else None
    end = dt.date.fromisoformat(args.end) if args.end else None
    last_n = args.days if not (start or end) else None
    if not (start or end or last_n):
        last_n = BACKTEST_DAYS

    history = build_candidate_history(start=start, end=end, last_n=last_n)
    dates = sorted(history)
    if len(dates) < 10:
        raise RuntimeError("At least 10 signal dates are required for a research/holdout split.")

    split = max(1, min(len(dates) - 1, int(len(dates) * 0.70)))
    train_dates = dates[:split]
    test_dates = dates[split:]
    model_names = list(next(iter(next(iter(history.values()))))["model_scores"].keys())

    results = {}
    for model in model_names:
        results[model] = {
            "full": summary(selected_trades(history, model, dates, direction="Long", limit=5)),
            "train": summary(selected_trades(history, model, train_dates, direction="Long", limit=5)),
            "test": summary(selected_trades(history, model, test_dates, direction="Long", limit=5)),
        }

    ranking_results = {}
    for model in model_names:
        ranking_results[model] = {
            "Rank 1 long": summary(selected_trades(history, model, test_dates, direction="Long", limit=1)),
            "Ranks 2–5 long": summary(selected_trades(history, model, test_dates, direction="Long", limit=4, skip_first=1)),
            "Top 5 long": summary(selected_trades(history, model, test_dates, direction="Long", limit=5)),
            "Top 5 short — research only": summary(selected_trades(history, model, test_dates, direction="Short", limit=5)),
        }

    factors = factor_analysis(history, test_dates)
    report = render_html(history, dates, train_dates, test_dates, results, ranking_results, factors)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print("\nMODEL COMPARISON — HOLDOUT PERIOD")
    baseline = results["Current score"]["test"]
    for model, result in results.items():
        test = result["test"]
        print(f"{model:18} P/L £{test['net_pnl']:+.2f} | win {test['win_rate']:.1f}% | PF {fmt_pf(test['profit_factor'])} | vs baseline £{test['net_pnl'] - baseline['net_pnl']:+.2f}")
    print("\nRANK 1 LONG — HOLDOUT PERIOD")
    for model in model_names:
        rank1 = ranking_results[model]["Rank 1 long"]
        print(f"{model:18} {rank1['wins']}/{rank1['executed']} winners | win {rank1['win_rate']:.1f}% | P/L £{rank1['net_pnl']:+.2f} | PF {fmt_pf(rank1['profit_factor'])}")
    if factors:
        p = factors[0]
        print(f"\nPrimary long-only factor: {p['factor']} | effect {p['effect']:+.2f}R | confidence {p['confidence']}")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
