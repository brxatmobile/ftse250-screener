# FILE_VERSION: FTSE350_UP_TO_FIVE_ACTIONABLE_LONGS_2026_08_04
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

CAPITAL = float(os.environ.get("CAPITAL", "5000"))
RISK_PCT = float(os.environ.get("RISK_PCT", "1"))
POOL_N = int(os.environ.get("CANDIDATE_POOL_N", "60"))
TARGET_R = float(os.environ.get("TARGET_R", "2"))
MAX_GAP_PCT = float(os.environ.get("MAX_GAP_PCT", "2.5"))
MAX_RANGE_PCT = float(os.environ.get("MAX_OPENING_RANGE_PCT", "5.0"))
MIN_RANGE_PCT = float(os.environ.get("MIN_OPENING_RANGE_PCT", "0.35"))
MIN_VOLUME_RATIO = float(os.environ.get("MIN_OPENING_VOLUME_RATIO", "0.80"))
MIN_OPENING_TURNOVER_GBP = float(os.environ.get("MIN_OPENING_TURNOVER_GBP", "100000"))
MAX_VWAP_DISTANCE_PCT = float(os.environ.get("MAX_VWAP_DISTANCE_PCT", "1.25"))
ENTRY_BUFFER_PCT = float(os.environ.get("ENTRY_BUFFER_PCT", "0.05"))
MAX_ENTRY_EXTENSION_R = float(os.environ.get("MAX_ENTRY_EXTENSION_R", "0.25"))
MIN_ACTIONABLE_BARS = int(os.environ.get("MIN_ACTIONABLE_BARS", "10"))
MAX_ACTIONABLE_TRADES = int(os.environ.get("MAX_ACTIONABLE_TRADES", "5"))
MIN_ACTIONABLE_SCORE = float(os.environ.get("MIN_ACTIONABLE_SCORE", "72"))

ROOT = Path(__file__).resolve().parent
DAILY_INDEX_PATH = ROOT / "docs" / "index.html"
OUTPUT_PATH = ROOT / "docs" / "intraday.html"

INK="#12161F"; PANEL="#1B2129"; HAIRLINE="#2C333D"; BRASS="#C9A24B"
SALMON="#E8A493"; BULL="#4FAE73"; BEAR="#D1594B"; PAPER="#ECE7DA"; MUTED="#8B92A0"

def gbx_to_gbp(value: float) -> float:
    return float(value) / 100.0

def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

def now_london() -> dt.datetime:
    return dt.datetime.now(UTC).astimezone(LONDON)

def normalise(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    required = ["Open","High","Low","Close","Volume"]
    if any(c not in frame.columns for c in required):
        return pd.DataFrame()
    frame = frame.dropna(subset=["Open","High","Low","Close"])
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
        page, flags=re.I | re.S
    )
    if not match:
        raise RuntimeError("The daily page contains no embedded intraday candidate pool.")
    payload = json.loads(match.group(1).replace("<\\/", "</"))
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise RuntimeError("The embedded intraday candidate pool is empty.")
    generated = dt.datetime.fromisoformat(str(payload["generated_at_utc"]).replace("Z","+00:00"))
    if generated.tzinfo is None:
        generated = generated.replace(tzinfo=UTC)
    if generated.astimezone(LONDON).date() != now_london().date():
        raise RuntimeError("The embedded candidate pool is not from today; run the daily screener first.")
    return [c for c in candidates[:POOL_N] if str(c.get("direction","")).lower()=="long"]

def history(ticker: str) -> pd.DataFrame:
    return normalise(yf.download(
        ticker, period="5d", interval="5m", progress=False,
        auto_adjust=False, prepost=False
    ))

def prior_close(frame: pd.DataFrame, today: dt.date) -> float | None:
    prior = frame[frame.index.date < today]
    return float(prior.iloc[-1]["Close"]) if not prior.empty else None

def prior_opening_volume(frame: pd.DataFrame, today: dt.date) -> float | None:
    totals=[]
    for d in sorted(set(frame.index.date)):
        if d >= today:
            continue
        day=frame[frame.index.date==d]
        opening=day[(day.index.time>=dt.time(8,0))&(day.index.time<dt.time(9,0))]
        total=float(opening["Volume"].fillna(0).sum())
        if total>0:
            totals.append(total)
    return float(np.mean(totals)) if totals else None

def opening_story(opening: pd.DataFrame) -> str:
    if opening.empty:
        return "No opening-hour bars were available."
    first=float(opening.iloc[0]["Open"]); last=float(opening.iloc[-1]["Close"])
    high=float(opening["High"].max()); low=float(opening["Low"].min())
    span=max(high-low,1e-9); body=abs(last-first)
    if last>first and body/span>=0.60:
        return "A strong bullish opening candle held near the upper part of the range."
    if last<first and body/span>=0.60:
        return "A strong bearish opening candle dominated the first hour."
    location=(last-low)/span*100
    if location>=70:
        return "Price finished in the upper part of the opening range, but without a decisive full-hour candle."
    if location<=30:
        return "Price finished in the lower part of the opening range."
    return "Price remained indecisive near the middle of the opening range."

def assess(candidate: dict[str, Any], now: dt.datetime) -> dict[str, Any]:
    ticker=str(candidate.get("yahoo_ticker") or f"{candidate['epic']}.L")
    frame=history(ticker)
    base={
        "epic":candidate.get("epic",ticker.replace(".L","")),
        "name":candidate.get("name",""),
        "ticker":ticker,
        "daily_score":float(candidate.get("score",0)),
        "daily_pattern":candidate.get("pattern","Long candidate"),
        "status":"REJECTED",
        "recommendation":"No trade.",
        "score":0.0,
        "reasons":[],
        "checks":[],
    }
    if frame.empty:
        base["reasons"]=["No intraday market data was available."]
        return base

    today=now.date()
    today_frame=frame[frame.index.date==today]
    opening=today_frame[(today_frame.index.time>=dt.time(8,0))&(today_frame.index.time<dt.time(9,0))]
    bars=len(opening)
    base["bars"]=bars
    base["story"]=opening_story(opening)

    pc=prior_close(frame,today)
    if opening.empty or pc is None or pc<=0:
        base["status"]="DATA ONLY"
        base["recommendation"]="No recommendation; the available data is insufficient."
        base["reasons"]=["Previous close or opening-hour data was unavailable."]
        return base

    open_px=float(opening.iloc[0]["Open"])
    high=float(opening["High"].max()); low=float(opening["Low"].min())
    close=float(opening.iloc[-1]["Close"])
    vol=float(opening["Volume"].fillna(0).sum())
    typical=(opening["High"]+opening["Low"]+opening["Close"])/3
    vol_series=opening["Volume"].fillna(0)
    vwap=float((typical*vol_series).sum()/vol_series.sum()) if vol_series.sum()>0 else close
    gap=((open_px-pc)/pc)*100
    move=((close-pc)/pc)*100
    rng=((high-low)/pc)*100
    location=((close-low)/max(high-low,1e-9))*100
    vwap_distance=abs((close-vwap)/vwap)*100 if vwap else 0
    avg_open_vol=prior_opening_volume(frame,today)
    vol_ratio=vol/avg_open_vol if avg_open_vol and avg_open_vol>0 else None
    turnover=vol*gbx_to_gbp(close)

    # Latest completed five-minute close after 09:00 anchors the executable plan.
    completed_cutoff = now.replace(second=0,microsecond=0) - dt.timedelta(
        minutes=now.minute % 5
    )
    completed=today_frame[(today_frame.index.time>=dt.time(9,0))&(today_frame.index<completed_cutoff)]
    purchase=float(completed.iloc[-1]["Close"]) if not completed.empty else close
    purchase_time=completed.index[-1] if not completed.empty else opening.index[-1]

    daily_stop=float(candidate.get("stop",low))
    stop=max(low,daily_stop) if daily_stop < purchase else low
    trigger=high*(1+ENTRY_BUFFER_PCT/100)
    initial_risk=trigger-stop
    max_purchase=trigger+MAX_ENTRY_EXTENSION_R*initial_risk if initial_risk>0 else trigger
    risk=purchase-stop
    target=purchase+TARGET_R*risk if risk>0 else purchase

    checks=[]; reasons=[]; score=0.0
    def good(condition: bool, points: float, yes: str, no: str):
        nonlocal score
        if condition:
            score += points; checks.append(yes)
        else:
            reasons.append(no)

    good(bars>=MIN_ACTIONABLE_BARS,10,"Enough five-minute bars for execution.","Too few opening-hour bars for a recommendation.")
    good(close>pc,15,"The first hour finished above the previous close.","The first hour did not finish above the previous close.")
    good(close>vwap,15,"Price finished above opening-hour VWAP.","Price finished below opening-hour VWAP.")
    good(abs(gap)<=MAX_GAP_PCT,10,"The opening gap was controlled.",f"The opening gap of {gap:+.2f}% was excessive.")
    good(MIN_RANGE_PCT<=rng<=MAX_RANGE_PCT,10,"The opening range was usable.",f"The opening range of {rng:.2f}% was not suitable.")
    good(location>=65,10,"Price held in the upper part of the opening range.","Price did not hold in the upper part of the opening range.")
    good(vol_ratio is not None and vol_ratio>=MIN_VOLUME_RATIO,10,
         f"Opening volume was {vol_ratio:.2f}× normal." if vol_ratio is not None else "",
         "Opening volume was unavailable or below the minimum.")
    good(turnover>=MIN_OPENING_TURNOVER_GBP,10,
         f"Opening turnover was about £{turnover:,.0f}.",
         f"Opening turnover of about £{turnover:,.0f} was too low.")
    good(vwap_distance<=MAX_VWAP_DISTANCE_PCT,5,"Price was not overextended from VWAP.",
         f"Price was {vwap_distance:.2f}% from VWAP and overextended.")
    good(risk>0 and purchase>stop,5,"The stop structure was valid.","The stop structure was invalid.")

    hard_failure = (
        bars < MIN_ACTIONABLE_BARS or close <= pc or close <= vwap or
        abs(gap)>MAX_GAP_PCT or rng<MIN_RANGE_PCT or rng>MAX_RANGE_PCT or
        turnover<MIN_OPENING_TURNOVER_GBP or risk<=0
    )

    if bars < MIN_ACTIONABLE_BARS:
        status="DATA ONLY"
        recommendation=(
            f"No recommendation — only {bars} of 12 expected opening-hour bars were available. "
            "The observations describe the partial price action only."
        )
    elif hard_failure:
        status="REJECTED"
        recommendation="No trade. One or more mandatory tradability checks failed."
    elif purchase < trigger:
        status="WAIT FOR BREAK"
        recommendation=(
            f"Not ready yet. Consider only if price breaks £{gbx_to_gbp(trigger):.2f}; "
            f"do not pay above £{gbx_to_gbp(max_purchase):.2f}."
        )
    elif purchase > max_purchase:
        status="DO NOT CHASE"
        recommendation=(
            f"No trade at the current price. It has moved beyond the maximum acceptable "
            f"purchase price of £{gbx_to_gbp(max_purchase):.2f}."
        )
    elif score >= MIN_ACTIONABLE_SCORE:
        status="ACTIONABLE"
        recommendation=(
            f"Executable long setup at approximately £{gbx_to_gbp(purchase):.2f}, "
            f"subject to checking the live broker quote and spread."
        )
    else:
        status="REJECTED"
        recommendation="No trade. The setup passed the hard filters but was not strong enough overall."

    risk_budget=CAPITAL*RISK_PCT/100
    risk_gbp=gbx_to_gbp(risk) if risk>0 else 0
    purchase_gbp=gbx_to_gbp(purchase)
    shares=max(0,min(
        math.floor(risk_budget/risk_gbp) if risk_gbp>0 else 0,
        math.floor(CAPITAL/purchase_gbp) if purchase_gbp>0 else 0
    ))

    base.update({
        "status":status,"recommendation":recommendation,"score":min(score,100),
        "checks":checks,"reasons":reasons,"previous_close":pc,"gap_pct":gap,
        "move_pct":move,"range_pct":rng,"close":close,"vwap":vwap,
        "location_pct":location,"volume_ratio":vol_ratio,"turnover_gbp":turnover,
        "purchase":purchase,"purchase_time":purchase_time,"trigger":trigger,
        "max_purchase":max_purchase,"stop":stop,"target":target,"risk":risk,
        "shares":shares,"position_value":shares*purchase_gbp,
        "planned_risk":shares*risk_gbp,
    })
    return base

def money(value: Any) -> str:
    return f"£{gbx_to_gbp(float(value)):.2f}" if finite(value) else "—"

def card(item: dict[str, Any], nap: bool=False) -> str:
    label=("NAP — " if nap else "")+str(item["status"])
    colour=BULL if item["status"]=="ACTIONABLE" else (BRASS if item["status"]=="WAIT FOR BREAK" else MUTED)
    checks="".join(f"<li>{html_lib.escape(x)}</li>" for x in item.get("checks",[])) or "<li>None</li>"
    reasons="".join(f"<li>{html_lib.escape(x)}</li>" for x in item.get("reasons",[])) or "<li>None</li>"
    vr=item.get("volume_ratio")
    vr_text=f"{vr:.2f}×" if finite(vr) else "—"
    return f"""
<section class="pick">
<div class="pick-head"><div><strong class="epic">{html_lib.escape(str(item['epic']))}</strong>
<span class="name">{html_lib.escape(str(item['name']))}</span>
<div class="pattern">{html_lib.escape(str(item.get('daily_pattern','')))} · daily pool {item['daily_score']:.0f}/100</div></div>
<div class="score"><span style="color:{colour}">{html_lib.escape(label)}</span><strong>{item['score']:.0f}/100</strong></div></div>
<p class="recommendation" data-original="{html_lib.escape(item['recommendation'],quote=True)}">{html_lib.escape(item['recommendation'])}</p>
<div class="metrics">
<div><span>Latest completed price</span><strong>{money(item.get('purchase'))} at {item.get('purchase_time').strftime('%H:%M') if item.get('purchase_time') else '—'}</strong></div>
<div><span>Maximum purchase</span><strong>{money(item.get('max_purchase'))}</strong></div>
<div><span>Stop / 2R target</span><strong>{money(item.get('stop'))} / {money(item.get('target'))}</strong></div>
<div><span>Indicative size</span><strong>{item.get('shares',0)} shares</strong></div>
<div><span>Opening gap</span><strong>{item.get('gap_pct',0):+.2f}%</strong></div>
<div><span>Opening range</span><strong>{item.get('range_pct',0):.2f}%</strong></div>
<div><span>Close / VWAP</span><strong>{money(item.get('close'))} / {money(item.get('vwap'))}</strong></div>
<div><span>Relative volume / turnover</span><strong>{vr_text} / £{item.get('turnover_gbp',0):,.0f}</strong></div>
</div>
<p class="story">{html_lib.escape(item.get('story',''))}</p>
<details><summary>Why</summary><div class="detail-grid">
<div><h3>Passed</h3><ul>{checks}</ul></div><div><h3>Failed or cautions</h3><ul>{reasons}</ul></div>
</div></details></section>"""

def rejected_row(item: dict[str, Any]) -> str:
    reason="; ".join(item.get("reasons",[])[:2]) or item.get("recommendation","No trade")
    return f"<li><strong>{html_lib.escape(str(item['epic']))}</strong> — {html_lib.escape(str(item['status']))}: {html_lib.escape(reason)}</li>"

def build_html(results: list[dict[str, Any]], generated: dt.datetime) -> str:
    actionable = sorted(
        [row for row in results if row["status"] == "ACTIONABLE"],
        key=lambda row: row["score"],
        reverse=True,
    )[:MAX_ACTIONABLE_TRADES]

    rejected = [row for row in results if row not in actionable]
    rejected.sort(key=lambda row: row["score"], reverse=True)

    nap_epic = actionable[0]["epic"] if actionable else None
    cards = "".join(
        card(row, nap=(row["epic"] == nap_epic))
        for row in actionable
    )

    if actionable:
        decision = (
            f"{len(actionable)} executable long trade"
            f"{'s' if len(actionable) != 1 else ''} passed every mandatory check. "
            "The highest-ranked trade is marked NAP."
        )
    else:
        decision = (
            "NO TRADE TODAY — none of the FTSE 350 candidates passed every "
            "mandatory tradability and execution check."
        )

    rejected_html = "".join(rejected_row(row) for row in rejected) or "<li>None</li>"
    expiry = dt.datetime.combine(
        generated.date(), dt.time(10, 0), tzinfo=LONDON
    ).isoformat()

    template = Template("""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>FTSE 350 intraday trades</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:$INK;color:$PAPER;font-family:Arial,sans-serif}
.wrap{max-width:920px;margin:auto;padding:22px 14px 56px}a{color:$BRASS}.header{display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;border-bottom:1px solid $HAIRLINE;padding-bottom:16px}
h1{margin:4px 0 0;font-size:27px}h3{font-size:13px;margin:0 0 5px}.kicker{color:$SALMON;font-size:12px;text-transform:uppercase;letter-spacing:.09em}
.time,.pattern,.name,.footer,.story{color:$MUTED;font-size:12px}.decision,.pick,.rejected,.expired{background:$PANEL;border:1px solid $HAIRLINE;border-radius:9px;padding:15px;margin:14px 0}
.decision strong{color:$BRASS}.pick-head{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}.epic{color:$BRASS;font-size:19px}.name{margin-left:8px}
.score{display:flex;flex-direction:column;text-align:right;font-size:13px;font-weight:700}.score strong{font-size:22px;margin-top:3px}.recommendation{font-size:14px;line-height:1.5}
.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;margin-top:12px}.metrics div{background:$INK;border:1px solid $HAIRLINE;border-radius:6px;padding:9px}
.metrics span{display:block;color:$MUTED;font-size:11px;margin-bottom:4px}.metrics strong{font-size:12px;line-height:1.35}details{margin-top:12px}summary{cursor:pointer;color:$BRASS;font-size:13px}
.detail-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:10px}ul{padding-left:18px;color:$MUTED;font-size:12px;line-height:1.5}
.expired{display:none;border-color:$SALMON}.expired h2{color:$SALMON;margin-top:0}.footer{line-height:1.6;border-top:1px solid $HAIRLINE;padding-top:14px;margin-top:20px}
@media(max-width:680px){.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.detail-grid{grid-template-columns:1fr}h1{font-size:22px}}
</style></head><body><main class="wrap">
<div class="header"><div><div class="kicker">FTSE 350 ex trusts · executable long trades</div><h1>Opening-hour intraday decisions</h1></div><div class="time">$DATE<br>$TIME</div></div>
<div class="decision"><strong>$DECISION</strong> <a href="index.html">Daily watchlist</a> · <a href="backtest.html">Backtest</a></div>
<div id="expired-content" class="expired"><h2>Past the 10:00 cutoff</h2><p>The trades below are retained for review only. Do not use their purchase, stop or target levels now.</p></div>
<div id="live-content">$CARDS
<details class="rejected"><summary>Other candidates assessed but not recommended</summary><ul>$REJECTED</ul></details></div>
<p class="footer">Only trades that pass every mandatory check are shown as recommendations. The engine requires adequate five-minute data, controlled gap and range, price above the previous close and VWAP, sufficient liquidity and turnover, a valid stop, and a current price within the permitted entry zone. It shows up to five trades and never fills the list with weaker names.</p>
</main><script>
(function(){const expiry=new Date('$EXPIRY');const expired=document.getElementById('expired-content');
function enforce(){if(new Date()>=expiry){expired.style.display='block';document.querySelectorAll('.recommendation').forEach(function(el){const original=el.dataset.original||el.textContent;el.dataset.original=original;el.textContent='PAST 10:00 — Do not follow this recommendation now. Original assessment: '+original;});}}
enforce();setInterval(enforce,30000);})();
</script></body></html>""")

    return template.substitute(
        INK=INK, PAPER=PAPER, BRASS=BRASS, HAIRLINE=HAIRLINE,
        SALMON=SALMON, MUTED=MUTED, PANEL=PANEL,
        DATE=generated.strftime("%a %d %b %Y"),
        TIME=generated.strftime("%H:%M %Z"),
        DECISION=html_lib.escape(decision),
        CARDS=cards or (
            "<section class='pick'><strong>NO TRADE TODAY</strong>"
            "<p>No candidate passed all mandatory checks.</p></section>"
        ),
        REJECTED=rejected_html,
        EXPIRY=expiry,
    )


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["auto","analyse","expire"],default="auto")
    p.add_argument("--force",action="store_true")
    return p.parse_args()

def main() -> int:
    args=parse_args(); now=now_london()
    OUTPUT_PATH.parent.mkdir(parents=True,exist_ok=True)
    mode=("expire" if now.hour>=10 else "analyse") if args.mode=="auto" else args.mode
    if mode=="expire":
        print("Retaining the existing intraday page; browser-side logic marks it expired.")
        return 0
    candidates=load_pool()
    print(f"Assessing {len(candidates)} liquid long candidates from the daily pool...")
    results=[assess(c,now) for c in candidates]
    OUTPUT_PATH.write_text(build_html(results,now),encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    for r in sorted(results,key=lambda x:x["score"],reverse=True):
        print(f"{r['epic']:<7} {r['status']:<15} {r['score']:>5.1f} {r['recommendation']}")
    return 0

if __name__=="__main__":
    sys.exit(main())
