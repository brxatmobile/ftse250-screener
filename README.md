# FTSE 250 daily screener

Screens the whole FTSE 250 automatically every weekday morning, scores candlestick
setups (engulfing, hammer, star, marubozu, doji) combined with RSI, trend, and
volume, and publishes the top 5 as a simple page you can bookmark on your phone.

No data upload, no manual work once it's set up — it runs itself.

## One-time setup (about 10 minutes, all free)

1. **Create a GitHub account** at github.com if you don't have one.
2. **Create a new repository**: click the "+" top-right → "New repository" →
   name it e.g. `ftse250-screener` → keep it **Public** (required for free
   GitHub Pages) → click "Create repository".
3. **Upload these files**: on the new repo's page, click "Add file" →
   "Upload files", then drag in `screener.py`, `backtest.py`,
   `requirements.txt`, `README.md`, and the `.github` folder. Commit the
   files. (You don't need to worry about the `docs` folder — the workflow
   creates it fresh on every run.)
4. **Turn on GitHub Pages**: go to the repo's Settings tab → Pages (left
   sidebar) → under "Build and deployment", set **Source to "GitHub
   Actions"** (not "Deploy from a branch" — that option relies on Jekyll and
   is what caused the earlier build errors). No further Pages config needed;
   the workflow itself handles publishing.
5. **Set your capital and risk % (optional)**: Settings → Secrets and
   variables → Actions → "Variables" tab → "New repository variable" → add
   `CAPITAL` (e.g. `5000`) and `RISK_PCT` (e.g. `1`). If you skip this it
   defaults to £5,000 and 1%.
6. **Run it once manually**: go to the "Actions" tab → "Daily FTSE 250 screen"
   → "Run workflow" → Run workflow. Wait a minute or two for it to finish
   (green tick).
7. **Find your page**: Settings → Pages will now show a URL like
   `https://yourusername.github.io/ftse250-screener/`. Open that on your
   phone and **add it to your home screen** (in Safari/Chrome: Share →
   "Add to Home Screen") so it behaves like an app icon.

After that, it re-runs automatically every weekday morning and the page
updates itself — just open the bookmark each day.

## Running a backtest with real data

Once the repo is set up (steps 1–4 above), you can test what the screener
would have recommended on real past days and what the P&L would have been:

1. Go to the **Actions** tab → **"Run backtest"** → **Run workflow**.
2. Either leave the default (last 3 trading days), or fill in `start` and
   `end` dates (e.g. `2026-07-27` to `2026-07-29`) to pick a specific week.
3. Wait for the run to finish (green tick), then click into the run →
   the "Run backtest" step → expand it to see the trade-by-trade report and
   total P&L, using your `CAPITAL`/`RISK_PCT` settings.

This uses the exact same scoring logic as the live daily screen, run against
real historical prices — genuine numbers, not estimates. Two honest caveats
baked into the method:

- **Entry is modelled as the signal day's closing price**, exit is evaluated
  on the *next* trading day (a same-day/day-trade round trip) — using that
  day's high/low to check whether the stop or target was hit, or the close
  if neither was.
- **Daily bars don't reveal intraday order.** If a day's range touches both
  the stop and the target, the script conservatively assumes the stop was
  hit first (worst case) and flags this in the output — real intraday
  execution could have gone either way.

## Adjusting risk later


Change the `CAPITAL` / `RISK_PCT` repository variables any time (step 5) —
the next scheduled or manual run will use the new values.

## Troubleshooting

**Pages build fails mentioning Jekyll, `style.scss`, or `.../docs: No such
file or directory`:** this means Settings → Pages → Source is still set to
"Deploy from a branch" instead of "GitHub Actions". Switch it to "GitHub
Actions" (step 4 above) — this removes Jekyll from the picture entirely and
lets the workflow publish the page itself, so it no longer matters whether a
`docs` folder happens to exist in your uploaded files.

**First visit to the Pages URL shows a 404:** the workflow needs to run
successfully at least once before there's anything to serve. Go to Actions →
"Daily FTSE 250 screen" → Run workflow, wait for the green tick, then reload
the Pages URL.

## Important caveats

- This uses free, delayed market data (Yahoo Finance), refreshed once a day —
  it is **not** a live intraday feed, and prices may lag the real market by
  up to a day.
- It cannot connect to your eToro account — there's no public eToro API for
  retail accounts. You'd still place any trade yourself.
- The scoring logic is a transparent, rules-based heuristic, not a backtested
  or guaranteed strategy. Treat the ranking as a shortlist to investigate,
  not an instruction to trade.
- The FTSE 250 constituent list is scraped live each run so it stays current
  as the index is reshuffled quarterly — if the London Stock Exchange changes
  their page layout, the run may fail; check the Actions tab for errors.
- This is not financial advice. Day trading carries a high risk of loss.
