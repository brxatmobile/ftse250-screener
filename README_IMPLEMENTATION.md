# Navigation and backtest implementation

Upload these files to the matching locations in the repository:

- `screener.py` → repository root
- `backtest.py` → repository root
- `intraday_daytrader.py` → repository root
- `.github/workflows/backtest-report.yml` → `.github/workflows/`

The three scripts generate:

- `docs/index.html`
- `docs/backtest.html`
- `docs/intraday.html`

Each generated page contains the same navigation links.

## GitHub variables

Under Settings → Secrets and variables → Actions → Variables, add or confirm:

- `CAPITAL` = `5000`
- `RISK_PCT` = `1`
- `BACKTEST_DAYS` = `75`
- `SPREAD_BPS` = `20`
- `COMMISSION_PER_TRADE` = `0`

## First run

1. Run the normal screener workflow to rebuild `docs/index.html`.
2. Run **FTSE Backtest Report** manually to create `docs/backtest.html`.
3. Run **FTSE Opening-Hour Day-Trade Review** with `analyse` to rebuild `docs/intraday.html`.
4. Open any one of the pages and use the navigation buttons.

Public URLs:

- `/index.html` or `/`
- `/backtest.html`
- `/intraday.html`

The backtest workflow also runs at 17:30 UTC each Saturday.
