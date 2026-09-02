# NSE Momentum (Clenow-style)

A long-only, cross-sectional momentum strategy for NSE equities, adapted from
the rules in Andreas Clenow's *Stocks on the Move* (as summarized by
[Teddy Koker](https://teddykoker.com/2019/05/momentum-strategy-from-stocks-on-the-move-in-python/)).
This is an independent implementation — no `backtrader` dependency, built
directly from the strategy's rules, adapted for Indian market realities.

## Strategy rules

1. **Momentum score**: annualized slope of a linear regression on `log(close)`
   over a 90-day window, multiplied by the regression's R². This rewards
   smooth, consistent trends over choppy ones with the same average slope.
2. **Regime filter**: only open new positions while NIFTY 50 (`^NSEI`) is
   above its 200-day SMA.
3. **Weekly rebalance**: drop any held stock that falls out of the top 20%
   momentum rank, or below its 100-day SMA. Fill freed cash with the
   highest-ranked names not already held.
4. **Position sizing (risk parity)**: `shares = portfolio_value × 0.001 / ATR(20)`
   — each position targets roughly equal rupee volatility, not equal capital.
5. **Bi-weekly resize**: every other rebalance, resize existing positions to
   the current target (ATR changes over time).

## India-specific adaptations

- Data via `yfinance` using NSE tickers (`.NS` suffix); benchmark defaults to
  `^NSEI` (NIFTY 50).
- A simple Indian cost model applied on every fill: brokerage + STT + stamp
  duty + slippage (see `CostModel` in the script — rates are indicative, tune
  to your actual broker).
- Universe is a static NIFTY 200-ish snapshot (see `NIFTY_UNIVERSE` in the
  script) — **not** a survivorship-bias-free, point-in-time index history.

## Known limitations (read before trusting the numbers)

- **Survivorship bias**: the universe list is current constituents, not the
  historical index membership at each point in time. Momentum strategies are
  particularly sensitive to this because index inclusion is itself
  correlated with momentum — expect backtest returns to be optimistic.
- **Costs are illustrative**: STT/brokerage/slippage rates are approximate.
  Adjust `CostModel` to your actual broker and account type before relying on
  the results.
- **Circuit filters**: NSE price bands (5–20%) can block entries/exits that
  this backtest assumes fill instantly at the close.
- **No point-in-time index reconstitution** for the 200/500 index membership.
- This is a research/backtesting tool, **not** a live trading system — there
  is no order execution, broker integration, or paper-trading loop here.

## Usage

```bash
pip install -r requirements.txt
python nse_momentum_clenow.py --start 2015-01-01 --end 2025-01-01 --capital 1000000
```

Outputs land in `results/`: `equity_curve.csv` and `summary.csv`
(CAGR, Sharpe, Max Drawdown). Downloaded OHLC data is cached to `data_cache/`
as Parquet so repeat runs don't re-hit the network.

### Options

| Flag | Default | Description |
|---|---|---|
| `--start` | `2015-01-01` | Backtest start date |
| `--end` | `2025-01-01` | Backtest end date |
| `--capital` | `1000000` | Starting capital (INR) |
| `--benchmark` | `^NSEI` | Regime-filter benchmark ticker |
| `--cache-dir` | `data_cache` | Local OHLC cache directory |
| `--out` | `results` | Output directory |
| `--cost-multiplier` | `1.0` | Scale all cost components (e.g. `2.0` to stress-test with double costs) |
| `--split-date` | none | Run an additional in-sample/out-of-sample walk-forward report split at this date |
| `--no-benchmark-compare` | off | Skip the benchmark buy-and-hold comparison |

### Validating a backtest result — recommended checklist

A single full-period run is easy to be fooled by. Before trusting a result:

1. **Benchmark comparison** (on by default) — writes `benchmark_summary.csv` /
   `benchmark_equity_curve.csv` for `^NSEI` buy-and-hold over the same
   period, net of one round-trip's cost, so you can see if the strategy is
   actually adding anything over just holding the index.
2. **Walk-forward split** — `--split-date 2021-01-01` runs the strategy
   separately on the two halves so a result that only worked in one regime
   (e.g. a strong bull run) doesn't get mistaken for a robust edge.
3. **Cost sensitivity** — rerun with `--cost-multiplier 2.0`. If CAGR
   collapses or goes negative, the edge is thin relative to real-world
   slippage on less liquid names.
4. **Position diagnostics** — `position_diagnostics.csv` reports average
   positions held, max positions, and `PctDaysFlat` (fraction of days with
   zero positions). A high `PctDaysFlat` means the regime filter is mostly
   keeping you in cash rather than the strategy doing real selection work —
   worth knowing before crediting the CAGR to stock-picking skill.

## Paper trading

`paper_trade.py` reuses the exact same momentum/ATR/regime functions as the
backtest engine, but runs forward against a persisted paper portfolio
instead of a historical loop. It never places a real order — it only tells
you what to do and, unless `--dry-run` is passed, updates a local JSON
state file to simulate the fill.

Run it once per week (matching the strategy's weekly cadence):

```bash
python paper_trade.py --init --capital 500000   # first run: start fresh
python paper_trade.py                            # every following week
python paper_trade.py --dry-run                  # preview signals only
```

Each run prints the regime status and any SELL/BUY/RESIZE signals, applies
them to `paper_state.json` (positions + cash), and appends fills to
`paper_trades_log.csv`. Both files are gitignored — they're your local,
personal paper-trading record, not something to commit.

Note: `--init` starts from a fresh 200-day-plus history window ending
"today", so early runs may show the regime filter or momentum ranks
shifting around simply because more history accumulates in the cache —
this settles down after the first few weeks.

## Dashboard

Since the repo stays private, `docs/index.html` is a **static, self-contained
dashboard** — no build step, no server. It reads `paper_state.json`,
`paper_trades_log.csv`, and `paper_equity_history.csv` (also under `docs/`)
directly via client-side `fetch()`. It's viewer-only, same as the Streamlit
version — it cannot place trades.

**Note on privacy**: GitHub Pages sites are publicly reachable by anyone with
the URL even when the source repo is private (private-Pages hosting is a paid
Enterprise-only feature). Deploying this way was a deliberate choice to test
the dashboard quickly — the plan is to move to the Streamlit version
(`dashboard.py`, described below) once testing is done, which supports
viewer-restricted access on a private repo.

**Enable Pages (one-time):**

1. Push the repo with the `docs/` folder included.
2. GitHub repo → Settings → Pages → Source: **Deploy from a branch** →
   Branch: `main`, folder: **`/docs`** → Save.
3. Your dashboard will be live at `https://g1bs.github.io/nse-momentum-clenow/`
   (a project page under your account — note this is a different, narrower
   thing than a `g1bs.github.io` *user site* repo, which would be your
   personal top-level GitHub Pages site).

**Weekly workflow:**

```bash
python paper_trade.py --push
```

`paper_trade.py` now writes its state/log/equity-history files into `docs/`
by default, so this one command updates the paper portfolio *and* pushes the
data the live Pages site reads — no separate sync step needed.

### Later: moving to Streamlit (private)

`dashboard.py` (Streamlit) is still in the repo, unused for now. When you're
ready to go private:

```bash
streamlit run dashboard.py       # test locally
```

Then deploy via [share.streamlit.io](https://share.streamlit.io) from this
private repo, and restrict viewers under App settings → Sharing. At that
point you can also turn off GitHub Pages (Settings → Pages → Source: None)
to stop publishing the public copy.

## File structure

```
nse_momentum_clenow.py   # strategy, backtest engine, CLI
paper_trade.py           # forward paper-trading signal generator (writes to docs/)
docs/index.html          # static dashboard for GitHub Pages (public, for now)
dashboard.py             # Streamlit dashboard (private, for later)
requirements.txt
README.md
```

## Attribution

Strategy rules originate from Andreas Clenow's *Stocks on the Move*, as
summarized in Teddy Koker's blog post (linked above). The code in this repo
is an independent implementation of those rules for the Indian market and
does not reuse code from that post.
