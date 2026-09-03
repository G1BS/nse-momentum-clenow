#!/usr/bin/env python3
"""
nse_momentum_clenow.py

Long-only cross-sectional momentum strategy for NSE equities, adapted from
the "Stocks on the Move" (Andreas Clenow) rules:

  - Rank stocks by annualized exponential-regression slope of log(close)
    over a lookback window, weighted by the regression's R^2.
  - Only take new positions while the benchmark index is above its own
    long moving average (trend regime filter).
  - Weekly: drop names that fall out of the top momentum percentile or
    below their medium-term moving average; buy replacements with the
    top-ranked names using freed-up cash.
  - Every other rebalance: resize existing positions to target risk
    (ATR-based) so each position contributes roughly equal volatility.

This is an independent implementation built from the strategy's published
rules -- it does not depend on backtrader and does not reuse any external
code. It is adapted for Indian markets:
  - NSE tickers (.NS suffix) via yfinance, NIFTY 50 (^NSEI) as the regime
    benchmark by default.
  - A simple Indian cost model (brokerage + STT + slippage) applied on
    every fill.
  - No survivorship-bias-free universe is available for free -- see the
    LIMITATIONS section in the README. Results here are indicative only.

Usage:
    python nse_momentum_clenow.py --start 2015-01-01 --end 2025-01-01

Requires: pandas, numpy, scipy, yfinance
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("nse_momentum_clenow")

# --------------------------------------------------------------------------
# Universe: NIFTY 200 constituents (as of file authoring). This is a static
# snapshot, NOT point-in-time / survivorship-bias-free -- see README.
# --------------------------------------------------------------------------
NIFTY_UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL",
    "SBIN", "LICI", "ITC", "HINDUNILVR", "LT", "BAJFINANCE", "HCLTECH",
    "MARUTI", "SUNPHARMA", "KOTAKBANK", "AXISBANK", "M&M", "ULTRACEMCO",
    "TITAN", "ADANIENT", "ADANIPORTS", "ONGC", "NTPC", "BAJAJFINSV",
    "ASIANPAINT", "COALINDIA", "POWERGRID", "WIPRO", "NESTLEIND", "JSWSTEEL",
    "BAJAJ-AUTO", "BALKRISIND", "TATASTEEL", "HAL", "GRASIM", "DMART",
    "TRENT", "SBILIFE", "HDFCLIFE", "TECHM", "CIPLA", "DRREDDY",
    "EICHERMOT", "APOLLOHOSP", "DIVISLAB", "BRITANNIA", "HINDALCO",
    "INDUSINDBK", "PIDILITIND", "GODREJCP", "DABUR", "SIEMENS", "PNB",
    "BANKBARODA", "CANBK", "AMBUJACEM", "SHREECEM", "IOC", "BPCL",
    "GAIL", "VEDL", "TATAPOWER", "ADANIPOWER", "ADANIGREEN", "DLF",
    "GODREJPROP", "OBEROIRLTY", "ETERNAL", "PAYTM", "NAUKRI", "INDIGO",
    "PIIND", "SRF", "UPL", "BEL", "BHEL", "IRCTC", "IRFC", "RVNL",
    "HAVELLS", "VOLTAS", "CROMPTON", "BOSCHLTD", "MOTHERSON", "MRF",
    "SHRIRAMFIN", "ASHOKLEY", "TVSMOTOR", "HEROMOTOCO", "BANDHANBNK",
    "FEDERALBNK", "IDFCFIRSTB", "AUBANK", "CHOLAFIN", "MUTHOOTFIN",
    "JIOFIN", "PERSISTENT", "COFORGE", "MPHASIS", "LTTS", "TATACOMM",
    "COLPAL", "MARICO", "BERGEPAINT", "PAGEIND", "JUBLFOOD", "UBL",
    "UNITDSPR", "VBL", "TORNTPHARM", "LUPIN", "AUROPHARMA", "ALKEM",
    "ZYDUSLIFE", "BIOCON", "GLENMARK", "SYNGENE", "MFSL", "ICICIPRULI",
    "ICICIGI", "SBICARD", "PFC", "RECLTD", "IEX", "CDSL", "BSE", "MCX",
    "POLYCAB", "KEI", "DIXON", "AMBER", "TIINDIA", "KPITTECH", "SONACOMS",
    "ABB", "CUMMINSIND", "SKFINDIA", "THERMAX", "GMRAIRPORT", "CONCOR",
]


@dataclasses.dataclass
class CostModel:
    """Approximate Indian equity delivery cost model (indicative)."""
    brokerage_pct: float = 0.0003     # discount broker delivery brokerage
    stt_pct: float = 0.001            # STT on delivery (sell side; approximated both sides)
    stamp_pct: float = 0.00015        # stamp duty (buy side)
    slippage_pct: float = 0.0005      # execution slippage

    def round_trip_cost_pct(self) -> float:
        return self.brokerage_pct + self.stt_pct + self.stamp_pct + self.slippage_pct


@dataclasses.dataclass
class Config:
    start: str
    end: str
    universe: list[str]
    benchmark: str = "^NSEI"
    momentum_window: int = 90
    trend_ma: int = 100
    regime_ma: int = 200
    atr_window: int = 20
    top_pct: float = 0.20
    risk_factor: float = 0.001       # 10 bps of portfolio value per ATR
    rebalance_every: int = 5         # trading days between rank/trim (weekly)
    resize_every: int = 10           # trading days between position resizing
    initial_capital: float = 1_000_000.0
    cost: CostModel = dataclasses.field(default_factory=CostModel)
    min_history: int = 100


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

EARLIEST_DATE = "2010-01-01"
CACHE_MAX_AGE_HOURS = 20  # refetch full history if cache is older than this


def download_data(cfg: Config, cache_dir: Path) -> dict[str, pd.DataFrame]:
    """Download OHLC data for the universe + benchmark via yfinance, with a
    local parquet cache so repeat runs don't re-hit the network.

    IMPORTANT: the cache always stores each ticker's FULL history
    (EARLIEST_DATE -> today), regardless of what --start/--end was passed.
    Every call then slices that full history down to cfg.start/cfg.end
    in-memory. This is what makes the cache safe to reuse across runs with
    different date ranges -- a previous bug cached whatever the first run's
    date range happened to be, so a later run with a different --start/--end
    silently got the wrong window back."""
    import yfinance as yf
    from datetime import datetime, timedelta

    cache_dir.mkdir(parents=True, exist_ok=True)
    data: dict[str, pd.DataFrame] = {}
    tickers = [cfg.benchmark] + [f"{t}.NS" for t in cfg.universe]
    today = datetime.now().strftime("%Y-%m-%d")
    stale_cutoff = datetime.now() - timedelta(hours=CACHE_MAX_AGE_HOURS)

    for ticker in tickers:
        cache_file = cache_dir / f"{ticker.replace('^', 'IDX_')}.parquet"
        df = None
        if cache_file.exists():
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if mtime >= stale_cutoff:
                df = pd.read_parquet(cache_file)

        if df is None:
            try:
                df = yf.download(
                    ticker, start=EARLIEST_DATE, end=today,
                    auto_adjust=True, progress=False,
                )
            except Exception as exc:  # pragma: no cover - network dependent
                log.warning("Download failed for %s: %s", ticker, exc)
                continue
            if df.empty:
                log.warning("No data returned for %s", ticker)
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.to_parquet(cache_file)

        sliced = df.loc[cfg.start:cfg.end]
        if len(sliced) >= cfg.min_history:
            data[ticker] = sliced
        else:
            log.info("Skipping %s: only %d bars in requested range", ticker, len(sliced))
    return data


# --------------------------------------------------------------------------
# Indicators (own implementation, vectorizable rolling apply)
# --------------------------------------------------------------------------

def exp_regression_momentum(close: pd.Series, window: int) -> pd.Series:
    """Annualized exponential regression slope * R^2 over a rolling window,
    computed on log(close). This is the core Clenow momentum score."""
    log_close = np.log(close)

    def _score(y: np.ndarray) -> float:
        if np.any(~np.isfinite(y)):
            return np.nan
        x = np.arange(len(y))
        slope, _, rvalue, _, _ = linregress(x, y)
        annualized = (1.0 + slope) ** 252
        return annualized * (rvalue ** 2)

    return log_close.rolling(window).apply(_score, raw=True)


def average_true_range(df: pd.DataFrame, window: int) -> pd.Series:
    high, low, prev_close = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


# --------------------------------------------------------------------------
# Backtest engine (simple event loop, no external backtesting dependency)
# --------------------------------------------------------------------------

@dataclasses.dataclass
class Position:
    ticker: str
    shares: float
    entry_price: float


class Portfolio:
    def __init__(self, cash: float):
        self.cash = cash
        self.positions: dict[str, Position] = {}

    def value(self, prices: dict[str, float]) -> float:
        v = self.cash
        for t, pos in self.positions.items():
            px = prices.get(t)
            if px is not None and np.isfinite(px):
                v += pos.shares * px
        return v

    def close_position(self, ticker: str, price: float, cost: CostModel) -> None:
        pos = self.positions.pop(ticker, None)
        if pos is None:
            return
        proceeds = pos.shares * price
        proceeds *= (1 - cost.round_trip_cost_pct())
        self.cash += proceeds

    def target_size(self, ticker: str, target_shares: float, price: float, cost: CostModel) -> None:
        current = self.positions.get(ticker)
        current_shares = current.shares if current else 0.0
        delta = target_shares - current_shares
        if abs(delta) < 1e-9:
            return
        trade_value = abs(delta) * price
        trade_cost = trade_value * cost.round_trip_cost_pct()
        if delta > 0:
            total_cost = trade_value + trade_cost
            if total_cost > self.cash:
                return  # not enough cash, skip rather than over-leverage
            self.cash -= total_cost
        else:
            self.cash += trade_value - trade_cost
        new_shares = current_shares + delta
        if new_shares <= 1e-9:
            self.positions.pop(ticker, None)
        else:
            self.positions[ticker] = Position(ticker, new_shares, price)


def run_backtest(cfg: Config, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if cfg.benchmark not in data:
        raise RuntimeError(f"Benchmark {cfg.benchmark} has no data; cannot run regime filter")

    bench = data[cfg.benchmark].copy()
    bench["sma_regime"] = bench["Close"].rolling(cfg.regime_ma).mean()

    stock_tickers = [t for t in data if t != cfg.benchmark]
    momentum: dict[str, pd.Series] = {}
    trend_sma: dict[str, pd.Series] = {}
    atr: dict[str, pd.Series] = {}
    for t in stock_tickers:
        df = data[t]
        momentum[t] = exp_regression_momentum(df["Close"], cfg.momentum_window)
        trend_sma[t] = df["Close"].rolling(cfg.trend_ma).mean()
        atr[t] = average_true_range(df, cfg.atr_window)

    all_dates = bench.index
    portfolio = Portfolio(cfg.initial_capital)
    equity_curve = []

    for i, date in enumerate(all_dates):
        prices = {t: data[t]["Close"].get(date, np.nan) for t in stock_tickers}
        prices = {t: p for t, p in prices.items() if np.isfinite(p)}

        do_rebalance = (i % cfg.rebalance_every == 0)
        do_resize = (i % cfg.resize_every == 0)

        if do_rebalance and i >= cfg.min_history:
            ranked = []
            for t in stock_tickers:
                if date not in data[t].index:
                    continue
                score = momentum[t].get(date, np.nan)
                px = prices.get(t)
                sma = trend_sma[t].get(date, np.nan)
                if not (np.isfinite(score) and np.isfinite(px) and np.isfinite(sma)):
                    continue
                ranked.append((t, score, px, sma))
            ranked.sort(key=lambda r: r[1], reverse=True)
            n = len(ranked)
            top_cutoff = max(1, int(n * cfg.top_pct))
            top_set = {r[0] for r in ranked[:top_cutoff]}
            rank_by_ticker = {r[0]: idx for idx, r in enumerate(ranked)}
            sma_by_ticker = {r[0]: r[3] for r in ranked}
            price_by_ticker = {r[0]: r[2] for r in ranked}

            # Exit: fell out of top percentile or below trend SMA
            for t in list(portfolio.positions.keys()):
                if t not in rank_by_ticker:
                    continue
                below_trend = prices.get(t, np.nan) < sma_by_ticker.get(t, np.inf)
                out_of_top = t not in top_set
                if out_of_top or below_trend:
                    portfolio.close_position(t, prices[t], cfg.cost)

            regime_ok = bench["Close"].get(date, np.nan) > bench["sma_regime"].get(date, np.inf)
            if regime_ok:
                for t in list(top_set):
                    if t in portfolio.positions or t not in prices:
                        continue
                    a = atr[t].get(date, np.nan)
                    if not np.isfinite(a) or a <= 0:
                        continue
                    value = portfolio.value(prices)
                    target_shares = (value * cfg.risk_factor) / a
                    portfolio.target_size(t, target_shares, prices[t], cfg.cost)

        elif do_resize and i >= cfg.min_history:
            regime_ok = bench["Close"].get(date, np.nan) > bench["sma_regime"].get(date, np.inf)
            if regime_ok:
                for t in list(portfolio.positions.keys()):
                    a = atr.get(t, pd.Series(dtype=float)).get(date, np.nan)
                    px = prices.get(t)
                    if not (np.isfinite(a) and a > 0 and px):
                        continue
                    value = portfolio.value(prices)
                    target_shares = (value * cfg.risk_factor) / a
                    portfolio.target_size(t, target_shares, px, cfg.cost)

        equity_curve.append({"date": date, "equity": portfolio.value(prices),
                              "n_positions": len(portfolio.positions)})

    return pd.DataFrame(equity_curve).set_index("date")


# --------------------------------------------------------------------------
# Performance metrics
# --------------------------------------------------------------------------

def performance_summary(equity: pd.Series, periods_per_year: int = 252) -> dict:
    returns = equity.pct_change().dropna()
    n_years = len(returns) / periods_per_year
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    sharpe = (returns.mean() / returns.std()) * np.sqrt(periods_per_year) if returns.std() > 0 else np.nan
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()
    return {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDrawdown": max_dd,
        "FinalEquity": equity.iloc[-1],
        "Years": n_years,
    }


def benchmark_buy_and_hold(bench: pd.DataFrame, initial_capital: float,
                            cost: CostModel) -> pd.Series:
    """Equity curve of buying the benchmark once and holding, net of one
    round-trip's worth of cost (entry + eventual exit), for a fair-ish
    comparison against the strategy's cost-inclusive numbers."""
    close = bench["Close"].dropna()
    entry_price = close.iloc[0]
    shares = (initial_capital * (1 - cost.round_trip_cost_pct())) / entry_price
    equity = shares * close
    equity.iloc[-1] = equity.iloc[-1] * (1 - cost.round_trip_cost_pct())
    return equity


def position_count_diagnostics(equity_df: pd.DataFrame) -> dict:
    n = equity_df["n_positions"]
    return {
        "AvgPositions": n.mean(),
        "MaxPositions": n.max(),
        "PctDaysFlat": float((n == 0).mean()),
    }


def walk_forward_report(cfg: Config, data: dict[str, pd.DataFrame], split_date: str) -> dict:
    """Run the strategy separately on the in-sample (start..split_date) and
    out-of-sample (split_date..end) windows, so a single lucky full-period
    run can't hide a strategy that only worked in one regime."""
    train_cfg = dataclasses.replace(cfg, end=split_date)
    test_cfg = dataclasses.replace(cfg, start=split_date)

    def _slice(d: dict[str, pd.DataFrame], start: str, end: str) -> dict[str, pd.DataFrame]:
        out = {}
        for t, df in d.items():
            sliced = df.loc[start:end]
            if len(sliced) >= cfg.min_history:
                out[t] = sliced
        return out

    train_data = _slice(data, cfg.start, split_date)
    test_data = _slice(data, split_date, cfg.end)

    train_eq = run_backtest(train_cfg, train_data)
    test_eq = run_backtest(test_cfg, test_data)

    return {
        "in_sample": performance_summary(train_eq["equity"]),
        "out_of_sample": performance_summary(test_eq["equity"]),
        "in_sample_curve": train_eq,
        "out_of_sample_curve": test_eq,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    parser.add_argument("--benchmark", default="^NSEI")
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--out", default="results")
    parser.add_argument("--cost-multiplier", type=float, default=1.0,
                         help="Scale all cost components (brokerage/STT/stamp/slippage) "
                              "by this factor, e.g. 2.0 to stress-test with double costs.")
    parser.add_argument("--split-date", default=None,
                         help="If set, also run an in-sample/out-of-sample walk-forward "
                              "report split at this date (YYYY-MM-DD).")
    parser.add_argument("--no-benchmark-compare", action="store_true",
                         help="Skip the benchmark buy-and-hold comparison.")
    args = parser.parse_args()

    cost = CostModel()
    if args.cost_multiplier != 1.0:
        cost = CostModel(*(v * args.cost_multiplier for v in dataclasses.astuple(cost)))

    cfg = Config(
        start=args.start,
        end=args.end,
        universe=NIFTY_UNIVERSE,
        benchmark=args.benchmark,
        initial_capital=args.capital,
        cost=cost,
    )

    log.info("Downloading data for %d tickers...", len(cfg.universe) + 1)
    data = download_data(cfg, Path(args.cache_dir))
    log.info("Loaded %d / %d tickers", len(data) - 1 if cfg.benchmark in data else len(data),
              len(cfg.universe))

    log.info("Running backtest...")
    equity_df = run_backtest(cfg, data)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    equity_df.to_csv(out_dir / "equity_curve.csv")

    stats = performance_summary(equity_df["equity"])
    print("\n--- NSE Momentum (Clenow-style) Backtest ---")
    if args.cost_multiplier != 1.0:
        print(f"(cost multiplier: {args.cost_multiplier}x)")
    for k, v in stats.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    pd.Series(stats).to_csv(out_dir / "summary.csv")

    diag = position_count_diagnostics(equity_df)
    print("\n--- Position diagnostics ---")
    for k, v in diag.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    pd.Series(diag).to_csv(out_dir / "position_diagnostics.csv")

    if not args.no_benchmark_compare and cfg.benchmark in data:
        bench_equity = benchmark_buy_and_hold(data[cfg.benchmark], cfg.initial_capital, cfg.cost)
        bench_stats = performance_summary(bench_equity)
        print(f"\n--- Benchmark buy-and-hold ({cfg.benchmark}) ---")
        for k, v in bench_stats.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        bench_equity.to_csv(out_dir / "benchmark_equity_curve.csv")
        pd.Series(bench_stats).to_csv(out_dir / "benchmark_summary.csv")

    if args.split_date:
        log.info("Running walk-forward split at %s...", args.split_date)
        wf = walk_forward_report(cfg, data, args.split_date)
        print(f"\n--- Walk-forward: in-sample ({cfg.start} to {args.split_date}) ---")
        for k, v in wf["in_sample"].items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        print(f"\n--- Walk-forward: out-of-sample ({args.split_date} to {cfg.end}) ---")
        for k, v in wf["out_of_sample"].items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        pd.Series(wf["in_sample"]).to_csv(out_dir / "walkforward_in_sample.csv")
        pd.Series(wf["out_of_sample"]).to_csv(out_dir / "walkforward_out_of_sample.csv")

    log.info("Results written to %s/", out_dir)


if __name__ == "__main__":
    main()
