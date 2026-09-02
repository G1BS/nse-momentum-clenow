#!/usr/bin/env python3
"""
paper_trade.py

Forward paper-trading signal generator for the NSE momentum strategy in
nse_momentum_clenow.py. Reuses the exact same momentum/ATR/regime logic as
the backtest engine (no separate reimplementation to drift out of sync),
but operates on a persisted paper-portfolio state instead of a backtest
loop, and never places a real order -- it only tells you what to do.

Run this once per week (matching the strategy's weekly cadence). Each run:
  1. Downloads the latest available OHLC data for the universe + benchmark.
  2. Recomputes momentum rank, trend SMA, ATR, and the regime filter as of
     the most recent close.
  3. Diffs that against your current paper positions (stored in
     paper_state.json) and prints what to sell / buy / resize.
  4. Unless --dry-run is passed, applies those signals to the paper state
     at the latest close price (simulated fill, same cost model as the
     backtest) and appends the trades to paper_trades_log.csv.

This is a signal generator and bookkeeping tool only. It does not connect
to a broker and does not place real orders.

Usage:
    python paper_trade.py                     # generate signals, apply them
    python paper_trade.py --dry-run            # just show what would happen
    python paper_trade.py --init --capital 500000   # start a fresh paper portfolio
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nse_momentum_clenow import (
    NIFTY_UNIVERSE, Config, CostModel, download_data,
    exp_regression_momentum, average_true_range,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("paper_trade")

STATE_FILE = Path("docs/paper_state.json")
LOG_FILE = Path("docs/paper_trades_log.csv")
EQUITY_HISTORY_FILE = Path("docs/paper_equity_history.csv")


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------

def load_state(path: Path, capital: float) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {
        "cash": capital,
        "positions": {},   # ticker -> {"shares": float, "entry_price": float}
        "week_count": 0,
        "created": datetime.now().isoformat(),
        "last_run": None,
    }


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2, default=str)


def portfolio_value(state: dict, prices: dict[str, float]) -> float:
    v = state["cash"]
    for t, pos in state["positions"].items():
        px = prices.get(t)
        if px is not None and np.isfinite(px):
            v += pos["shares"] * px
    return v


# --------------------------------------------------------------------------
# Signal computation (mirrors run_backtest's per-step logic, but for the
# single most recent date only)
# --------------------------------------------------------------------------

def compute_signals(cfg: Config, data: dict[str, pd.DataFrame]) -> dict:
    bench = data[cfg.benchmark].copy()
    bench["sma_regime"] = bench["Close"].rolling(cfg.regime_ma).mean()
    latest_date = bench.index[-1]

    regime_ok = bool(bench["Close"].iloc[-1] > bench["sma_regime"].iloc[-1])

    stock_tickers = [t for t in data if t != cfg.benchmark]
    ranked = []
    prices, atrs = {}, {}
    for t in stock_tickers:
        df = data[t]
        if len(df) < cfg.min_history:
            continue
        mom = exp_regression_momentum(df["Close"], cfg.momentum_window).iloc[-1]
        sma = df["Close"].rolling(cfg.trend_ma).mean().iloc[-1]
        atr = average_true_range(df, cfg.atr_window).iloc[-1]
        px = df["Close"].iloc[-1]
        prices[t] = float(px)
        if not (np.isfinite(mom) and np.isfinite(sma) and np.isfinite(px)):
            continue
        atrs[t] = float(atr) if np.isfinite(atr) else np.nan
        ranked.append((t, float(mom), float(px), float(sma)))

    ranked.sort(key=lambda r: r[1], reverse=True)
    n = len(ranked)
    top_cutoff = max(1, int(n * cfg.top_pct))
    top_set = {r[0] for r in ranked[:top_cutoff]}
    sma_by_ticker = {r[0]: r[3] for r in ranked}

    return {
        "date": str(latest_date.date()),
        "regime_ok": regime_ok,
        "ranked": ranked,
        "top_set": top_set,
        "sma_by_ticker": sma_by_ticker,
        "prices": prices,
        "atrs": atrs,
    }


# --------------------------------------------------------------------------
# Apply signals to paper state
# --------------------------------------------------------------------------

def generate_actions(state: dict, sig: dict, cfg: Config, do_resize: bool) -> list[dict]:
    actions = []
    prices, atrs = sig["prices"], sig["atrs"]

    # Exits: fell out of top percentile or below trend SMA
    for t in list(state["positions"].keys()):
        if t not in prices:
            continue
        below_trend = prices[t] < sig["sma_by_ticker"].get(t, np.inf)
        out_of_top = t not in sig["top_set"]
        if out_of_top or below_trend:
            reason = "below_trend_sma" if below_trend else "out_of_top_momentum"
            actions.append({"action": "SELL", "ticker": t, "reason": reason,
                             "price": prices[t]})

    # Entries: top-ranked names not already held, only if regime is OK
    if sig["regime_ok"]:
        for t in sig["top_set"]:
            if t in state["positions"] or t not in prices:
                continue
            a = atrs.get(t, np.nan)
            if not np.isfinite(a) or a <= 0:
                continue
            value = portfolio_value(state, prices)
            target_shares = (value * cfg.risk_factor) / a
            actions.append({"action": "BUY", "ticker": t, "reason": "top_momentum_new",
                             "price": prices[t], "target_shares": round(target_shares, 2)})
    else:
        actions.append({"action": "NOTE", "ticker": None,
                         "reason": f"regime filter OFF ({cfg.benchmark} below its "
                                   f"{cfg.regime_ma}d SMA) -- no new entries this week",
                         "price": None})

    # Resize existing positions to current target ATR risk (every other week)
    if do_resize and sig["regime_ok"]:
        for t in list(state["positions"].keys()):
            if t in {a["ticker"] for a in actions if a["action"] == "SELL"}:
                continue
            a = atrs.get(t, np.nan)
            px = prices.get(t)
            if not (np.isfinite(a) and a > 0 and px):
                continue
            value = portfolio_value(state, prices)
            target_shares = (value * cfg.risk_factor) / a
            current_shares = state["positions"][t]["shares"]
            if abs(target_shares - current_shares) / max(current_shares, 1e-9) > 0.05:
                actions.append({"action": "RESIZE", "ticker": t, "reason": "atr_resize",
                                 "price": px, "target_shares": round(target_shares, 2),
                                 "current_shares": round(current_shares, 2)})

    return actions


def apply_actions(state: dict, actions: list[dict], cfg: Config, sig: dict) -> list[dict]:
    fills = []
    for act in actions:
        if act["action"] == "NOTE":
            continue
        t, px = act["ticker"], act["price"]

        if act["action"] == "SELL":
            pos = state["positions"].pop(t, None)
            if pos is None:
                continue
            proceeds = pos["shares"] * px * (1 - cfg.cost.round_trip_cost_pct())
            state["cash"] += proceeds
            fills.append({**act, "shares": pos["shares"], "cash_after": state["cash"]})

        elif act["action"] == "BUY":
            target_shares = act["target_shares"]
            cost_value = target_shares * px
            total_cost = cost_value * (1 + cfg.cost.round_trip_cost_pct())
            if total_cost > state["cash"]:
                fills.append({**act, "shares": 0, "note": "SKIPPED: insufficient paper cash"})
                continue
            state["cash"] -= total_cost
            state["positions"][t] = {"shares": target_shares, "entry_price": px}
            fills.append({**act, "shares": target_shares, "cash_after": state["cash"]})

        elif act["action"] == "RESIZE":
            target_shares = act["target_shares"]
            current_shares = state["positions"][t]["shares"]
            delta = target_shares - current_shares
            trade_value = abs(delta) * px
            trade_cost = trade_value * cfg.cost.round_trip_cost_pct()
            if delta > 0:
                total = trade_value + trade_cost
                if total > state["cash"]:
                    fills.append({**act, "note": "SKIPPED: insufficient paper cash"})
                    continue
                state["cash"] -= total
            else:
                state["cash"] += trade_value - trade_cost
            state["positions"][t]["shares"] = target_shares
            fills.append({**act, "cash_after": state["cash"]})

    return fills


def append_log(fills: list[dict], sig: dict, log_path: Path) -> None:
    if not fills:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{**f, "date": sig["date"]} for f in fills]
    df = pd.DataFrame(rows)
    header = not log_path.exists()
    df.to_csv(log_path, mode="a", header=header, index=False)


def append_equity_history(sig: dict, state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "date": sig["date"],
        "equity": portfolio_value(state, sig["prices"]),
        "cash": state["cash"],
        "n_positions": len(state["positions"]),
    }
    df = pd.DataFrame([row])
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def git_push_state(paths: list[str]) -> None:
    """Best-effort commit + push of the paper-trading data files, so a
    dashboard reading from the GitHub repo sees fresh data. Assumes the
    caller already has push access configured (same as any other git push
    from this machine) -- this does not handle auth."""
    import subprocess
    try:
        subprocess.run(["git", "add", *paths], check=True)
        result = subprocess.run(["git", "status", "--porcelain", *paths],
                                 capture_output=True, text=True, check=True)
        if not result.stdout.strip():
            log.info("No changes to paper-trading data files; skipping push.")
            return
        subprocess.run(["git", "commit", "-m",
                         f"Paper trading update {datetime.now().strftime('%Y-%m-%d %H:%M')}"],
                        check=True)
        subprocess.run(["git", "push"], check=True)
        log.info("Pushed paper-trading data to GitHub.")
    except subprocess.CalledProcessError as exc:
        log.warning("git push failed (%s) -- push manually if needed.", exc)
    except FileNotFoundError:
        log.warning("git not found on PATH -- push manually if needed.")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capital", type=float, default=500_000.0,
                         help="Starting paper capital (only used on first run / --init)")
    parser.add_argument("--benchmark", default="^NSEI")
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--state", default=str(STATE_FILE))
    parser.add_argument("--log", default=str(LOG_FILE))
    parser.add_argument("--equity-history", default=str(EQUITY_HISTORY_FILE))
    parser.add_argument("--dry-run", action="store_true",
                         help="Show signals only, do not update the paper portfolio")
    parser.add_argument("--init", action="store_true",
                         help="Explicitly start a fresh paper portfolio (overwrites existing state)")
    parser.add_argument("--push", action="store_true",
                         help="Commit and push the updated state/log/equity-history files to "
                              "the git remote after this run, so a dashboard reading from "
                              "GitHub sees fresh data. Requires push access already configured.")
    args = parser.parse_args()

    state_path = Path(args.state)
    if args.init and state_path.exists():
        state_path.unlink()
    state = load_state(state_path, args.capital)

    cfg = Config(
        start="2018-01-01",  # need enough history for 200d regime SMA etc.
        end=datetime.now().strftime("%Y-%m-%d"),
        universe=NIFTY_UNIVERSE,
        benchmark=args.benchmark,
        initial_capital=args.capital,
    )

    log.info("Fetching latest data...")
    data = download_data(cfg, Path(args.cache_dir))
    if cfg.benchmark not in data:
        raise RuntimeError(f"No data for benchmark {cfg.benchmark}")

    sig = compute_signals(cfg, data)
    do_resize = (state["week_count"] % 2 == 1)  # every other run

    print(f"\n=== Paper trading signals for {sig['date']} ===")
    print(f"Regime filter ({cfg.benchmark} > {cfg.regime_ma}d SMA): "
          f"{'ON (new entries allowed)' if sig['regime_ok'] else 'OFF (no new entries)'}")
    print(f"Current paper portfolio: {len(state['positions'])} positions, "
          f"cash Rs.{state['cash']:,.0f}, "
          f"value Rs.{portfolio_value(state, sig['prices']):,.0f}")

    actions = generate_actions(state, sig, cfg, do_resize)
    if not actions:
        print("\nNo actions this week.")
    else:
        print(f"\n{'Action':<8} {'Ticker':<12} {'Price':>10} {'Detail':<40}")
        for a in actions:
            detail = a["reason"]
            if "target_shares" in a:
                detail += f" -> target {a['target_shares']} shares"
            px_str = f"{a['price']:.2f}" if a["price"] is not None else ""
            print(f"{a['action']:<8} {str(a['ticker']):<12} {px_str:>10} {detail:<40}")

    if args.dry_run:
        print("\n[DRY RUN] Paper state was not modified.")
        return

    fills = apply_actions(state, actions, cfg, sig)
    append_log(fills, sig, Path(args.log))

    state["week_count"] += 1
    state["last_run"] = sig["date"]
    save_state(state_path, state)
    append_equity_history(sig, state, Path(args.equity_history))

    final_value = portfolio_value(state, sig["prices"])
    print(f"\nApplied. New paper portfolio value: Rs.{final_value:,.0f} "
          f"({len(state['positions'])} positions, cash Rs.{state['cash']:,.0f})")
    print(f"State saved to {state_path}, trades logged to {args.log}")

    if args.push:
        git_push_state([args.state, args.log, args.equity_history])


if __name__ == "__main__":
    main()
