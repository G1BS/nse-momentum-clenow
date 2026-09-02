#!/usr/bin/env python3
"""
dashboard.py

Streamlit dashboard for tracking the NSE momentum paper-trading portfolio.
Reads paper_state.json, paper_trades_log.csv, and paper_equity_history.csv
from this repo -- it does not fetch data or place orders itself. Run
`paper_trade.py --push` locally each week to keep these files (and the
deployed dashboard) up to date.

Local usage:
    streamlit run dashboard.py

Deployed usage: point Streamlit Community Cloud at this repo/file (see
README.md "Dashboard deployment" section).
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="NSE Momentum Paper Trading", layout="wide")

STATE_PATH = Path("paper_state.json")
LOG_PATH = Path("paper_trades_log.csv")
EQUITY_PATH = Path("paper_equity_history.csv")


@st.cache_data(ttl=300)
def load_state():
    if not STATE_PATH.exists():
        return None
    with open(STATE_PATH) as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_log():
    if not LOG_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(LOG_PATH)


@st.cache_data(ttl=300)
def load_equity_history():
    if not EQUITY_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(EQUITY_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


st.title("NSE Momentum — Paper Trading Tracker")
st.caption(
    "Forward paper-trading record for the Clenow-style momentum strategy. "
    "No real orders are placed; this reflects simulated fills only."
)

state = load_state()
log_df = load_log()
equity_df = load_equity_history()

if state is None:
    st.warning(
        "No paper_state.json found yet. Run `python paper_trade.py --init` "
        "locally, then `python paper_trade.py --push` to publish data here."
    )
    st.stop()

# ---- Top-line metrics ----
col1, col2, col3, col4 = st.columns(4)
current_value = equity_df["equity"].iloc[-1] if not equity_df.empty else state["cash"]
initial_value = equity_df["equity"].iloc[0] if not equity_df.empty else state["cash"]
total_return_pct = (current_value / initial_value - 1) * 100 if initial_value else 0.0

col1.metric("Portfolio Value", f"Rs.{current_value:,.0f}",
            f"{total_return_pct:+.2f}% since start")
col2.metric("Cash", f"Rs.{state['cash']:,.0f}")
col3.metric("Open Positions", len(state["positions"]))
col4.metric("Last Run", state.get("last_run", "—"))

st.divider()

# ---- Equity curve ----
st.subheader("Equity curve")
if equity_df.empty:
    st.info("No equity history yet — accumulates one point per weekly run.")
else:
    st.line_chart(equity_df.set_index("date")[["equity"]])
    with st.expander("Cash and position count over time"):
        st.line_chart(equity_df.set_index("date")[["cash"]])
        st.bar_chart(equity_df.set_index("date")[["n_positions"]])

st.divider()

# ---- Current holdings ----
st.subheader("Current holdings")
if not state["positions"]:
    st.info("No open positions.")
else:
    holdings = pd.DataFrame([
        {"Ticker": t, "Shares": p["shares"], "Entry Price": p["entry_price"],
         "Cost Basis": p["shares"] * p["entry_price"]}
        for t, p in state["positions"].items()
    ]).sort_values("Cost Basis", ascending=False)
    st.dataframe(holdings, use_container_width=True, hide_index=True)

st.divider()

# ---- Trade log ----
st.subheader("Trade log")
if log_df.empty:
    st.info("No trades logged yet.")
else:
    st.dataframe(log_df.sort_values("date", ascending=False),
                 use_container_width=True, hide_index=True)
    csv = log_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download full trade log (CSV)", csv, "paper_trades_log.csv")

st.caption(
    "Data refreshes on each `paper_trade.py --push` run (weekly cadence). "
    "This dashboard is read-only — it cannot place trades."
)
