"""
validate_rsi.py — Separate real edge from curve-fit for RSI GBPUSD M5.

Walk-forward across 3 independent 2-month periods, testing:
  BOTH (baseline) | SELL-only | BUY-only | TREND-filtered (200-EMA)
and checks whether the "toxic hours" are consistently negative across periods
(robust) or only in one (overfit). Throwaway diagnostic.
"""
from __future__ import annotations
import datetime as dt
import MetaTrader5 as mt5
import pandas as pd

from config import Config, TIMEFRAME_MAP
from strategies import get_strategy
from ab_backtest import run

cfg = Config()
SYMBOL, TF = "GBPUSD", "M5"

PERIODS = [
    ("P1 Jan-Feb", "2025-12-31", "2026-02-28"),
    ("P2 Mar-Apr", "2026-03-01", "2026-04-30"),
    ("P3 May-Jun", "2026-05-01", "2026-06-24"),
]


def fetch(date_from, date_to):
    rates = mt5.copy_rates_range(
        SYMBOL, TIMEFRAME_MAP[TF],
        dt.datetime.fromisoformat(date_from), dt.datetime.fromisoformat(date_to))
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    return df[["time", "open", "high", "low", "close", "volume"]].copy()


def line(tag, r):
    return (f"  {tag:16} trades={r['trades']:>4}  win%={r['win_rate']:>5}  "
            f"net=${r['net']:>9}  PF={str(r['pf']):>5}  maxDD=${r['max_dd']:>8}")


def main():
    if not mt5.initialize(path=cfg.mt5_path, login=cfg.mt5_login,
                          password=cfg.mt5_password, server=cfg.mt5_server):
        print("MT5 init failed:", mt5.last_error()); return
    si = mt5.symbol_info(SYMBOL)
    pt, dg = si.point, si.digits

    print(f"\n{'='*72}\nWALK-FORWARD VALIDATION — RSI {SYMBOL} {TF}, fixed 50/100\n{'='*72}")
    hour_nets = {}  # period -> {hour: net}
    for tag, d0, d1 in PERIODS:
        df = fetch(d0, d1)
        if df is None or len(df) < 300:
            print(f"\n{tag}: no data"); continue
        both = run(df, get_strategy("rsi"), SYMBOL, pt, dg, "OLD")
        sell = run(df, get_strategy("rsi"), SYMBOL, pt, dg, "OLD", direction_filter="SELL")
        buy = run(df, get_strategy("rsi"), SYMBOL, pt, dg, "OLD", direction_filter="BUY")
        trend = run(df, get_strategy("rsi"), SYMBOL, pt, dg, "OLD", trend_filter=True)
        print(f"\n{tag}  ({len(df)} bars)")
        print(line("BOTH", both)); print(line("SELL-only", sell))
        print(line("BUY-only", buy)); print(line("TREND-filtered", trend))
        # hour nets from the BOTH run
        hn = {}
        for h, pnl in both["trade_log"]:
            hn[h] = hn.get(h, 0) + pnl
        hour_nets[tag] = hn

    # Hour robustness: which hours are negative in ALL periods?
    print(f"\n{'='*72}\nHOUR ROBUSTNESS — net $ by hour per period (server time)\n{'='*72}")
    print(f"  {'hr':>3} " + "".join(f"{t.split()[0]:>10}" for t in hour_nets))
    all_neg, ever_pos = [], []
    for h in range(24):
        vals = [hour_nets.get(t, {}).get(h, 0) for t in hour_nets]
        if any(v != 0 for v in vals):
            print(f"  {h:>3} " + "".join(f"{v:>10.0f}" for v in vals))
        nz = [v for v in vals if v != 0]
        if nz and all(v < 0 for v in nz):
            all_neg.append(h)
        if any(v > 0 for v in nz):
            ever_pos.append(h)
    print(f"\n  ROBUST toxic hours (negative in EVERY period traded): {all_neg}")
    print(f"  Hours positive in at least one period: {ever_pos}")
    mt5.shutdown()


if __name__ == "__main__":
    main()
