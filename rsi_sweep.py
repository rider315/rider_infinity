"""
rsi_sweep.py — Find the best timeframe/symbol for the RSI strategy.

Runs RSI (fixed 50/100 stops = the recommended config) across several
timeframes and symbols on real MT5 history, both unrestricted and restricted
to the active London+NY window. Throwaway diagnostic.
"""
from __future__ import annotations
import datetime as dt
import MetaTrader5 as mt5
import pandas as pd

from config import Config, TIMEFRAME_MAP
from strategies import get_strategy
from ab_backtest import run  # reuse the same simulation engine

cfg = Config()

SYMBOLS = ["GBPUSD", "EURUSD", "USDJPY", "AUDUSD", "XAUUSD"]
TFS = ["M5", "M15", "M30", "H1"]
DAYS = 120
HOURS = (7, 21)  # London + New York (server time ~UTC+2/3, directional)


def fetch(symbol, tf_key):
    rates = mt5.copy_rates_range(symbol, TIMEFRAME_MAP[tf_key],
                                 dt.datetime.now() - dt.timedelta(days=DAYS),
                                 dt.datetime.now())
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    return df[["time", "open", "high", "low", "close", "volume"]].copy()


def main():
    if not mt5.initialize(path=cfg.mt5_path, login=cfg.mt5_login,
                          password=cfg.mt5_password, server=cfg.mt5_server):
        print("MT5 init failed:", mt5.last_error()); return

    print(f"\n{'='*86}\nRSI SWEEP — fixed 50/100 stops, {DAYS} days, lot 0.10\n"
          f"net$ = all hours | net$(7-21) = London+NY only\n{'='*86}")
    print(f"{'symbol':8} {'TF':4} {'trades':>7} {'win%':>6} {'net$':>10} "
          f"{'PF':>6} {'maxDD$':>9} | {'tr(7-21)':>8} {'net$(7-21)':>11} {'PF':>6}")
    best = []
    for symbol in SYMBOLS:
        si = mt5.symbol_info(symbol)
        if si is None:
            print(f"{symbol:8} — no symbol info"); continue
        for tf in TFS:
            df = fetch(symbol, tf)
            if df is None or len(df) < 300:
                print(f"{symbol:8} {tf:4} — no data"); continue
            a = run(df, get_strategy("rsi"), symbol, si.point, si.digits, "OLD")
            b = run(df, get_strategy("rsi"), symbol, si.point, si.digits, "OLD",
                    hour_window=HOURS)
            print(f"{symbol:8} {tf:4} {a['trades']:>7} {a['win_rate']:>6} "
                  f"{a['net']:>10} {str(a['pf']):>6} {a['max_dd']:>9} | "
                  f"{b['trades']:>8} {b['net']:>11} {str(b['pf']):>6}")
            best.append((b["net"], symbol, tf, b))

    print(f"\n{'='*86}\nTOP 5 by net$ (London+NY window):\n{'='*86}")
    for net, symbol, tf, r in sorted(best, key=lambda x: x[0], reverse=True)[:5]:
        print(f"  {symbol:8} {tf:4}  net=${net:>9}  trades={r['trades']:>4}  "
              f"win%={r['win_rate']:>5}  PF={r['pf']}  maxDD=${r['max_dd']}")
    mt5.shutdown()


if __name__ == "__main__":
    main()
