"""
search_walkforward.py — Hunt for a robust config.

Runs every (strategy x symbol x timeframe) through 3 independent 2-month
periods with fixed 50/100 stops, and flags only the configs that are
profitable in ALL THREE periods (the real bar for an edge). Throwaway.
"""
from __future__ import annotations
import datetime as dt
import MetaTrader5 as mt5
import pandas as pd

from config import Config, TIMEFRAME_MAP
from strategies import get_strategy
from ab_backtest import run

cfg = Config()

STRATS = ["rsi", "macd", "bollinger", "ema"]
SYMBOLS = ["GBPUSD", "EURUSD", "AUDUSD", "USDJPY"]
TFS = ["M15", "M30", "H1"]
PERIODS = [
    ("P1", "2025-12-31", "2026-02-28"),
    ("P2", "2026-03-01", "2026-04-30"),
    ("P3", "2026-05-01", "2026-06-24"),
]


def fetch(symbol, tf, d0, d1):
    rates = mt5.copy_rates_range(symbol, TIMEFRAME_MAP[tf],
                                 dt.datetime.fromisoformat(d0),
                                 dt.datetime.fromisoformat(d1))
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

    results = []
    for symbol in SYMBOLS:
        si = mt5.symbol_info(symbol)
        if si is None:
            continue
        pt, dg = si.point, si.digits
        for tf in TFS:
            dfs = [fetch(symbol, tf, d0, d1) for _, d0, d1 in PERIODS]
            if any(d is None or len(d) < 300 for d in dfs):
                continue
            for strat in STRATS:
                nets, pfs, trades = [], [], 0
                for df in dfs:
                    r = run(df, get_strategy(strat), symbol, pt, dg, "OLD")
                    nets.append(r["net"]); trades += r["trades"]
                    pfs.append(r["pf"] if r["pf"] != "inf" else 9.99)
                results.append({
                    "cfg": f"{strat:10}{symbol:8}{tf:4}",
                    "n": trades, "nets": nets, "total": round(sum(nets), 0),
                    "all_pos": all(x > 0 for x in nets),
                    "min_pf": min(pfs) if pfs else 0,
                })

    print(f"\n{'='*82}\nWALK-FORWARD SEARCH — fixed 50/100, 3 periods, net $ per period\n{'='*82}")
    print(f"{'config':24}{'tot$':>8}  {'P1':>7}{'P2':>7}{'P3':>7}  {'trades':>7} {'allpos':>7}")
    for r in sorted(results, key=lambda x: x["total"], reverse=True):
        flag = "  <== ROBUST" if r["all_pos"] and r["n"] >= 30 else ""
        print(f"{r['cfg']}{r['total']:>8.0f}  "
              f"{r['nets'][0]:>7.0f}{r['nets'][1]:>7.0f}{r['nets'][2]:>7.0f}  "
              f"{r['n']:>7} {str(r['all_pos']):>7}{flag}")

    robust = [r for r in results if r["all_pos"] and r["n"] >= 30]
    print(f"\n{'='*82}\nROBUST configs (positive in ALL 3 periods, >=30 trades): "
          f"{len(robust)}\n{'='*82}")
    for r in sorted(robust, key=lambda x: x["total"], reverse=True):
        print(f"  {r['cfg']}  total=${r['total']:.0f}  periods={r['nets']}  "
              f"trades={r['n']}  minPF={r['min_pf']}")
    if not robust:
        print("  NONE. No simple config is profitable across all 3 periods.")
    mt5.shutdown()


if __name__ == "__main__":
    main()
