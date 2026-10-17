"""
ab_backtest.py — One-off A/B harness comparing the OLD vs NEW trade logic.

Replays real MT5 history bar-by-bar (entry at next bar's open, intrabar SL/TP)
and runs each strategy twice:

* OLD  — fixed 50/100-pip SL/TP, EMA trend filter OFF, Bollinger gate OFF.
* NEW  — ATR-based SL (1.5xATR, clamped 8-60), TP = SL x 1.8, EMA trend
         filter ON, Bollinger confidence gate ON.

This is a diagnostic script, not part of the bot. Delete when done.
"""
from __future__ import annotations
import datetime as dt
import MetaTrader5 as mt5
import pandas as pd

from backtest_core import (
    pip_value_per_lot as _pip_value_per_lot,
    profit_factor,
)
from config import Config, TIMEFRAME_MAP
from strategies import get_strategy
from strategies.base import SignalAction
from utils import compute_atr, pips_to_price, price_to_pips

cfg = Config()


def pip_value_per_lot(symbol: str, point: float) -> float:
    """$ per pip per lot, via the shared instrument definitions.

    Previously computed with its own JPY special case, which disagreed
    with both the live engine and the backtester.
    """
    return _pip_value_per_lot(symbol, point, contract_size=100_000.0)


def resolve_sl_tp_new(window, symbol, point):
    """Mirror multi_bot._resolve_sl_tp for ATR stops + RR floor."""
    atr_price = compute_atr(window, cfg.atr_period)
    sl = 50.0
    if atr_price > 0:
        atr_pips = price_to_pips(atr_price * cfg.atr_sl_mult, symbol, point)
        if atr_pips > 0:
            sl = atr_pips
    sl = max(cfg.min_sl_pips, min(cfg.max_sl_pips, sl))
    return round(sl, 1), round(sl * cfg.min_rr_ratio, 1)


def session_for_hour(h):
    if 7 <= h < 12: return "London"
    if 12 <= h < 16: return "London/NY overlap"
    if 16 <= h < 21: return "New York"
    return "Asian/Off-hrs"


def run(df, strategy, symbol, point, digits, mode, lot=0.10, max_open=5,
        hour_window=None, direction_filter=None, trend_filter=False,
        trend_period=200):
    pv = pip_value_per_lot(symbol, point)
    total = len(df)
    warmup = min(200, total // 3)
    equity = 10000.0
    peak = equity
    max_dd = 0.0
    open_tr, closed, pending = [], [], None
    trades = []  # (entry_hour, pnl)

    for i in range(warmup, total):
        bar = df.iloc[i]
        # execute pending at this open
        if pending is not None and len(open_tr) < max_open:
            sig, sl_pips, tp_pips = pending
            ep = bar["open"]
            if sig.action == SignalAction.BUY:
                sl_p = round(ep - pips_to_price(sl_pips, symbol, point), digits)
                tp_p = round(ep + pips_to_price(tp_pips, symbol, point), digits)
                d = "BUY"
            else:
                sl_p = round(ep + pips_to_price(sl_pips, symbol, point), digits)
                tp_p = round(ep - pips_to_price(tp_pips, symbol, point), digits)
                d = "SELL"
            open_tr.append({"dir": d, "ep": ep, "sl": sl_p, "tp": tp_p,
                            "hour": int(bar["time"].hour)})
        pending = None

        # check SL/TP
        still = []
        for t in open_tr:
            hi, lo = bar["high"], bar["low"]
            exit_p = None
            if t["dir"] == "BUY":
                if lo <= t["sl"]:
                    exit_p = t["sl"]
                elif hi >= t["tp"]:
                    exit_p = t["tp"]
                delta = (exit_p - t["ep"]) if exit_p else 0
            else:
                if hi >= t["sl"]:
                    exit_p = t["sl"]
                elif lo <= t["tp"]:
                    exit_p = t["tp"]
                delta = (t["ep"] - exit_p) if exit_p else 0
            if exit_p is not None:
                pnl = price_to_pips(delta, symbol, point) * pv * lot
                equity += pnl
                peak = max(peak, equity)
                max_dd = max(max_dd, peak - equity)
                closed.append(pnl)
                trades.append((t["hour"], pnl))
            else:
                still.append(t)
        open_tr = still

        # signal on window
        window = df.iloc[max(0, i - 499):i + 1].reset_index(drop=True)
        try:
            sig = strategy.analyze(window, symbol)
        except Exception:
            sig = None
        if sig and sig.action in (SignalAction.BUY, SignalAction.SELL):
            h = int(bar["time"].hour)
            in_window = hour_window is None or (hour_window[0] <= h < hour_window[1])
            if direction_filter and sig.action.value != direction_filter:
                in_window = False
            if trend_filter and in_window:
                ema = window["close"].ewm(span=trend_period, adjust=False).mean().iloc[-1]
                price = window["close"].iloc[-1]
                if sig.action == SignalAction.BUY and not price > ema:
                    in_window = False
                if sig.action == SignalAction.SELL and not price < ema:
                    in_window = False
            if in_window:
                if mode == "NEW":
                    sl_pips, tp_pips = resolve_sl_tp_new(window, symbol, point)
                else:
                    sl_pips, tp_pips = 50.0, 100.0
                pending = (sig, sl_pips, tp_pips)

    # metrics
    n = len(closed)
    wins = [p for p in closed if p > 0]
    pf = profit_factor(closed)
    return {
        "mode": mode, "trades": n,
        "win_rate": round(len(wins) / n * 100, 1) if n else 0.0,
        "net": round(sum(closed), 2),
        "pf": round(pf, 2),
        "max_dd": round(max_dd, 2),
        "worst": round(min(closed), 2) if closed else 0.0,
        "trade_log": trades,
    }


def make(strategy_key, mode):
    filtered = mode in ("FILT", "NEW")
    if strategy_key == "ema":
        return get_strategy("ema", use_trend_filter=filtered, min_confidence=0.0)
    if strategy_key == "bollinger":
        return get_strategy("bollinger", min_confidence=(0.20 if filtered else 0.0))
    return get_strategy(strategy_key)  # rsi / macd unchanged


def main():
    ok = mt5.initialize(path=cfg.mt5_path, login=cfg.mt5_login,
                        password=cfg.mt5_password, server=cfg.mt5_server)
    if not ok:
        print("MT5 init failed:", mt5.last_error()); return

    date_to = dt.datetime.now()
    date_from = date_to - dt.timedelta(days=120)
    tf = TIMEFRAME_MAP["M15"]

    tests = [
        ("ema", "GBPUSD"), ("ema", "EURUSD"),
        ("bollinger", "XAUUSD"), ("bollinger", "GBPUSD"),
        ("rsi", "GBPUSD"), ("macd", "EURUSD"),
    ]

    print(f"\n{'='*78}\nA/B BACKTEST — M15, last 120 days, lot 0.10\n{'='*78}")
    session_agg = {}  # session -> [pnl] aggregated across all OLD runs
    rsi_session = {}  # RSI-only session breakdown
    for key, symbol in tests:
        si = mt5.symbol_info(symbol)
        rates = mt5.copy_rates_range(symbol, tf, date_from, date_to)
        if si is None or rates is None or len(rates) == 0:
            print(f"\n{key:10} {symbol:8} — no data ({mt5.last_error()})"); continue
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        df = df[["time", "open", "high", "low", "close", "volume"]].copy()
        point, digits = si.point, si.digits

        old = run(df, make(key, "OLD"), symbol, point, digits, "OLD")
        filt = run(df, make(key, "FILT"), symbol, point, digits, "FILT")
        new = run(df, make(key, "NEW"), symbol, point, digits, "NEW")
        print(f"\n{key.upper()} on {symbol}  ({len(df)} bars)")
        for r in (old, filt, new):
            print(f"  {r['mode']:4} trades={r['trades']:>4}  win%={r['win_rate']:>5}  "
                  f"net=${r['net']:>9}  PF={str(r['pf']):>5}  maxDD=${r['max_dd']:>8}  "
                  f"worst=${r['worst']:>8}")
        # accumulate session P&L from the recommended (OLD) config
        for h, pnl in old["trade_log"]:
            s = session_for_hour(h)
            session_agg.setdefault(s, []).append(pnl)
            if key == "rsi":
                rsi_session.setdefault(s, []).append(pnl)

    def show(title, agg):
        print(f"\n{title}")
        order = ["London", "London/NY overlap", "New York", "Asian/Off-hrs"]
        for s in order:
            v = agg.get(s)
            if not v:
                continue
            wins = sum(1 for p in v if p > 0)
            print(f"  {s:20} trades={len(v):>4}  win%={wins/len(v)*100:>5.1f}  "
                  f"net=${sum(v):>9.2f}  avg=${sum(v)/len(v):>7.2f}")

    print(f"\n{'='*78}\nTIME-OF-DAY (entry session, UTC) — recommended config across all tests\n{'='*78}")
    show("ALL STRATEGIES combined:", session_agg)
    show("RSI only (your edge):", rsi_session)
    mt5.shutdown()


if __name__ == "__main__":
    main()
