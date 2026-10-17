"""
backtest_report.py — Backtest every strategy and rank them.

Answers "which of my strategies actually make money, on my broker's
data, after costs" — across any set of symbols and timeframes, in one
run, as a sortable table plus a CSV.

**Run this on the machine with MetaTrader 5.** It pulls bars through the
terminal, so it needs Windows and a logged-in MT5 with the symbols open
in Market Watch. There is no way around that: the broker's own history
is the only data that tells you anything about your own trading.

Usage
-----
    python scripts/backtest_report.py
    python scripts/backtest_report.py --symbols EURUSD,GBPUSD,XAUUSD
    python scripts/backtest_report.py --timeframes M15,H1 --days 365
    python scripts/backtest_report.py --strategies ema,rsi,scalper
    python scripts/backtest_report.py --walk-forward

Reading the output
------------------
``profit factor`` is gross profit over gross loss; above 1.0 is
profitable, and anything above ~3 on a few dozen trades is usually a
small sample rather than an edge.  ``trades`` is the column to check
first: a result built on under ~30 trades says almost nothing, however
good it looks.

``--walk-forward`` is the one that matters before risking money. A
plain backtest reports how a strategy did on the period you chose, with
parameters you may have picked because they suited that period.
Walk-forward optimises on one window and scores the next, so its
``robustness`` column is how much of the performance survived on data
the optimiser never saw. Below ~0.5 means the backtest was curve-fitting.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# MT5-dependent imports happen inside main() so --help works on any
# machine; only the actual run needs the Windows terminal.
from strategies import STRATEGY_REGISTRY  # noqa: E402

DEFAULT_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
DEFAULT_TIMEFRAMES = ["M15", "H1"]

#: Below this, a result is a small sample rather than a measurement.
MIN_MEANINGFUL_TRADES = 30


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtest every strategy across symbols and timeframes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--symbols", default=",".join(DEFAULT_SYMBOLS),
        help=f"Comma-separated (default: {','.join(DEFAULT_SYMBOLS)}).",
    )
    p.add_argument(
        "--timeframes", default=",".join(DEFAULT_TIMEFRAMES),
        help=f"Comma-separated (default: {','.join(DEFAULT_TIMEFRAMES)}).",
    )
    p.add_argument(
        "--strategies", default="",
        help="Comma-separated registry keys. Default: all of them.",
    )
    p.add_argument(
        "--days", type=int, default=180,
        help="How far back to test (default: 180).",
    )
    p.add_argument("--lot", type=float, default=0.10, help="Lot size.")
    p.add_argument(
        "--balance", type=float, default=10_000.0, help="Starting balance.",
    )
    p.add_argument(
        "--walk-forward", action="store_true",
        help="Also run walk-forward and report the robustness ratio. Much "
             "slower, and the number that actually matters.",
    )
    p.add_argument(
        "--csv", default="backtest_report.csv",
        help="Where to write the full results.",
    )
    return p.parse_args()


def _await_run(engine, run_id: str, timeout: float = 600.0):
    """Block until a backtest finishes, then return its results.

    ``BacktestEngine.run_async`` starts a worker thread and returns the
    run id straight away, so the results are not there yet.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = engine.get_status(run_id)
        if status and status.get("status") in ("completed", "failed"):
            return engine.get_results(run_id) or status
        time.sleep(0.25)
    return None


def _fmt(value, width=9, places=2):
    if value is None:
        return "—".rjust(width)
    return f"{value:>{width}.{places}f}"


def main() -> int:
    args = _parse_args()

    try:
        from config import Config, ConfigError, TIMEFRAME_MAP
        from mt5_connector import MT5Connector
    except ModuleNotFoundError as exc:
        if "MetaTrader5" in str(exc):
            print(
                "\n  This script needs the MetaTrader5 package, which is\n"
                "  Windows-only. Run it on the machine where your MT5\n"
                "  terminal is installed and logged in.\n"
            )
            return 1
        raise

    try:
        config = Config()
    except ConfigError as exc:
        print(f"[CONFIG ERROR] {exc}")
        return 1

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    timeframes = [t.strip().upper() for t in args.timeframes.split(",") if t.strip()]
    if args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
        unknown = [s for s in strategies if s not in STRATEGY_REGISTRY]
        if unknown:
            print(f"Unknown strategies: {', '.join(unknown)}")
            print(f"Available: {', '.join(sorted(STRATEGY_REGISTRY))}")
            return 1
    else:
        strategies = list(STRATEGY_REGISTRY)

    bad_tf = [t for t in timeframes if t not in TIMEFRAME_MAP]
    if bad_tf:
        print(f"Unknown timeframes: {', '.join(bad_tf)}")
        print(f"Available: {', '.join(TIMEFRAME_MAP)}")
        return 1

    to_date = datetime.now(timezone.utc).date()
    from_date = to_date - timedelta(days=args.days)

    print(f"\n  Connecting to MetaTrader 5 …")
    connector = MT5Connector(config)
    if not connector.connect():
        print(
            "  Could not connect. This script needs a running, logged-in MT5\n"
            "  terminal on this machine — check MT5_PATH, MT5_LOGIN and\n"
            "  MT5_SERVER in your .env."
        )
        return 1

    total = len(strategies) * len(symbols) * len(timeframes)
    print(
        f"\n  {len(strategies)} strategies x {len(symbols)} symbols x "
        f"{len(timeframes)} timeframes = {total} backtests\n"
        f"  Period: {from_date} to {to_date}  |  lot {args.lot}  |  "
        f"balance {args.balance:,.0f}\n"
    )

    # Imported here so a config or connection failure reports first.
    import validation
    from backtester import BacktestEngine

    engine = BacktestEngine(config=config, connector=connector)
    rows: list[dict] = []

    for symbol in symbols:
        for timeframe in timeframes:
            print(f"  ── {symbol} {timeframe} " + "─" * 46)
            header = (
                f"  {'strategy':<12} {'trades':>7} {'win %':>7} "
                f"{'net $':>10} {'PF':>7} {'max DD %':>9}"
            )
            if args.walk_forward:
                header += f" {'robustness':>11}"
            print(header)

            for key in strategies:
                try:
                    run_id = engine.run_async(
                        strategy_name=key, symbol=symbol,
                        timeframe_key=timeframe,
                        date_from=from_date.isoformat(),
                        date_to=to_date.isoformat(),
                        initial_balance=args.balance, lot_size=args.lot,
                    )
                    results = _await_run(engine, run_id)
                except Exception as exc:
                    print(f"  {key:<12} failed: {str(exc)[:50]}")
                    continue
                if results is None:
                    print(f"  {key:<12} (timed out or no result)")
                    continue
                if results.get("status") == "failed":
                    msg = str(results.get("error_msg") or "")[:50]
                    print(f"  {key:<12} failed: {msg}")
                    continue

                robustness = None
                if args.walk_forward:
                    try:
                        wf = validation.run_walk_forward(
                            key, symbol, timeframe,
                            from_date.isoformat(), to_date.isoformat(),
                            config=config, connector=connector,
                        )
                        robustness = wf.robustness_ratio
                    except Exception:
                        robustness = None

                row = {
                    "symbol": symbol, "timeframe": timeframe, "strategy": key,
                    "trades": results.get("total_trades", 0),
                    "win_rate": results.get("win_rate", 0.0),
                    "net_money": results.get("total_pnl_money", 0.0),
                    "profit_factor": results.get("profit_factor", 0.0),
                    "max_drawdown_pct": results.get("max_drawdown_pct", 0.0),
                    "robustness_ratio": robustness,
                }
                rows.append(row)

                thin = "" if row["trades"] >= MIN_MEANINGFUL_TRADES else "  (thin)"
                line = (
                    f"  {key:<12} {row['trades']:>7} "
                    f"{_fmt(row['win_rate'], 7, 1)} "
                    f"{_fmt(row['net_money'], 10)} "
                    f"{_fmt(row['profit_factor'], 7)} "
                    f"{_fmt(row['max_drawdown_pct'], 9, 1)}"
                )
                if args.walk_forward:
                    line += f" {_fmt(robustness, 11, 3)}"
                print(line + thin)
            print()

    connector.disconnect()

    if not rows:
        print("  No results. Check that the symbols exist on your broker.")
        return 1

    # ── Ranking ──────────────────────────────────────────────────────
    print("  " + "=" * 64)
    print(f"  Best by profit factor (>= {MIN_MEANINGFUL_TRADES} trades only)")
    print("  " + "=" * 64)
    solid = [r for r in rows if r["trades"] >= MIN_MEANINGFUL_TRADES]
    if not solid:
        print(
            f"  Nothing cleared {MIN_MEANINGFUL_TRADES} trades. Widen --days "
            f"or use a faster timeframe before reading anything into these."
        )
    else:
        solid.sort(key=lambda r: r["profit_factor"], reverse=True)
        print(f"  {'strategy':<12} {'symbol':<9} {'tf':<5} {'trades':>7} "
              f"{'net $':>10} {'PF':>7}")
        for r in solid[:15]:
            print(f"  {r['strategy']:<12} {r['symbol']:<9} {r['timeframe']:<5} "
                  f"{r['trades']:>7} {_fmt(r['net_money'], 10)} "
                  f"{_fmt(r['profit_factor'], 7)}")

        profitable = [r for r in solid if r["net_money"] > 0]
        print(
            f"\n  {len(profitable)} of {len(solid)} well-sampled combinations "
            f"were profitable over this period."
        )
        if not args.walk_forward:
            print(
                "  These are in-sample results. Re-run with --walk-forward\n"
                "  before trusting any of them: a strategy that looks good on\n"
                "  one period and fails out of sample was curve-fitted."
            )

    out = Path(args.csv)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Full results: {out.resolve()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
