"""Guards the single pip definition across every layer.

Pip size used to be computed inline in six places by five different
rules.  They disagreed by 10x on gold and on every 3-digit JPY pair, and
by 1000x on gold in the cloud worker.  Because the risk layer sized lots
with one rule while the order layer placed stops with another, a trade
carried different risk than it was sized for.

These tests fail if any layer grows its own pip arithmetic again.
"""

from __future__ import annotations

import pytest

import backtest_core
import instruments
import trade_manager
import utils

# (symbol, point) pairs spanning every quote convention in the project.
INSTRUMENTS = [
    ("EURUSD", 0.00001),   # 5-digit major
    ("GBPUSD", 0.0001),    # legacy 4-digit major
    ("USDJPY", 0.001),     # 3-digit JPY
    ("GBPJPY", 0.001),
    ("EURJPY", 0.01),      # legacy 2-digit JPY
    ("XAUUSD", 0.01),      # gold
    ("XAGUSD", 0.001),     # silver
    ("BTCUSD", 0.01),      # crypto
]

#: Every module that exposes pip arithmetic to callers.
PIP_MODULES = [instruments, utils, backtest_core, trade_manager]


@pytest.mark.parametrize("symbol,point", INSTRUMENTS)
@pytest.mark.parametrize("module", PIP_MODULES, ids=lambda m: m.__name__)
def test_every_layer_agrees_on_pip_size(module, symbol, point):
    assert module.pip_size(symbol, point) == instruments.pip_size(
        symbol, point,
    )


@pytest.mark.parametrize("symbol,point", INSTRUMENTS)
@pytest.mark.parametrize("module", PIP_MODULES, ids=lambda m: m.__name__)
def test_every_layer_agrees_on_pips_to_price(module, symbol, point):
    assert module.pips_to_price(25, symbol, point) == pytest.approx(
        instruments.pips_to_price(25, symbol, point)
    )


@pytest.mark.parametrize("symbol,point", INSTRUMENTS)
def test_risk_sizing_and_stop_placement_use_one_pip(symbol, point, mt5,
                                                    monkeypatch, config):
    """The core consistency requirement.

    The distance ``order_manager`` places a stop at must equal the
    distance ``risk_manager`` sized the lot for, or the realised loss
    will not match the intended risk percentage.
    """
    import order_manager
    import risk_manager as rm

    sl_pips = 20.0
    # What order_manager.set_breakeven / open_position place:
    placed_distance = order_manager.pips_to_price(sl_pips, symbol, point)

    # What risk_manager implicitly assumes when converting pips to money:
    sized_pip = rm.pip_value_per_lot(
        symbol, point, contract_size=100_000.0,
    ) / 100_000.0

    assert placed_distance == pytest.approx(sized_pip * sl_pips)


@pytest.mark.parametrize("symbol,point", INSTRUMENTS)
def test_backtest_and_live_measure_the_same_move(symbol, point):
    """A strategy validated in a backtest must trade the same way live.

    The backtest reports a move in pips; the live engine places stops in
    pips. If those units differ, backtest results do not transfer.
    """
    move = 10 * instruments.pip_size(symbol, point)
    backtest_pips = backtest_core.calc_pnl_pips(
        "BUY", 1000.0, 1000.0 + move, symbol, point,
    )
    live_pips = trade_manager.price_to_pips(move, symbol, point)
    assert backtest_pips == pytest.approx(live_pips) == pytest.approx(10.0)


def test_no_module_reintroduces_inline_pip_math():
    """A literal 100*point / 0.0001 pip rule must not come back."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    # The one legitimate home for the rule.
    allowed = {"instruments.py"}
    patterns = [
        re.compile(r"100\s*\*\s*point"),
        re.compile(r"point\s*\*\s*100\b"),
        re.compile(r'"JPY"\s+in\s+\w+'),
    ]
    for path in root.glob("**/*.py"):
        if "tests" in path.parts or path.name in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("*"):
                continue
            if any(p.search(line) for p in patterns):
                offenders.append(f"{path.relative_to(root)}:{line_no}: {stripped}")

    assert not offenders, (
        "Inline pip arithmetic outside instruments.py:\n  "
        + "\n  ".join(offenders)
    )
