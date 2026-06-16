# Rider Infinity (RI)

> **Rider Infinity (RI)** — *Ride on the Infinity edge of Markets.* An automated Forex trading platform built on MetaTrader 5 with Python, featuring interactive CLI menus, pluggable strategies, real-time web dashboard, and comprehensive risk management.

---

## ✨ Features

| Category | Details |
|---|---|
| **Interactive CLI** | Colourful menu system for selecting symbols, timeframes, strategies, and lot sizes at runtime |
| **CLI / Headless Mode** | Pass `--strategy`, `--symbol`, `--timeframe`, `--lot`, `--dry-run` flags to skip menus |
| **Pluggable Strategies** | Register new strategies by dropping a file in `strategies/` — auto-discovered via `STRATEGY_REGISTRY` |
| **Risk Management** | Per-trade risk % sizing, daily drawdown cap, max open positions, spread filter |
| **Dry-Run Mode** | Simulate trades without touching your live account — toggle on/off from the menu |
| **Session Awareness** | Detects active Forex sessions (Sydney, Tokyo, London, New York) and overlap windows |
| **Trade Journal** | Every trade is logged to a local journal file for review |
| **Reconnection** | Auto-reconnects to MetaTrader 5 if the connection drops mid-session |

---

## 📋 Prerequisites

| Requirement | Minimum Version |
|---|---|
| Python | 3.8+ |
| MetaTrader 5 Terminal | Installed & logged in to a broker account |
| Broker Account | Demo or Live (demo recommended while testing) |
| OS | Windows (MT5 terminal is Windows-only) |

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/rider-infinity.git
cd rider-infinity

# 2. Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your .env file
copy .env.example .env
```

### `requirements.txt` (typical)

```
MetaTrader5>=5.0.45
pandas>=1.5
python-dotenv>=1.0
colorama>=0.4
```

---

## ⚙️ Configuration

All settings live in a `.env` file (loaded by `config.py`). Create one in the project root:

```dotenv
# ── MT5 Connection ──────────────────────────────
MT5_LOGIN=12345678
MT5_PASSWORD=YourPassword
MT5_SERVER=YourBroker-Demo
MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe

# ── Trading Defaults ────────────────────────────
DEFAULT_SYMBOL=EURUSD
DEFAULT_TIMEFRAME=M5
DEFAULT_LOT_SIZE=0.01

# ── Risk Management ────────────────────────────
MAX_RISK_PER_TRADE=1.0        # % of equity risked per trade
MAX_DAILY_DRAWDOWN=5.0        # % max daily loss before bot stops
MAX_OPEN_POSITIONS=3
MAX_SPREAD_POINTS=30          # reject trades if spread exceeds this

# ── SL / TP Defaults ───────────────────────────
DEFAULT_SL_PIPS=20
DEFAULT_TP_PIPS=40

# ── Misc ────────────────────────────────────────
MAGIC_NUMBER=234000
DRY_RUN=false                 # set to "true" to start in simulation mode
```

| Variable | Description |
|---|---|
| `MT5_LOGIN` | Your MT5 account number |
| `MT5_PASSWORD` | Account password |
| `MT5_SERVER` | Broker server name (visible in MT5 → File → Login) |
| `MT5_PATH` | Full path to `terminal64.exe` |
| `DEFAULT_SYMBOL` | Symbol the bot selects on startup |
| `DEFAULT_TIMEFRAME` | Candle timeframe key (`M1`, `M5`, `M15`, `M30`, `H1`, `H4`, `D1`) |
| `DEFAULT_LOT_SIZE` | Starting lot size for fixed-lot mode |
| `MAX_RISK_PER_TRADE` | Percentage of equity risked on a single trade |
| `MAX_DAILY_DRAWDOWN` | Daily drawdown limit (% of starting equity) |
| `MAX_OPEN_POSITIONS` | Maximum concurrent positions |
| `MAX_SPREAD_POINTS` | Spread threshold in points — trades are skipped when the spread is wider |
| `DEFAULT_SL_PIPS` | Default stop-loss distance in pips |
| `DEFAULT_TP_PIPS` | Default take-profit distance in pips |
| `MAGIC_NUMBER` | Unique identifier attached to every order placed by this bot |
| `DRY_RUN` | `true` to simulate all orders without touching the live account |

---

## 🖥️ Usage

### Interactive Mode

```bash
python bot.py
```

You'll see a colour-coded menu:

```
  ╔══════════════════════════════════════╗
  ║       RIDER INFINITY (RI) v1.0       ║
  ╠══════════════════════════════════════╣
  ║  1. Start Trading (Single Bot)       ║
  ║  2. Select Symbol                    ║
  ║  3. Select Timeframe                 ║
  ║  4. Select Strategy                  ║
  ║  5. Set Lot Size                     ║
  ║  6. View Account Info                ║
  ║  7. View Open Positions              ║
  ║  8. Close All Positions              ║
  ║  9. Toggle Dry Run                   ║
  ║ 10. Market Session Info              ║
  ║ 11. ⚡ Multi-Bot Manager             ║
  ║  0. Exit                             ║
  ╚══════════════════════════════════════╝
```

1. Pick a **symbol** (option 2) — categorised into Majors, Crosses, Metals, Exotics.
2. Pick a **timeframe** (option 3) — M1 through MN1.
3. Pick a **strategy** (option 4) — any strategy registered in `strategies/`.
4. Optionally set **lot size** (option 5) — fixed or risk-based auto sizing.
5. Hit **1** to begin automated trading.

### CLI / Non-Interactive Mode

```bash
# Fixed lot
python bot.py --strategy ema --symbol EURUSD --timeframe M5 --lot 0.01

# Auto lot sizing with dry run
python bot.py --strategy rsi --symbol GBPUSD --timeframe H1 --auto-lot --dry-run
```

| Flag | Description |
|---|---|
| `--strategy` | Strategy name (required for CLI mode) |
| `--symbol` | Trading symbol |
| `--timeframe` | Candle timeframe key |
| `--lot` | Fixed lot size |
| `--auto-lot` | Enable risk-based automatic lot sizing |
| `--dry-run` | Simulate trades without real execution |

---

## 📈 Strategies

Strategies live in the `strategies/` directory and are auto-registered. Each strategy subclasses `BaseStrategy` and implements an `analyze(df, symbol)` method that returns a `Signal`.

| Strategy | Description |
|---|---|
| **EMA Crossover** | Enters on fast/slow EMA crossover with trend-direction filter |
| **RSI Reversal** | Buys oversold / sells overbought RSI levels with confirmation candle |
| **MACD Histogram** | Trades MACD histogram zero-line crosses with signal-line confirmation |
| **Bollinger Bands** | Mean-reversion entries when price touches the outer bands |

> **Adding a new strategy:** Create `strategies/my_strategy.py`, subclass `BaseStrategy`, implement `analyze()`, and register it in `strategies/__init__.py`'s `STRATEGY_REGISTRY`.

---

## 🛡️ Risk Management

The `RiskManager` class enforces multiple safety layers **before** any order is sent:

1. **Per-Trade Risk Sizing** — Calculates lot size so that the stop-loss distance risks exactly *N%* of account equity (configurable via `MAX_RISK_PER_TRADE`).
2. **Daily Drawdown Cap** — Tracks equity at the start of each day. If the drawdown exceeds `MAX_DAILY_DRAWDOWN` %, all new trades are blocked.
3. **Max Open Positions** — No new positions when the limit (`MAX_OPEN_POSITIONS`) is reached.
4. **Spread Filter** — Skips trades when the current spread exceeds `MAX_SPREAD_POINTS`.
5. **Weekend Guard** — Prevents orders when the Forex market is closed.

---

## 🕐 Forex Sessions

The `SessionManager` detects which major sessions are currently active and suggests the best currency pairs:

| Session | Hours (UTC) | Best Pairs |
|---|---|---|
| **Sydney** | 22:00 – 07:00 | AUD/USD, NZD/USD, AUD/JPY |
| **Tokyo** | 00:00 – 09:00 | USD/JPY, EUR/JPY, AUD/JPY |
| **London** | 08:00 – 17:00 | EUR/USD, GBP/USD, EUR/GBP |
| **New York** | 13:00 – 22:00 | EUR/USD, USD/CAD, GBP/USD |

Session **overlaps** (e.g. London–New York, 13:00–17:00 UTC) typically offer the highest liquidity and tighter spreads.

---

## 📁 Project Structure

```
rider-infinity/
├── bot.py               # Main entry point — interactive & CLI modes
├── config.py            # Loads .env and exposes Config class
├── mt5_connector.py     # MT5 connection management
├── market_data.py       # Candle, tick, and symbol info retrieval
├── session_manager.py   # Forex session detection
├── order_manager.py     # Trade execution (open/close/modify)
├── risk_manager.py      # Pre-trade risk checks and lot sizing
├── utils.py             # Logging, formatting, helpers
├── strategies/
│   ├── __init__.py      # Strategy registry
│   ├── base.py          # BaseStrategy ABC and Signal dataclass
│   ├── ema_strategy.py  # EMA Crossover strategy
│   ├── rsi_strategy.py  # RSI Reversal strategy
│   └── ...
├── .env                 # Your local configuration (git-ignored)
├── .env.example         # Template configuration
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## ⚠️ Disclaimer

> **This software is provided for educational and research purposes only.**
>
> Forex trading involves substantial risk of loss and is not suitable for every investor. Past performance is not indicative of future results. The developers of this bot make **no guarantees** regarding the profitability or accuracy of any trading signals.
>
> - **Never trade with money you cannot afford to lose.**
> - Always test with a **demo account** before going live.
> - You are solely responsible for any financial decisions made using this software.
> - The authors and contributors accept **no liability** for losses incurred through the use of this bot.

---

## 📄 License

© Rider Infinity (RI). This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
