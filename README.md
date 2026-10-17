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

### Running it

```bash
# Web dashboard (binds to 127.0.0.1:5000 by default)
python dashboard.py

# Interactive CLI
python bot.py
```

> **Linux / macOS:** `MetaTrader5` is a Windows-only package, so the
> live trading paths need Windows. Everything else — the backtester,
> optimiser, validation tools and the test-suite — runs anywhere.

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

| Key | Strategy | Description |
|---|---|---|
| `ema` | EMA Crossover | Fast/slow EMA cross with a trend filter |
| `rsi` | RSI Reversal | Mean reversion from oversold / overbought |
| `bollinger` | Bollinger Bands | Breakout and squeeze entries |
| `macd` | MACD | Histogram zero-cross momentum |
| `donchian` | Donchian Momentum | CTA-style channel breakout |
| `meanrev` | Statistical Mean Reversion | Session fade on a z-score |
| `orb` | Opening Range Breakout | Session-open volatility expansion |
| `sweep` | Liquidity Sweep | Stop-run reversal |
| `regime` | Regime Switch | Adapts between trend and range |
| `scalper` | Micro-Trend Scalper | Fast breakout; pair with Quick Exit |
| `grid` | Grid Trading | Harvests volatility around a centre |
| `dca` | DCA | Dollar-cost averaging into a position |
| `news` | News Intelligence | Claude-scored sentiment signals |
| `dsl` | DSL | Your own rules, no Python required |

> **Adding a new strategy:** Create `strategies/my_strategy.py`, subclass `BaseStrategy`, implement `analyze()`, and register it in `strategies/__init__.py`'s `STRATEGY_REGISTRY`.

---

## 🛡️ Risk Management

Risk is enforced in two layers, and both engines (the CLI and the
dashboard fleet) go through the same code.

**Before an order is sent** — `RiskManager`:

1. **Market open** — delegates to `SessionManager`; crypto trades 24/7.
2. **Trading-hours window** — optional UTC gate (`TRADE_HOUR_START/END`).
3. **Per-trade risk sizing** — lot sized so the stop risks exactly
   *N%* of equity (`MAX_RISK_PER_TRADE`).
4. **Hard per-trade loss cap** — shrinks or rejects the lot so one trade
   can never lose more than `MAX_LOSS_PER_TRADE_PCT`.
5. **Daily drawdown cap** — blocks new trades past `MAX_DAILY_DRAWDOWN`.
6. **Position limits** — `MAX_POSITIONS_PER_BOT` per bot, plus the
   fleet-wide `MAX_OPEN_POSITIONS` ceiling.
7. **Spread filter** — skips entries wider than `MAX_SPREAD_POINTS`.
8. **Correlation awareness** — flags same-direction exposure in
   correlated pairs.

**While a trade is open** — `trade_manager`:

1. **Volatility-sized stops** — `ATR_SL_MULT × ATR`, clamped to
   `[MIN_SL_PIPS, MAX_SL_PIPS]`, with a `MIN_RR_RATIO` floor on the
   target. Crypto is clamped to a percentage of price instead, since a
   60-pip stop on a $60k instrument is dust.
2. **Breakeven** — once a trade is `BREAKEVEN_TRIGGER_R` of its risk in
   profit, the stop moves to entry. This is what stops losers running to
   the full stop while winners get banked early.
3. **Time-stop** — force-closes positions held past `MAX_TRADE_HOURS`,
   which will hit neither stop nor target and only tie up margin.
4. **Post-loss cooldown** — after a *losing* stop-out, blocks re-entry
   for `SL_COOLDOWN_MIN`. A breakout signal stays valid for several
   candles, so without this the bot re-enters the move that just
   stopped it out.
5. **Quick Exit** — optional per-bot scalp exit that ratchets the stop
   behind price (`QUICK_EXIT_TRAIL_PIPS`) instead of hard-closing.
6. **Global pause** — an automation rule or the dashboard can block all
   new entries while leaving open positions managed.

---

## 🔒 Dashboard Security

The dashboard can place orders, flip bots to live and close every
position, so reaching it is equivalent to holding the account.

* It binds to **`127.0.0.1` by default** — this machine only.
* Binding to a reachable address **requires `DASHBOARD_PASSWORD`**;
  without one, startup fails rather than exposing the account.
* The password enables HTTP Basic auth on every route, so a new
  endpoint cannot forget it. The browser prompts once and the existing
  UI keeps working unchanged.
* State-changing requests from another origin are rejected.
* The Werkzeug debugger allows arbitrary code execution and is refused
  on any non-loopback bind.

To reach it from another machine, prefer an SSH tunnel:

```bash
ssh -L 5000:127.0.0.1:5000 user@trading-host
```

---

## 🧪 Validation

Beyond a single backtest, three tools test whether a result is real
(CLI: option 14; dashboard: `/api/validate/*`):

| Tool | Answers |
|---|---|
| **Genetic search** | Finds good parameters without enumerating a grid too large to run |
| **Walk-forward** | Do these parameters hold up *out of sample*? `robustness_ratio` is how much of the optimised result survived on data the optimiser never saw |
| **Monte Carlo** | How much of this equity curve was trade ordering? Size risk from the 5th percentile, not the median |

A grid sweep reports the best parameters *in hindsight* — the number
most likely to be curve-fitted. Walk-forward is the check on it.

---

## 🧰 Running the Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite runs anywhere, including Linux and CI: `tests/mt5_stub.py`
stands in for the Windows-only `MetaTrader5` package, so no terminal or
broker account is needed. Warnings are errors.

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
├── bot.py               # Entry point — interactive CLI & single-bot mode
├── dashboard.py         # Flask web dashboard (REST API + SPA)
├── wsgi.py              # WSGI entry point (gunicorn / waitress)
├── config.py            # Loads .env and exposes the Config class
│
│   # ── Core ───────────────────────────────────────────────────────
├── mt5_connector.py     # MT5 connection + broker symbol resolution
├── market_data.py       # Candle, tick and symbol-info retrieval
├── session_manager.py   # Forex session / market-open calendar
├── order_manager.py     # Trade execution (open/close/modify/brackets)
├── risk_manager.py      # Pre-trade gates, lot sizing, exposure
├── multi_bot.py         # Thread-per-bot fleet engine
│
│   # ── Shared rules (one definition, every engine) ────────────────
├── instruments.py       # Symbol families + pip arithmetic
├── trade_manager.py     # Stops, breakeven, time-stop, cooldown, pause
├── backtest_core.py     # The simulation kernel + performance stats
├── web_security.py      # Dashboard auth, CSRF and safe bind
│
│   # ── Analysis ───────────────────────────────────────────────────
├── backtester.py        # Backtest engine + SQLite results store
├── optimizer.py         # Grid search + genetic optimiser
├── validation.py        # Genetic / walk-forward / Monte Carlo runners
├── performance.py       # Trade metrics and CSV export
├── logic_version.py     # Fingerprints code+settings onto each trade
│
│   # ── Subsystems ─────────────────────────────────────────────────
├── strategies/          # BaseStrategy ABC, registry, 14 strategies
├── automation/          # Trigger → condition → action rule engine
├── news_intelligence/   # RSS/API ingestion + Claude sentiment analysis
├── cloud/               # AWS spot-fleet distributed optimisation
├── templates/           # dashboard.html (single-page UI)
│
├── tests/               # pytest suite + MetaTrader5 stub
├── pyproject.toml       # Packaging and tool config
├── pytest.ini           # Test configuration
├── .env                 # Your local configuration (git-ignored)
├── .env.example         # Template configuration
└── requirements.txt     # Python dependencies
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
