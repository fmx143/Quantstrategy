<h1 align="center">Quantstrategy 📉</h1>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3" />
  <img src="https://img.shields.io/badge/backtrader-event--driven-2b8a3e?style=for-the-badge" alt="backtrader" />
  <img src="https://img.shields.io/badge/vectorbt-vectorized-6f42c1?style=for-the-badge" alt="vectorbt" />
  <img src="https://img.shields.io/badge/Optuna-optimization-1f6feb?style=for-the-badge" alt="Optuna" />
  <img src="https://img.shields.io/badge/MetaTrader%205-live-e67e22?style=for-the-badge" alt="MetaTrader 5" />
</div>

<div align="center">
  <br />
  <img src="https://img.shields.io/badge/status-WIP%20research%20sandbox-yellow.svg" alt="WIP research sandbox" />
  <img src="https://img.shields.io/badge/asset%20class-Forex-lightgrey.svg" alt="Forex" />
  <br />
  <br />
  <i>A personal research sandbox for algorithmic Forex trading strategies.</i>
  <br />
  <i>Full lifecycle: data cleaning → backtesting → Bayesian optimization → live/demo execution.</i>
  <br />
  <i>Mainly EUR/USD and GBP/USD, from daily bars down to 1-minute scalping.</i>
  <br />
</div>

<hr />

### In plain terms — what is this?

**Backtesting** is replaying a trading strategy over past market data to see how it *would* have
performed — no money at risk. This repo is where strategies are hand-coded on currency pairs,
backtested, hyperparameter-tuned, and (for the ones that survive) wired up to a broker for
demo/live execution. Strategies here are **classical technical rules** (moving averages,
Bollinger bands, ADX/RSI filters, VWAP) — **no machine learning**.

**Who is it for?** A **personal, single-developer** research repo. It's a working sandbox, not a
packaged product — expect scripts you run directly rather than a polished CLI.

> ⚠️ **WIP sandbox.** No automated tests yet; some files carry hardcoded local paths. One
> strategy (engulfing) is explicitly marked by the author as unprofitable and kept only for
> reference. Treat everything here as research in progress.

> 🔐 **Secrets stay out of git.** Broker credentials belong in an **untracked** `apy.py` (or a
> `.env`), never committed. Do not paste real logins/passwords into any tracked file. If a
> credential has ever been committed, rotate it and add the file to `.gitignore`.

---

## 🧭 Table of contents

1. [✨ What it does](#-what-it-does)
2. [🔁 How it works](#-how-it-works)
3. [🏗️ Repository layout](#-repository-layout)
4. [📋 Prerequisites](#-prerequisites)
5. [⚙️ Setup](#-setup)
6. [⌨️ Running things](#-running-things)
7. [📈 Strategies](#-strategies)
8. [🔬 Optimization](#-optimization)
9. [🔴 Live / demo trading (MT5)](#-live--demo-trading-mt5)

---

## ✨ What it does

- 🧹 **Cleans** raw historical CSVs (Tickstory / ForexSB) — normalizes OHLC headers, reorders
  columns, writes `cleaned_*.csv`.
- 🧪 **Backtests** rule-based strategies with **backtrader** (event-driven) — bracket orders,
  stops, and analyzers.
- 🔬 **Optimizes** parameters with **vectorbt** + **Optuna** — multi-objective Bayesian search
  (Sharpe, win rate, drawdown…), with results viewable in the **Optuna Dashboard**.
- 🔴 **Executes** on a broker via **MetaTrader 5** — pulls live rates, builds orders, and runs a
  live scalping loop (order sending is partly gated/commented while under test).

---

## 🔁 How it works

The typical path from raw data to a running strategy:

```
raw CSV  →  cleaned CSV  →  strategy (backtrader)  →  optimization  →  live / demo
(Tickstory   (clean_data)   (backtest + analyzers)   (vectorbt +      (MetaTrader 5)
 / ForexSB)                                           Optuna)
```

1. **Clean.** Point `clean_data.py` at a raw export; it standardizes the OHLC layout and saves a
   `cleaned_*.csv`.
2. **Backtest.** Run a strategy file (e.g. `strategy.py`, `Strategy_EMA.py`) through backtrader's
   Cerebro to get trades and performance analyzers.
3. **Optimize.** Feed the strategy into an `Opt_Strategy_*.py` script — Optuna searches the
   parameter space, persists studies to SQLite, and you inspect the Pareto front in the dashboard.
4. **Go live.** The MT5 scripts connect to a (demo) broker account, pull rates, and place orders
   on a schedule.

---

## 🏗️ Repository layout

A **flat repo** (no packages) — files grouped here by role:

```
Quantstrategy/
  # Data layer
  apy.py                 config/secrets — file paths + MT5 login  (KEEP UNTRACKED)
  clean_data.py          interactive CSV cleaner → cleaned_*.csv
  clean_data_wrangler.ipynb   notebook version of the data wrangling

  # Backtesting strategies (backtrader)
  strategy.py            Bollinger breakout + re-entry, ADX>30 filter, bracket orders
  Strategy_EMA.py        triple-EMA crossover trend strategy
  Strategy_EMA copy.py   duplicate/scratch copy of the above
  Strategy_engulfing.py  engulfing-candle pattern — experimental, marked unprofitable

  # Optimization (vectorbt + Optuna)
  Opt_Strategy_EMA.py    multi-objective Optuna study over the triple-EMA strategy
  Opt_Strategy_VWAP.py   same framework over a custom VWAP/VWMA-bands + ATR strategy

  # Live / demo trading (MetaTrader 5)
  mt5_ema_crossover.py   EMA 8/21/200 crossover on MT5 (order sends gated while testing)
  Scalper_GG.py          multi-indicator scalper (ADX/RSI/MA/ATR), ported from an MQL4 EA
  Scalper_GG_MT5.py      MT5 variant of the scalper (adds credential import + docs)
```

---

## 📋 Prerequisites

- 🐍 **Python 3** with a virtual environment.
- 📦 Dependencies are pinned in **`requirements.txt`**: `numpy`, `pandas`, `matplotlib`,
  `yfinance`, `backtrader`, `vectorbt`, `optuna`, `optuna-dashboard`, `TA-Lib`.
- 🧱 **TA-Lib** needs its native C library installed first (OS-level), before `pip install TA-Lib`.
- 🖥️ **MetaTrader 5** terminal installed and logged in (for the live scripts). The `MetaTrader5`
  package is Windows-only and is listed as a comment in `requirements.txt` — install it
  separately on a Windows machine.

---

## ⚙️ Setup

```bash
cd Quantstrategy
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# TA-Lib needs its native C library installed first (OS-level), then:
pip install -r requirements.txt

# Windows only, for the live MT5 scripts (see comment in requirements.txt):
pip install MetaTrader5
```

> 🔐 Create your own **untracked** `apy.py` with your data paths and MT5 login. Never commit it.

---

## ⌨️ Running things

Most scripts execute their work on run (module-level `cerebro.run()`, `study.optimize()`, or a
live loop) — there's no unified CLI. Run them directly:

```bash
python clean_data.py            # clean a raw CSV → cleaned_*.csv
python strategy.py              # backtest the Bollinger strategy (backtrader)
python Opt_Strategy_EMA.py      # run the Optuna study for the triple-EMA strategy
```

---

## 📈 Strategies

| Strategy | File(s) | Idea |
|----------|---------|------|
| Bollinger breakout | `strategy.py` | Band breakout + re-entry, ADX>30 trend filter, bracket orders |
| Triple-EMA crossover | `Strategy_EMA.py` | Fast/mid/slow EMA trend-following (uses Optuna-tuned params) |
| EMA 8/21/200 (live) | `mt5_ema_crossover.py` | Crossover with fixed SL/TP, wired to MT5 |
| VWAP/VWMA bands | `Opt_Strategy_VWAP.py` | Custom VWAP/VWMA bands + ATR stops |
| Scalper-GG-14 | `Scalper_GG*.py` | ADX/RSI/MA/ATR scalper with trailing stops (ported from MQL4) |
| Engulfing pattern | `Strategy_engulfing.py` | ⚠️ Experimental — marked unprofitable, kept for reference |

**Risk management** across strategies: bracket orders, fixed-pip stop-losses, reward:risk ratios,
ATR-based stops, and percent-of-equity position sizing.

---

## 🔬 Optimization

The `Opt_Strategy_*.py` scripts run **multi-objective Bayesian optimization** with Optuna over
**vectorbt** backtests — optimizing across Sharpe, win rate and drawdown with a Pareto-front
analysis. Studies persist to SQLite; inspect them with the dashboard:

```bash
optuna-dashboard sqlite:///path/to/your_study.db
```

---

## 🔴 Live / demo trading (MT5)

The MetaTrader 5 scripts connect to a broker **demo** account, pull rates, and place orders. The
scalper (`Scalper_GG_MT5.py`) runs a live loop with spread checks, dynamic lot sizing and trailing
stops. Some order-send calls are intentionally gated/commented while the logic is under test.

> 🔐 Reminder: MT5 login/password/server go in your **untracked** `apy.py` only. Rotate any
> credential that has previously been committed.
